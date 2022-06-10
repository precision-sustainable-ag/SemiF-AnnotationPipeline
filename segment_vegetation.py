from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import rank
from skimage.measure import label
from skimage.morphology import disk
from skimage.segmentation import watershed
from tqdm import tqdm

from semif_utils.datasets import CUTOUT_PROPS, BatchMetadata, Cutout
from semif_utils.mongo_utils import Connect, to_db
from semif_utils.synth_utils import save_dataclass_json
from semif_utils.utils import (apply_mask, clear_border, crop_cutouts,
                               dilate_erode, get_site_id, get_upload_datetime,
                               make_exg, reduce_holes, rescale_bbox)


class SegmentVegetation:

    def __init__(self, db, cfg: DictConfig) -> None:
        """_summary_

        """
        self.batchdir = Path(cfg.data.batchdir)
        self.data_root = Path(cfg.data.developeddir)
        self.batch_id = self.batchdir.name
        self.metadata = self.batchdir / "metadata"
        self.autosfm = self.batchdir / "autosfm"
        self.cutoutdir = Path(cfg.data.cutoutdir, self.batch_id)
        self.labels = [x for x in (self.metadata).glob("*.json")]

        if not self.cutoutdir.exists():
            self.cutoutdir.mkdir(parents=True, exist_ok=True)

        self.clear_border = cfg.segment.clear_border
        self.site_id = self.batchdir.stem.split("_")[0]
        self.vi = cfg.segment.vi
        self.class_algorithm = cfg.segment.class_algorithm

        self.db = db

        self.cutout_pipeline()

    def process_domain(self, img):
        """First cutout processing step of the full image."""
        v_index = getattr(VegetationIndex(), self.vi)
        vi = v_index(img)
        th_vi = self.thresh_vi(vi)
        # Get classified mask
        clalgo = getattr(ClassifyMask(), self.class_algorithm)
        mask = clalgo(th_vi)
        return mask

    def process_cutout(self, cutout):
        """Second cutout processing step.
        """
        vi = make_exg(cutout, thresh=True)
        thresh_vi = np.where(vi <= 0, 0, vi)
        thresh_vi = np.where((thresh_vi > 10) & (thresh_vi < 100),
                             thresh_vi * 5, thresh_vi)
        markers = rank.gradient(thresh_vi, disk(1)) < 12
        markers = ndi.label(markers)[0]
        gradient = rank.gradient(thresh_vi, disk(10))
        # process the watershed
        labels = watershed(gradient, markers)
        seg1 = label(labels <= 0)
        labels = label2rgb(seg1, image=thresh_vi, bg_label=0) * 2.5
        lb = rescale_intensity(labels, in_range=(0, 1), out_range=(0, 1))
        mask = np.where(lb <= 0.3, 0., 1)
        dil_erod_mask = dilate_erode(mask[:, :, 0],
                                     kernel_size=3,
                                     dil_iters=5,
                                     eros_iters=6,
                                     hole_fill=True)
        reduced_mask = reduce_holes(dil_erod_mask * 255) * 255
        return reduced_mask

    def cutout_pipeline(self):
        """ Main Processing pipeline. Reads images from list of labels in
            labeldir,             
        """
        cutouts = []
        for label_set in tqdm(self.labels, desc="Segmenting Vegetation"):
            imgdata = self.get_image_meta(label_set)
            dt = datetime.strptime(imgdata.exif_meta.DateTime,
                                   "%Y:%m:%d %H:%M:%S")
            # Call image array
            rgb_array = imgdata.array
            ## Process on images by individual bbox detection
            cutout_num = 0
            cutout_ids = []
            bboxes = imgdata.bboxes
            for box in bboxes:
                # if not box.is_primary:
                # continue
                # Only scale the box that will be used for the cutout
                image_width = imgdata.fullres_width
                image_height = imgdata.fullres_height
                scale = [image_width, image_height]
                box = rescale_bbox(box, scale)
                x1, y1 = box.local_coordinates["top_left"]
                x2, y2 = box.local_coordinates["bottom_right"]
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)

                # Crop image to bbox
                rgb_crop = rgb_array[y1:y2, x1:x2]
                mask = self.process_domain(rgb_crop)
                # Clear borders
                if self.clear_border:
                    mask = clear_border(mask) * 255
                # Separate components
                list_cutouts_masks = self.seperate_components(mask)
                # Create RGB cutout for second round of processing
                # cutout_0 = apply_mask(rgb_crop, mask, "black")
                # Second round of processing
                for cut_mask in list_cutouts_masks:
                    preproc_cutout = apply_mask(rgb_crop, cut_mask, "black")
                    mask2 = self.process_cutout(preproc_cutout)
                    if np.sum(mask2) == 0:
                        continue

                    new_cutout = apply_mask(rgb_crop, mask2, "black")
                    new_cropped_cutout = crop_cutouts(new_cutout)
                    # Get regionprops
                    cutprops = GenCutoutProps(mask2).to_dataclass()
                    if type(cutprops.area) is not list and cutprops.area < 500:
                        continue
                    cutout_path = self.save_cutout(new_cropped_cutout,
                                                   Path(imgdata.image_path),
                                                   cutout_num)
                    cutrelpath = Path(cutout_path).relative_to(
                        Path(cutout_path).parent.parent)
                    # Save cutout ids for DB
                    cutout_ids.append(cutout_path.stem)

                    # Create dataclass
                    cutout = Cutout(data_root=self.data_root.name,
                                    batch_id=self.batch_id,
                                    site_id=self.site_id,
                                    cutout_num=cutout_num,
                                    cutout_path=str(cutrelpath),
                                    image_id=imgdata.image_id,
                                    cutout_props=cutprops,
                                    is_primary=box.is_primary,
                                    datetime=dt)
                    cutouts.append(cutout)
                    cutout_num += 1
                    # Move cutout to database
                    cutout = asdict(cutout)
                    if self.db is not None:
                        to_db(self.db, "Cutouts", cutout)
                    self.save_cutout_json(cutout, cutout_path)

            # To database
            if self.db is not None:
                imgdata.cutout_ids = cutout_ids
                # imgdata.cutouts = cutouts
                imgdata.image_path = str(
                    Path(imgdata.image_path).relative_to(
                        Path(imgdata.image_path).parent.parent))
                imgdata = asdict(imgdata)
                to_db(self.db, "Images", imgdata)
            # To json
            imgdata = asdict(imgdata)
            jsparents = Path(self.data_root, self.batch_id, "metadata")
            jsonpath = Path(jsparents, imgdata["image_id"] + ".json")
            save_dataclass_json(imgdata, jsonpath)


def main(cfg: DictConfig) -> None:
    # Create batch metadata
    data_root = str(Path(cfg.data.developeddir).name)
    batch_id = str(Path(cfg.data.batchdir).name)
    site_id = get_site_id(cfg.data.batchdir)
    upload_datetime = get_upload_datetime(cfg.data.batchdir)

    batch = BatchMetadata(data_root=data_root,
                          batch_id=batch_id,
                          site_id=site_id,
                          upload_datetime=upload_datetime)
    # Save To json
    batch = asdict(batch)
    jsparents = Path(batch["blob_root"], data_root, batch_id)
    jsonpath = Path(jsparents, batch_id + ".json")
    save_dataclass_json(batch, jsonpath)

    # Connect to database
    if cfg.general.save_to_database:
        db = Connect.get_connection()
        db = getattr(db, cfg.general.db)
        # To DB
        to_db(db, "Batches", batch)

    else:
        db = None

    # Run pipeline
    vegseg = SegmentVegetation(db, cfg)

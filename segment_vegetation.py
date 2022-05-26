import json
from dataclasses import asdict, make_dataclass
from datetime import datetime
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
from dacite import Config, from_dict
from omegaconf import DictConfig
from scipy import ndimage as ndi
from skimage import measure
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import rank
from skimage.measure import label
from skimage.morphology import disk
from skimage.segmentation import watershed
from tqdm import tqdm

from semif_utils.datasets import (CUTOUT_PROPS, BatchMetadata, BBoxMetadata,
                                  Cutout, CutoutProps, ImageData)
from semif_utils.mongo_utils import Connect, _id_query, to_db
from semif_utils.synth_utils import save_dataclass_json
from semif_utils.utils import (apply_mask, clear_border, crop_cutouts,
                               dilate_erode, get_site_id, get_upload_datetime,
                               make_exg, make_exg_minus_exr, make_exr,
                               make_kmeans, make_ndi, otsu_thresh, parse_dict,
                               reduce_holes)


class VegetationIndex:

    def exg(self, img):
        exg_vi = make_exg(img, thresh=True)
        return exg_vi

    def exr(self, img):
        exr_vi = make_exr(img)
        return exr_vi

    def exg_minus_exr(self, img):
        gmr_vi = make_exg_minus_exr(img)
        return gmr_vi

    def ndi(self, img):
        ndi_vi = make_ndi(img)
        return ndi_vi


class ClassifyMask:

    def otsu(self, vi):
        # Otsu's thresh
        vi_mask = otsu_thresh(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255
        return reduce_holes_mask

    def kmeans(self, vi):
        vi_mask = make_kmeans(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255

        return reduce_holes_mask


class GenCutoutProps:

    def __init__(self, mask):
        """ Generate cutout properties and returns them as a dataclass.
        """
        self.mask = mask

    def from_regprops_table(self, connectivity=2):
        """Generates list of region properties for each cutout mask
        """
        labels = measure.label(self.mask, connectivity=connectivity)
        props = [measure.regionprops_table(labels, properties=CUTOUT_PROPS)]
        # Parse regionprops_table
        nprops = [parse_dict(d) for d in props][0]
        return nprops

    def to_dataclass(self):
        table = self.from_regprops_table()
        cutout_props = from_dict(data_class=CutoutProps, data=table)
        return cutout_props


class SegmentVegetation:

    def __init__(self, db, cfg: DictConfig) -> None:
        """_summary_

        """
        self.batchdir = Path(cfg.data.batchdir)
        self.data_root = Path(cfg.data.developeddir)
        self.batch_id = self.batchdir.name
        self.metadata = self.batchdir / "metadata"
        self.autosfm = self.batchdir / "autosfm"
        self.imagedir = self.batchdir / "images"
        self.cutoutdir = Path(cfg.data.cutoutdir, self.batch_id)
        self.detectioncsv = self.autosfm / "detections.csv"

        self.labels = [x for x in (self.metadata).glob("*.json")]

        if not self.cutoutdir.exists():
            self.cutoutdir.mkdir(parents=True, exist_ok=True)

        self.clear_border = cfg.segment.clear_border
        self.sitedir, self.site_id = self.get_siteinfo()
        self.vi = cfg.segment.vi
        self.class_algorithm = cfg.segment.class_algorithm

        self.db = db
        self.cutout_pipeline()

    def get_siteinfo(self):
        """Uses image directory to gather site specific information.
            Agnostic to what relative path structure is used. As in it does
            not matter whether parent directory of images is sitedir or "developed". 

        Returns:
            sitedir: developed image parent directory name
            site_id: state id takend from sitedir
        """
        states = ['TX', 'NC', 'MD']
        sitedir = [p for st in states for p in self.imagedir.parts
                   if st in p][0]
        site_id = [st for st in states if st in sitedir][0]
        return sitedir, site_id

    def save_cutout(self, cutout, imgpath, cutout_num):
        """Saves cutouts to cutoutdir.

        Args:
            cutout (np.ndarray): final processed cutout array in RGB
            imgpath (str): filename of the original developed image
            cutout_num (int): unique number per image

        Returns:
            cutoutpath: used for saving information to mongodb
        """
        fname = f"{imgpath.stem}_{cutout_num}.png"
        cutout_path = Path(self.cutoutdir, fname)
        # return cutout_path
        cv2.imwrite(str(cutout_path), cv2.cvtColor(cutout, cv2.COLOR_RGB2BGRA))
        return cutout_path

    def save_cutout_json(self, cutout, cutoutpath):
        cutout_json_path = Path(
            Path(cutoutpath).parent,
            Path(cutoutpath).stem + ".json")
        with open(cutout_json_path, 'w') as j:
            json.dump(cutout, j, indent=4, default=str)

    def seperate_components(self, mask):
        """ Seperates multiple unconnected components in a mask
            for seperate processing. 
        """
        # Store individual plant components in a list
        mask = mask.astype(np.uint8)
        nb_components, output, _, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        # Remove background component
        nb_components = nb_components - 1
        list_filtered_masks = []
        for i in range(0, nb_components):
            filtered_mask = np.zeros((output.shape))
            filtered_mask[output == i + 1] = 255
            list_filtered_masks.append(filtered_mask)
        return list_filtered_masks

    def thresh_vi(self, vi, low=20, upper=100, sigma=2):
        """
        Args:
            vi (np.ndarray): vegetation index single channel
            low (int, optional): lower end of vi threshold. Defaults to 20.
            upper (int, optional): upper end of vi threshold. Defaults to 100.
            sigma (int, optional): multiplication factor applied to range within
                                   "low" and "upper". Defaults to 2.
        """
        thresh_vi = np.where(vi <= 0, 0, vi)
        thresh_vi = np.where((thresh_vi > low) & (thresh_vi < upper),
                             thresh_vi * sigma, thresh_vi)
        return thresh_vi

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

    def get_bboxmeta(self, path):
        with open(path) as f:
            j = json.load(f)

            bbox_meta = BBoxMetadata(**j)
        return bbox_meta

    def get_image_meta(self, path):
        with open(path) as f:
            j = json.load(f)
            imgdata = from_dict(data_class=ImageData,
                                data=j,
                                config=Config(check_types=False))
        return imgdata

    def cutout_pipeline(self):
        """ Main Processing pipeline. Reads images from list of labels in
            labeldir,             
        """
        cutouts = []
        for label_set in tqdm(self.labels,
                              desc="Segmenting Vegetation",
                              colour="green"):
            imgdata = self.get_image_meta(label_set)
            dt = datetime.strptime(imgdata.exif_meta.DateTime,
                                   "%Y:%m:%d %H:%M:%S")
            # Call image array
            rgb_array = imgdata.array
            ## Process on images by individual bbox detection
            cutout_num = 0
            cutout_ids = []
            bboxes = imgdata.bboxes
            for box in tqdm(bboxes,
                            leave=False,
                            colour="#6dbc90",
                            desc="Generating Cutouts"):

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
                cutout_0 = apply_mask(rgb_crop, mask, "black")
                # Second round of processing
                for cut_mask in list_cutouts_masks:
                    preproc_cutout = apply_mask(cutout_0, cut_mask, "black")
                    mask2 = self.process_cutout(preproc_cutout)
                    if np.sum(mask2) == 0:
                        continue

                    new_cutout = apply_mask(preproc_cutout, mask2, "black")
                    new_cropped_cutout = crop_cutouts(new_cutout)
                    # Get regionprops
                    cutprops = GenCutoutProps(mask2).to_dataclass()
                    if type(cutprops.area
                            ) is not list and cutprops.area < 3000:
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
    # Connect to database
    if cfg.general.save_to_database:
        db = Connect.get_connection()
        db = getattr(db, cfg.general.db)
        batch = asdict(batch)
        # To DB
        to_db(db, "Batches", batch)
        # Save To json
        jsparents = Path(batch["blob_root"], data_root, batch_id)
        jsonpath = Path(jsparents, batch_id + ".json")
        save_dataclass_json(batch, jsonpath)

    else:
        db = None
    # Run pipeline
    vegseg = SegmentVegetation(db, cfg)

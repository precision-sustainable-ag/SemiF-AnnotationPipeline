import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

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

from semif_utils.datasets import (CUTOUT_PROPS, BatchMetadata, Cutout,
                                  CutoutProps, ImageData)
from semif_utils.utils import (apply_mask, clear_border, crop_cutouts,
                               dilate_erode, get_upload_datetime, make_exg,
                               make_exg_minus_exr, make_exr, make_kmeans,
                               make_ndi, otsu_thresh, parse_dict, reduce_holes,
                               rescale_bbox)


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
        # Used to check schema
        table = self.from_regprops_table()
        cutout_props = from_dict(data_class=CutoutProps, data=table)
        return cutout_props


class SegmentVegetation:

    def __init__(self, cfg: DictConfig) -> None:
        """_summary_

        """
        self.blob_home = Path(cfg.blob_storage.blobhome)
        self.cutout_dir = Path(cfg.data.cutoutdir)
        self.batchdir = Path(cfg.data.batchdir)
        self.batch_id = self.batchdir.name
        self.metadata = self.batchdir / "metadata"
        self.cutout_batch_dir = self.cutout_dir / self.batch_id
        self.labels = [x for x in (self.metadata).glob("*.json")]
        self.clear_border = cfg.segment.clear_border
        self.vi = cfg.segment.vi
        self.class_algorithm = cfg.segment.class_algorithm

        if not self.cutout_batch_dir.exists():
            self.cutout_batch_dir.mkdir(parents=True, exist_ok=True)

        self.cutout_pipeline()

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
                if not box.is_primary:
                    continue
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

                    if type(cutprops.area) is not list and cutprops.area < 500:
                        continue

                    # Create dataclass
                    cutout = Cutout(blob_home=self.blob_home.name,
                                    data_root=self.cutout_dir.name,
                                    batch_id=self.batch_id,
                                    image_id=imgdata.image_id,
                                    cutout_num=cutout_num,
                                    datetime=dt,
                                    cutout_props=asdict(cutprops)
                                    #cutout_species=species_cls
                                    )
                    cutout.save_cutout(new_cropped_cutout)
                    cutout.save_config(self.cutout_dir)

                    cutout_ids.append(cutout.cutout_id)
                    cutout_num += 1

            # To json
            imgdata.cutout_ids = cutout_ids
            imgdata.save_config(self.metadata)


def main(cfg: DictConfig) -> None:
    # Create batch metadata
    blob_home = Path(cfg.blob_storage.blobhome)
    data_root = Path(cfg.data.developeddir)
    batch_dir = Path(cfg.data.batchdir)
    upload_datetime = get_upload_datetime(cfg.data.batchdir)

    BatchMetadata(blob_home=blob_home.name,
                  data_root=data_root.name,
                  batch_id=batch_dir.name,
                  upload_datetime=upload_datetime).save_config()

    # Run pipeline
    SegmentVegetation(cfg)

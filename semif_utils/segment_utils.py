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
                               reduce_holes, rescale_bbox)

################################################################
########################## SETUP ###############################
################################################################


def get_siteinfo(imagedir):
    """Uses image directory to gather site specific information.
            Agnostic to what relative path structure is used. As in it does
            not matter whether parent directory of images is sitedir or "developed". 

        Returns:
            sitedir: developed image parent directory name
            site_id: state id takend from sitedir
        """
    imagedir = Path(imagedir)
    states = ['TX', 'NC', 'MD']
    sitedir = [p for st in states for p in imagedir.parts if st in p][0]
    site_id = [st for st in states if st in sitedir][0]
    return sitedir, site_id


def get_image_meta(path):
    with open(path) as f:
        j = json.load(f)
        imgdata = from_dict(data_class=ImageData,
                            data=j,
                            config=Config(check_types=False))
    return imgdata


def save_cutout_json(cutout, cutoutpath):
    cutout_json_path = Path(
        Path(cutoutpath).parent,
        Path(cutoutpath).stem + ".json")
    with open(cutout_json_path, 'w') as j:
        json.dump(cutout, j, indent=4, default=str)


def get_bboxmeta(path):
    with open(path) as f:
        j = json.load(f)

        bbox_meta = BBoxMetadata(**j)
    return bbox_meta


################################################################
######################## PROCESSING ############################
################################################################


class GenCutoutProps:

    def __init__(self, mask):
        """ Generate cutout properties and returns them as a dataclass.
        """
        self.mask = mask

    def from_regprops_table(mask, connectivity=2):
        """Generates list of region properties for each cutout mask
        """
        labels = measure.label(mask, connectivity=connectivity)
        props = [measure.regionprops_table(labels, properties=CUTOUT_PROPS)]
        # Parse regionprops_table
        nprops = [parse_dict(d) for d in props][0]
        return nprops

    def to_dataclass(self):
        table = self.from_regprops_table()
        cutout_props = from_dict(data_class=CutoutProps, data=table)
        return cutout_props


class ClassifyMask:

    def otsu(vi):
        # Otsu's thresh
        vi_mask = otsu_thresh(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255
        return reduce_holes_mask

    def kmeans(vi):
        vi_mask = make_kmeans(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255

        return reduce_holes_mask


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


def thresh_vi(vi, low=20, upper=100, sigma=2):
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


def seperate_components(mask):
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

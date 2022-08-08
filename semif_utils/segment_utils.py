import json
from difflib import get_close_matches
from pathlib import Path

import cv2
import numpy as np
from dacite import Config, from_dict
from skimage import measure

from semif_utils.datasets import CUTOUT_PROPS, CutoutProps, ImageData
from semif_utils.utils import (get_watershed, make_exg, make_exg_minus_exr,
                               make_exr, make_kmeans, make_ndi, multiple_otsu,
                               otsu_thresh, parse_dict, read_json, apply_mask,
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
    states = ["TX", "NC", "MD"]
    sitedir = [p for st in states for p in imagedir.parts if st in p][0]
    site_id = [st for st in states if st in sitedir][0]
    return sitedir, site_id


def save_cutout_json(cutout, cutoutpath):
    cutout_json_path = Path(
        Path(cutoutpath).parent,
        Path(cutoutpath).stem + ".json")
    with open(cutout_json_path, "w") as j:
        json.dump(cutout, j, indent=4, default=str)


def get_species_info(path, cls, default_species="grass"):
    with open(path) as f:
        spec_info = json.load(f)
        spec_info = (spec_info["species"][cls] if cls
                     in spec_info["species"].keys() else default_species)
    return spec_info


################################################################
######################## PROCESSING ############################
################################################################


class GenCutoutProps:

    def __init__(self, img, mask):
        """Generate cutout properties and returns them as a dataclass."""
        self.img = img
        self.mask = mask
        self.green_thresh = 1000 # TODO change to normalized based on size of image
        self.green_sum = int(np.sum(self.green_mask()))
        self.is_green = True if self.green_sum > self.green_thresh else False
        
    def green_mask(self):
        """Returns binary mask if values are within certain "green" HSV range."""
        cutout = apply_mask(self.img, self.mask, "black")
        hsv = cv2.cvtColor(cutout, cv2.COLOR_RGB2HSV)
        lower = np.array([40, 70, 120])
        upper = np.array([90, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        hsv_mask = np.where(hsv_mask == 255, 1, 0)
        return hsv_mask

    def from_regprops_table(self, connectivity=2):
        """Generates list of region properties for each cutout mask"""
        labels = measure.label(self.mask, connectivity=connectivity)
        props = [measure.regionprops_table(labels, properties=CUTOUT_PROPS)]
        # Parse regionprops_table
        nprops = [parse_dict(d) for d in props][0]
        nprops["is_green"] = self.is_green
        nprops["green_sum"] = self.green_sum
        return nprops

    def to_dataclass(self):
        table = self.from_regprops_table()
        cutout_props = from_dict(data_class=CutoutProps, data=table)
        return cutout_props

class SegmentMask:

    def otsu(self, vi):
        # Otsu's thresh
        vi_mask = otsu_thresh(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255
        return reduce_holes_mask

    def kmeans(self, vi):
        vi_mask = make_kmeans(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255
        return reduce_holes_mask

    def watershed(self, vi):
        vi_mask = get_watershed(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255
        return reduce_holes_mask

    def multi_otsu(self, vi):
        vi_mask = multiple_otsu(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255
        return reduce_holes_mask


class VegetationIndex:

    def exg(self, img, thresh=0):
        exg_vi = make_exg(img, thresh=0)
        return exg_vi

    def exr(self, img, thresh=0):
        exr_vi = make_exr(img, thresh=0)
        return exr_vi

    def exg_minus_exr(self, img, thresh=0):
        gmr_vi = make_exg_minus_exr(img, thresh=0)
        return gmr_vi

    def ndi(self, img, thresh=0):
        ndi_vi = make_ndi(img, thresh=0)
        return ndi_vi


def prep_bbox(box, scale):
    box = rescale_bbox(box, scale)
    x1, y1 = box.local_coordinates["top_left"]
    x2, y2 = box.local_coordinates["bottom_right"]
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    return box, x1, y1, x2, y2


def species_info(speciesjson, df, default_species="grass"):
    """Compares entries in user provided species map csv with those from a common
       species data model (json). Uses 'get_close_matches' to get the best match.
       Is meant to create flexibility in how users are defining "species" in their
       maps.s

    Args:
        speciesjson (str): json of common specie data model (data/species.json)
        species_mapcsv (str): csv of user provided species map by row (data/developed/[batch_id]/autosfm/specie_map.csv)
        default_species (str, optional): Defaults to "grass". For testing purposes, if species column is left blank,
        or if 'get_close_matches' returns an empty list.

    Returns:
        updated_species_map: dictionary of "row:common name" key-value pairs
    """

    # get species map dictionary unique to batch

    spec_map = df.set_index("row").T.to_dict("records")[0]
    spec_map = eval(repr(spec_map).lower())
    spec_map_copy = spec_map.copy()

    # get species common names
    species_data = read_json(speciesjson)
    common_names = []
    spec_idx = species_data["species"].keys()
    common_name_list = [
        species_data["species"][x]["common_name"] for x in spec_idx
    ]
    # Get copy species map to update
    update_specmap = spec_map.copy()

    # Compare each value in species map with common name list from species data
    spec_dict = spec_map_copy["species"]
    for row in spec_map:
        comm_name = spec_map[row]
        match = get_close_matches(comm_name, common_name_list, n=1)
        comm_match = match if match else default_species

        for x in spec_idx:
            if species_data["species"][x]["common_name"] == comm_match:
                species_data["species"][x]

    return update_specmap

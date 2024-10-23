import json
import logging
import random
from difflib import get_close_matches
from pathlib import Path

import cv2
import numpy as np
import numpy.ma as ma
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.stats import kurtosis, skew
from semif_utils.datasets import CUTOUT_PROPS, CutoutProps
from semif_utils.utils import (apply_mask, get_watershed,
                               make_exg, make_exg_minus_exr, make_exr,
                               make_gli, make_kmeans, make_ndi, multiple_otsu,
                               otsu_thresh, parse_dict, read_json,
                               reduce_holes, rescale_bbox)
from skimage import measure

log = logging.getLogger(__name__)

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
    cutout_json_path = Path(Path(cutoutpath).parent, Path(cutoutpath).stem + ".json")
    with open(cutout_json_path, "w") as j:
        json.dump(cutout, j, indent=4, default=str)


def load_speciesinfo(path):
    with open(path) as f:
        spec_info = json.load(f)
    return spec_info


def get_species_info(path, cls, default_species="plant"):

    spec_info = (
        spec_info["species"][cls]
        if cls in spec_info["species"].keys()
        else spec_info["species"][default_species]
    )
    return spec_info


def get_bboxarea(bbox):
    x1, y1, x2, y2 = bbox
    width = float(x2) - float(x1)
    length = float(y2) - float(y1)
    area = width * length
    return area


################################################################
######################## PROCESSING ############################
################################################################


class GenCutoutProps:
    def __init__(self, img, mask):
        """Generate cutout properties and returns them as a dataclass."""
        self.img = img  # RGB
        self.mask = mask
        self.cutout = apply_mask(self.img, self.mask, "black")
        self.green_thresh = 100  # TODO change to normalized based on size of image
        self.green_sum = int(np.sum(self.green_mask()))
        self.is_green = True if self.green_sum > self.green_thresh else False
        self.color_dist_tol = 12

    def get_blur_effect(self):
        # 0 for no blur, 1 for maximal blur
        blur_effect = measure.blur_effect(self.cutout, channel_axis=2)
        if np.isnan(blur_effect):
            blur_effect = None
        return blur_effect

    def exg_sum(self):
        sumexg = np.sum(make_exg(self.cutout, normalize=True))
        return sumexg

    def green_mask(self):
        """Returns binary mask if values are within certain "green" HSV range."""
        hsv = cv2.cvtColor(self.cutout, cv2.COLOR_RGB2HSV)
        lower = np.array([40, 70, 120])
        upper = np.array([90, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        hsv_mask = np.where(hsv_mask == 255, 1, 0)
        return hsv_mask

    def num_connected_components(self):
        if len(self.mask.shape) > 2:
            mask = self.mask[..., 0]
        else:
            mask = self.mask
        _, num = measure.label(mask, background=0, connectivity=2, return_num=True)
        return num

    def calc_ch_means(self):
        """Calculates the mean of each channel passed as


        Args:
            img (np.array): rgb input image

        Returns:
            list: list of means
        """
        r, g, b = cv2.split(self.img)

        r_mean = np.mean(r, dtype=np.float32)
        g_mean = np.mean(g, dtype=np.float32)
        b_mean = np.mean(b, dtype=np.float32)

        return [r_mean, g_mean, b_mean]

    def calc_ch_stds(self):
        """calculates standard deviation of each channel of the
        croptout array, not cutout.

        Returns:
            list: list of std for r,g,b in that order
        """
        r, g, b = cv2.split(self.img)

        r_std = np.std(r, dtype=np.float32)
        g_std = np.std(g, dtype=np.float32)
        b_std = np.std(b, dtype=np.float32)

        return [r_std, g_std, b_std]

    def descriptive_stats(self, rgb_img, ignore_zeros=False):
        rgb_img = rgb_img.astype(np.float64)

        if ignore_zeros:
            #     # Mask out zero values for descriptive stats
            rgb_img[rgb_img == [0, 0, 0]] = np.nan

        rgb_channels = cv2.split(rgb_img)
        str_channels = ["r", "g", "b"]
        desc_stats = dict()
        for img_c, str_c in zip(rgb_channels, str_channels):
            rec_key = f"channel_{str_c}"
            # dataframe describe automatically ignores nans
            c_df = pd.DataFrame(img_c.flatten(), columns=[rec_key]).describe()
            # scipy describe stats ignore nans
            c_scipy_desc = stats.describe(img_c.flatten(), nan_policy="omit")
            c_df.loc["variance"] = c_scipy_desc.variance
            c_df.loc["skewness"] = c_scipy_desc.skewness
            c_df.loc["kurtosis"] = c_scipy_desc.kurtosis
            desc_stats.update(c_df.to_dict())

        return desc_stats

    def analyze_image(self, rgb_image, ignore_zeros=False):
        # Split the image into individual channels
        r, g, b = cv2.split(rgb_image)

        all_zeros = not np.any(rgb_image)

        # Create a mask to identify non-black pixels
        mask = None

        if ignore_zeros:
            mask = cv2.bitwise_or(cv2.bitwise_or(b, g), r)
            #     # Mask out zero values for descriptive stats
        if all_zeros:
            all_zero_result = self.all_zero_props()
            return all_zero_result
        else:
            # Calculate the mean, count, standard deviation, minimum, and maximum values for each channel
            b_mean, b_std = cv2.meanStdDev(b, mask=mask)
            g_mean, g_std = cv2.meanStdDev(g, mask=mask)
            r_mean, r_std = cv2.meanStdDev(r, mask=mask)

            del rgb_image
            del b, g, r
            del mask

            rgb_mean = [float(r_mean[0][0]), float(g_mean[0][0]), float(b_mean[0][0])]
            rgb_std = [float(r_std[0][0]), float(g_std[0][0]), float(b_std[0][0])]
            return rgb_mean, rgb_std

    def all_zero_props(self):
        return {
            "exg_mean": None,
            "exg_std": None,
            "gli_mean": None,
            "gli_std": None,
            "channel_r": {
                "mean": None,
                "std": None,
                "skewness": None,
                "kurtosis": None,
                "variance": None,
            },
            "channel_g": {
                "mean": None,
                "std": None,
                "skewness": None,
                "kurtosis": None,
                "variance": None,
            },
            "channel_b": {
                "mean": None,
                "std": None,
                "skewness": None,
                "kurtosis": None,
                "variance": None,
            },
        }


    def from_regprops_table(self, connectivity=2):
        """Generates list of region properties for each cutout mask"""
        labels = measure.label(self.mask, connectivity=connectivity)
        props = [measure.regionprops_table(labels, properties=CUTOUT_PROPS)]
        # Parse regionprops_table
        nprops = [parse_dict(d) for d in props][0]
        nprops["green_sum"] = self.green_sum
        nprops["blur_effect"] = self.get_blur_effect()
        nprops["num_components"] = self.num_connected_components()

        rgb_mean, rgb_std = self.analyze_image(self.img)
        nprops["cropout_rgb_mean"] = rgb_mean
        nprops["cropout_rgb_std"] = rgb_std

        return nprops

    def to_dataclass(self):
        table = self.from_regprops_table()
        cutout_props = CutoutProps(**table)
        return cutout_props
    
    def to_regprops_table(self):
        table = self.from_regprops_table()
        return table


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


def prep_bbox(box, scale, save_changes=True):
    # box = rescale_bbox(box, scale, save_changes=save_changes)
    w, h = scale
    x1, y1 = box.local_coordinates["top_left"]
    x2, y2 = box.local_coordinates["bottom_right"]
    x1, y1 = int(x1 * w), int(y1 * h)
    x2, y2 = int(x2 * w), int(y2 * h)
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
    common_name_list = [species_data["species"][x]["common_name"] for x in spec_idx]
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


################################################################
######################## Colors ############################
################################################################


def get_random_color(pastel_factor=0.5):
    return [
        ((x + pastel_factor) / (1.0 + pastel_factor)) * 255
        for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]
    ]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def to_rgb(color):
    return [int(x) for x in color]


def generate_new_color(existing_colors, pastel_factor=0.5):
    """https://gist.github.com/adewes/5884820"""
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return to_rgb(best_color)

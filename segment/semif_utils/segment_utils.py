import json
import random

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage import measure

from semif_utils.utils import apply_mask, make_exg

def load_speciesinfo(path):
    with open(path) as f:
        spec_info = json.load(f)
    return spec_info


################################################################
######################## PROCESSING ############################
################################################################


class GenCutoutProps:
    def __init__(self, img, mask):
        """Generate cutout properties and returns them as a dataclass."""
        self.img = img  # RGB
        self.mask = mask
        self.cutout = apply_mask(self.img, self.mask, "black")
        # self.green_thresh = 100  # TODO change to normalized based on size of image
        # self.green_sum = int(np.sum(self.green_mask()))
        # self.is_green = True if self.green_sum > self.green_thresh else False
        # self.color_dist_tol = 12

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

        # Check if the image is completely black
        if not np.any(rgb_image):
            return self.all_zero_props()

        mask = None
        if ignore_zeros:
            # Create a binary mask where any nonzero pixel is marked as 255
            mask = cv2.bitwise_or(cv2.bitwise_or(b, g), r)
            mask = (mask > 0).astype(np.uint8) * 255

        # Compute mean and standard deviation for each channel using the mask if provided.
        b_mean, b_std = cv2.meanStdDev(b, mask=mask)
        g_mean, g_std = cv2.meanStdDev(g, mask=mask)
        r_mean, r_std = cv2.meanStdDev(r, mask=mask)

        # (The explicit deletion is optional, Python's GC will handle this.)
        del rgb_image, b, g, r, mask

        # Return the results in RGB order
        # Normalize the results by dividing by 255 so that values are in [0, 1].
        rgb_mean = [
            float(r_mean[0][0]) / 255,
            float(g_mean[0][0]) / 255,
            float(b_mean[0][0]) / 255,
        ]
        rgb_std = [
            float(r_std[0][0]) / 255,
            float(g_std[0][0]) / 255,
            float(b_std[0][0]) / 255,
        ]
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
        # labels = measure.label(self.mask, connectivity=connectivity)
        # props = [measure.regionprops_table(labels, properties=CUTOUT_PROPS)]
        # Parse regionprops_table
        # nprops = [parse_dict(d) for d in props][0]
        nprops = {}
        # nprops["green_sum"] = self.green_sum
        nprops["blur_effect"] = self.get_blur_effect()
        nprops["num_components"] = self.num_connected_components()
        rgb_mean, rgb_std = self.analyze_image(self.img)
        nprops["cropout_rgb_mean"] = rgb_mean
        nprops["cropout_rgb_std"] = rgb_std

        return nprops
    
    def to_regprops_table(self):
        table = self.from_regprops_table()
        return table


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

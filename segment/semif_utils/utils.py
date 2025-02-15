import copy
import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd

from PIL import Image
from scipy import ndimage
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import rank
from skimage.measure import label
from skimage.morphology import disk
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
from tqdm import tqdm

log = logging.getLogger(__name__)


def read_json(path):
    # Opening JSON file
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def parse_dict(props_tabl):
    """
    Used to parse regionprops table dictionary.
    Sums region props if multiple exists in one
    mask.
    """
    ndict = {}
    for key, val in props_tabl.items():
        key = key.replace("-", "") if "-" in key else key
        sum_val_entry = []
        if isinstance(val, np.ndarray) or val.shape[0] > 1:
            for i, v in enumerate(val):
                sum_val_entry.append(float(v))
            ndict[key] = np.sum(np.array(sum_val_entry))
        elif val.shape[0] == 0:
            ndict[key] = None
    return ndict


def flatten_json(nested_json, separator='_', prefix=''):
    """Recursively flatten a nested JSON."""
    flattened = {}
    for key, value in nested_json.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_json(value, separator, new_key))
        else:
            flattened[new_key] = value
    return flattened

def cutoutmeta2csv(cutoutdir, batch_id, csv_savepath, save_df=True):
    # Get all json files
    metas = [x for x in Path(cutoutdir, batch_id).glob("*.json")]
    cutouts = []

    for meta in tqdm(metas):
        
        # Get dictionaries
        with open(meta) as f:
            j = json.load(f)
            data = flatten_json(j)
            cutouts.append(data)
    df = pd.DataFrame(cutouts)
    if save_df:
        df.to_csv(csv_savepath, index=False)
    return df
    
######################################################
############### VEGETATION INDICES ###################
######################################################


def make_exg(rgb_image, normalize=False, thresh=0):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # EXG = 2 * G - R - B
    np.seterr(divide="ignore", invalid="ignore")
    rgb_image = rgb_image.astype(float)
    r, g, b = cv2.split(rgb_image)

    if normalize:
        total = r + g + b
        exg = 2 * (g / total) - (r / total) - (b / total)
    else:
        exg = 2 * g - r - b
    if thresh is not None and normalize == False:
        exg = np.where(exg < thresh, 0, exg)
        return exg.astype("uint8")


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
    thresh_vi = np.where(
        (thresh_vi > low) & (thresh_vi < upper), thresh_vi * sigma, thresh_vi
    )
    return thresh_vi


######################################################
################# MORPHOLOGICAL ######################
######################################################


def clear_border(mask):
    mask = segmentation.clear_border(mask)
    return mask


def reduce_holes(mask, min_object_size=1000, min_hole_size=1000):
    mask = mask.astype(np.bool8)
    # mask = measure.label(mask, connectivity=2)
    mask = morphology.remove_small_holes(
        morphology.remove_small_objects(mask, min_hole_size), min_object_size
    )
    # mask = morphology.opening(mask, morphology.disk(3))
    return mask

######################################################
########### CLASSIFIERS AND THRESHOLDING #############
######################################################

def make_kmeans(exg_mask):
    # Use kmeans and find the cluster that
    # is the highest (or greenest) to have consistent labels
    n_classes = 2
    # Reshape the data to a 2D array of pixels
    pixels = exg_mask.reshape((-1, 1))
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=3)
    labels = kmeans.fit_predict(pixels)
    # Identify which cluster corresponds to the green plant material
    green_cluster_index = np.argmax(kmeans.cluster_centers_)
    # Create a binary mask based on the cluster labels
    binary_mask = labels == green_cluster_index
    # Reshape the binary mask to have the same shape as the original image
    reshaped_mask = binary_mask.reshape(exg_mask.shape)
    return reshaped_mask.astype(np.uint8)


def otsu_thresh(mask, kernel_size=(3, 3)):
    mask_blur = cv2.GaussianBlur(mask, kernel_size, 0).astype("uint16")
    ret3, mask_th3 = cv2.threshold(
        mask_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask_th3


######################################################
##################### MASKING ########################
######################################################


def apply_mask(img, mask, mask_color):
    """Apply white image mask to image, with bitwise AND operator bitwise NOT operator and ADD operator.
    Inputs:
    img        = RGB image data
    mask       = Binary mask image data
    mask_color = 'white' or 'black'
    Returns:
    masked_img = masked image data
    :param img: numpy.ndarray
    :param mask: numpy.ndarray
    :param mask_color: str
    :return masked_img: numpy.ndarray
    """
    if mask_color.upper() == "WHITE":
        color_val = 255
    elif mask_color.upper() == "BLACK":
        color_val = 0

    array_data = img.copy()

    # Mask the array
    array_data[np.where(mask == 0)] = color_val
    return array_data


def calculate_bbox_area_cm2(image_height_m, image_width_m, cutout_height, cutout_width, fullres_width, fullres_height):
    """
    Calculate bbox area in cm² using cutout dimensions (pixels) and global bounding box coordinates (meters).
    
    Args:
        global_coordinates (dict): Global bounding box coordinates with 'top_left' and 'bottom_right' in meters.
        cutout_height (int): The height of the cutout in pixels.
        cutout_width (int): The width of the cutout in pixels.
        fullres_width (int): Full-resolution image width in pixels.
        fullres_height (int): Full-resolution image height in pixels.
        
    Returns:
        float: Bounding box area in cm².
    """
    
    # Calculate the pixel-to-meter scaling factors
    pixel_width_m = image_width_m / fullres_width
    pixel_height_m = image_height_m / fullres_height

    # Convert cutout width and height from pixels to meters
    cutout_width_m = cutout_width * pixel_width_m
    cutout_height_m = cutout_height * pixel_height_m

    # Calculate the area in cm² (1 m² = 10,000 cm²)
    bbox_area_cm2 = cutout_width_m * cutout_height_m * 10000  # Convert m² to cm²
    return bbox_area_cm2

def match_season_to_date_ranges(season, date_ranges_seasons):
    """
    Matches a single season string to its corresponding date range season from a list.
    Args:
    - season (str): The season string to match (e.g., 'cool_season_covers_2022_2023').
    - date_ranges_seasons (list of str): A list of date range season strings (e.g., ['weeds 2022', 'cover crops 2022/2023']).
    Returns:
    - str: The matched date range season or None if no match is found.
    """
    # Extract the year or year range from the season string
    year_match = re.search(r'\d{4}(?:[_/]\d{4})?', season)
    if not year_match:
        return None
    season_year = year_match.group().replace('_', '/')
    # Identify the crop type from the season string
    crop_type = None
    if 'weeds' in season:
        crop_type = 'weeds'
    elif 'cover' in season:
        crop_type = 'cover crops'
    elif 'cash' in season:
        crop_type = 'cash crops'
    # Find the closest match in the date_ranges_seasons
    for entry in date_ranges_seasons:
        # Extract the year or year range from the entry
        entry_year_match = re.search(r'\d{4}(?:[/]\d{4})?', entry)
        if entry_year_match:
            entry_year = entry_year_match.group()
            if season_year == entry_year and crop_type in entry:
                return entry
    return None

def match_single_date_range_to_season(date_range_season, seasons):
    """
    Matches a single date range season to its corresponding season from a list.

    Args:
    - date_range_season (str): The date range season to match (e.g., 'cover crops 2022/2023').
    - seasons (list of str): A list of season strings (e.g., ['cool_season_covers_2022_2023']).

    Returns:
    - str: The matched season or None if no match is found.
    """
    # Extract the year or year range from the date_range_season
    year_match = re.search(r'\d{4}(?:[/]\d{4})?', date_range_season)
    if not year_match:
        return None
    
    date_range_year = year_match.group().replace('/', '_')

    # Identify the crop type from the date_range_season
    crop_type = None
    if 'weeds' in date_range_season:
        crop_type = 'weeds'
    elif 'cover crops' in date_range_season:
        crop_type = 'cover'
    elif 'cash crops' in date_range_season:
        crop_type = 'cash'
    
    # Find the closest match in the seasons
    for season in seasons:
        # Extract the year or year range from the season
        season_year_match = re.search(r'\d{4}(?:[_]\d{4})?', season)
        if season_year_match:
            season_year = season_year_match.group()
            if date_range_year == season_year and crop_type in season:
                return season

    return None

def find_date_range_season(input_date, location, date_ranges):
    """
    Finds the specific date range season that contains the given date for a specific location.

    Args:
    - input_date (str): The date to check (in "YYYY-MM-DD" format).
    - location (str): The location key (e.g., "MD", "NC", "TX").
    - date_ranges (dict): The nested dictionary containing date range information.

    Returns:
    - str: The matched date range season or None if no match is found.
    """
    input_date_obj = datetime.strptime(input_date, "%Y-%m-%d")
    

    location_ranges = date_ranges.get(location, {})
    for season, details in location_ranges.items():
        start_date = datetime.strptime(details["start"], "%Y-%m-%d")
        end_date = datetime.strptime(details["end"], "%Y-%m-%d")
        if start_date <= input_date_obj <= end_date:
            return season

    return None

def read_json_file(json_file: Path) -> dict:
    """Read a JSON file and return its contents as a dictionary."""
    with open(json_file, 'r') as f:
        return json.load(f)

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



# Flatten the nested JSON
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

# Function to rename columns dynamically
def rename_columns(columns):
    rename_map = {
        'cropout_rgb_mean_0': 'cropout_rgb_mean_r',
        'cropout_rgb_mean_1': 'cropout_rgb_mean_g',
        'cropout_rgb_mean_2': 'cropout_rgb_mean_b',
        'cropout_rgb_std_0': 'cropout_rgb_std_r',
        'cropout_rgb_std_1': 'cropout_rgb_std_g',
        'cropout_rgb_std_2': 'cropout_rgb_std_b',
        'category_rgb_0': 'category_rgb_r',
        'category_rgb_1': 'category_rgb_g',
        'category_rgb_2': 'category_rgb_b'
    }

    new_columns = []
    for col in columns:
        # Keep the format for category_rgb columns
        if 'category_rgb' in col:
            if 'category_rgb_0' == col:
                col = 'category_rgb_r'
            if 'category_rgb_1' == col:
                col = 'category_rgb_g'
            if 'category_rgb_2' == col:
                col = 'category_rgb_b'

            new_columns.append(col)
        else:
            # Remove nested prefixes like "cutout_props_" or "category_"
            col = col.replace('cutout_props_', '').replace('category_', '')
            # Replace specific mappings if present
            if col in rename_map:
                col = rename_map[col]
            new_columns.append(col)
    return new_columns

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
    df.columns = rename_columns(list(df.columns))
    if save_df:
        df.to_csv(csv_savepath, index=False)
    return df
    

def is_green(green_sum, image_shape, percent_thresh=0.2):
    """Returns true if number of green pixels is
    above certain threshold percentage based on
    total number of pixels.
    """
    # check threshold value
    assert (
        percent_thresh <= 1
    ), "green sum percent threshold is greater than 1. Must be less than or equal to 1."
    total_pixels = image_shape[0] * image_shape[1]
    green_percent = green_sum / total_pixels
    is_green = True if green_percent > percent_thresh else False
    return is_green, green_percent


def is_mask_empty(mask):
    """Returns true if mask is empty along
    with a logging message
    """
    if mask.max() == 0:
        result = True
        log.info(f"Mask is empty")
    else:
        result = False

    return result


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


def make_gli(rgb_image):
    # Green Leaf Index (GLI) is another vegetation index that helps quantify the greenness of vegetation in an image.
    # It is particularly useful for estimating chlorophyll content and assessing the health of vegetation.

    # Split the image into individual channels
    r, g, b = cv2.split(rgb_image)

    # Calculate the Green Leaf Index
    gli = (2 * g) - (r + b)

    return gli


def make_exr(rgb_img, thresh=0):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # EXR = 1.4 * R - G
    img = rgb_img.astype(float)

    blue = img[:, :, 2]
    green = img[:, :, 1]
    red = img[:, :, 0]

    exr = 1.4 * red - green
    if thresh is not None:
        exr = np.where(exr < thresh, 0, exr)  # Thresholding removes low negative values
    return exr.astype("uint8")


def make_exg_minus_exr(img, thresh=0):
    img = img.astype(float)  # Rgb image
    exg = make_exg(img)
    exr = make_exr(img)
    exgr = exg - exr
    if thresh is not None:
        exgr = np.where(exgr < thresh, 0, exgr)
    exgr = cv2.bitwise_not(exgr)
    return exgr.astype("uint8")


def make_ndi(rgb_img, thresh=0):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # NDI = 128 * (((G - R) / (G + R)) + 1)
    img = rgb_img.astype(float)

    blue = img[:, :, 2]
    green = img[:, :, 1]
    red = img[:, :, 0]
    gminr = green - red
    gplusr = green + red
    gdivr = np.true_divide(gminr, gplusr, out=np.zeros_like(gminr), where=gplusr != 0)
    ndi = 128 * (gdivr + 1)
    # print("Max ndi: ", ndi.max())
    # print("Min ndi: ", ndi.min())

    return ndi


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
###################### BBOX ##########################
######################################################


def rescale_bbox(box, scale, save_changes):
    """Rescales local bbox coordinates, that were first scaled to "downscaled_photo" size (height=3184, width=4796),
       to original image size (height=6368, width=9592). Takes in and returns "Box" dataclass.

    Args:
        box (dataclass): box metedata from bboxes from image metadata
        scale: np.ndarray: scaling dimensions of the image to be scaled to (width, height)

    Returns:
        box (dataclass): box metadata with scaled/updated bbox
    """
    if not save_changes:
        box = copy.deepcopy(box)

    box.local_coordinates = replace(
        box.local_coordinates,
        top_left=[c * s for c, s in zip(box.local_coordinates["top_left"], scale)],
    )
    box.local_coordinates = replace(
        box.local_coordinates,
        top_right=[c * s for c, s in zip(box.local_coordinates["top_right"], scale)],
    )
    box.local_coordinates = replace(
        box.local_coordinates,
        bottom_left=[
            c * s for c, s in zip(box.local_coordinates["bottom_left"], scale)
        ],
    )
    box.local_coordinates = replace(
        box.local_coordinates,
        bottom_right=[
            c * s for c, s in zip(box.local_coordinates["bottom_right"], scale)
        ],
    )

    return box


######################################################
################# MORPHOLOGICAL ######################
######################################################


def region_props(img, label):
    props = [x.area for x in measure.regionprops(label, img)]
    return props


def clean_mask(mask, kernel_size=3, iterations=1, dilation=True):
    if int(kernel_size):
        kernel_size = (kernel_size, kernel_size)
    mask = morphology.opening(mask, morphology.disk(3))
    mask = mask.astype("float32")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    if dilation:
        mask = cv2.dilate(mask, kernel, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))
    mask = cv2.erode(mask, (5, 5), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (7, 7))
    return mask


def dilate_erode(mask, kernel_size=3, dil_iters=5, eros_iters=3, hole_fill=True):
    mask = mask.astype(np.float32)

    if int(kernel_size):
        kernel_size = (kernel_size, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    mask = cv2.dilate(mask, kernel, iterations=dil_iters)
    if hole_fill:
        mask = ndimage.binary_fill_holes(mask.astype(np.int32))
    mask = mask.astype("float")
    mask = cv2.erode(mask, kernel, iterations=eros_iters)

    cleaned_mask = clean_mask(mask)
    return cleaned_mask


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


def seperate_components(mask):
    """Seperates multiple unconnected components in a mask
    for seperate processing.
    """
    # Store individual plant components in a list
    mask = mask.astype(np.uint8)
    nb_components, output, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Remove background component
    nb_components = nb_components - 1
    list_filtered_masks = []
    for i in range(0, nb_components):
        filtered_mask = np.zeros((output.shape))
        filtered_mask[output == i + 1] = 255
        list_filtered_masks.append(filtered_mask)
    return list_filtered_masks


######################################################
########### CLASSIFIERS AND THRESHOLDING #############
######################################################


def check_kmeans(mask):
    max_sum = mask.shape[0] * mask.shape[1]
    ones_sum = np.sum(mask)
    if ones_sum > max_sum / 2:
        mask = np.where(mask == 1, 0, 1)
    return mask


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


def get_watershed(vi, disk1=1, grad1_thresh=12, disk2=10, lbl_fact=2.5):
    # process the watershed
    markers = rank.gradient(vi, disk(disk1)) < grad1_thresh
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(vi, disk(disk2))
    labels = watershed(gradient, markers)
    seg1 = label(labels <= 0)
    lbls = label2rgb(seg1, image=vi, bg_label=0) * lbl_fact
    wtrshed_lbls = rescale_intensity(lbls, in_range=(0, 1), out_range=(0, 1))
    return wtrshed_lbls


def multiple_otsu(vi, classes=3):
    thresholds = filters.threshold_multiotsu(vi, classes=3)
    regions = np.digitize(vi, bins=thresholds)
    return regions


######################################################
##################### CONTOURS #######################
######################################################
def contour_mask(img, mode="biggest"):
    # For using find_contours
    # get most significant contours
    contours_mask, hierachy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    mask = np.zeros(img.shape, np.uint8)
    # find the biggest countour (c) by the area
    if mode == "biggest":
        c = max(contours_mask, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, (255), 1)
    return mask


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


def trans_cutout(img):
    """Get transparent cutout from cutout image with black background. Requires RGB image"""

    # img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    # threshold on black to make a mask
    color = (0, 0, 0)
    mask = np.where((img == color).all(axis=2), 0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result


######################################################
#################### CUTOUTS #########################
######################################################


def crop_cutouts(img, add_padding=False):
    if len(img.shape) == 2:
        foreground = Image.fromarray(img.astype(np.uint8))
    else:
        foreground = Image.fromarray(img)
    pil_crop_frground = foreground.crop(foreground.getbbox())
    array = np.array(pil_crop_frground)
    if add_padding:
        pil_crop_frground = foreground.crop(
            (
                foreground.getbbox()[0] - 2,
                foreground.getbbox()[1] - 2,
                foreground.getbbox()[2] + 2,
                foreground.getbbox()[3] + 2,
            )
        )
    return array


## For metadata converters



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
    
def get_process_all_dict(semif_utils_dir):
    data_schema_storage_dict = {
            "GROW_semifield-developed": {
                "storage": "GROW_DATA",
                "data_type": "semifield-developed-images", 
                "schema_path": Path(semif_utils_dir, "fullsized_schema.json")
                },
            "longterm_semifield-developed": {
                "storage": "longterm_images",
                "data_type": "semifield-developed-images", 
                "schema_path": Path(semif_utils_dir, "fullsized_schema.json")
                },
            "GROW_semifield-cutouts": {
                "storage": "GROW_DATA",
                "data_type": "semifield-cutouts", 
                "schema_path": Path(semif_utils_dir, "cutout_schema.json")
                },
            "longterm_semifield-cutouts": {
                "storage": "longterm_images",
                "data_type": "semifield-cutouts", 
                "schema_path": Path(semif_utils_dir, "cutout_schema.json")
                },
                }
    return data_schema_storage_dict
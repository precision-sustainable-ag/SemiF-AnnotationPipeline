import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# Function to classify each area
def classify_area(area):
    low_xs_small_bounds = 0
    up_xs_small_bounds = 1000
    up_s_bounds = 5000
    up_m_bounds = 100000
    up_l_bounds = 1000000
    up_xl_bounds = 100000000

    if low_xs_small_bounds < area <= up_xs_small_bounds:
        return f"{low_xs_small_bounds} - {up_xs_small_bounds}"

    elif up_xs_small_bounds < area <= up_s_bounds:
        return f"{up_xs_small_bounds+1} - {up_s_bounds}"

    elif up_s_bounds < area <= up_m_bounds:
        return f"{up_s_bounds+1} - {up_m_bounds}"

    elif up_m_bounds < area <= up_l_bounds:
        return f"{up_m_bounds+1} - {up_l_bounds}"

    elif up_l_bounds < area <= up_xl_bounds:
        return f"{up_l_bounds+1} - {up_xl_bounds}"

    elif up_xl_bounds < area:
        return f"greater than {up_xl_bounds+1}"

def get_image_info(row, idx):
    cutpath = row["cutout_path"]
    common_name = row["common_name"]
    genus = row["genus"]
    species = row["species"]

    info_dict = {
        "index": idx,
        "common_name": common_name,
        "genus": genus,
        "species": species,
        "cutout_path": cutpath,
    }
    return info_dict


def filter_area(df, lower, upper):
    filtered_dfs = []
    for spec in df.common_name.unique():
        temp = df[df["common_name"] == spec]

        mean = temp.area.describe()["mean"]
        min = temp.area.describe()["min"]
        max = temp.area.describe()["max"]
        per25 = temp.area.describe()["25%"]
        per50 = temp.area.describe()["50%"]
        per75 = temp.area.describe()["75%"]

        if type(lower) is int:
            lower_area_limit = lower
        if lower is None:
            lower_area_limit = 0
        if lower == "mean":
            lower_area_limit = mean
        if lower == "min":
            lower_area_limit = min
        if lower == "max":
            lower_area_limit = max
        if lower == "per25":
            lower_area_limit = per25
        if lower == "per50":
            lower_area_limit = per50
        if lower == "per75":
            lower_area_limit = per75

        if type(upper) is int:
            upper_area_limit = upper
        if upper == "mean":
            upper_area_limit = mean
        if upper == "min":
            upper_area_limit = min
        if upper == "max":
            upper_area_limit = max
        if upper == "per25":
            upper_area_limit = per25
        if upper == "per50":
            upper_area_limit = per50
        if upper == "per75":
            upper_area_limit = per75

        temp = temp[
            (temp["area"] < upper_area_limit) & (lower_area_limit < temp["area"])
        ]
        filtered_dfs.append(temp)
    filtered_df = pd.concat(filtered_dfs)
    return filtered_df

def filter_by_common_name(df,common_name):
    if type(common_name) is list:
        tempdfs = []
        for cname in common_name:
            tempdf = df[df["common_name"]==cname]
            tempdfs.append(tempdf)
        df = pd.concat(tempdfs)
        return df
    else:
        df = df[df["common_name"]==common_name]
        return df

def filter_green_sum(df, lower, upper):
    filtered_dfs = []
    for spec in df.common_name.unique():
        temp = df[df["common_name"] == spec]

        mean = temp.green_sum.describe()["mean"]
        min = temp.green_sum.describe()["min"]
        max = temp.green_sum.describe()["max"]
        per25 = temp.green_sum.describe()["25%"]
        per50 = temp.green_sum.describe()["50%"]
        per75 = temp.green_sum.describe()["75%"]
        if type(lower) is int:
            lower_green_sum_limit = lower
        if lower is None:
            lower_green_sum_limit = 0
        if lower == "mean":
            lower_green_sum_limit = mean
        if lower == "min":
            lower_green_sum_limit = min
        if lower == "max":
            lower_green_sum_limit = max
        if lower == "per25":
            lower_green_sum_limit = per25
        if lower == "per50":
            lower_green_sum_limit = per50
        if lower == "per75":
            lower_green_sum_limit = per75

        if type(upper) is int:
            upper_green_sum_limit = upper
        if upper == "mean":
            upper_green_sum_limit = mean
        if upper == "min":
            upper_green_sum_limit = min
        if upper == "max":
            upper_green_sum_limit = max
        if upper == "per25":
            upper_green_sum_limit = per25
        if upper == "per50":
            upper_green_sum_limit = per50
        if upper == "per75":
            upper_green_sum_limit = per75

        temp = temp[
            (temp["green_sum"] < upper_green_sum_limit)
            & (lower_green_sum_limit < temp["green_sum"])
        ]
        filtered_dfs.append(temp)
    filtered_df = pd.concat(filtered_dfs)
    return filtered_df

def minor_df_cleaning(cfg, df):
    df =df[["common_name", "image_id", "general_season", "cutout_path","cutout_id", "area", "batch_id", "is_primary", "extends_border", "group", "growth_habit", "class_id", "green_sum"]]
    df = df[df["common_name"] != "Unknown"]
    df = df[df["common_name"] != "unknown"]
    df = df[df["common_name"] != "Colorchecker"]
    df = df[df["common_name"] != "colorchecker"]

    df["file_path"] = cfg.data.longterm_storage + "/semifield-cutouts/" + df["cutout_path"]
    print("creating area class")
    # df["area_class"] = df["area"].apply(classify_area)
    return df

def df_cleaning(cfg, df):
    cols = [
    "EPPO",
    "USDA_symbol",
    "area",
    "authority",
    "axis_major_length",
    "axis_minor_length",
    "b",
    "b_mean",
    "b_std",
    "batch_id",
    "bbox",
    "blob_home",
    "blur_effect",
    "camera_info",
    "category",
    "class",
    "class_id",
    "collection_location",
    "collection_timing",
    "common_name",
    "cropout_b_mean",
    "cropout_b_std",
    "cropout_exg_mean",
    "cropout_exg_std",
    "cropout_g_mean",
    "cropout_g_std",
    "cropout_r_mean",
    "cropout_r_std",
    "cutout_b_mean",
    "cutout_b_std",
    "cutout_exg_mean",
    "cutout_exg_std",
    "cutout_g_mean",
    "cutout_g_std",
    "cutout_id",
    "cutout_num",
    "cutout_path",
    "cutout_r_mean",
    "cutout_r_std",
    "cutout_version",
    "data_root",
    "date",
    "datetime",
    "dt",
    "duration",
    "eccentricity",
    "exg_sum",
    "exif_meta",
    "extends_border",
    "extent",
    "family",
    "g",
    "g_max",
    "g_mean",
    "g_min",
    "g_std",
    "general_season",
    "genus",
    "green_sum",
    "group",
    "growth_habit",
    "hex",
    "hwc",
    "image_id",
    "is_primary",
    "link",
    "multi_species_USDA_symbol",
    "note",
    "num_components",
    "order",
    "perimeter",
    "r",
    "r_max",
    "r_mean",
    "r_min",
    "r_std",
    "schema_version",
    "season",
    "shape",
    "solidity",
    "species",
    "state_id",
    "subclass",
    "synth",
]
    df = df[cols]
    df = df[df["common_name"] != "Unknown"]
    df = df[df["common_name"] != "Colorchecker"]
    df["file_path"] = cfg.data.longterm_storage + "/semifield-cutouts/" + df["cutout_path"]
    df["area_class"] = df["area"].apply(classify_area)
    df["full_size_file_path"] = cfg.data.longterm_storage + "/semifield-developed-images/" + df["batch_id"] +  "/images/" + df["image_id"] + ".jpg"
    df["full_size_file_path_masks"] = cfg.data.longterm_storage + "/semifield-developed-images/" + df["batch_id"] +  "/meta_masks/semantic_masks/" + df["image_id"] + ".png"
    return df

def color_mapper(cfg):
    color_map = {}
    with open(cfg.data.species) as outfile:
        data = json.load(outfile)
        spec = data["species"]
        for i in spec.keys():
            class_id = spec[i]["class_id"]
            rgb = spec[i]["rgb"]
            color_map[class_id] = rgb
    return color_map

def get_cutoutpath(row):
    cutout_path = row["file_path"]
    if not Path(cutout_path).exists():
        cutout_path = cutout_path.replace("longterm_images", "GROW_DATA")
    cutout_metadata_path = cutout_path.replace(".png", ".json")
    return cutout_path, cutout_metadata_path

def get_paths(row):
    fullsize_path = row["full_size_file_path"]
    fullsize_mask_path = row["full_size_file_path_masks"]
    fullsize_instancemask_path = row["full_size_file_path_masks"].replace("semantic_masks", "instance_masks")
    fullsize_metadata_path = fullsize_path.replace("/images/", "/metadata/").replace("jpg", "json")
    
    if (not Path(fullsize_path).exists()) or (not Path(fullsize_mask_path).exists()):
        fullsize_path = fullsize_path.replace("longterm_images", "GROW_DATA")
        fullsize_mask_path = fullsize_mask_path.replace("longterm_images", "GROW_DATA")
        fullsize_instancemask_path = fullsize_instancemask_path.replace("longterm_images", "GROW_DATA")
        fullsize_metadata_path = fullsize_metadata_path.replace("longterm_images", "GROW_DATA")
    return fullsize_path, fullsize_mask_path, fullsize_instancemask_path, fullsize_metadata_path

def get_bboxes(image_metadata_path, image):
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        height, width, _ = image.shape

    with open(image_metadata_path) as outfile:
        data = json.load(outfile)
        bounding_boxes = []
        for box in data["bboxes"]:
            top_left_normalized = tuple(box["local_coordinates"]["top_left"])
            bottom_right_normalized = tuple(box["local_coordinates"]["bottom_right"])
            top_left = (int(top_left_normalized[0] * width), int(top_left_normalized[1] * height))
            bottom_right = (int(bottom_right_normalized[0] * width), int(bottom_right_normalized[1] * height))
            bounding_boxes.append((top_left, bottom_right))
    return bounding_boxes

# Load the image in grayscale to get the pixel values as keys for your dictionary
def remap_images(img, color_map):
    # img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # Check if the image was loaded correctly
    if img is None:
        raise ValueError("Image not loaded correctly")
    # Create an empty array for the output image with 3 channels for RGB
    remapped_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # Iterate over each pixel and remap the color
    for key, new_color in color_map.items():
        remapped_image[img == key] = new_color

    return remapped_image

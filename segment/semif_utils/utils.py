import copy
import json
import logging
import operator
import os
import platform
import random
from dataclasses import asdict, replace
from datetime import datetime
from multiprocessing import Manager, Pool, Process, cpu_count
from pathlib import Path

import cv2
import extcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from colormap import rgb2hex
from dacite import Config, from_dict
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy import ndimage
from scipy import ndimage as ndi
from semif_utils.datasets import Cutout, ImageData
from skimage import filters, measure, morphology, segmentation
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import rank
from skimage.measure import label
from skimage.morphology import disk
from skimage.segmentation import watershed
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)

######################################################
################### GENERAL ##########################
######################################################


def read_json(path):
    # Opening JSON file
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=str)
        
######################################################
################## GET METADATA ######################
######################################################

def create_dataclasses(metadir, cfg):
    labels = sorted([str(x) for x in (metadir).glob("*.json")])
    return_list = []
    procs = cpu_count() - cfg.segment.cpus_left
    with Pool(processes=procs) as pool:
        imap_unordered_it = pool.imap_unordered(get_image_meta, labels)
        for x in imap_unordered_it:
            return_list.append(x)
    log.info(f"Converted metadata to {len(return_list)} dataclasses")
    return return_list

def get_bbox_info(csv_path):

    df = pd.read_csv(csv_path).drop(columns=["Unnamed: 0"])
    bbox_dict = df.groupby(
        by="imgname", sort=True).apply(lambda x: x.to_dict(orient="records"))
    img_list = list(bbox_dict.keys())
    return bbox_dict, img_list


def get_site_id(imagedir):
    # Must be in TX_2022-12-31 format
    imgstem = Path(imagedir).stem
    siteid = imgstem.split("_")[0]
    return siteid


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == "Windows":
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


def get_upload_datetime(imagedir):

    creation_dt = creation_date(imagedir)
    creation_dt = datetime.fromtimestamp(creation_dt).strftime(
        "%Y-%m-%d_%H:%M:%S")
    return creation_dt


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


def get_image_meta(path):
    with open(path) as f:
        j = json.load(f)
        imgdata = from_dict(data_class=ImageData,
                            data=j,
                            config=Config(check_types=False))
    return imgdata


def get_cutout_meta(path):
    with open(path) as f:
        j = json.load(f)
        cutout = from_dict(data_class=Cutout,
                           data=j,
                           config=Config(check_types=False))
    return cutout


def cutoutmeta2csv(cutoutdir, batch_id, save_df=False):
    # Get all json files
    metas = [x for x in Path(cutoutdir, batch_id).glob("*.json")]
    cutouts = []
    for meta in metas:
        # Get dictionaries
        cutout = asdict(get_cutout_meta(meta))
        row = cutout["cutout_props"]
        cls = cutout["cls"]
        # Extend nested dicts to single column header
        for ro in row:
            rec = {ro: row[ro]}
            cutout.update(rec)
            for cl in cls:
                spec = {cl: cls[cl]}
                cutout.update(spec)
        # Remove duplicate nested dicts
        cutout.pop("cutout_props")
        cutout.pop("cls")
        # Create and append df
        cutdf = pd.DataFrame(cutout)
        cutouts.append(cutdf)
    # Concat and reset index of main df
    cutouts_df = pd.concat(cutouts)
    cutouts_df = cutouts_df.reset_index()
    # Save dataframe
    if save_df:
        cutouts_df.to_csv(f"{cutoutdir}/{batch_id}.csv", index=False)
    return cutouts_df


def growth_stage(batch_date, plant_date_list):
    """ Gets rough approximation of growth stage by comparing the batch upload date
        with a list of "planting dates" in config.planting. Uses a threshold value,
        "coty_thresh" to differentiate between cotyledon and vegetative. 
        Returns growth stage and planting date.
        "" Classifies growth stage and approximates planting date by comparing the 
        batch upload date with a list of "planting dates" in config.planting.

        Input:
        batch_date(str)      -  Gathered from cfg.general.batch_id.strip("_)[0] (ex. 2022-06-28)
        plant_date_list(list)-  List of dates, by location, taken from cfg.planting that 
                                represent all planting dates by locations
        
        Returns:
        pl_dt(int)           -  Planting date in config that is closest to, but not more recent than,
                                the batch date. Planting dates before the batch date are excluded. 
        g_stage(str)         -  Growth stage classification based on numbers of days after planting date
                                (subject to change).
        """

    batch_date = datetime.strptime(batch_date, "%Y-%m-%d")
    plant_date_list = [
        datetime.strptime(x, "%Y-%m-%d") for x in plant_date_list
    ]
    # Remove plant dates that are newer than batch date
    plant_date_list = [x for x in plant_date_list if x <= batch_date]
    # Difference and get indices
    deltas = [abs(ti - batch_date) for ti in plant_date_list]
    min_index, min_delta = min(enumerate(deltas), key=operator.itemgetter(1))

    pl_dt = plant_date_list[min_index].strftime('%Y-%m-%d')
    days_after_planting = min_delta.days
    if days_after_planting < 2:
        g_stage = "seed"
    elif days_after_planting < 10:
        g_stage = "cotyledon"
    elif days_after_planting < 20:
        g_stage = "seedling"
    else:
        g_stage = "vegetative"
    return g_stage, pl_dt, days_after_planting


def is_green(green_sum, image_shape, percent_thresh=.2):
    """ Returns true if number of green pixels is 
        above certain threshold percentage based on
        total number of pixels.
    """
    # check threshold value
    assert percent_thresh <= 1, "green sum percent threshold is greater than 1. Must be less than or equal to 1."
    total_pixels = image_shape[0] * image_shape[1]
    green_percent = green_sum / total_pixels
    is_green = True if green_percent > percent_thresh else False
    return is_green, green_percent


def is_mask_empty(mask):
    """ Returns true if mask is empty along
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


def make_exg(img, normalize=False, thresh=0):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # EXG = 2 * G - R - B
    np.seterr(divide='ignore', invalid='ignore')
    img = img.astype(float)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if normalize:
        total = r + g + b
        exg = 2 * (g / total) - (r / total) - (b / total)
    else:
        exg = 2 * g - r - b
    if thresh is not None and normalize == False:
        exg = np.where(exg < thresh, 0, exg)
        return exg.astype("uint8")


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
        exr = np.where(exr < thresh, 0,
                       exr)  # Thresholding removes low negative values
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
    gdivr = np.true_divide(gminr,
                           gplusr,
                           out=np.zeros_like(gminr),
                           where=gplusr != 0)
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
    thresh_vi = np.where((thresh_vi > low) & (thresh_vi < upper),
                         thresh_vi * sigma, thresh_vi)
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
        top_left=[
            c * s for c, s in zip(box.local_coordinates["top_left"], scale)
        ])
    box.local_coordinates = replace(
        box.local_coordinates,
        top_right=[
            c * s for c, s in zip(box.local_coordinates["top_right"], scale)
        ])
    box.local_coordinates = replace(
        box.local_coordinates,
        bottom_left=[
            c * s for c, s in zip(box.local_coordinates["bottom_left"], scale)
        ])
    box.local_coordinates = replace(
        box.local_coordinates,
        bottom_right=[
            c * s for c, s in zip(box.local_coordinates["bottom_right"], scale)
        ])

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


def dilate_erode(mask,
                 kernel_size=3,
                 dil_iters=5,
                 eros_iters=3,
                 hole_fill=True):
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
        morphology.remove_small_objects(mask, min_hole_size), min_object_size)
    # mask = morphology.opening(mask, morphology.disk(3))
    return mask


def seperate_components(mask):
    """Seperates multiple unconnected components in a mask
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
    rows, cols = exg_mask.shape
    n_classes = 2
    exg = exg_mask.reshape(rows * cols, 1)
    kmeans = KMeans(n_clusters=n_classes, random_state=3).fit(exg)
    mask = kmeans.labels_.reshape(rows, cols)
    mask = check_kmeans(mask)
    return mask.astype("uint64")


def otsu_thresh(mask, kernel_size=(3, 3)):
    mask_blur = cv2.GaussianBlur(mask, kernel_size, 0).astype("uint16")
    ret3, mask_th3 = cv2.threshold(mask_blur, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    contours_mask, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
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
    """ Get transparent cutout from cutout image with black background. Requires RGB image"""

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
        pil_crop_frground = foreground.crop((
            foreground.getbbox()[0] - 2,
            foreground.getbbox()[1] - 2,
            foreground.getbbox()[2] + 2,
            foreground.getbbox()[3] + 2,
        ))
    return array


# ------------------- Helper functions --------------------------------
def bbox_iou(box1, box2):
    box1 = torch.tensor([box1], dtype=torch.float)
    box2 = torch.tensor([box2], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    return iou


def get_img_bbox(x, y, imgshape):
    pot_h, pot_w, _ = imgshape
    x0, x1, y0, y1 = x, x + pot_w, y, y + pot_h
    bbox = [x0, y0, x1, y1]  # top right corner, bottom left corner
    return bbox


def center2topleft(x, y, background_imgshape):
    """Gets top left coordinates of an image from center point coordinate"""
    back_h, back_w, _ = background_imgshape
    y = y - int(back_h / 2)
    x = x - int(back_w / 2)
    return x, y


def transform_position(points, imgshape, spread_factor):
    """Applies random jitter factor to points and transforms them to top left image coordinates."""
    y, x = points

    x, y = x + random.randint(
        -spread_factor, spread_factor), y + random.randint(
            -int(spread_factor / 3), int(spread_factor / 3))

    x, y = center2topleft(x, y, imgshape)

    return x, y


def center_on_background(y, x, back_shape, fore_shape):
    # pot positions and shape top left corner
    back_h, back_w, _ = back_shape
    fore_h, fore_w, _ = fore_shape
    newx = int(((back_w - fore_w) / 2) + x)
    newy = int(((back_h - fore_h) / 2) + y)
    return newx, newy


def img2RGBA(img):
    alpha = np.sum(img, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    img = np.dstack((img, alpha))
    return img


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rect(object):
    def __init__(self, p1, p2):
        """Store the top, bottom, left and right values for points
        p1 and p2 are the (corners) in either order
        """
        self.left = min(p1.x, p2.x)
        self.right = max(p1.x, p2.x)
        self.bottom = min(p1.y, p2.y)
        self.top = max(p1.y, p2.y)


def overlap(r1, r2):
    """Overlapping rectangles overlap both horizontally & vertically"""
    return range_overlap(r1.left, r1.right,
                         r2.left, r2.right) and range_overlap(
                             r1.bottom, r1.top, r2.bottom, r2.top)


def range_overlap(a_min, a_max, b_min, b_max):
    """Neither range is completely greater than the other"""
    return (a_min <= b_max) and (b_min <= a_max)


def dict_to_json(dic, path):
    json_path = Path(path)
    with open(json_path, "w") as j:
        json.dump(dic, j, indent=4, default=str)


def clean_data(data):
    """Convert absolute pot and background paths to relative.
    Takes the last two components of a path object for each.

    Takes in and returns a dictionary of dataclass to be
    stored in json and db.
    """
    data["background"]["background_path"] = "/".join(
        Path(data["background"]["background_path"]).parts[-2:])
    pots = data["pots"]
    for pot in pots:
        pot["pot_path"] = "/".join(Path(pot["pot_path"]).parts[-2:])

    for cutout in data["cutouts"]:
        cutout["cutout_path"] = "/".join(
            Path(cutout["cutout_path"]).parts[-2:])

    return data


def save_dataclass_json(data_dict, path):
    json_path = Path(path)
    with open(json_path, "w") as j:
        json.dump(data_dict, j, indent=4, default=str)


def get_cutout_dir(batch_dir, cutout_dir):
    batch = Path(batch_dir).name
    cutout_dir = Path(cutout_dir, batch)
    return cutout_dir


######################################################
################### COLOR STATS ######################
######################################################


def color_to_df(input, ignore_black=False):
    if ignore_black:
        colors_pre_list = str(input).replace('[(', '').split(', (')[0:-1]
    else:
        colors_pre_list = str(input).replace('([(', '').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')', '') for i in colors_pre_list]

    #convert RGB to HEX code
    df_color_up = [
        rgb2hex(int(i.split(", ")[0].replace("(", "")), int(i.split(", ")[1]),
                int(i.split(", ")[2].replace(")", ""))) for i in df_rgb
    ]

    df = pd.DataFrame(zip(df_color_up, df_rgb, df_percent),
                      columns=['hex', 'rgb', 'occurence'])
    return df


def exact_color(pilimg,
                resize,
                tolerance,
                zoom,
                save=False,
                show=False,
                ignore_black=False,
                return_colors=False,
                transparent=True):

    colors_x = extcolors.extract_from_image(pilimg,
                                            tolerance=tolerance,
                                            limit=13)
    if ignore_black:
        colors_x = [(colo, num) for colo, num in colors_x[0]
                    if colo != (0, 0, 0)]
    #crate dataframe

    df_color = color_to_df(colors_x, ignore_black=True)
    if return_colors and not save and not show:
        return df_color
    # Plotting and saving
    if show or save:
        #background
        bg = 'bg.png'
        fig, ax = plt.subplots(figsize=(192, 108), dpi=10)
        if not transparent:
            fig.set_facecolor('white')
        plt.savefig(bg, transparent=transparent)
        plt.close(fig)
        #resize
        output_width = resize
        img = Image.open(input_image)
        wpercent = (output_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((output_width, hsize), Image.Resampling.LANCZOS)
        # Change transparency of image in donut
        if transparent and ignore_black:
            datas = img.getdata()
            newData = []
            for item in datas:
                if item[0] == 0 and item[1] == 0 and item[
                        2] == 0:  # finding black colour by its RGB value
                    # storing a transparent value when we find a black colour
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)  # other colours remain unchanged
            img.putdata(newData)

        #annotate text
        list_color = list(df_color['hex'])
        list_precent = [int(i) for i in list(df_color['occurence'])]
        text_c = [
            c + ' ' + str(round(p * 100 / sum(list_precent), 1)) + '%'
            for c, p in zip(list_color, list_precent)
        ]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160, 120), dpi=10)
        #donut plot
        wedges, text = ax1.pie(list_precent,
                               labels=text_c,
                               labeldistance=1.05,
                               colors=list_color,
                               textprops={
                                   'fontsize': 150,
                                   'color': 'black'
                               })
        plt.setp(wedges, width=0.3)
        #add image in the center of donut plot
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (0, 0), xycoords='data', frameon=False)
        ax1.add_artist(ab)
        #color palette
        x_posi, y_posi, y_posi2 = 160, -170, -170
        for c in list_color:
            if list_color.index(c) <= 5:
                y_posi += 180
                rect = patches.Rectangle((x_posi, y_posi),
                                         360,
                                         160,
                                         facecolor=c)
                ax2.add_patch(rect)
                ax2.text(x=x_posi + 400,
                         y=y_posi + 100,
                         s=c,
                         fontdict={'fontsize': 190})
            else:
                y_posi2 += 180
                rect = patches.Rectangle((x_posi + 1000, y_posi2),
                                         360,
                                         160,
                                         facecolor=c)
                ax2.add_artist(rect)
                ax2.text(x=x_posi + 1400,
                         y=y_posi2 + 100,
                         s=c,
                         fontdict={'fontsize': 190})
        if not transparent:
            fig.set_facecolor('white')
        ax2.axis('off')
        bg = plt.imread('bg.png')
        plt.imshow(bg)
        plt.tight_layout()
        if save:
            plt.savefig(Path(input_image).name,
                        dpi=100,
                        bbox_inches="tight",
                        transparent=transparent)
        if return_colors:
            return colors_x
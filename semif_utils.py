import csv
import dataclasses
import glob
import itertools
import json
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFilter, ImageStat
from shapely.geometry import Polygon
from skimage import measure, morphology, segmentation
from sklearn.cluster import KMeans

######################################################
############### Directory Organization ###############
######################################################


def dt_str(format="%m%d%Y_%H%M%S"):
    now = datetime.now()
    dt = now.strftime(format)
    return dt


def increment_path(path, new_path=True, sep="_", mkdir=False):
    """Increment file or directory path if directory already exists.
    Uses a time stamp (format mmddYYYY_HHMMSS) to increment path.
    i.e. data/masks --> data/masks{sep}03052022_140133, data/masks{sep}03052022_140349, ... etc."""
    path = Path(path)  # os-agnostic
    # "exist_ok=False" means you want to create a new incremented path object maybe or maybe not
    # for creating a new directory
    if path.exists() and new_path:
        path, suffix = ((path.with_suffix(""),
                         path.suffix) if path.is_file() else (path, ""))
        # path = Path(f"{path}{sep}{dt_str()}{suffix}")  # increment path
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


######################################################
####################### SAVING #######################
######################################################
# class EnhancedJSONEncoder(json.JSONEncoder):

#     def default(self, o):
#         if dataclasses.is_dataclass(o):
#             return dataclasses.asdict(o)
#         return super().default(o)


def save_metadata(data_dict, save_path):
    with open(save_path, 'w') as f:
        # f.write(json.dumps(data_dict))
        json.dumps(data_dict, cls=EnhancedJSONEncoder)
        # json.dumps(data_dict)
        # yaml.dump(data_dict,
        # f,
        # default_flow_style=False,
        # sort_keys=False)
        f.close()


######################################################
################### PREPROCESSINGS ###################
######################################################


def read_img(imgpath, mode="RGB"):
    assert Path(imgpath).exists, "Image path doe not exist."
    if mode.upper() == "RGB":
        img = cv2.imread(str(imgpath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mode.upper() == "MASK":
        img = cv2.imread(str(imgpath), 0)

    return img


def load_rgba_img(image_path):
    # Load images
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    bgra_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    return bgra_img


def get_images(
        parent_dir,
        extensions=["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"],
        as_strs=False,
        recursively=False,
        sort=False):

    # Creates list of image paths of multiple file tpyes in directoty
    """TODO _summary_"""
    # Check file and directory exists
    assert type(extensions) is list, "Extensions are not in a list."
    assert Path(
        parent_dir).exists(), f"Image directory {parent_dir} is not valid."

    # Parse locations and collection image files
    files = []
    if not recursively:
        for ext in extensions:
            files.extend(Path(parent_dir).glob(ext))
    else:
        for ext in extensions:
            files.extend(Path(parent_dir).rglob(ext))

    if as_strs:
        files = [str(x) for x in files]
    if sort:
        files = sorted([x for x in files])
    return files


def check_data(self):
    """Checks that number and names of image and label files match"""
    imgfilesstem = [x.stem for x in self.imgs]
    lblfilesstem = [x.stem for x in self.lbls]
    assert imgfilesstem == lblfilesstem, "Items in image directory and label directory do not match. Check contents"

    imgdir_parent = Path(self.imgs[0]).parts[-2]
    lbldir_parent = Path(self.lbls[0]).parts[-2]
    print("\nDATA CHECK")
    print(
        f"Contents in '{imgdir_parent}' directory and '{lbldir_parent}' directory match."
    )
    print(f"\n{len(imgfilesstem)} items in '{imgdir_parent}'")
    print(f"{len(lblfilesstem)} items in '{lbldir_parent}'")
    print(f"Item 0 in '{imgdir_parent}': {self.imgs[0].name}")
    print(f"Item 0 in '{lbldir_parent}': {self.lbls[0].name}\n")


def check_paths(self, savedir):
    assert self.datadir.exists(), "Data directory does not exist"
    assert Path(self.datadir,
                "labels").exists(), "Label directory does not exist"
    assert Path(self.datadir,
                "images").exists(), "Image directory does not exist"

    if not savedir.exists():
        savedir.mkdir(parents=True, exist_ok=False)
        Path(savedir, "masks").mkdir(parents=True, exist_ok=False)
        Path(savedir, "cutouts").mkdir(parents=True, exist_ok=False)
    else:
        savedir = increment_path(savedir, sep="_", mkdir=True)
        if not Path(savedir, "masks").exists():
            Path(savedir, "masks").mkdir(parents=True, exist_ok=False)
        if not Path(savedir, "cutouts").exists():
            Path(savedir, "cutouts").mkdir(parents=True, exist_ok=False)
    return savedir


def get_image_stats(img, mask=False):
    img_stat = Image.fromarray(img)
    print("#####################################")
    print(img_stat)
    print()
    print("Shape            ", img.shape)
    print("Max value        ", img.max())
    print("Min value        ", img.min())
    if mask:
        print("Unique values    ", np.unique(img))
    print("Sum              ", ImageStat.Stat(img_stat).sum)
    print("Mean             ", ImageStat.Stat(img_stat).mean)
    print("STD              ", ImageStat.Stat(img_stat).stddev)
    print("Variance         ", ImageStat.Stat(img_stat).var)
    print()


def copy_bench_images(img_parent_dir, dst_dir):
    ## Copies files using glob
    imgs = get_images(img_parent_dir)
    for img in imgs:
        split_path = img.split("/")
        newfname = split_path[-2] + "_" + split_path[-1]
        # fname = os.path.basename(img)
        dst = os.path.join(dst_dir, newfname)
        shutil.copy(img, dst)


def pluck_bench_imgs(bench_dir, dest_dir):
    """Removes one bench images per folder"""
    # sub_dirs = [x[0] for x in os.walk(bench_dir)]
    sub_dirs = glob.glob(bench_dir + "/*")
    img_list = []
    for i in sub_dirs[1:]:
        imgs = get_images(i, start_one_up=True)
        rndm_img = random.choice(imgs[10:])
        new_fname = get_bench_basename(rndm_img)
        dest_path = os.path.join(dest_dir, new_fname)
        shutil.copy(imgs[-2], dest_path)


def get_bench_basename(bench_img_path):
    bench_split = bench_img_path.split("/")
    bench_base_name = bench_split[-1]
    bench_parent = bench_split[-2]
    new_name = bench_parent + "_" + bench_base_name
    return new_name


def hist_equa(bgr_img, color_CLAHE=False):
    # brg_img: np array in [BGR] channel order
    b, g, r = cv2.split(bgr_img)

    if color_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equ_r = clahe.apply(r)
        equ_g = clahe.apply(g)
        equ_b = clahe.apply(b)
    else:
        equ_r = cv2.equalizeHist(r)
        equ_g = cv2.equalizeHist(g)
        equ_b = cv2.equalizeHist(b)

    # RGB
    equ = cv2.merge((equ_r, equ_g, equ_b))

    # Returns RGB color image
    return equ


def CLAHE_hist(img: "np.int8"):
    # Contrast limiting adaptive histogram equalization (CLAHE).
    # Contrast amplification is limited to reduce noise amplification.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)

    return clahe_img


def improve_bright(img, alpha=1.0, beta=10):

    # alpha = 1.0 # Simple contrast control
    # beta = 10    # Simple brightness control

    # try:
    #     alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    #     beta = int(input('* Enter the beta value [0-100]: '))
    # except ValueError:
    #     print('Error, not a number')
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return new_img


######################################################
############# VEGETATION INDICES #####################
######################################################


def make_exg(img, normalize=False, thresh=0):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # EXG = 2 * G - R - B
    img = img.astype(float)
    b, g, r = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    if normalize:
        total = r + g + b
        exg = 2 * (g / total) - (r / total) - (b / total)
    else:
        exg = 2 * g - r - b
    if thresh is not None and normalize == False:
        exg = np.where(exg < thresh, 0, exg)
        return exg.astype("uint8")


def make_exr(rgb_img):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # EXR = 1.4 * R - G
    img = rgb_img.astype(float)

    blue = img[:, :, 2]
    green = img[:, :, 1]
    red = img[:, :, 0]

    exr = 1.4 * red - green
    exr = np.where(exr < 0, 0, exr)  # Thresholding removes low negative values
    return exr.astype("uint8")


def make_exg_minus_exr(img):
    img = img.astype(float)  # Rgb image
    exg = make_exg(img)
    exr = make_exr(img)
    exgr = exg - exr
    exgr = np.where(exgr < 25, 0, exgr)
    return exgr.astype("uint8")


def make_ndi(rgb_img):
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


########################################################
###################### CLASSIFY  #######################
########################################################


def check_kmeans(mask):
    max_sum = mask.shape[0] * mask.shape[1]
    ones_sum = np.sum(mask)
    if ones_sum > max_sum / 2:
        mask = np.where(mask == 1, 0, 1)
    return mask


def make_kmeans(exg_mask):
    rows, cols = exg_mask.shape
    n_classes = 2
    X = exg_mask.reshape(rows * cols, 1)
    kmeans = KMeans(n_clusters=n_classes, random_state=3).fit(X)
    mask = kmeans.labels_.reshape(rows, cols)
    mask = check_kmeans(mask)
    return mask.astype("uint8")


def otsu_thresh(mask, kernel_size=(3, 3)):
    mask_blur = cv2.GaussianBlur(mask, kernel_size, 0).astype("uint16")
    ret3, mask_th3 = cv2.threshold(mask_blur, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask_th3


######################################################
############ MORPHOLOGICAL OPERATIONS ################
######################################################


def clear_border(mask):
    mask = segmentation.clear_border(mask)
    return mask


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


def reduce_holes(mask, min_object_size=1000, min_hole_size=1000):

    mask = mask.astype(np.bool8)
    # mask = measure.label(mask, connectivity=2)
    mask = morphology.remove_small_holes(
        morphology.remove_small_objects(mask, min_hole_size), min_object_size)
    # mask = morphology.opening(mask, morphology.disk(3))
    return mask


def clean_edges(mask, kernel=(3, 3), erode_iterations=1, dilate_iterations=1):
    # Dilate the image to join small pieces to the larger white sections
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    erosion = cv2.erode(mask, kernel, iterations=erode_iterations)
    dilate = cv2.dilate(erosion, kernel, iterations=dilate_iterations)
    return dilate


def filter_topn_components(mask, top_n=3) -> "list[np.ndarray]":
    # input must be single channel array
    # calculate size of individual components and chooses based on min size
    mask = mask.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    # size of components except 0 (background)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # Determines number of components to segment
    # Sort components from largest to smallest
    top_n_sizes = sorted(sizes, reverse=True)[:top_n]
    try:
        min_size = min(top_n_sizes) - 1
    except:
        min_size = 0
    list_filtered_masks = []
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered_mask = np.zeros((output.shape))
            filtered_mask[output == i + 1] = 255
            list_filtered_masks.append(filtered_mask)
    return list_filtered_masks


def filter_by_component_size(mask: np.int8, top_n: int) -> "list[np.ndarray]":
    # calculate size of individual components and chooses based on min size
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    # size of components except 0 (background)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # Determines number of components to segment
    # Sort components from largest to smallest
    top_n_sizes = sorted(sizes, reverse=True)[:top_n]
    try:
        min_size = min(top_n_sizes) - 1
    except:
        min_size = 0
    list_filtered_masks = []
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered_mask = np.zeros((output.shape))
            filtered_mask[output == i + 1] = 255
            list_filtered_masks.append(filtered_mask)

    return list_filtered_masks


######################################################
############## CONTOUR OPERATIONS ####################
######################################################


def contour_list2array(contour_list) -> list["np.array"]:
    contour = np.squeeze(contour_list)
    arr_contours = Polygon(contour).exterior.coords
    return np.array(arr_contours, dtype=np.int32)


def crop_contours2contents():  # TODO create this
    # """ Crop contours to contents """
    # contour = contour[0]
    # x, y, w, h = cv2.boundingRect(contour)
    # # then crop it to bounding rectangle
    # crop = cutout[y:y + h, x:x + w]
    # fnd_crop_contours = cv2.findContours(crop, cv2.RETR_EXTERNAL,
    #                                      cv2.CHAIN_APPROX_SIMPLE)
    # crop_contours = fnd_crop_contours[0] if len(
    #     fnd_crop_contours) == 2 else fnd_crop_contours[1]
    # return crop
    pass


def crop_img2contents(img):
    """Crop mask component to contents"""
    img = img.astype(np.uint16)
    x, y, w, h = cv2.boundingRect(img)
    cropped_img = img[y:y + h, x:x + w]
    return cropped_img


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


##########################################################
################### EXTRACT FOREGROUND ###################
##########################################################


def create_foreground(img, mask, add_padding=False, crop_to_content=True):
    # applys mask to create RGBA foreground using PIL

    if len(np.array(mask).shape) == 3:
        mask = np.asarray(mask)[:, :, 0]
    else:
        mask = np.asarray(mask)
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    # extract from image using mask
    rgba[:, :, 3][mask == 0] = 0

    foreground = Image.fromarray(rgba)
    # crop foreground to content
    if add_padding:
        pil_crop_frground = foreground.crop((
            foreground.getbbox()[0] - 3,
            foreground.getbbox()[1] - 3,
            foreground.getbbox()[2] + 3,
            foreground.getbbox()[3] + 3,
        ))
    else:
        if crop_to_content:
            pil_crop_frground = foreground.crop(foreground.getbbox())
        else:
            pil_crop_frground = foreground
    return pil_crop_frground


def crop_cutouts(img, add_padding=False):
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


def clean_foreground_edge(pil_img, smoothing=5):
    img = pil_img.filter(ImageFilter.ModeFilter(size=smoothing))
    return img


def extract_save_ind_frgrd(
    img: np.array,
    top_masks: "list(np.ndarray)",
    imgp: str,
    save_frgd_dir: str,
    testing=False,
    start_one_up=False,
    save_bbox=False,
):

    assert type(top_masks) is list, "top_masks is not a list of np.arrays"

    frgd_id = 0
    # Crop image using mask to create RGBA foreground (optional: 3 px padding)
    for component_mask in top_masks:
        exg_frgd = create_foreground(img, component_mask, add_padding=True)
        exg_frgd = clean_foreground_edge(exg_frgd)
        # exg_frgd = exg_frgd.filter(ImageFilter.DETAIL)

        if not os.path.exists(save_frgd_dir):
            os.makedirs(save_frgd_dir)

        if start_one_up:
            new_fname = imgp
            fname_prefix = (os.path.splitext(os.path.basename(new_fname))[0] +
                            "_" + str(frgd_id) + ".png")
        else:
            new_fname = imgp.split("/")[-2:][0] + "_" + imgp.split("/")[-2:][1]
            fname_prefix = (os.path.splitext(os.path.basename(new_fname))[0] +
                            "_" + str(frgd_id) + ".png")
        if save_bbox:
            bbox_txt = fname_prefix.split(".")[0] + ".txt"
            bbox_path = os.path.join(save_frgd_dir, bbox_txt)
            get_bbox(top_masks, bbox_path)

        frgd_path = os.path.join(save_frgd_dir, fname_prefix)

        exg_frgd.save(frgd_path)
        frgd_id += 1
        if testing:
            bgr_imgpath = os.path.join(save_frgd_dir,
                                       fname_prefix + "_ORIGINAL" + ".png")
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(bgr_imgpath, img_bgr)
    return exg_frgd


def extract_save_frgrd(img: np.array,
                       mask,
                       imgp: str,
                       save_frgd_dir: str,
                       testing=False):

    exg_frgd = create_foreground(img, mask, add_padding=True)
    # if not os.path.exists(save_frgd_dir):
    #     os.makedirs(save_frgd_dir)

    # fname_prefix = os.path.splitext(os.path.basename(imgp))[0]

    # frgd_path = os.path.join(save_frgd_dir, fname_prefix + ".png" )
    exg_frgd.save(save_frgd_dir)

    # if testing:
    # bgr_imgpath = os.path.join(save_frgd_dir, fname_prefix + "_ORIGINAL" + ".png" )
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(bgr_imgpath, img_bgr)
    return exg_frgd


def extract_detection_foreground(img,
                                 imgp,
                                 component_mask,
                                 save_frgd_dir,
                                 detection_id,
                                 testing=False):

    # Crop image using mask to create RGBA foreground (optional: 3 px padding)
    # for component_mask in top_masks:
    exg_frgd = create_foreground(img, component_mask, add_padding=True)

    if not os.path.exists(save_frgd_dir):
        os.makedirs(save_frgd_dir)

    fname_prefix = os.path.splitext(os.path.basename(imgp))[0]

    frgd_path = os.path.join(
        save_frgd_dir,
        fname_prefix + "_" + "detection_" + str(detection_id) + ".png")
    exg_frgd.save(frgd_path)
    if testing:
        bgr_imgpath = os.path.join(save_frgd_dir,
                                   fname_prefix + "_ORIGINAL" + ".png")
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(bgr_imgpath, img_bgr)


########################################################
############### OBJECT DETECTIOON UTILS ################
########################################################


def xywh_2_xyxy(x, y, w, h, img_h, img_w):
    dh, dw = img_h, img_w
    # Split string to float
    x1 = int((x - w / 2) * dw)  # top left x
    y1 = int((y - h / 2) * dh)  # top left y
    x2 = int((x + w / 2) * dw)  # bottom right x
    y2 = int((y + h / 2) * dh)  # bottom right y
    # Make sure dimensions aren't too big
    if x1 < 0:
        x1 = 0
    if x2 > dw - 1:
        x2 = dw - 1
    if y1 < 0:
        y1 = 0
    if y2 > dh - 1:
        y2 = dh - 1
    return [x1, y1, x2, y2]


def xywh2nxywh(x,
               y,
               sub_img_w,
               sub_img_h,
               species,
               super_img_w=1920,
               super_img_h=1080):
    # Converts non-normalized    coordinates and dimensions to normalized version for yolov5
    """Dont use"""
    norm = []

    x2 = x + sub_img_w
    y2 = y + sub_img_h

    new_x = round(((x + x2) / 2) / super_img_w, 6)  # x center
    new_y = round(((y + y2) / 2) / super_img_h, 6)  # y center

    w = round(sub_img_w / super_img_w, 6)
    h = round(sub_img_h / super_img_h, 6)

    norm.append(new_x)
    norm.append(new_y)
    norm.append(w)
    norm.append(h)
    new_norm = (species, *norm)

    return new_norm


def get_bbox_dep(label_path):
    """deprecated"""
    # Open label
    fl = open(label_path, "r")
    data = fl.readlines()
    fl.close()
    for dt in data:
        # Split string to float
        species, x, y, w, h, c = map(float, dt.split(" "))
    return species, x, y, w, h, c


def get_bbox(label_path, imgh, imgw, contains_conf=False):
    """For each yolo detectionresult in a text file, converts to pixel coordinates"""
    # Open label
    if contains_conf:
        arr = np.empty((0, 6), float)  # cls, x, y, w, h, conf
    else:
        arr = np.empty((0, 5), float)

    fl = open(label_path, "r")
    data = fl.readlines()
    fl.close()
    for dt in data:
        # Split string to float
        if contains_conf:
            species, x, y, w, h, conf = map(float, dt.split(" "))
            x1, y1, x2, y2 = xywh_2_xyxy(x, y, w, h, imgh, imgw)
            arr = np.append(arr,
                            np.array([[species, x1, y1, x2, y2, conf]]),
                            axis=0)
        else:
            species, x, y, w, h = map(float, dt.split(" "))
            x1, y1, x2, y2 = xywh_2_xyxy(x, y, w, h)
            arr = np.append(arr, np.array([[species, x1, y1, x2, y2]]), axis=0)
    return arr


def xywh_2_xyxyimg(img_path, label_path, show_image=False):
    """Reads in paths to image and yolo label txt file and outputs either;
    1. RGB numpy image cropped to bbox area or
    2. matplotlib image with bbox reactangle over original iamge"""

    # Read image data
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dh, dw, _ = img.shape
    # Open label
    fl = open(label_path, "r")
    data = fl.readlines()
    fl.close()
    for dt in data:
        # Split string to float
        species, x, y, w, h = map(float, dt.split(" "))
        x1 = int((x - w / 2) * dw)  # top left x
        y1 = int((y - h / 2) * dh)  # top left y
        x2 = int((x + w / 2) * dw)  # bottom right x
        y2 = int((y + h / 2) * dh)  # bottom right y

        # Make sure dimensions aren't too big
        if x1 < 0:
            x1 = 0
        if x2 > dw - 1:
            r = dw - 1
        if y1 < 0:
            t = 0
        if y2 > dh - 1:
            b = dh - 1
        # Create numpy image
        if not show_image:
            return species, img[y1:y2, x1:x2]
        else:
            return species, cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255),
                                          5)


def vegimg2label_path(img_path):
    # Define label paths as a function of image paths
    # Get single labels, some images don't have detection results
    label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    img_dir = os.path.dirname(img_path)
    label_dir = os.path.join(img_dir, "labels")
    label_path = os.path.join(label_dir, label_name)
    if not os.path.exists(label_path):
        return None
    return label_path


def bbox_areas(bboxes):
    areas = []
    for box in bboxes:
        # grab the coordinates of the bounding boxes
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        # Compute the area of each bounding boxes and store in list
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
    return areas


def box_area(box):

    # grab the coordinates of the bounding boxes
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    # Compute the area of each bounding boxes and store in list
    area = (x2 - x1) * (y2 - y1)
    return area


######################################################
############## TABULER CONVERSIONS ###################
######################################################


def csv2dict(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        result = {}
        for row in reader:
            for column, value in row.items():
                result[column] = value
    return result


######################################################
####################### VIZ ##########################
######################################################


class Relabel_Bbox_Extract_Mask_Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self, colortheme="lancet"):
        #          0        1         2         3          4        5          6         7
        if colortheme == "lancet":
            # ggsci.lancet.oncology.colorblind
            hex = ("00468b", "ed0000", "42b540", "0099b4", "925e9f", "fdaf91",
                   "ad002a")
        if colortheme == "journal_of_medicine":
            # ggsci.new.england.journal.of.medicine.colorblind
            hex = ("bc3c29", "0272b5", "e18727", "20854e", "7876b1", "6f99ad",
                   "ee4c97")
        if colortheme == "nature":
            # ggsci.nature.review.cancer.colorblind
            hex = ("f39b7f", "4dbbd5", "00a087", "3c5488", "91d1c2", "8491b4",
                   "dc0000")

        self.palette = [self.hex2rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def show(img, title=None):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis(False)
    plt.show()


def get_exg_histogram(exg):
    hist = cv2.calcHist([exg], [0], None, [256], [0, 256])
    # plot the histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()


def overlay_polygon(img, polygon):
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(polygon.exterior.coords)]
    return exterior


######################################################
################# ERROR HANDLING ####################
######################################################

# class CustomError(Exception):
#     """Creates custom error using inherited Exception object"""

#     def __init__(self, message):
#         super().__init__(message)

########################################################
####################### LOGGING  #######################
########################################################


def imgdata2df(glob_dir):
    """Takes image given a specific filename (wk_row_stop_timestamp.png)
    and turns that info into a pandas dataframe"""
    n = 1 if "week1" in glob_dir or "week2" in glob_dir else 0
    # Get list of images in week directory
    imgs = glob.glob(glob_dir)
    # Create pandas dataframe for easy handling
    df = pd.DataFrame(imgs, columns=["img"])
    # Split img columns to make row and stop columns
    df = df.img.str.split("/", expand=True)
    # Remove extraneous columns and split img path parts using location indexing
    ndf = df.iloc[:, 4].str.split("_", expand=True)
    # Create new date and time columns
    ndf["date"] = ndf.iloc[:, 3 - n] + ndf.iloc[:, 4 -
                                                n] + ndf.iloc[:, 5]  # yyyymmdd
    ndf["time"] = ndf.iloc[:, 6 - n] + ndf.iloc[:,
                                                7 - n] + ndf.iloc[:, 8 -
                                                                  n]  # hhmmss
    ndf["week"] = ndf.iloc[:, 0]
    # Create location columns
    ndf["row"] = ndf.iloc[:, 1]

    if n == 0:
        ndf["stop"] = ndf.iloc[:, 2]
        # Create 'image' column for filename
        ndf["image"] = (ndf["week"] + "_" + ndf["row"] + "_" + ndf["stop"] +
                        "_" + ndf["date"] + "_" + ndf["time"] + "_" +
                        ndf.iloc[:, 9])
    else:
        # Create 'image' column for filename
        ndf["image"] = (ndf["week"] + "_" + ndf["row"] + "_" + ndf["date"] +
                        "_" + ndf["time"] + "_" + ndf.iloc[:, 8])
        # Create empty 'stop' column name to concatenate with others
        ndf["stop"] = np.nan

    # Remove extraneous columns and organzie df
    ndf = ndf[["date", "time", "week", "row", "stop", "image"]]
    return ndf


########################################################
################ SYNTHETIC DATA TOOLS  #################
########################################################


def specify_species(img_list, species):
    """Gets list of image paths by specific row numbers.
    Uses species name to call specific rows."""

    all_imgs = [str(k) for k in img_list]

    if species == "clover":
        row1 = [str(k) for k in all_imgs if "row1_" in k]
        row2 = [str(k) for k in all_imgs if "row2_" in k]
        row3 = [str(k) for k in all_imgs if "row3_" in k]
        row4 = [str(k) for k in all_imgs if "row4_" in k]
        row5 = [str(k) for k in all_imgs if "row5_" in k]
        target = itertools.chain(row1, row2, row3, row4, row5)
        target = list(set(target))
        target_posix = [Path(k) for k in target]

    if species == "cowpea":
        row1 = [str(k) for k in all_imgs if "row5" in k]
        row2 = [str(k) for k in all_imgs if "row6" in k]
        row3 = [str(k) for k in all_imgs if "row7" in k]
        row4 = [str(k) for k in all_imgs if "row8" in k]
        target = itertools.chain(row1, row2, row3, row4)
        target = list(set(target))
        target_posix = [Path(k) for k in target]

    if species == "horseweed":
        row1 = [str(k) for k in all_imgs if "row4" in k]
        row2 = [str(k) for k in all_imgs if "row5" in k]
        row3 = [str(k) for k in all_imgs if "row6" in k]
        row4 = [str(k) for k in all_imgs if "row7" in k]
        row5 = [str(k) for k in all_imgs if "row8" in k]
        target = itertools.chain(row1, row2, row3, row4, row5)
        target = list(set(target))
        target_posix = [Path(k) for k in target]

    if species == "goosefoot" or species == "sunflower" or species == "velvetleaf":
        row1 = [str(k) for k in all_imgs if "row4" in k]
        row2 = [str(k) for k in all_imgs if "row5" in k]
        row3 = [str(k) for k in all_imgs if "row6" in k]
        row4 = [str(k) for k in all_imgs if "row7" in k]
        row5 = [str(k) for k in all_imgs if "row8" in k]
        row6 = [str(k) for k in all_imgs if "row9" in k]
        target = itertools.chain(row1, row2, row3, row4, row5, row6)
        target = list(set(target))
        target_posix = [Path(k) for k in target]

    if species == "grasses":
        row1 = [str(k) for k in all_imgs if "row8" in k]
        row2 = [str(k) for k in all_imgs if "row9" in k]
        row3 = [str(k) for k in all_imgs if "row10" in k]
        row4 = [str(k) for k in all_imgs if "row11" in k]
        row5 = [str(k) for k in all_imgs if "row12" in k]
        target = itertools.chain(row1, row2, row3, row4, row5)
        target = list(set(target))
        target_posix = [Path(k) for k in target]

    if species == "all":
        row1 = [str(k) for k in all_imgs if "row1" in k]
        row2 = [str(k) for k in all_imgs if "row2" in k]
        row3 = [str(k) for k in all_imgs if "row3" in k]
        row4 = [str(k) for k in all_imgs if "row4" in k]
        row5 = [str(k) for k in all_imgs if "row5" in k]
        row6 = [str(k) for k in all_imgs if "row6" in k]
        row7 = [str(k) for k in all_imgs if "row7" in k]
        row8 = [str(k) for k in all_imgs if "row8" in k]
        row9 = [str(k) for k in all_imgs if "row9" in k]
        row10 = [str(k) for k in all_imgs if "row10" in k]
        row11 = [str(k) for k in all_imgs if "row11" in k]
        row12 = [str(k) for k in all_imgs if "row12" in k]

        target = itertools.chain(row1, row2, row3, row4, row5, row6, row7,
                                 row8, row9, row10, row11, row12)

        target = list(set(target))
        target_posix = [Path(k) for k in target]

    return target_posix


########################################################
################### MAP VEGETATION  ####################
########################################################
"""Failed"""


def grid_image(img):
    """Returns an image broken up in six even portions.
    For image shape of (1920, 1080)
    """
    h, w = img.shape[:2]

    w1 = int(np.floor(w / 3))
    w2 = int(w1 + w1)
    h1 = int(h / 2)

    img1 = img[:h1, :w1]
    img2 = img[:h1, w1:w2]
    img3 = img[:h1, w2:]
    img4 = img[h1:, :w1]
    img5 = img[h1:, w1:w2]
    img6 = img[h1:, w2:]

    return [img1, img2, img3, img4, img5, img6]


def move_bboximgs(img_dir):
    imgs = sorted(glob.glob(img_dir + "/*rgb.png"), reverse=True)

    # Create list of txt file based on presence of matching png file
    imgs2txts = []
    for img in imgs:
        img2txt = str(Path(img).parent / Path(img).stem) + ".txt"
        if os.path.exists(img2txt):
            imgs2txts.append(img2txt)
    # Move images and text files to new directory
    for box in imgs2txts:
        imgbbox_dir = Path(box).parent
        # Create destination directory
        dst_dir = (Path(box).parts[0] + "/" + Path(box).parts[1] + "/" +
                   "bbox_package" + "/" + Path(box).parts[3])
        if not Path(dst_dir).is_file():
            Path(dst_dir).mkdir(exist_ok=True)
        # Get image and txt filenames
        imgbbox_fname = Path(box).stem + ".png"
        bbox_fname = Path(box).stem + ".txt"
        # Create sources and destination paths
        srcbbox = os.path.join(str(imgbbox_dir), bbox_fname)
        dstbbox = os.path.join(str(dst_dir), bbox_fname)
        srcimgs = os.path.join(str(imgbbox_dir), imgbbox_fname)
        dstimgs = os.path.join(str(dst_dir), imgbbox_fname)
        # Check that images and text files match
        if srcbbox in imgs2txts:
            # Check text is not empty
            if os.stat(srcbbox).st_size != 0:
                try:
                    # Copy files
                    shutil.copy(srcbbox, dstbbox)
                    shutil.copy(srcimgs, dstimgs)
                except:
                    continue

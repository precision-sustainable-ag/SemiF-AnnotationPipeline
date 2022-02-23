import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pandas import array
from shapely.geometry import Polygon
from skimage import morphology
from sklearn.cluster import KMeans

######## Get Images ##########


def read_img(imgpath, mode="RGB"):
    assert Path(imgpath).exists, "Image path doe not exist."
    if mode.upper() == "RGB":
        img = cv2.imread(str(imgpath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mode.upper() == "MASK":
        img = cv2.imread(str(imgpath), 0)

    return img


def get_imgs(
        parent_dir,
        extensions=["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]):
    # Check file and directory exists
    assert type(extensions) is list, "Input is not a list"
    assert Path(parent_dir).exists(), "Image directory is not valid.adasd"
    files = []
    # Parse locations and collection image files
    for ext in extensions:
        files.extend(Path(parent_dir).rglob(ext))
    return files


######## Vegetation Indices ##########


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
    return exr.astype('uint8')


def exg_minus_exr(img):
    img = img.astype(float)  # Rgb image
    exg = make_exg(img)
    exr = make_exr(img)
    exgr = exg - exr
    exgr = np.where(exgr < 25, 0, exgr)
    return exgr.astype('uint8')


######### Classify Masks #########


def make_kmeans(exg_mask):
    rows, cols = exg_mask.shape
    n_classes = 2
    X = exg_mask.reshape(rows * cols, 1)
    kmeans = KMeans(n_clusters=n_classes, random_state=3).fit(X)
    mask = kmeans.labels_.reshape(rows, cols)
    return mask.astype("uint8")


def check_kmeans(mask):
    max_sum = mask.shape[0] * mask.shape[1]
    ones_sum = np.sum(mask)
    if ones_sum > max_sum / 2:
        mask = np.where(mask == 1, 0, 1)
    return mask


def make_otsu(mask, kernel_size=(3, 3)):
    mask_blur = cv2.GaussianBlur(mask, kernel_size, 0).astype('uint8')
    _, mask_th3 = cv2.threshold(mask_blur, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask_th3


######## Clean Masks ##########
def reduce_holes(mask,
                 kernel_size=(3, 3),
                 min_object_size=1000,
                 min_hole_size=1000,
                 iterations=1,
                 dilation=True):
    mask = morphology.opening(mask, morphology.disk(3))
    mask = mask.astype('float32')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    if dilation:
        mask = cv2.dilate(mask, kernel, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))
    mask = cv2.erode(mask, (5, 5), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (7, 7))
    return mask


######## Sort Components ###########


def filter_topn_components(mask, top_n=3) -> 'list[np.ndarray]':
    # input must be single channel array
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


######## Contours #########
def contour_list2array(contour_list) -> list['np.array']:
    contour = np.squeeze(contour_list)
    arr_contours = Polygon(contour).exterior.coords
    return np.array(arr_contours, dtype=np.int32)


def crop_contours2contents():  # TODO create this
    # class Crop contours to contents
    # self.contour = contour[0]
    # x,y,w,h = cv2.boundingRect(self.contour)
    # # then crop it to bounding rectangle
    # crop = cutout[y:y+h, x:x+w]
    # fnd_crop_contours = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # self.crop_contours = fnd_crop_contours[0] if len(fnd_crop_contours) == 2 else fnd_crop_contours[1]
    # return crop
    pass


######## Tabular Conversion #########


def csv2dict(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        result = {}
        for row in reader:
            for column, value in row.items():
                result[column] = value
    return result


######## Visualize ##########


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


######## Error Handling #########


class CustomError(Exception):
    """Creates custom error using inherited Exception object"""

    def __init__(self, message):
        super().__init__(message)

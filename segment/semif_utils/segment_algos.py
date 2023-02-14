import sys

sys.path.append("..")
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.feature import peak_local_max
from skimage.filters import gaussian, rank
from skimage.segmentation import active_contour, watershed
from sklearn.cluster import KMeans

from semif_utils.utils import (clean_mask, clear_border, crop_cutouts,
                               dilate_erode, make_exg, make_kmeans,
                               reduce_holes, thresh_vi)


def process_general(img, vi_algo, vi_thresh, class_algo, reduce_holes=False):
    """ Vegetation index followed by classification algorithm, 
        and some morphological operations
    """
    vi = vi_algo(img, thresh=vi_thresh)
    th_vi = thresh_vi(vi)
    mask = class_algo(th_vi)
    mask = dilate_erode(mask)
    if reduce_holes:
        mask = reduce_holes(mask * 255)
    return mask


def watershed_vi_simple(img,
                        kernel=3,
                        clear_borders=True,
                        disk_size=4,
                        grad_thres=12):
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    vi = make_exg(img, thresh=True)
    th_vi = thresh_vi(vi)
    kmeans = make_kmeans(th_vi)
    if clear_borders:
        mask = clear_border(kmeans) * 255

    gradient = rank.gradient(thresh_vi, disk(disk_size)) < grad_thres
    markers, _ = ndi.label(mask)

    labels = watershed(gradient, markers)
    return labels


def watershed_vi_dist(img, kernel=3):
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    vi = make_exg(img, thresh=True)
    th_vi = thresh_vi(vi)
    kmeans = make_kmeans(th_vi)
    distance = ndi.distance_transform_edt(vi)
    coords = peak_local_max(distance,
                            footprint=np.ones((kernel, kernel)),
                            labels=img)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=exg)
    return labels


def process_cotyledon(img, bbox):
    """ Applies active contour models to fit open or closed splines to lines or edges in an image.
        Takes in cropped images from bbox.
    """

    y1, y2, x1, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
    # img = img[y1:y2, x1:x2]

    exg = make_exg(img)
    exg = thresh_vi(exg, low=20, upper=100, sigma=2)

    # initialize circle
    extent_h, extent_w = exg.shape[0], exg.shape[1]
    center_h, center_w = int(extent_h / 2), int(extent_w / 2)
    s = np.linspace(0, 2 * np.pi, 400)
    r = center_h + int(extent_h / 2) * np.sin(s)
    c = center_w + int(extent_w / 2) * np.cos(s)
    init = np.array([r, c]).T

    # Find contours
    snake = active_contour(gaussian(exg, 3, preserve_range=False),
                           init,
                           alpha=0.0005,
                           beta=2,
                           gamma=0.001,
                           w_edge=1,
                           boundary_condition="periodic",
                           w_line=0)

    # Mask using contours
    mask = np.zeros(exg.shape)
    rr, cc = polygon(snake[:, 0], snake[:, 1], mask.shape)

    mask[rr, cc] = 255
    mask = mask.astype("uint8")
    # Mask the array
    array_data = img.copy()
    array_data[np.where(mask == 0)] = 0
    cutout = crop_cutouts(array_data)
    kmeans_model = KMeans(n_clusters=3)  # we shall retain only 7 colors

    (h, w, c) = cutout.shape
    img2D = cutout.reshape(h * w, c)

    cluster_labels = kmeans_model.fit_predict(img2D)

    rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int)

    colors = [50, 150, 250]

    rgb_cols[0] = np.array([50, 50, 50])
    rgb_cols[1] = np.array([150, 150, 150])
    rgb_cols[2] = np.array([250, 250, 250])
    img_quant = np.reshape(rgb_cols[cluster_labels], (h, w, c))

    color_sums = {num: (img_quant == num).sum() for num in colors}
    color_sums = sorted(color_sums.items(), key=lambda x: x[1], reverse=True)
    threshold_val = color_sums[1][0]
    quant_mask = np.where(img_quant == threshold_val, 255, 0)[:, :, 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    closing = cv2.morphologyEx(quant_mask.astype("int16"), cv2.MORPH_CLOSE,
                               kernel)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

    cleaned_mask = clean_mask(closing)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # dilated_mask = cv2.dilate(cleaned_mask, kernel2, iterations = 1)
    # dilate_erode_mask = dilate_erode(cleaned_mask, kernel_size=5, dil_iters=5, eros_iters=5, hole_fill=True)

    # Mask the array
    array_data = cutout.copy()
    array_data[np.where(cleaned_mask == 0)] = 0
    cutout2 = crop_cutouts(array_data)
    return cutout2

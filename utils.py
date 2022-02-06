import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import csv
from datasets import ImageData


class ExtractVegetation:
    """ Extracts vegetation segments
    Inputs:
    data            = Color image (np.array) as an ImageData instance
    Returns:
    index_array     = Cutout images as an instance of ImageData

    :param rgb_img: np.array
    :return index_array: np.array
    """

    def __init__(self, data):
        self.data = data

    def exg(self):
        # Get exg
        print()

    def otsu(self):
        # Get Otsu
        print()

    def kmeans(self):
        print()


def get_files(
        parent_dir,
        extensions=["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]):
    """
    Get files with multiple extensions using Path.glob
    
    Inputs:
    parent_dir   = parent directory of images
    extensions   = list of extensions to include in search

    Returns:
    files        = list of file names

    :param parent_dir: str
    :param extensions: list of str
    :return files: list of str
    """
    # Check file and directory exists
    assert type(extensions) is list, "Input is not a list"
    assert Path(parent_dir).exists(), "Image directory is not valid.adasd"
    files = []
    # Parse locations and collection image files
    for ext in extensions:
        files.extend(Path(parent_dir).rglob(ext))
    return files


def showimg(img):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axes(False)
    plt.show()


def csv2dict(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        result = {}
        for row in reader:
            for column, value in row.items():
                result[column] = value
    return result


def exg(rgb_img):
    """Excess Green Index.
    r = R / (R + G + B)
    g = G / (R + G + B)
    b = B / (R + G + B)
    EGI = 2g - r - b
    The theoretical range for ExG is (-1, 2).
    Inputs:
    rgb_img      = Color image (np.array)
    Returns:
    index_array    = Index data as a Spectral_data instance
    :param rgb_img: np.array
    :return index_array: np.array
    """

    # Split the RGB image into component channels
    blue, green, red = cv2.split(rgb_img)
    # Calculate float32 sum of all channels
    total = red.astype(np.float32) + green.astype(np.float32) + blue.astype(
        np.float32)
    # Calculate normalized channels
    r = red.astype(np.float32) / total
    g = green.astype(np.float32) / total
    b = blue.astype(np.float32) / total
    index_array_raw = (2 * g) - r - b

    hsi = Spectral_data(array_data=None,
                        max_wavelength=0,
                        min_wavelength=0,
                        max_value=255,
                        min_value=0,
                        d_type=np.uint8,
                        wavelength_dict={},
                        samples=None,
                        lines=None,
                        interleave=None,
                        wavelength_units=None,
                        array_type=None,
                        pseudo_rgb=None,
                        filename=None,
                        default_bands=None)

    return _package_index(hsi=hsi, raw_index=index_array_raw, method="EGI")
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import csv
from datasets import ImageData


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


def show(img, title=None):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis(False)
    plt.show()
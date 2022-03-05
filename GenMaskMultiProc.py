import multiprocessing
from pathlib import Path

import cv2
from mongoengine import DynamicDocument, StringField, connect

from OpenCV2021DataBase import Mask
from utils import (check_kmeans, exg_minus_exr, make_exg, make_kmeans,
                   make_otsu, read_img, reduce_holes)


class Mask(DynamicDocument):
    meta = {
        "collection": "Mask",
    }


class MaskGenerator(object):
    DISPLAY_NAME = "invalid"

    def __init__(self):
        self.meta = None
        # self.params = cfg

    def read_image(self, imgpath):
        self.input_img = read_img(imgpath)

    def write_image(self, save_path, img):
        cv2.imwrite(save_path, img)

    def write_metadata(self):
        pass

    def process(self):
        pass  # override in derived classes to perform an actual segmentation

    def start_pipeline(self, data):
        self.img_path, self.params = data  # TODO make checks for paths

        self.clsproc = self.params.gen_mask.classify
        self.mask_savedir = self.params.general.mask_savedir

        self.read_image(self.img_path)
        return self.process()


class ExGMaskGenerator(MaskGenerator):
    DISPLAY_NAME = "ExG"

    def process(self):
        self.vi = make_exg(self.input_img)

        if "OTSU" in str.upper(self.clsproc):
            self.mask = make_otsu(self.vi)

        elif "KMEANS" in str.upper(self.clsproc):
            self.mask = check_kmeans(make_kmeans(self.vi)) * 255

        self.cleaned_mask = reduce_holes(self.mask)
        mask_savepath = str(Path(self.mask_savedir, Path(self.img_path).name))
        self.write_image(mask_savepath, self.cleaned_mask)
        # print(f"ExG saved to : {mask_savepath}\n")


class ExGRMaskGenerator(MaskGenerator):
    DISPLAY_NAME = 'ExGmR'

    def process(self):
        self.vi = exg_minus_exr(self.input_img)

        if "OTSU" in str.upper(self.clsproc):
            self.mask = make_otsu(self.vi)

        elif "KMEANS" in str.upper(self.clsproc):
            self.mask = check_kmeans(make_kmeans(self.vi)) * 255

        self.cleaned_mask = reduce_holes(self.mask)
        mask_savepath = str(Path(self.mask_savedir, Path(self.img_path).name))
        self.write_image(mask_savepath, self.cleaned_mask)
        # print(f"ExGmR saved to : {mask_savepath}\n")

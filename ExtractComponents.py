import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import cv2
import numpy as np
from shapely.geometry import Polygon

from OpenCV2021DataBase import CUTOUT_DICT
from utils import filter_topn_components, read_img

# TODO improve segmentation results be using outlier component extraction not just top 3


class ExtractAllCutouts:

    def __init__(self, mask_path):
        # TODO doc string
        """Extracts vegetation components from input image mask and returns list of np.arrays (refered to as "cutouts")
        """
        self.mask = read_img(mask_path, mode="MASK")
        self.mask_components = self.extract_mask_components(
        )  # Full framed : list[np.array]

    def extract_mask_components(self):
        return filter_topn_components(self.mask)


class ExtractIndCutouts(ExtractAllCutouts):
    # TODO doc string
    """Operates on individual componentsPer individual component

    Args:
        ExtractCutouts (_type_): _description_
    """

    def __init__(self, params, image_path):

        self.image_path = image_path
        self.mask_path = os.path.join(params.general.mask_savedir,
                                      Path(image_path).name)

        self.img = read_img(image_path, mode="RGB")
        self.cutout_savedir = params.general.cutout_savedir
        self.mask_components = ExtractAllCutouts(
            self.mask_path).mask_components
        self.crop_imgs = params.cutouts.crop
        self.mask_color = params.cutouts.background_color

    def extract_and_save_cutout(self):
        for idx, mask_comp in enumerate(self.mask_components):
            cutout_array = self.apply_mask(mask_comp)
            cutout_array = cv2.cvtColor(cutout_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.name_cutout(idx), cutout_array)

    def name_cutout(self, idx):
        img_stem = Path(self.image_path).stem
        cutout_name = f"{img_stem}_{str(idx).zfill(3)}.png"
        cutout_path = Path(self.cutout_savedir, cutout_name)
        return str(cutout_path)

    def crop2contents(self, mask_comp):
        x, y, w, h = cv2.boundingRect(mask_comp.astype(np.uint8))
        cropped_img = self.img[y:y + h, x:x + w]
        cropped_mask = mask_comp[y:y + h, x:x + w]
        return cropped_img, cropped_mask

    def apply_mask(self, mask_comp):

        if self.crop_imgs:
            img, mask = self.crop2contents(mask_comp)
        else:
            img, mask = self.img, read_img(self.mask_path, mode="MASK")

        if self.mask_color.upper() == "WHITE":
            color_val = 255
        elif self.mask_color.upper() == "BLACK":
            color_val = 0

        cutout_array = img.copy()
        mask_array = mask.copy()

        # Mask the array
        cutout_array[np.where(mask_array == 0)] = color_val
        return cutout_array


class ContourCollection(ExtractAllCutouts):

    def __init__(self, mask_dir):  #, imgobj):
        super().__init__(mask_dir)  #, imgobj)
        """
        Adds ListField ("cutouts"). Each list contains dictionaries of modified version of original image file name, 
        hashed new file name (uuid), and contours of mask components.
        Dictionary keys are "cutout_fname", "cutout_uuid", and "contours".

        Args:
            imgobj (object): mongoDB collection object
        """
        self.cutout_list = self.set_contours()

    def set_contours(self):
        cutout_list = []
        for idx, cutout in enumerate(self.cutout_contours):
            cutout_fname = f"{Path(self.mask_path).stem}_{str(idx)}.png"
            cutout_uuid = uuid5(NAMESPACE_URL, cutout_fname).hex
            cutout_as_list = cutout.tolist()

            cutout_dict = CUTOUT_DICT.copy()
            # Create and populate dictionary to store in collection
            cutout_dict["cutout_fname"] = cutout_fname
            cutout_dict["cutout_uuid"] = cutout_uuid
            cutout_dict["contours"] = cutout_as_list
            # Add to list variable that will be used to create new field
            cutout_list.append(cutout_dict)
        return cutout_list

    def get_contours(self):
        """
            """
        cutout_contours = []
        for idx, component in enumerate(self.mask_components):
            # get contours (presumably just one around the nonzero pixels)
            contours_tuple = cv2.findContours(component.astype(np.uint8),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

            cutout_contour = contours_tuple[0] if len(
                contours_tuple) == 2 else contours_tuple[1]

            contours_arr = self.contour_list2array(cutout_contour)

            cutout_contours.append(contours_arr)

        return cutout_contours

    def contour_list2array(self, contour_list) -> list['np.array']:
        contour = np.squeeze(contour_list)
        arr_contours = Polygon(contour).exterior.coords
        return np.array(arr_contours, dtype=np.int32)

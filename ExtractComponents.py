from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import cv2
import numpy as np
from shapely.geometry import Polygon

from ImageCollection import CUTOUT_DICT
from utils import filter_topn_components, read_img

# TODO improve segmentation results be using outlier component extraction not just top 3
# TODO Create scheme for saving cutouts to folder


class ExtractCutouts:

    def __init__(self, mask_dir, imgobj):
        """Extracts vegetation components from input mask and returns list of np.arrays (refered to as "cutouts")

        Args:
            mask_dir (str): location of mask imags
            imgobj (object): mongonDB collection object
        """

        self.imgobj = imgobj
        self.mask_path = Path(mask_dir, imgobj.file_name)
        self.pk = imgobj.pk
        self.mask = self.read_image()[:, :, 0]

        self.mask_components = self.extract_mask_components(
        )  # Full framed : list[np.array]

        self.cutout_contours: np.array = self.get_contours(
        )  # multiple contours within array

    def read_image(self):
        return read_img(self.mask_path)

    def extract_mask_components(self):
        """ Process for extracting vegetation components from an input mask
        Returns:
            list[np.arrays]: list of vegetation components
        """
        return filter_topn_components(self.mask.astype("uint8"))

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


class ContourCollection(ExtractCutouts):

    def __init__(self, mask_dir, imgobj):
        super().__init__(mask_dir, imgobj)
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

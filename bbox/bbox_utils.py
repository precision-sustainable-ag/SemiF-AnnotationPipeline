from typing import Callable, Tuple, List
from dataclasses import dataclass, field
import os
import numpy as np
import cv2


@dataclass
class _BBoxes:
    bboxes: List[object]


@dataclass
class Image:
    path: str
    bboxes: _BBoxes

    @property
    def array(self):
        img_array = cv2.imread(self.path)
        img_array = np.ascontiguousarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        return img_array


@dataclass
class ImageData:
    path: str
    images: list=field(init=False, default_factory=list)

    def __post_init__(self):
        image_list = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        self.images = [Image(image_path) for image_path in image_list]


@dataclass
class BBox:
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_right: np.ndarray
    bottom_left: np.ndarray
    image: Image


@dataclass
class BBoxes(_BBoxes):
    bboxes: List[BBox]


class BBoxMapper():

    def __init__(self, image_path: str, bbox_path: str):
        """Class to map bounding box coordinates from image cordinates
           to global coordinates
        """
        self.image_data = ImageData(image_path)

    def map(self):
        """
        Maps all the bounding boxes to a global coordinate space
        """
        pass
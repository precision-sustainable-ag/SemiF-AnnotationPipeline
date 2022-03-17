from ast import If
from typing import Callable, Tuple, List
from dataclasses import dataclass, field
import os
import numpy as np
import cv2

EPS = 1e-16

@dataclass
class BoxCoordinates:
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_left: np.ndarray
    bottom_right: np.ndarray

    def __eq__(self, __o: object) -> bool:
        
        if not isinstance(__o, BoxCoordinates):
            return False

        return all([
            np.sum((c1 - c2)**2) < EPS 
            for c1, c2 in zip(
                    [self.top_left, self.top_right, self.bottom_left, self.bottom_right], 
                    [__o.top_left, __o.top_right, __o.bottom_left, __o.bottom_right]
                )
            ])


def init_empty():
    empty_array = np.array([])
    # Initialize with an empty array
    return BoxCoordinates(empty_array, empty_array, empty_array, empty_array)

@dataclass
class BBox:
    id: str
    image_id: str
    cls: str
    local_coordinates: BoxCoordinates=field(init=True, default_factory=init_empty)
    global_coordinates: BoxCoordinates=field(init=True, default_factory=init_empty)
    is_primary: bool=field(init=False, default=False)

    def __eq__(self, __o: object) -> bool:
        
        if not isinstance(__o, BBox):
            return False
        is_eq = False
        if self.local_coordinates and self.local_coordinates == __o.local_coordinates:
            is_eq = True
        return is_eq


@dataclass
class BBoxes:
    bboxes: List[BBox]
    image_id: str


@dataclass
class Image:
    id: str
    path: str
    bboxes: BBoxes
    fov: BoxCoordinates
    width: int=field(init=False)
    height: int=field(init=False)

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


# class BBoxMapper():

#     def __init__(self, image_path: str, bbox_path: str):
#         """Class to map bounding box coordinates from image cordinates
#            to global coordinates
#         """
#         self.image_data = ImageData(image_path)

#     def map(self):
#         """
#         Maps all the bounding boxes to a global coordinate space
#         """
#         pass

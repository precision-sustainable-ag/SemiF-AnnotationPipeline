from typing import Callable, Tuple, List
from dataclasses import dataclass, field
import os
import numpy as np
import cv2


@dataclass
class BoxCoordinates:
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_left: np.ndarray
    bottom_right: np.ndarray


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

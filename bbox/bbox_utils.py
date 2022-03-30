from typing import List, Dict
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

    def __bool__(self):
        return all([len(coord) == 2 for coord in [self.top_left, self.top_right, self.bottom_left, self.bottom_right]])


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
    local_centroid: np.ndarray=field(init=False, default_factory=lambda : np.array([]))
    global_centroid: np.ndarray=field(init=False, default_factory=lambda : np.array([]))
    is_primary: bool=field(init=False, default=False)

    @property
    def local_area(self):
        if self._local_area is None:
            if self.local_coordinates:
                height = self.local_coordinates.bottom_left[1] - self.local_coordinates.top_left[1]
                width = self.local_coordinates.bottom_right[0] - self.local_coordinates.bottom_left[0]
                self._local_area = height * width
            else:
                raise AttributeError("local coordinates have to be defined for local area to be calculated.")
        return self._local_area

    @property
    def global_area(self):
        if self._global_area is None:
            if self.global_coordinates:
                height = self.global_coordinates.bottom_left[1] - self.global_coordinates.top_left[1]
                width = self.global_coordinates.bottom_right[0] - self.global_coordinates.bottom_left[0]
                self._global_area = height * width
            else:
                raise AttributeError("Global coordinates have to be defined for the global area to be calculated.")
        return self._global_area

    def __post_init__(self):

        self._local_area = None
        self._global_area = None

        if self.local_coordinates:
            self.local_centroid = self.get_centroid(self.local_coordinates)
        
        if self.global_coordinates:
            self.global_centroid = self.get_centroid(self.global_coordinates)

        # A list of all overlapping bounding boxes
        self._overlapping_bboxes: List[BBox] = []
    
    def add_box(self, box):
        self._overlapping_bboxes.append(box)

    def get_centroid(self, coords):
        # print(coords)
        centroid_x = (coords.bottom_right[0] + coords.bottom_left[0]) / 2.
        centroid_y = (coords.bottom_left[1] + coords.top_left[1]) / 2.
        centroid = np.array([centroid_x, centroid_y])

        return centroid

    def update_global_coordinates(self, global_coordinates: BoxCoordinates):

        assert not self.global_coordinates

        self.global_coordinates = global_coordinates
        self.global_centroid = self.get_centroid(self.global_coordinates)

    def bb_iou(self, comparison_box, type="global"):
    
        if type == "global":
            _boxA = self.global_coordinates
            _boxB = comparison_box.global_coordinates
        elif type == "local":
            _boxA = self.local_coordinates
            _boxB = comparison_box.local_coordinates
        else:
            raise ValueError(f"Type {type} not supported.")

        boxA = [_boxA.top_left[0], -_boxA.top_left[1], _boxA.bottom_right[0], -_boxA.bottom_right[1]]
        boxB = [_boxB.top_left[0], -_boxB.top_left[1], _boxB.bottom_right[0], -_boxB.bottom_right[1]]
        
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou


@dataclass
class Image:
    id: str
    path: str
    bboxes: List[BBox]
    fov: BoxCoordinates
    camera_location: np.ndarray
    pixel_width: float
    pixel_height: float
    yaw: float
    pitch: float
    roll: float
    focal_length: float
    width: int=field(init=False, default=-1)
    height: int=field(init=False, default=-1)

    @property
    def array(self):
        img_array = cv2.imread(self.path)
        img_array = np.ascontiguousarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        return img_array

    def __post_init__(self):

        image_array = self.array
        self.width = image_array.shape[1]
        self.height = image_array.shape[0]


def bb_iou(_boxA: BBox, _boxB: BBox):
    
    boxA = [_boxA.top_left[0], -_boxA.top_left[1], _boxA.bottom_right[0], -_boxA.bottom_right[1]]
    boxB = [_boxB.top_left[0], -_boxB.top_left[1], _boxB.bottom_right[0], -_boxB.bottom_right[1]]
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def generate_hash(box: BBox, auxilliary_hash=None):

    box_hash = box.id
    if auxilliary_hash is not None:
        
        box_hash = ",".join(sorted([auxilliary_hash, box_hash]))
    return box_hash

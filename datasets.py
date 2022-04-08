import uuid
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import yaml

from semif_utils import get_images

# from connectors import BBoxComponents

EPS = 1e-16


@dataclass
class BoxCoordinates:
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_left: np.ndarray
    bottom_right: np.ndarray

    def __bool__(self):
        # The bool function is to check if the coordinates are populated or not
        return all([
            len(coord) == 2 for coord in [
                self.top_left, self.top_right, self.bottom_left,
                self.bottom_right
            ]
        ])

    @property
    def config(self):
        _config = {
            "top_left": self.top_left.tolist(),
            "top_right": self.top_right.tolist(),
            "bottom_left": self.bottom_left.tolist(),
            "bottom_right": self.bottom_right.tolist()
        }

        return _config


def init_empty():
    empty_array = np.array([])
    # Initialize with an empty array
    return BoxCoordinates(empty_array, empty_array, empty_array, empty_array)


@dataclass
class BBox:
    id: str
    image_id: str
    cls: str
    local_coordinates: BoxCoordinates = field(init=True,
                                              default_factory=init_empty)
    global_coordinates: BoxCoordinates = field(init=True,
                                               default_factory=init_empty)
    local_centroid: np.ndarray = field(init=False,
                                       default_factory=lambda: np.array([]))
    global_centroid: np.ndarray = field(init=False,
                                        default_factory=lambda: np.array([]))
    is_primary: bool = field(init=False, default=False)

    @property
    def local_area(self):
        if self._local_area is None:
            if self.local_coordinates:
                height = self.local_coordinates.bottom_left[
                    1] - self.local_coordinates.top_left[1]
                width = self.local_coordinates.bottom_right[
                    0] - self.local_coordinates.bottom_left[0]
                self._local_area = height * width
            else:
                raise AttributeError(
                    "local coordinates have to be defined for local area to be calculated."
                )
        return self._local_area

    @property
    def global_area(self):
        if self._global_area is None:
            if self.global_coordinates:
                height = self.global_coordinates.bottom_left[
                    1] - self.global_coordinates.top_left[1]
                width = self.global_coordinates.bottom_right[
                    0] - self.global_coordinates.bottom_left[0]
                self._global_area = height * width
            else:
                raise AttributeError(
                    "Global coordinates have to be defined for the global area to be calculated."
                )
        return self._global_area

    @property
    def config(self):
        _config = {
            "id": self.id,
            "image_id": self.image_id,
            "local_coordinates": self.local_coordinates.config,
            "global_coordinates": self.global_coordinates.config,
            "is_primary": self.is_primary,
            "overlapping_bbox_ids":
            [box.id for box in self._overlapping_bboxes]
        }
        return _config

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
        """Adds a box as an overlapping box

        Args:
            box (BBox): BBox to add as an overlapping box
        """
        self._overlapping_bboxes.append(box)

    def get_centroid(self, coords: BoxCoordinates) -> np.ndarray:
        """Get the centroid of the bounding box based on the coordinates passed

        Args:
            coords (BoxCoordinates): Bounding box coordinates

        Returns:
            np.ndarray: Centroid
        """
        centroid_x = (coords.bottom_right[0] + coords.bottom_left[0]) / 2.
        centroid_y = (coords.bottom_left[1] + coords.top_left[1]) / 2.
        centroid = np.array([centroid_x, centroid_y])

        return centroid

    def update_global_coordinates(self, global_coordinates: BoxCoordinates):
        """Update the global coordinates of the bounding box

        Args:
            global_coordinates (BoxCoordinates): Global bounding box coordinates
        """
        assert not self.global_coordinates

        self.global_coordinates = global_coordinates
        self.global_centroid = self.get_centroid(self.global_coordinates)

    def bb_iou(self, comparison_box, type="global"):
        """Function to calculate the IoU of this bounding box
           with another bbox 'comparison_box'.

        Args:
            comparison_box (BBox): Another bounding box
            type (str, optional): IoU in global or local coordinates. Defaults to "global".

        Returns:
            float: IoU of the two boxes
        """
        if type == "global":
            _boxA = self.global_coordinates
            _boxB = comparison_box.global_coordinates
        elif type == "local":
            _boxA = self.local_coordinates
            _boxB = comparison_box.local_coordinates
        else:
            raise ValueError(f"Type {type} not supported.")

        boxA = [
            _boxA.top_left[0], -_boxA.top_left[1], _boxA.bottom_right[0],
            -_boxA.bottom_right[1]
        ]
        boxB = [
            _boxB.top_left[0], -_boxB.top_left[1], _boxB.bottom_right[0],
            -_boxB.bottom_right[1]
        ]

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


def bb_iou(boxA: BBox, boxB: BBox):
    """Function to calculate the IoU of two bounding boxes

    Args:
        _boxA (BBox): First bounding box
        _boxB (BBox): Secong bounding box

    Returns:
        _type_: _description_
    """

    _boxA = [
        boxA.top_left[0], -boxA.top_left[1], boxA.bottom_right[0],
        -boxA.bottom_right[1]
    ]
    _boxB = [
        boxB.top_left[0], -boxB.top_left[1], boxB.bottom_right[0],
        -boxB.bottom_right[1]
    ]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(_boxA[0], _boxB[0])
    yA = max(_boxA[1], _boxB[1])
    xB = min(_boxA[2], _boxB[2])
    yB = min(_boxA[3], _boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((_boxA[2] - _boxA[0]) * (_boxA[3] - _boxA[1]))
    boxBArea = abs((_boxB[2] - _boxB[0]) * (_boxB[3] - _boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def generate_hash(box: BBox, auxilliary_hash=None):
    """Generate a unique ID for the BBox. For now, the hash is the BBox.id,
       since the id is assumed to be unique (composed from the image_id).

    Returns:
        str: The unique ID for the box
    """

    box_hash = box.id
    # If another has is given, incorporate that
    # This is handy if the hash for a pair of bounding boxes is to be
    # determined
    if auxilliary_hash is not None:

        box_hash = ",".join(sorted([auxilliary_hash, box_hash]))
    return box_hash


@dataclass
class EnvironmentalMetadata:
    site_id: str
    hardiness_zone: str = field(init=False)
    humidity: float = field(default=None)
    precipitation: float = field(default=None)

    def __post_init__(self):
        sid = self.site_id.upper()
        hd_zone = {"TX": "8b", "NC": "7b", "MD": "7a"}
        self.hardiness_zone = hd_zone[sid]


@dataclass
class BatchMetadata:
    """ Batch metadata class for yaml loader"""
    upload_dir: str
    site_id: str
    upload_datetime: str
    batch_id: str = field(init=False)
    environmental_metadata: EnvironmentalMetadata = field(init=False)
    image_list: List = field(init=False)
    schema_version: str = "v1"

    def __post_init__(self):
        self.batch_id = str(uuid.uuid4())
        self.environmental_metadata = EnvironmentalMetadata(self.site_id)
        self.image_list = self.get_batch_images()

    def get_batch_images(self):
        images = get_images(self.upload_dir, sort=True)
        image_ids = [image.stem for image in images]

        image_list = [{
            "id": image_id,
            "path": str(path)
        } for image_id, path in zip(image_ids, images)]
        return image_list


def batchmetadata_constructor(loader: yaml.SafeLoader,
                              node: yaml.nodes.MappingNode) -> BatchMetadata:
    return BatchMetadata(**loader.construct_mapping(node))


def get_batchloader():
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    loader.add_constructor("!BatchMetadata", batchmetadata_constructor)
    return loader


@dataclass
class CameraInfo:
    "Per individual image."
    camera_location: np.ndarray
    pixel_width: float
    pixel_height: float
    yaw: float
    pitch: float
    roll: float
    focal_length: float
    # fov: BBoxComponents = None


@dataclass
class ImageData:
    """ Data and metadata for individual images """
    image_id: str
    image_path: str
    batch_id: str
    bboxes: BBox = None
    camera_info: CameraInfo = None
    width: int = field(init=False, default=-1)
    height: int = field(init=False, default=-1)

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        img_array = cv2.imread(self.image_path)
        img_array = np.ascontiguousarray(
            cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        return img_array

    def __post_init__(self):
        image_array = self.array
        self.width = image_array.shape[1]
        self.height = image_array.shape[0]

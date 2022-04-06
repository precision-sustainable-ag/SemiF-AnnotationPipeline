import csv
import dataclasses
import glob
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml

from semif_utils import get_bbox, get_images

# from chinmay_dummy_data.bbox.bbox_utils import BBox, BoxCoordinates


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
    """ Input batch metadata """
    site_id: str
    upload_datetime: str
    upload_dir: str
    env_metadata: EnvironmentalMetadata
    upload_id: str = field(init=False)
    schema_version: str = field(default="v1")

    def __post_init__(self):
        self.upload_id = str(uuid.uuid4())


@dataclass
class CameraData:
    "Per individual image."
    # fov: BoxCoordinates
    camera_location: np.ndarray
    pixel_width: float
    pixel_height: float
    yaw: float
    pitch: float
    roll: float
    focal_length: float


@dataclass
class ImageData:
    """ Data and metadata for individual images """
    image_id: str
    image_path: str
    upload_id: str
    # bboxes: List[BBox]
    camera_data: CameraData = field(default=None)
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


@dataclass
class BatchImageList:
    """List of images in batch upload"""

    batch_metadata: BatchMetadata
    image_list: List = field(init=False)

    def __post_init__(self):
        image_dir = self.batch_metadata.upload_dir
        images = get_images(image_dir, sort=True)
        image_ids = [image.stem for image in images]

        self.image_list = [{
            "id": image_id,
            "path": str(path)
        } for image_id, path in zip(image_ids, images)]

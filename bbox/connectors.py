from typing import Callable
import os
import numpy as np
import pandas as pd
from .bbox_utils import Image, BBox, BBoxes, BoxCoordinates

class SfMComponents:

    def __init__(self, base_path, 
                 camera_reference_file="camera_reference.csv", 
                 fov_reference_file="fov.csv", 
                 gcp_reference_file="gcp_reference.csv"):
        
        self.base_path = base_path
        self.camera_reference_file = camera_reference_file
        self.fov_reference_file = fov_reference_file
        self.gcp_reference_file = gcp_reference_file

        self._camera_reference = None
        self._fov_reference = None
        self._gcp_reference = None

    @property
    def camera_reference(self):
        if self._camera_reference is None:
            self._camera_reference = pd.read_csv(os.path.join(self.base_path, self.camera_reference_file))
        return self._camera_reference

    @property
    def fov_reference(self):
        if self._fov_reference is None:
            self._fov_reference = pd.read_csv(os.path.join(self.base_path, self.fov_reference_file))
        return self._fov_reference

    @property
    def gcp_reference(self):
        if self._gcp_reference is None:
            self._gcp_reference = pd.read_csv(os.path.join(self.base_path, self.gcp_reference_file))
        return self._gcp_reference


class BBoxComponents:
    """Reads bounding box coordinate files and converts to BBox class
    """

    def __init__(self, reader: Callable, *args, **kwargs):
        
        self.reader = reader
        self.image_list, self.bounding_boxes = self.reader(*args, **kwargs)

        self._bboxes = dict()
        self._images = []

    def convert_bboxes(self):
        
        assert not self._bboxes
        for image_id, bboxes in self.bounding_boxes.items():
            boxes = []
            for bbox in bboxes:

                unique_box_id = "_".join([image_id, bbox["id"]])

                # Convert the coordinates
                top_left = np.array(bbox["top_left"])
                top_right = np.array(bbox["top_right"])
                bottom_left = np.array(bbox["bottom_left"])
                bottom_right = np.array(bbox["bottom_right"])

                box_coordinates = BoxCoordinates(top_left, top_right, bottom_left, bottom_right)
                box = BBox(id=unique_box_id, image_id=image_id, local_coordinates=box_coordinates, cls=bbox["cls"])
                boxes.append(box)
            bounding_boxes = BBoxes(bboxes=boxes, image_id=image_id)
            self._bboxes[image_id] = bounding_boxes

    @property
    def bboxes(self):
        if not self._bboxes:
            # Convert
            self.convert_bboxes()
        return self._bboxes
        
    @property
    def images(self):
        if not self._images:
            bboxes = self.bboxes
            for image in self.image_list:
                image_id = image["id"]
                path = image["path"]
                image = Image(path=path, id=image_id, bboxes=bboxes[image_id], fov=[])
                self._images.append(image)
        return self._images

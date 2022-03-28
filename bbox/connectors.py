from typing import Callable
import os
import numpy as np
import pandas as pd
from .bbox_utils import Image, BBox, BoxCoordinates

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

    def __init__(self, fov_reference: pd.DataFrame, camera_reference: pd.DataFrame, 
                 reader: Callable, *args, **kwargs):
        
        self.fov_reference = fov_reference
        self.camera_reference = camera_reference
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
            # bounding_boxes = BBoxes(bboxes=boxes, image_id=image_id)
            self._bboxes[image_id] = boxes # bounding_boxes

    def get_fov(self, image_id):
        
        row = self.fov_reference[self.fov_reference["label"] == image_id].reset_index(drop=True)

        top_left_x = float(row.loc[0, "top_left_x"])
        top_left_y = float(row.loc[0, "top_left_y"])

        bottom_left_x = float(row.loc[0, "bottom_left_x"])
        bottom_left_y = float(row.loc[0, "bottom_left_y"])
        
        top_right_x = float(row.loc[0, "top_right_x"])
        top_right_y = float(row.loc[0, "top_right_y"])
        
        bottom_right_x = float(row.loc[0, "bottom_right_x"])
        bottom_right_y = float(row.loc[0, "bottom_right_y"])

        top_left = np.array([top_left_x, top_left_y])
        bottom_left = np.array([bottom_left_x, bottom_left_y])
        top_right = np.array([top_right_x, top_right_y])
        bottom_right = np.array([bottom_right_x, bottom_right_y])

        fov = BoxCoordinates(top_left, top_right, bottom_left, bottom_right)

        return fov

    def get_camera_location(self, image_id):

        row = self.camera_reference[self.camera_reference["label"] == image_id].reset_index(drop=True)
        camera_x = float(row.loc[0, "Estimated_X"])
        camera_y = float(row.loc[0, "Estimated_Y"])
        camera_z = float(row.loc[0, "Estimated_Z"])

        return np.array([camera_x, camera_y, camera_z])

    def get_pixel_dims(self, image_id):

        row = self.camera_reference[self.camera_reference["label"] == image_id].reset_index(drop=True)
        pixel_width = float(row.loc[0, "pixel_width"])
        pixel_height = float(row.loc[0, "pixel_height"])

        return pixel_width, pixel_height

    def get_orientation_angles(self, image_id):

        row = self.camera_reference[self.camera_reference["label"] == image_id].reset_index(drop=True)
        yaw = float(row.loc[0, "Estimated_Yaw"])
        pitch = float(row.loc[0, "Estimated_Pitch"])
        roll = float(row.loc[0, "Estimated_Roll"])

        return yaw, pitch, roll

    def get_focal_length(self, image_id):

        row = self.camera_reference[self.camera_reference["label"] == image_id].reset_index(drop=True)
        focal_length = float(row.loc[0, "f"])
        return focal_length

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
                fov = self.get_fov(image_id)
                camera_location = self.get_camera_location(image_id)
                pixel_width, pixel_height = self.get_pixel_dims(image_id)
                yaw, pitch, roll = self.get_orientation_angles(image_id)
                focal_length = self.get_focal_length(image_id)

                image = Image(
                    path=path, id=image_id, bboxes=bboxes[image_id], 
                    fov=fov, camera_location=camera_location, 
                    pixel_width=pixel_width, pixel_height=pixel_height,
                    yaw=yaw, pitch=pitch, roll=roll, focal_length=focal_length
                )
                self._images.append(image)
        return self._images

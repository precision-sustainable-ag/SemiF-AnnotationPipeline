import glob
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from mongoengine import (Document, IntField, ListField, StringField, UUIDField,
                         connect)

CUTOUT_DICT = {"cutout_fname": None, "cutout_uuid": None, "contours": None}

connect(db="opencv2021", host="localhost", port=27017)


class Image(Document):
    file_name = StringField(required=True, unique=True)
    uuid = UUIDField(required=True)
    date = IntField(required=True)
    time = IntField(required=True)
    week = IntField()
    row = IntField()
    stop = IntField()
    meta = {'allow_inheritance': True}


# TODO reassess need for everything below
class InputMetadata:
    """ Input metadata to create a metadata yaml file. """

    def __init__(self, output_path):
        self.output_path = output_path
        self.upload_id = self.upload_id()
        input_metadata = self.inputdata()

    def inputdata(self):
        image_dir = input("image directory: ")
        upload_id = self.upload_id
        date = input("date: ")
        start_time = input("Start time: ")
        end_time = input("End time: ")
        location = input("location: ")
        cloud_cover = input("Cloud cover: ")
        camera_height = input("Camera height (m): ")
        camera_lens = input("Camera lens (mm): ")
        pot_height = input("Pot height (m): ")

        data_dict = [{
            "upload_path": self.output_path,
            "image_dir": image_dir,
            "upload_id": upload_id,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "location": location,
            "cloud_cover": cloud_cover,
            "camera_height": camera_height,
            "camera_lens": camera_lens,
            "pot_height": pot_height
        }]

        with open(self.output_path, 'w') as f:
            # f.write(json.dumps(data_dict))
            yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)
            f.close()

    def upload_id(self):
        uid = str(uuid.uuid4())
        return uid


@dataclass
class ImageData:
    """ Loads images and metadata. Is also iterable to access image arrays
    Parameters:
    upload_id = uuid str
    path = path to image directory
    date = date of image acquisition from metadata
    time = time of image acquisition from metadata
    location = TX, MD, or NC
    cloud_cover = brief description of cloud cover
    camera_height = height of camera on benchbot in meters
    camera_lens = 35 or 55 in millimeteres
    pot_height = from top to bottom in meters
    files = list of image file paths
    nf = number of image files
    
    : params upload_id: str
    : params path: str
    : params date: str
    : params time: str
    : params location: str
    : params cloud_cover: str
    : params camera_height: float
    : params camera_lens: float
    : params pot_height: float
    : params files: list[str]
    : params nf: int
    """
    upload_id: str
    path: str
    date: str
    time: str
    location: str
    cloud_cover: str
    camera_height: str
    camera_lens: str
    pot_height: str
    files: list = field(default_factory=list)
    nf: int = field(default_factory=int)

    def __post_init__(self):
        IMG_FORMATS = ['JPG', 'jpg', 'JPEG', 'jpeg', 'PNG',
                       'png']  # include image suffixes

        p = str(Path(self.path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        # Number of images
        self.nf = len(images)
        self.files = images
        assert self.nf > 0, f'No images found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, f'Image Not Found {path}'
        s = f'image {self.count}/{self.nf} {path}: '

        # Convert
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img)
        return path, img, img0, s

    def __len__(self):
        return self.nf  # number of files

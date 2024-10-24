import datetime
import json
import os
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import cv2
import exifread
import numpy as np

SCHEMA_VERSION = "1.0"


@dataclass
class BoxCoordinates:
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_left: np.ndarray
    bottom_right: np.ndarray
    is_scaleable: bool = field(init=True, default=True)

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
        if isinstance(self.top_left, np.ndarray):
            _top_left = self.top_left.tolist()
            _top_right = self.top_right.tolist()
            _bottom_left = self.bottom_left.tolist()
            _bottom_right = self.bottom_right.tolist()
        else:
            _top_left = self.top_left
            _top_right = self.top_right
            _bottom_left = self.bottom_left
            _bottom_right = self.bottom_right

        _config = {
            "top_left": _top_left,
            "top_right": _top_right,
            "bottom_left": _bottom_left,
            "bottom_right": _bottom_right
        }

        return _config

    def set_scale(self, new_scale: np.ndarray):

        if not self.is_scaleable:
            raise ValueError(
                "is_scalable set to False, coordinates cannot be scaled.")
        self.scale = new_scale

        self.top_left = self.top_left * self.scale
        self.top_right = self.top_right * self.scale
        self.bottom_left = self.bottom_left * self.scale
        self.bottom_right = self.bottom_right * self.scale

    def copy(self):

        return self.__class__(top_left=self.top_left.copy(),
                              top_right=self.top_right.copy(),
                              bottom_left=self.bottom_left.copy(),
                              bottom_right=self.bottom_right.copy(),
                              is_scaleable=self.is_scaleable)

    def __getitem__(self, key):

        if not hasattr(self, key):
            raise AttributeError(f"{self.__class__.__name__} has not attribute {key}")

        return getattr(self, key)


def init_empty():
    empty_array = np.array([])
    # Initialize with an empty array
    return BoxCoordinates(empty_array, empty_array, empty_array, empty_array)


@dataclass
class BBox:
    bbox_id: str
    image_id: str
    cls: str
    local_coordinates: BoxCoordinates = field(init=True,
                                              default_factory=init_empty)
    global_coordinates: BoxCoordinates = field(init=True,
                                               default_factory=init_empty)
    is_normalized: bool = field(init=True, default=True)
    local_centroid: np.ndarray = field(init=False,
                                       default_factory=lambda: np.array([]))
    global_centroid: np.ndarray = field(init=False,
                                        default_factory=lambda: np.array([]))
    is_primary: bool = field(init=False, default=False)
    norm_local_coordinates: BoxCoordinates = field(init=False,
                                                   default_factory=init_empty)

    @property
    def local_area(self):
        if self.local_coordinates:
            local_area = self.get_area(self.local_coordinates)
        else:
            raise AttributeError(
                "Local coordinates have to be defined for local area to be calculated."
            )
        return local_area

    @property
    def norm_local_area(self):
        if self.norm_local_coordinates:
            norm_local_area = self.get_area(self.norm_local_coordinates)
        else:
            raise AttributeError(
                "Normalized local coordinates have to be defined for local area to be calculated."
            )
        return norm_local_area

    @property
    def global_area(self):
        if self.global_coordinates:
            global_area = self.get_area(self.global_coordinates)
        else:
            raise AttributeError(
                "Global coordinates have to be defined for the global area to be calculated."
            )
        return global_area

    @property
    def config(self):
        _config = {
            "bbox_id":
            self.bbox_id,
            "image_id":
            self.image_id,
            "local_centroid":
            list(
                self.norm_local_centroid),  # Always use normalized coordinates
            "local_coordinates":
            self.norm_local_coordinates.
            config,  # Always use normalized coordinates
            "global_centroid":
            list(self.global_centroid),
            "global_coordinates":
            self.global_coordinates.config,
            "is_primary":
            self.is_primary,
            "cls":
            self.cls,
            "overlapping_bbox_ids":
            [box.bbox_id for box in self._overlapping_bboxes],
            "num_overlapping_bboxes":
            len(self._overlapping_bboxes)
        }
        return _config

    def __post_init__(self):

        if self.local_coordinates:
            self.set_local_centroid()
            if self.is_normalized:
                self.norm_local_coordinates = self.local_coordinates.copy()
                self.norm_local_coordinates.is_scaleable = False
                self.set_norm_local_centroid()

        if self.global_coordinates:
            self.set_global_centroid()

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

    def get_area(self, coordinates: BoxCoordinates) -> float:
        height = coordinates.bottom_left[1] - coordinates.top_left[1]
        width = coordinates.bottom_right[0] - coordinates.bottom_left[0]
        return float(height * width)

    def set_local_centroid(self):
        self.local_centroid = self.get_centroid(self.local_coordinates)

    def set_norm_local_centroid(self):
        self.norm_local_centroid = self.get_centroid(
            self.norm_local_coordinates)

    def set_global_centroid(self):
        self.global_centroid = self.get_centroid(self.global_coordinates)

    def set_local_scale(self, new_scale):
        self.local_coordinates.set_scale(new_scale)
        self.set_local_centroid()

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

        # compute the area of intersection rectanglee
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


# Batch Metadata ---------------------------------------------------------------------------


@dataclass
class BatchMetadata:
    """ Batch metadata class for yaml loader"""
    blob_home: str
    data_root: str
    batch_id: str
    upload_datetime: str
    image_list: List = field(init=False)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        self.image_list = self.get_batch_images()

    def get_batch_images(self):
        imgdir = Path(self.blob_home, self.data_root, self.batch_id, "images")
        extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        images = []
        for ext in extensions:
            images.extend(Path(imgdir).glob(ext))
        image_ids = [image.stem for image in images]
        image_list = []

        for ids, imgp in zip(image_ids, images):
            image_list.append(str("/".join(imgp.parts[-2:])))

        return image_list

    @property
    def config(self):
        _config = {
            "blob_home": self.blob_home,
            "data_root": self.data_root,
            "batch_id": self.batch_id,
            "upload_datetime": self.upload_datetime,
            "image_list": self.image_list,
            "schema_version": self.schema_version
        }

        return _config

    def save_config(self):
        try:
            save_batch_path = Path(self.blob_home, self.data_root,
                                   self.batch_id, self.batch_id + ".json")
            with open(save_batch_path, "w") as f:
                json.dump(self.config, f, indent=4, default=str)
        except Exception as e:
            raise e
        return True


# Image dataclasses ----------------------------------------------------------


@dataclass
class ImageMetadata:
    ImageWidth: int
    ImageLength: int
    BitsPerSample: int
    Compression: int
    PhotometricInterpretation: int
    Make: str
    Model: str
    Orientation: int
    SamplesPerPixel: int
    XResolution: str
    YResolution: str
    PlanarConfiguration: int
    ResolutionUnit: int
    Software: str
    DateTime: str
    Rating: int
    ExifOffset: int
    ExposureTime: str
    FNumber: str
    ExposureProgram: int
    ISOSpeedRatings: int
    RecommendedExposureIndex: int
    ExifVersion: list
    DateTimeOriginal: str
    DateTimeDigitized: str
    BrightnessValue: str
    ExposureBiasValue: str
    MaxApertureValue: str
    MeteringMode: int
    LightSource: int
    Flash: int
    FocalLength: str
    FileSource: int
    SceneType: int
    CustomRendered: int
    ExposureMode: int
    WhiteBalance: int
    DigitalZoomRatio: str
    FocalLengthIn35mmFilm: int
    SceneCaptureType: int
    Contrast: int
    Saturation: int
    Sharpness: int
    LensSpecification: list
    LensModel: str
    BodySerialNumber: Optional[str] = None
    MakerNote: Optional[str] = None
    ImageDescription: Optional[str] = None
    UserComment: Optional[str] = None
    ApplicationNotes: Optional[str] = None
    Tag: Optional[int] = None
    SubIFDs: Optional[int] = None


@dataclass
class CameraInfo:
    """ 
    """
    camera_location: np.ndarray
    pixel_width: float
    pixel_height: float
    yaw: float
    pitch: float
    roll: float
    focal_length: float
    fov: BoxCoordinates.config = None


@dataclass
class Box:
    bbox_id: str
    image_id: str
    local_centroid: list
    local_coordinates: BoxCoordinates
    global_centroid: list
    global_coordinates: BoxCoordinates
    cls: str
    is_primary: bool
    overlapping_bbox_ids: List[BBox] = field(init=False, default_factory=lambda : [])

    def assign_species(self, species):
        self.cls = species


@dataclass
class BBoxFOV:
    top_left: list
    top_right: list
    bottom_left: list
    bottom_right: list


@dataclass
class BBoxMetadata:
    data_root: str
    batch_id: str
    image_path: str
    image_id: str
    width: int
    height: int
    camera_info: CameraInfo
    exif_meta: ImageMetadata
    bboxes: list[Box]


@dataclass
class Image:
    """Parent class for RemapImage and ImageData.

    """
    blob_home: str
    data_root: str
    batch_id: str
    image_path: str
    image_id: str
    plant_date: str
    growth_stage: str

    def __post_init__(self):
        image_array = self.array
        self.width = image_array.shape[1]
        self.height = image_array.shape[0]

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        img_path = Path(self.blob_home, self.data_root, self.batch_id,
                        self.image_path)
        img_array = cv2.imread(str(img_path))
        img_array = np.ascontiguousarray(
            cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        return img_array

    @property
    def config(self):
        _config = {
            "blob_home": self.blob_home,
            "data_root": self.data_root,
            "batch_id": self.batch_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "plant_date": self.plant_date,
            "growth_stage": self.growth_stage,
            "width": self.width,
            "height": self.height,
            "exif_meta": asdict(self.exif_meta),
            "camera_info": asdict(self.camera_info),
            "bboxes": [box.config for box in self.bboxes],
            "schema_version": self.schema_version
        }

        return _config

    def save_config(self, save_path):
        try:
            save_file = os.path.join(save_path, f"{self.image_id}.json")
            with open(save_file, "w") as f:
                json.dump(self.config, f, indent=4, default=str)
        except Exception as e:
            raise e
        return True


@dataclass
class RemapImage(Image):
    """ For remapping labels (remap_labels) """
    bboxes: list[BBox]
    camera_info: CameraInfo
    fullres_path: str
    width: int = field(init=False, default=-1)
    height: int = field(init=False, default=-1)
    exif_meta: Optional[ImageMetadata] = field(init=False, default=None)
    fullres_height: Optional[int] = field(init=False, default=-1)
    fullres_width: Optional[int] = field(init=False, default=-1)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        self.height, self.width = self.array.shape[:2]
        self.set_fullres_dims(self.width, self.height)
        self.exif_meta = self.get_exif()

    def set_fullres_dims(self, fullres_width, fullres_height):
        self.fullres_width = fullres_width
        self.fullres_height = fullres_height

    def get_exif(self):
        """Creates a dataclass by reading exif metadata, creating a dictionary, and creating dataclass form that dictionary
        """
        # Open image file for reading (must be in binary mode)
        f = open(self.image_path, 'rb')
        # Return Exif tags
        tags = exifread.process_file(f, details=False)
        f.close()
        meta = {}
        for x, y in tags.items():
            newval = y.values[0] if type(y.values) == list and len(
                y.values) == 1 else y.values
            if type(newval) == exifread.utils.Ratio:
                newval = str(newval)
            meta[x.rsplit(" ")[1]] = newval
        imgmeta = ImageMetadata(**meta)
        return imgmeta

    @property
    def config(self):
        _config = super(RemapImage, self).config
        _config["fullres_width"] = self.fullres_width
        _config["fullres_height"] = self.fullres_height
        

        return _config


@dataclass
class ImageData(Image):
    """ Dataclass for segmentation data generation"""

    width: int
    height: int
    exif_meta: ImageMetadata
    cutout_ids: List[str] = None
    camera_info: CameraInfo = None
    bboxes: list[Box] = None
    fullres_height: int = -1
    fullres_width: int = -1
    schema_version: str = "1.0"

    def __post_init__(self):
        # Overload the post init of the super class
        # which reads the array for the width and height.
        # The width and height will be available in the metadata
        pass
    @property
    def config(self):
        _config = {
            "blob_home": self.blob_home,
            "data_root": self.data_root,
            "batch_id": self.batch_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "plant_date": self.plant_date,
            "growth_stage": self.growth_stage,
            "width": self.width,
            "height": self.height,
            "exif_meta": asdict(self.exif_meta),
            "camera_info": asdict(self.camera_info),
            "cutout_ids": self.cutout_ids,
            "bboxes": [asdict(x) for x in self.bboxes],
            "fullres_height": self.fullres_height,
            "fullres_width": self.fullres_width,
            "schema_version": self.schema_version
        }

        return _config

    def save_config(self, save_path):
        try:
            save_image_path = Path(save_path, self.image_id + ".json")
            with open(save_image_path, "w") as f:
                json.dump(self.config, f, indent=4, default=str)
        except Exception as e:
            raise e
        return True


@dataclass
class Mask:
    mask_id: str
    mask_path: str
    image_id: str
    width: int = field(init=False, default=-1)
    height: int = field(init=False, default=-1)

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        mask_array = cv2.imread(self.mask_path)
        mask_array = np.ascontiguousarray(
            cv2.cvtColor(mask_array, cv2.COLOR_BGR2RGB))
        return mask_array

    def __post_init__(self):
        mask_array = self.array
        self.width = mask_array.shape[1]
        self.height = mask_array.shape[0]

    def save_mask(self, save_path):
        try:
            save_file = os.path.join(save_path, f"{self.image_id}.png")
            cv2.imwrite(save_file, self.array)

        except Exception as e:
            raise e
        return True


# Cutouts -------------------------------------------------------------------------------------


@dataclass
class CutoutProps:
    """Region properties for cutouts
    "area",  # float Area of the region i.e. number of pixels of the region scaled by pixel-area.
    "area_bbox",  # float Area of the bounding box i.e. number of pixels of bounding box scaled by pixel-area.
    "area_convex",  # float Are of the convex hull image, which is the smallest convex polygon that encloses the region.
    "axis_major_length",  # float The length of the major axis of the ellipse that has the same normalized second central moments as the region.
    "axis_minor_length",  # float The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
    "centroid",  # array Centroid coordinate list [row, col].
    "eccentricity",  # float Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
    "extent",  # float Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
    "solidity",  # float Ratio of pixels in the region to pixels of the convex hull image.
    "perimeter",  # float Perimeter of object which approximates the contour as a line 
    """
    area: Union[float, list]
    area_bbox: Union[float, list]
    area_convex: Union[float, list]
    axis_major_length: Union[float, list]
    axis_minor_length: Union[float, list]
    centroid0: Union[float, list]
    centroid1: Union[float, list]
    eccentricity: Union[float, list]
    solidity: Union[float, list]
    perimeter: Union[float, list]


@dataclass
class Cutout:
    """Per cutout. Goes to PlantCutouts"""
    blob_home: str
    data_root: str
    batch_id: str
    image_id: str
    cutout_num: int
    datetime: datetime.datetime  # Datetime of original image creation
    cutout_props: CutoutProps
    cutout_id: str = field(init=False)
    cutout_path: str = field(init=False)
    cls: str = None
    is_primary: bool = False
    cutout_version: str = "1.0"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        self.cutout_id = self.image_id + "_" + str(self.cutout_num)
        self.cutout_path = str(Path(self.batch_id, self.cutout_id + ".png"))

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        cut_array = cv2.imread(self.cutout_path)
        cut_array = np.ascontiguousarray(
            cv2.cvtColor(cut_array, cv2.COLOR_BGR2RGB))
        return cut_array

    @property
    def config(self):
        _config = {
            "blob_home": self.blob_home,
            "data_root": self.data_root,
            "batch_id": self.batch_id,
            "image_id": self.image_id,
            "cutout_id": self.cutout_id,
            "cutout_path": self.cutout_path,
            "cls": self.cls,
            "cutout_num": self.cutout_num,
            "is_primary": self.is_primary,
            "datetime": self.datetime,
            "cutout_props": self.cutout_props,
            "schema_version": self.schema_version
        }

        return _config

    def save_config(self, save_path):
        try:
            save_cutout_path = Path(self.blob_home, self.data_root,
                                    self.batch_id, self.cutout_id + ".json")
            with open(save_cutout_path, "w") as f:
                json.dump(self.config, f, indent=4, default=str)
        except Exception as e:
            raise e
        return True

    def save_cutout(self, cutout_array):

        fname = f"{self.image_id}_{self.cutout_num}.png"
        cutout_path = Path(self.blob_home, self.data_root, self.batch_id,
                           fname)
        cv2.imwrite(str(cutout_path),
                    cv2.cvtColor(cutout_array, cv2.COLOR_RGB2BGRA))
        return True


# GLOBALS -------------------------------------------------------------------------

CUTOUT_PROPS = [
    "area",  # float Area of the region i.e. number of pixels of the region scaled by pixel-area.
    "area_bbox",  # float Area of the bounding box i.e. number of pixels of bounding box scaled by pixel-area.
    "area_convex",  # float Are of the convex hull image, which is the smallest convex polygon that encloses the region.
    "axis_major_length",  # float The length of the major axis of the ellipse that has the same normalized second central moments as the region.
    "axis_minor_length",  # float The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
    "centroid",  # array Centroid coordinate tuple (row, col).
    "eccentricity",  # float Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
    "extent",  # float Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
    "solidity",  # float Ratio of pixels in the region to pixels of the convex hull image.
    # "label",  # int The label in the labeled input image.
    "perimeter",  # float Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
]

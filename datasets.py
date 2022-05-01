import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
from dacite import from_dict
from omegaconf import DictConfig


@dataclass
class BoxCoordinates:
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_left: np.ndarray
    bottom_right: np.ndarray

    # scale: np.ndarray = field(init=False, default=np.array([]))

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
            "bottom_right": self.bottom_right.tolist(),
            # "scale": self.scale.tolist()
        }

        return _config

    # def set_scale(self, new_scale: np.ndarray):
    #     self.scale = new_scale
    #     self.top_left = self.top_left * self.scale
    #     self.top_right = self.top_right * self.scale
    #     self.bottom_left = self.bottom_left * self.scale
    #     self.bottom_right = self.bottom_right * self.scale


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
    is_normalized: bool = field(init=True, default=False)
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
            "bbox_id":
            self.bbox_id,
            "image_id":
            self.image_id,
            "local_coordinates":
            self.local_coordinates.config,
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

        self._local_area = None
        self._global_area = None

        if self.local_coordinates:
            self.set_local_centroid()

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

    def set_local_centroid(self):
        self.local_centroid = self.get_centroid(self.local_coordinates)

    def set_global_centroid(self):
        self.global_centroid = self.get_centroid(self.global_coordinates)

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


# Batch Config from YAML ---------------------------------------------------------------------------


class BatchConfigImages:

    def __init__(self, batchdata, cfg: DictConfig) -> None:
        # From yaml config file
        self.num_class = cfg.segment.num_classes
        self.task = cfg.general.task
        self.clear_border = cfg.segment.clear_border
        self.workdir = cfg.general.workdir

        self.batchdir = cfg.general.batchdir
        self.datadir = cfg.general.datadir
        self.imagedir = cfg.general.imagedir
        self.autosfmdir = Path(cfg.autosfm.autosfmdir)
        self.model_path = cfg.detect.model_path

        self.detections_csv = cfg.detect.detections_csv
        self.reference_path = self.autosfmdir / "reference"
        self.raw_label_dir = self.autosfmdir / "annotations"
        self.asfm_metadata = self.autosfmdir / "metadata"
        self.vi = cfg.segment.vi
        self.class_algorithm = cfg.segment.class_algorithm
        self.num_classes = cfg.segment.num_classes

        # Batch Metadata
        self.batch_id = batchdata.batch_id
        self.upload_dir = batchdata.upload_dir
        self.site_id = batchdata.site_id
        self.upload_datetime = batchdata.upload_datetime
        self.batch_id = batchdata.batch_id
        self.image_list = batchdata.image_list
        self.schema_version = batchdata.schema_version


# Batch Metadata ---------------------------------------------------------------------------


@dataclass
class BatchMetadata:
    """ Batch metadata class for yaml loader"""
    upload_dir: str
    site_id: str
    upload_datetime: str
    batch_id: uuid = field(init=False)
    image_list: List = field(init=False)
    schema_version: str = "v1"

    def __post_init__(self):
        self.batch_id = uuid.uuid4()
        self.image_list = self.get_batch_images()

    def get_batch_images(self):
        extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        images = []
        for ext in extensions:
            images.extend(Path(self.upload_dir, "developed").glob(ext))
        image_ids = [image.stem for image in images]

        self.image_list = [{
            "id": image_id,
            "path": str(path)
        } for image_id, path in zip(image_ids, images)]
        # image_list = images
        return images


# Image dataclasses ----------------------------------------------------------


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
    fov: BoxCoordinates = None


@dataclass
class Box:
    bbox_id: str
    image_id: str
    local_coordinates: dict
    global_coordinates: dict


@dataclass
class BBoxFOV:
    top_left: list
    top_right: list
    bottom_left: list
    bottom_right: list


@dataclass
class BBoxMetadata:
    id: str
    width: int
    height: int
    field_of_view: BBoxFOV
    pixel_width: float
    pixel_height: float
    yaw: float
    pitch: float
    roll: float
    focal_length: float
    camera_location: list
    bboxes: List[Box]


@dataclass
class Image:
    """Parent class for RemapImage and ImageData.

    """
    image_id: str
    image_path: str

    def __post_init__(self):
        image_array = self.array
        self.width = image_array.shape[1]
        self.height = image_array.shape[0]

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        img_array = cv2.imread(str(self.image_path))
        img_array = np.ascontiguousarray(
            cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        return img_array

    @property
    def config(self):
        _config = {
            "id": self.image_id,
            "width": self.width,
            "height": self.height,
            "field_of_view": self.camera_info.fov.config,
            "pixel_width": self.camera_info.pixel_width,
            "pixel_height": self.camera_info.pixel_height,
            "yaw": self.camera_info.yaw,
            "pitch": self.camera_info.pitch,
            "roll": self.camera_info.roll,
            "focal_length": self.camera_info.focal_length,
            "camera_location": self.camera_info.camera_location.tolist(),
            "bboxes": [box.config for box in self.bboxes]
        }

        return _config

    def save_config(self, save_path):
        try:
            save_file = os.path.join(save_path, f"{self.image_id}.json")
            with open(save_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            raise e
        return True


@dataclass
class RemapImage(Image):
    """ For remapping labels (remap_labels) """
    bboxes: BBox
    camera_info: CameraInfo
    width: int = field(init=False, default=-1)
    height: int = field(init=False, default=-1)


@dataclass
class ImageData(Image):
    """ Dataclass for segmentation and synthetic data generation"""
    batch_id: uuid
    cutout_ids: List[str] = None
    camera_info: CameraInfo = None

    @property
    def bbox(self):
        json_path = Path(self.image_path.parent.parent, "labels",
                         self.image_id + ".json")

        with open(json_path) as f:
            j = json.load(f)
            boxset = from_dict(data_class=BBoxMetadata, data=j)
        return boxset


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
    area: float
    area_bbox: float
    area_convex: float
    axis_major_length: float
    axis_minor_length: float
    centroid: List
    eccentricity: float
    solidity: float
    perimeter: float

@dataclass
class Cutout:
    """Per cutout. Goes to PlantCutouts"""
    cutout_path: str
    cutout_num: int
    image_id: str
    site_id: str
    date: str
    cutout_props: CutoutProps
    cutout_id: uuid = field(init=False, default=None)
    species: str = None
    days_after_planting: int = None
    

    def __post_init__(self):
        ct_hash = self.cutout_path + self.date
        self.cutout_id = uuid.uuid3(uuid.NAMESPACE_DNS, ct_hash)

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        cut_array = cv2.imread(self.cutout_path)
        cut_array = np.ascontiguousarray(
            cv2.cvtColor(cut_array, cv2.COLOR_BGR2RGB))
        return cut_array


# Synthetic Data Generation -------------------------------------------------------------------------


@dataclass
class Pot:
    pot_path: str
    pot_id: uuid = None

    def __post_init__(self):
        self.pot_id = uuid.uuid4()

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        pot_array = cv2.imread(self.pot_path)
        pot_array = np.ascontiguousarray(
            cv2.cvtColor(pot_array, cv2.COLOR_BGR2RGB))
        return pot_array

@dataclass
class Background:
    background_path: str
    background_id: uuid = None

    def __post_init__(self):
        self.back_id = uuid.uuid4()

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        back_array = cv2.imread(self.background_path)
        back_array = np.ascontiguousarray(
            cv2.cvtColor(back_array, cv2.COLOR_BGR2RGB))
        return back_array


@dataclass
class PlantCutouts:
    """ Per benchbot image. Standalone class for processing"""
    batch_id: str
    image_id: str
    cutouts: List[Cutout]


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

    # "equivalent_diameter_area",  # float The diameter of a circle with the same area as the region.
    # "feret_diameter_max",  # float Maximum Feret’s diameter computed as the longest distance between points around a region’s convex hull contour as determined by find_contours. [5]
    # "perimeter_crofton",  # float Perimeter of object approximated by the Crofton formula in 4 directions.
    # "num_pixels",  # int Number of foreground pixels.
    # "area_filled",  # float Area of the region with all the holes filled in.
    # "bbox",  # tuple Bounding box (min_row, min_col, max_row, max_col). Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col).
    # "centroid_local",  # array Centroid coordinate tuple (row, col), relative to region bounding box.
    # "centroid_weighted",  # array Centroid coordinate tuple (row, col) weighted with intensity image.
    # "centroid_weighted_local",  #array Centroid coordinate tuple (row, col), relative to region bounding box, weighted with intensity image.
    # "coords_scaled",  # (N, 2) ndarray Coordinate list (row, col)``of the region scaled by ``spacing.
    # "coords",  # (N, 2) ndarray Coordinate list (row, col) of the region.
    # "euler_number",  # int Euler characteristic of the set of non-zero pixels. Computed as number of connected components subtracted by number of holes (input.ndim connectivity). In 3D, number of connected components plus number of holes subtracted by number of tunnels.
    # "image",  # (H, J) ndarray Sliced binary region image which has the same size as bounding box.
    # "image_convex",  # (H, J) ndarray Binary convex hull image which has the same size as bounding box.
    # "image_filled",  # (H, J) ndarray Binary region image with filled holes which has the same size as bounding box.
    # "image_intensity",  # ndarray Image inside region bounding box.
    # "inertia_tensor",  # ndarray Inertia tensor of the region for the rotation around its mass.
    # "inertia_tensor_eigvals",  # tuple The eigenvalues of the inertia tensor in decreasing order.
    # "intensity_mean",  # float Value with the mean intensity in the region.
    # "intensity_min",  # float Value with the least intensity in the region.
    # "moments",  # (3, 3) ndarray Spatial moments up to 3rd order
    # "moments_central",  # (3, 3) ndarray
    # "moments_hu",  # tuple Hu moments (translation, scale and rotation invariant).
    # "moments_normalized",  # (3, 3) ndarray where m_00 is the zeroth spatial moment.
    # "moments_weighted",  # (3, 3) ndarray Spatial moments of intensity image up to 3rd order:
    # "moments_weighted_central",  # (3, 3) ndarray
    # "moments_weighted_hu",  # tuple Hu moments (translation, scale and rotation invariant) of intensity image.
    # "moments_weighted_normalized",  # (3, 3) ndarray where wm_00 is the zeroth spatial moment (intensity-weighted area).
    # "orientation",  # float Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.
    # "slice",  # tuple of slices A slice to extract the object from the source image.
]

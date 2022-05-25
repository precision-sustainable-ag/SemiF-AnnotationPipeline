import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import cv2
import exifread
import numpy as np

from semif_utils.mongo_utils import Connect


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
    data_root: str
    batch_id: str
    site_id: str
    upload_datetime: str
    image_list: List = field(init=False)
    blob_root: str = "data"
    schema_version: str = "v1"

    def __post_init__(self):
        self.image_list = self.get_batch_images()

    def get_batch_images(self):
        imgdir = Path(self.blob_root, self.data_root, self.batch_id, "images")
        extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        images = []
        for ext in extensions:
            images.extend(Path(imgdir).glob(ext))
        image_ids = [image.stem for image in images]
        image_list = []

        for ids, imgp in zip(image_ids, images):
            image_list.append(str("/".join(imgp.parts[-2:])))

        return image_list


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
    ImageDescription: list = None
    MakerNote: list = None
    UserComment: list = None
    ApplicationNotes: list = None
    Tag: int = None
    SubIFDs: int = None


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
    local_coordinates: dict
    global_centroid: list
    global_coordinates: dict
    is_normalized: bool
    cls: str
    is_primary: bool


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
    image_id: str
    image_path: str
    data_root: str
    batch_id: str

    def __post_init__(self):
        image_array = self.array
        self.width = image_array.shape[1]
        self.height = image_array.shape[0]

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        img_path = Path("data", self.data_root, self.batch_id, self.image_path)
        img_array = cv2.imread(str(img_path))
        img_array = np.ascontiguousarray(
            cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        return img_array

    @property
    def config(self):
        _config = {
            "data_root": self.data_root,
            "batch_id": self.batch_id,
            "image_path": self.image_path,
            "image_id": self.image_id,
            "exif_meta": asdict(self.exif_meta),
            "camera_info": asdict(self.camera_info),
            "width": self.width,
            "height": self.height,
            "bboxes": [box.config for box in self.bboxes]
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
    width: int = field(init=False, default=-1)
    height: int = field(init=False, default=-1)
    exif_meta: Optional[ImageMetadata] = field(init=False, default=None)

    def __post_init__(self):
        self.width = self.array.shape[1]
        self.height = self.array.shape[0]
        self.exif_meta = self.get_exif()

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
            pop_list = [
                "MakerNote", "UserComment", "ImageDescription",
                "ApplicationNotes"
            ]
            meta.pop(
                x.rsplit(" ")[1]) if x.rsplit(" ")[1] in pop_list else None
        imgmeta = ImageMetadata(**meta)
        return imgmeta


@dataclass
class ImageData(Image):
    """ Dataclass for segmentation and synthetic data generation"""
    data_root: str
    image_path: str
    batch_id: str
    width: int
    height: int
    exif_meta: ImageMetadata
    # cutouts: list[Cutout] = None
    cutout_ids: List[str] = None
    camera_info: CameraInfo = None
    bboxes: list[Box] = None


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
    data_root: str
    batch_id: str
    cutout_path: str
    cutout_num: int
    image_id: str
    site_id: str
    datetime: datetime.datetime  # Datetime of original image creation
    cutout_props: CutoutProps
    cutout_id: uuid = field(init=False)
    species: str = None
    schema_version: str = "1.0"

    def __post_init__(self):
        self.cutout_id = uuid.uuid4()

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
    # TODO complete attributes
    pot_path: str
    pot_id: uuid = None

    def __post_init__(self):
        self.pot_id = uuid.uuid4()

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        pot_array = cv2.imread(self.pot_path, cv2.IMREAD_UNCHANGED)
        pot_array = np.ascontiguousarray(
            cv2.cvtColor(pot_array, cv2.COLOR_BGR2RGBA))
        return pot_array


@dataclass
class Background:
    # TODO complete attributes
    background_path: str
    background_id: uuid = None

    def __post_init__(self):
        self.background_id = uuid.uuid4()

    @property
    def array(self):
        # Read the image from the file and return the numpy array
        background_array = cv2.imread(self.background_path)
        background_array = np.ascontiguousarray(
            cv2.cvtColor(background_array, cv2.COLOR_BGR2RGB))
        return background_array


@dataclass
class SynthImage:
    data_root: str
    synth_path: str
    synth_maskpath: str
    pots: list[Pot]
    background: Background
    cutouts: list[Cutout]
    synth_id: str = field(init=False)

    def __post_init__(self):
        self.synth_id = uuid.uuid4()


@dataclass
class SynthDataContainer:
    """Combines documents in a database with items in a directory to form data container for generating synthetic bench images. Includes lists of dataclasses for cutouts, pots, and backgrounds.
    
    Args:
        datadir (Path): Parent directory that should contain "cutouts", "pots", and "backgrounds".
        cutouts list[Cutouts]: list of Cutout dataclasses
        pots list[Pot]: list of Pot dataclasses
        backs list[Background]: list of Background dataclasses

    Returns:
        data_container (object): Dataclass with Cutouts, Pots, and Backgrounds itemized in a mongoDB.
    """
    synthdir: str
    db_db: str = None
    from_db: bool = True
    background_dir: str = None
    pot_dir: str = None
    cutout_dir: str = None
    cutouts: list[Cutout] = field(init=False, default=None)
    pots: list[Pot] = field(init=False, default=None)
    backgrounds: list[Background] = field(init=False, default=None)

    def __post_init__(self):
        self.backgrounds = self.get_dcs("Backgrounds")
        self.pots = self.get_dcs("Pots")
        self.cutouts = self.get_dcs("Cutouts")

    def get_data_from_json(self, jsun):
        """ Open json and create dictionary
        """
        # Opening JSON file
        with open(jsun) as json_file:
            data = json.load(json_file)
        return data

    def get_jsons(self, collection):
        """ Gets json files
        """
        datas = []
        collection = collection.lower().rstrip(collection[-1])
        collection_dir = collection + "_dir"
        if collection == "cutout":
            jsondir = Path(self.cutout_dir, getattr(self, collection_dir))
            jsons = jsondir.glob("*.json")
            jsons = [x for x in jsons]
        else:
            jsons = Path(self.synthdir, getattr(self,
                                                collection_dir)).glob("*.json")
        jsons = [x for x in jsons]
        for jsun in jsons:
            data = self.get_data_from_json(jsun)
            datas.append(data)
        return datas

    def query_db(self, db_collection):
        connection = Connect.get_connection()
        db = getattr(connection, self.db_db)
        cursor = getattr(db, db_collection).find()
        return cursor

    def get_dcs(self, collection_str):
        """Connnects documents in a database collection with items in a directory.
        Places connected items in a list of dataclasses.
        """
        syn_datacls = {"cutout": Cutout, "pot": Pot, "background": Background}
        if self.from_db:
            cursor = self.query_db(collection_str.title())
        else:
            cursor = self.get_jsons(collection_str)

        if collection_str == "Cutouts":
            docdir = Path(self.cutout_dir).parent
        else:
            docdir = Path(self.synthdir)

        path_str_ws = collection_str.lower().replace("s", "")
        path_str = f"{path_str_ws}_path"

        docs = []
        for doc in cursor:
            doc_path = docdir / doc[path_str]
            doc[path_str] = str(doc_path)

            assert Path(doc_path).exists(
            ), f"Image with path {str(doc_path)} does not exist."

            # Clean up doc and json by removoing _id from db
            if "_id" in doc:
                doc.pop("_id")
            if "cutout_id" in doc:
                cut_id = doc["cutout_id"]
                doc.pop("cutout_id")

            data_cls = syn_datacls[path_str_ws]
            dc = data_cls(**doc)
            if hasattr(dc, "cutout_id"):
                dc.cutout_id = cut_id

            docs.append(dc)

        return docs


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

# Pot positioning
sixpot = {
    0: (1592, 1599),
    1: (1592, 4796),
    2: (1592, 7993),
    3: (4776, 1599),
    4: (4776, 4796),
    5: (4776, 7993)
}

POTMAPS = [sixpot]

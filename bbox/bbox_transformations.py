from typing import Dict, List, Type

import numpy as np
from scipy.spatial.transform import Rotation
from semif_utils.datasets import BoxCoordinates, ImageData, BBox

from .bbox_utils import bb_iou, generate_hash

import Metashape

FOV_IOU_THRESH = 0.1
BBOX_OVERLAP_THRESH = 0.3


def image_to_global_transform(focal_length: float, pixel_dim: float,
                              coords: np.ndarray, camera_height: float):
    """Find the object dimensions using the camera model

    Args:
        focal_length (float): Focal length of the camera (in pixels)
        pixel_dim (float): pixel width and pixel height
        coords (np.ndarray): local coordinates to transform to global coordinates
        camera_height (float): Height of the camera (in global coordinate units, i.e., meters)

    Returns:
        _type_: _description_
    """
    f = focal_length * pixel_dim
    # Distances from the center
    distances = np.abs(coords)
    signs = np.sign(coords)
    global_distances = distances * (camera_height / f)
    global_coords = signs * global_distances

    return global_coords


def global_to_image_transform(focal_length: float, pixel_dim: float,
                              coords: np.ndarray, camera_height: float):
    """Find the object dimensions in the image plane using the camera model

    Args:
        focal_length (float): Focal length of the camera (in pixels)
        pixel_dim (float): pixel width and pixel height
        coords (np.ndarray): global coordinates to transform
        camera_height (float): Height of the camera (in global coordinate units, i.e., meters)

    Returns:
        _type_: _description_
    """
    f = focal_length * pixel_dim
    # Distances from the center
    distances = np.abs(coords)
    signs = np.sign(coords)

    image_distances = distances * (f / camera_height)

    # This is in global units. Convert to pixels
    image_distances = image_distances / pixel_dim

    image_coords = signs * image_distances

    return image_coords


def find_global_coords(unrotated_coords: np.ndarray, yaw_angle: float,
                       focal_length: float, pixel_height: float,
                       pixel_width: float, camera_height: float, pitch_angle,
                       roll_angle) -> np.ndarray:
    """Function to find the translated global coordinates

    Args:
        center_coords (np.ndarray): Image center coordinates in global coordinate system
        unrotated_coords (np.ndarray): Unrotated coordinates in global coordinate system
        yaw_angle (float): Yaw angle in degrees
        is_bbox (bool, optional): Do the coordinates describe a boudig box? Defaults to True.

    Returns:
        np.ndarray: Translated global coordinates
    """

    global_unrotated_coords = unrotated_coords.copy()

    global_unrotated_coords[:, 0] = global_unrotated_coords[:, 0] * pixel_width
    global_unrotated_coords[:,
                            1] = global_unrotated_coords[:, 1] * pixel_height

    # Find the "object dimensions" in the global coordinates
    global_unrotated_coords[:, 0] = image_to_global_transform(
        focal_length, pixel_width, global_unrotated_coords[:, 0],
        camera_height)
    global_unrotated_coords[:, 1] = image_to_global_transform(
        focal_length, pixel_height, global_unrotated_coords[:, 1],
        camera_height)

    # Add the z coordinate for 3D rotation
    # The z coordinate is the -camera_height since the 
    # z vector is assumed to point downward, and the global plane
    # is on the ground
    global_unrotated_coords = np.concatenate(
        (global_unrotated_coords, -camera_height * np.ones((4, 1))), axis=1)

    if yaw_angle < 180:
        _yaw_angle = 360. - yaw_angle
    else:
        _yaw_angle = yaw_angle

    # Apply rotations to the objects
    # This gives the coordinates with the origin shifted to
    # the camera location
    R = Rotation.from_euler("ZYX",
                            np.array([_yaw_angle, roll_angle, pitch_angle]),
                            degrees=True)
    R_quat = R.as_quat()
    rotation = Rotation.from_quat(R_quat)
    rotated_coordinates = rotation.apply(global_unrotated_coords)

    # Project onto z = -camera_height plane
    rotated_coordinates[:, 0] = -camera_height * (rotated_coordinates[:, 0] / rotated_coordinates[:, 2])
    rotated_coordinates[:, 1] = -camera_height * (rotated_coordinates[:, 1] / rotated_coordinates[:, 2])

    return rotated_coordinates[:, :2]


def img_to_global_coord(image_coordinates: np.ndarray,
                        camera_center: np.ndarray,
                        pixel_width: float,
                        pixel_height: float,
                        focal_length: float,
                        image_width: float,
                        image_height: float,
                        camera_height: float,
                        yaw_angle: float,
                        pitch_angle: float,
                        roll_angle: float,
                        is_bbox: bool = True) -> np.ndarray:
    """Map the bounding box points form local coordinates to gobal coordinates

    Args:
        image_coordinates (np.ndarray): Image coordinates in image space
        camera_center (np.ndarray): Image center coordinates in global coordinate space
        pixel_width (float): Pixel width
        pixel_height (float): Pixel height
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
        yaw_angle (float): Yaw angle in degrees
        is_bbox (bool, optional): Are the image coordinates bounding box coordinates? Defaults to True.

    Returns:
        np.ndarray: Global coordinates
    """

    _image_coordinates = image_coordinates.copy()
    _camera_center = camera_center.copy()

    if np.ndim(image_coordinates) == 1:
        _image_coordinates = np.expand_dims(image_coordinates, axis=0)
    assert _image_coordinates.shape[1] == 2

    if np.ndim(_camera_center) == 1:
        _camera_center = np.expand_dims(_camera_center, axis=0)
    assert _camera_center.shape[1] == 2

    # Shift the origin to the image center: The image center (in local coordinates)
    # corresponds to the camera locatio in the global coordinates
    image_center = np.array(
        [[float(image_width // 2),
          float(image_height // 2)]])
    _image_coordinates -= image_center

    # Find the coordinates with the origin at the camera location
    # using the camera model
    global_coordinates = find_global_coords(_image_coordinates, yaw_angle,
                                            focal_length, pixel_width,
                                            pixel_height, camera_height,
                                            pitch_angle, roll_angle)

    # Shift the origin from the camera location back to the global origin (0, 0)
    global_coordinates += _camera_center

    if is_bbox:

        # To stay consistent with the convention,
        # assign top_left, top_right, bottom_left, bottom_right
        # the respective x and y coordinates
        mask = np.argsort(global_coordinates[:, 0])

        # Determine which is the top_left and bottom_left
        left_coordinates = global_coordinates[mask[:2], :]
        top_sort = np.argsort(left_coordinates[:, 1])
        bottom_left = left_coordinates[top_sort[0], :]
        top_left = left_coordinates[top_sort[1], :]
        # Determine which is the top_right and bottom_right
        right_coordinates = global_coordinates[mask[2:], :]
        top_sort = np.argsort(right_coordinates[:, 1])
        bottom_right = right_coordinates[top_sort[0], :]
        top_right = right_coordinates[top_sort[1], :]

        global_coordinates[0, :] = top_right
        global_coordinates[1, :] = bottom_left
        global_coordinates[2, :] = top_left
        global_coordinates[3, :] = bottom_right

    return global_coordinates


def bbox_to_global(top_left: np.ndarray, top_right: np.ndarray,
                   bottom_left: np.ndarray, bottom_right: np.ndarray,
                   camera_center: np.ndarray, pixel_width: float,
                   pixel_height: float, focal_length: float, image_width: int,
                   image_height: int, camera_height: float, yaw_angle: float,
                   pitch_angle: float, roll_angle: float) -> BoxCoordinates:
    """Map the bonding box points form local coordinates to gobal coordinates, and apply
       roll and pitch corrections

    Args:
        top_left (np.ndarray): Top left x and y coordinates
        top_right (np.ndarray): Top right x and y coordinates
        bottom_left (np.ndarray): Bottom left x and y coordinates
        bottom_right (np.ndarray): Bottom right x and y coordinates
        camera_center (np.ndarray): x and y coordinates of the camera location (in global coordinates)
        pixel_width (float): Pixel width found using SfM
        pixel_height (float): Pixel height found using SfM
        focal_length (float): Focal length (in pixels) found using SfM
        image_width (int): Image width (in pixels)
        image_height (int): Image height (in pixels)
        camera_height (float): Camera height (in global coordinates) found using SfM
        yaw_angle (float): Yaw angle of the camera found using SfM
        pitch_angle (float): Pitch angle of the camera found using SfM
        roll_angle (float): Roll angle of the camera found using SfM

    Returns:
        BoxCoordinates: Global coordinates of the bounding box
    """

    assert all([
        len(coord) == 2 for coord in
        [top_left, top_right, bottom_left, bottom_right, camera_center]
    ])

    _top_right = np.array([top_right])
    _bottom_left = np.array([bottom_left])
    _top_left = np.array([top_left])
    _bottom_right = np.array([bottom_right])
    _camera_center = np.array([camera_center])

    image_coordinates = np.concatenate(
        [_top_right, _bottom_left, _top_left, _bottom_right])

    # Change the origin to bottom-left of the image
    image_coordinates[:, 1] = image_height - image_coordinates[:, 1]

    bbox_global_coordinates = img_to_global_coord(image_coordinates,
                                                  _camera_center,
                                                  pixel_width,
                                                  pixel_height,
                                                  focal_length,
                                                  image_width,
                                                  image_height,
                                                  camera_height,
                                                  yaw_angle,
                                                  pitch_angle,
                                                  roll_angle,
                                                  is_bbox=True)

    # Unpack
    top_left = bbox_global_coordinates[2, :]
    top_right = bbox_global_coordinates[0, :]
    bottom_left = bbox_global_coordinates[1, :]
    bottom_right = bbox_global_coordinates[3, :]

    global_coordinates = BoxCoordinates(top_left, top_right, bottom_left,
                                        bottom_right)

    return global_coordinates


class BBoxFilter:

    def __init__(self, images: List[ImageData]):
        self.images = images
        self.image_map = {image.image_id: image for image in images}
        self.total_bboxes = sum([len(image.bboxes) for image in self.images])
        self.primary_boxes = []
        self.primary_box_ids = set()

    def deduplicate_bboxes(self):
        """Calculates the ideal bounding box and the associated image from all the
           bounding boxes
        """
        comparisons = self.filter_images()
        self.filter_bounding_boxes(comparisons)

    def filter_images(self) -> Dict[str, List[str]]:
        """Filter the images to compare based on the overlap between their fields of
           view

        Returns:
            Dict[str, List[str]]: A dictionary containing the image IDs as keys, and
                                  a list of image IDs each key overlaps with
        """
        image_ids = list(self.image_map.keys())
        comparisons = dict()
        # Find the overlap between FOVs of the images
        for i, image_id in enumerate(image_ids):
            image = self.image_map[image_id]
            comparisons[image_id] = []
            for j in range(i + 1, len(image_ids)):
                compare_image_id = image_ids[j]
                compare_image = self.image_map[compare_image_id]
                fov_iou = bb_iou(image.camera_info.fov,
                                 compare_image.camera_info.fov)
                if fov_iou > FOV_IOU_THRESH:
                    comparisons[image_id].append(compare_image_id)

        return comparisons

    def filter_bounding_boxes(self, comparisons: Dict[str, List[str]]):
        """Find overlapping bounding boxes from the images to compare

        Args:
            comparisons (Dict[str, List[str]]): Images to compare, found via
                                                filter_images
        """
        # For all the overlapping images
        visited_bboxes = set()
        areas = []

        for image_id, image_ids_for_comparison in comparisons.items():
            # For each bounding box in the key image
            for box in self.image_map[image_id].bboxes:
                if box.bbox_id not in visited_bboxes:
                    visited_bboxes.add(box.bbox_id)
                    areas.append(box.local_area)
                compared = set()
                # A unique ID for the bounding box in question
                box_hash = generate_hash(box)
                # For each overlapping image
                for compare_image_id in image_ids_for_comparison:
                    boxes = self.image_map[compare_image_id].bboxes
                    # And each of its bounding box
                    for _box in boxes:

                        if _box.bbox_id not in visited_bboxes:
                            visited_bboxes.add(_box.bbox_id)
                            areas.append(_box.local_area)

                        # A unique ID for a pair of bounding boxes
                        # Note that the order of the boxes does not matter
                        # i.e. Box_A,Box_B is the same as Box_B,Box_A
                        _box_hash = generate_hash(_box, box_hash)
                        if _box_hash in compared:
                            continue
                        compared.add(_box_hash)
                        iou = box.bb_iou(_box)
                        if iou > BBOX_OVERLAP_THRESH:
                            # Set the two boxes as overlapping
                            box.add_box(_box)
                            _box.add_box(box)

        self.select_best_bbox()
        self.cleanup_primary_boxes()

    def select_best_bbox(self):
        # visited will be a set of boxes that have been compared
        visited = set()
        for image in self.images:
            # If all the boxes have been checked, no need to
            # check the other images
            if len(visited) == self.total_bboxes:
                break
            bboxes = image.bboxes
            for box in bboxes:
                box_hash = generate_hash(box)
                if box_hash in visited:
                    continue
                all_boxes = [box] + box._overlapping_bboxes
                box_hashes = [generate_hash(_box) for _box in all_boxes]
                visited = visited.union(set(box_hashes))
                # Find the best bounding box
                centers = np.array([
                    self.image_map[_box.image_id].camera_info.camera_location
                    for _box in all_boxes
                ])
                centroids = np.array(
                    [_box.global_centroid for _box in all_boxes])
                distances = ((centroids - centers[:, :2])**2).sum(axis=-1)
                min_idx = np.argmin(distances)
                all_boxes[min_idx].is_primary = True
                if all_boxes[min_idx].bbox_id not in self.primary_box_ids:
                    self.primary_boxes.append(all_boxes[min_idx])
                    self.primary_box_ids.add(all_boxes[min_idx].bbox_id)

    def cleanup_primary_boxes(self):

        _primary_boxes = []
        for i, box in enumerate(self.primary_boxes):
            image_width = self.image_map[box.image_id].width
            image_height = self.image_map[box.image_id].height

            if box.local_centroid[0] < image_width // 4 or \
               box.local_centroid[0] > 3 * image_width // 4 or \
               box.local_centroid[1] < image_height // 4 or \
               box.local_centroid[1] > 3 * image_height // 4:

                box.is_primary = False
                # del self.primary_boxes[i]
            else:
                _primary_boxes.append(box)

        # Revisit all bounding boxes identified as primary and
        # remove the overlapping ones
        for i in range(len(_primary_boxes)):
            box1 = _primary_boxes[i]
            camera_location1 = self.image_map[
                box1.
                image_id].camera_info.camera_location[:2]  # get just x and y
            for j in range(i + 1, len(_primary_boxes)):
                box2 = _primary_boxes[j]
                camera_location2 = self.image_map[
                    box2.
                    image_id].camera_info.camera_location[:
                                                          2]  # get just x and y
                iou = box1.bb_iou(box2)
                if iou > BBOX_OVERLAP_THRESH:
                    # De-duplicate
                    distance1 = ((box1.global_centroid -
                                  camera_location1)**2).sum(axis=-1)
                    distance2 = ((box2.global_centroid -
                                  camera_location2)**2).sum(axis=-1)

                    if distance1 < distance2:
                        box2.is_primary = False
                    else:
                        box1.is_primary = False


class BBoxMapper():

    def __init__(self, project_path: str, images: List[ImageData]):
        """Class to map bounding box coordinates from image cordinates
           to global coordinates
        """
        self.images = images
        self.doc = Metashape.Document()
        self.doc.open(project_path)

    def map(self):
        """
        Maps all the bounding boxes to a global coordinate space
        """

        for image in self.images:

            # base, row, pot, timestamp = image.image_id.split("_")
            # image_id = "_".join([base, pot])
            image_id = image.image_id

            # Isolate the chunk
            camera_chunk = None
            for chunk in self.doc.chunks:
                cameras = [camera.label for camera in chunk.cameras]
                if image_id in cameras:
                    camera_chunk = chunk
                    camera = [cam for cam in chunk.cameras if cam.label == image_id][0]
                    break
            
            assert camera_chunk is not None

            # From: https://www.agisoft.com/forum/index.php?topic=13875.0
            surface = camera_chunk.point_cloud

            global_coordinates = dict()
            for bbox in image.bboxes:

                top_left = bbox.local_coordinates.top_left
                bottom_left = bbox.local_coordinates.bottom_left
                top_right = bbox.local_coordinates.top_right
                bottom_right = bbox.local_coordinates.bottom_right
                
                mapped_coordinates = []

                co_type = ["top_left", "bottom_left", "top_right", "bottom_right"]

                for co, coords in zip(co_type, [top_left, bottom_left, top_right, bottom_right]):

                    x_coord = coords[0]
                    y_coord = coords[1]
                    
                    ray_origin = camera.center # camera.unproject(Metashape.Vector([x_coord, y_coord, 0]))
                    ray_target = camera.unproject(Metashape.Vector([x_coord, y_coord]))

                    point_internal = surface.pickPoint(ray_origin, ray_target)

                    if point_internal is None:
                        raise TypeError()
                    
                    # From https://www.agisoft.com/forum/index.php?topic=12781.0
                    global_coord = camera_chunk.crs.project(camera_chunk.transform.matrix.mulp(point_internal))[:2]
                    mapped_coordinates.append(global_coord)
                    
                top_left = np.array(mapped_coordinates[0])
                top_right = np.array(mapped_coordinates[2])
                bottom_left = np.array(mapped_coordinates[1])
                bottom_right = np.array(mapped_coordinates[3])

                global_coordinates = BoxCoordinates(top_left, top_right, bottom_left, bottom_right)
                bbox.update_global_coordinates(global_coordinates)


class GlobalToLocalMaper:

    def __init__(self, project_path: str, images: List[ImageData]):
        """Class to map bounding box coordinates from global cordinates
           to image coordinates
        """
        self.images = images
        self.doc = Metashape.Document()
        self.doc.open(project_path)

    def map(self, global_box_coordinates: BBox, map_to: str):

        local_coordinates = self.bbox_to_local(global_box_coordinates, map_to)
        return local_coordinates

    def bbox_to_local(self, global_bbox: BBox, map_to_image: ImageData):

        assert not global_bbox.local_coordinates, "The global box contains existing local coordinates"\
                                                  "Mapping operation will overwrite them."

        # base, row, pot, timestamp = map_to.split("_")
        # image_id = "_".join([base, pot])
        image_id = map_to_image.image_id

        # Isolate the chunk
        camera_chunk = None
        for chunk in self.doc.chunks:
            cameras = [camera.label for camera in chunk.cameras]
            if image_id in cameras:
                camera_chunk = chunk
                camera = [cam for cam in chunk.cameras if cam.label == image_id][0]
                break
        
        assert camera_chunk is not None

        # From: https://www.agisoft.com/forum/index.php?topic=13875.0
        crs = camera_chunk.crs
        T = camera_chunk.transform.matrix

        top_left = global_bbox.global_coordinates.top_left
        bottom_left = global_bbox.global_coordinates.bottom_left
        top_right = global_bbox.global_coordinates.top_right
        bottom_right = global_bbox.global_coordinates.bottom_right

        mapped_coordinates = dict()

        co_type = ["top_left", "bottom_left", "top_right", "bottom_right"]

        scale = np.array([map_to_image.width, map_to_image.height])

        for co, coords in zip(co_type, [top_left, bottom_left, top_right, bottom_right]):

            x_coord = coords[0]
            y_coord = coords[1]
            z_coord = 0. # Assuming 0 height. This could be tuned further

            point = Metashape.Vector([x_coord, y_coord, z_coord]) # point of interest in geographic coord
            point_internal = T.inv().mulp(crs.unproject(point))
            coords_2D = camera.project(point_internal)
            # Normalize
            mapped_coordinates[co] = np.array(coords_2D) / scale

        box_coordinates = BoxCoordinates(top_left=mapped_coordinates["top_left"], 
                                         top_right=mapped_coordinates["top_right"], 
                                         bottom_left=mapped_coordinates["bottom_left"], 
                                         bottom_right=mapped_coordinates["bottom_right"])
        
        return box_coordinates

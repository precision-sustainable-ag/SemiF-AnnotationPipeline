from typing import Dict, List
import math
import numpy as np
from .bbox_utils import BBox, BoxCoordinates, Image, bb_iou, generate_hash


FOV_IOU_THRESH = 0.1
BBOX_OVERLAP_THRESH = 0.2


def get_rotation_matrix(angle: float) -> np.ndarray:
    """Get the rotation matrix for the Yaw
    Args:
        angle (float): Yaw angle in degrees
    """
    alpha = angle * (math.pi / 180.)
    cosa = math.cos(alpha)
    sina = math.sin(alpha)
    R = np.array([
        [cosa, -sina],
        [sina, cosa]
    ])

    return R


def rotation_transform(coord: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation transform on the coordinates
    Args:
        coord (np.ndarray): Coordinate vector to rotate
        R (np.ndarray): Rotation matrix
    Returns:
        np.ndarray: Transformed coordinates
    """
    _coord = coord.copy()
    if np.ndim(coord) == 1:
        _coord = np.expand_dims(_coord, axis=0)
    assert _coord.shape[1] == 2

    rotated_coord = np.dot(R, _coord.T).T
    
    return rotated_coord

def image_to_global_transform(focal_length, pixel_dim, coords, camera_height):
    
    f = focal_length * pixel_dim
    # Distances from the center
    distances = np.abs(coords)
    signs = np.sign(coords)
    global_distances = distances * (camera_height / f)
    global_coords = signs * global_distances

    return global_coords


def correct_xy_plane(camera_center, camera_height, coordinates, angle):

    num_coordinates = len(coordinates)
    # Change the origin to camera_center, camera_height
    origin = np.array([[camera_center, camera_height]])
    # Assume that the coordinates are present at height=0
    _coordinates = np.hstack([coordinates[:, np.newaxis], np.zeros((num_coordinates, 1))])
    # Shift the origin
    coord = _coordinates - origin

    # Get the rotation matrix
    R = get_rotation_matrix(angle)
    # Apply rotation on the vector. This is wrt origin=0,0
    rotated_coord = rotation_transform(coord, R)

    # Solve using two-point form
    target_coord = np.zeros_like(rotated_coord)
    target_coord[:, 0] = -(rotated_coord[:, 0] * camera_height) / rotated_coord[:, 1]
    # Shift the origin back
    target_coord += origin

    return target_coord[:, 0]


def pitch_correction(camera_center, camera_height, bbox_global_coordinates, pitch_angle):

    coordinates_to_correct = bbox_global_coordinates[:, 0]
    _camera_center = camera_center[0]
    corrected_coord = correct_xy_plane(_camera_center[0], camera_height, coordinates_to_correct, pitch_angle)
    return corrected_coord

def roll_correction(camera_center, camera_height, bbox_global_coordinates, roll_angle):

    coordinates_to_correct = bbox_global_coordinates[:, 1]
    _camera_center = camera_center[0]
    corrected_coord = correct_xy_plane(_camera_center[1], camera_height, coordinates_to_correct, roll_angle)
    return corrected_coord


def find_global_coords(unrotated_coords: np.ndarray, 
                       yaw_angle: float, focal_length: float,
                       pixel_height: float, pixel_width: float,
                       camera_height: float) -> np.ndarray:
    """_summary_

    Args:
        center_coords (np.ndarray): Image center coordinates in global coordinate system
        unrotated_coords (np.ndarray): Unrotated coordinates in global coordinate system
        yaw_angle (float): Yaw angle in degrees
        is_bbox (bool, optional): Do the coordinates describe a boudig box? Defaults to True.

    Returns:
        np.ndarray: _description_
    """

    global_unrotated_coords = unrotated_coords.copy()

    global_unrotated_coords[:, 0] = global_unrotated_coords[:, 0] * pixel_width
    global_unrotated_coords[:, 1] = global_unrotated_coords[:, 1] * pixel_height

    global_unrotated_coords[:, 0] = image_to_global_transform(focal_length, pixel_width, global_unrotated_coords[:, 0], camera_height)
    global_unrotated_coords[:, 1] = image_to_global_transform(focal_length, pixel_height, global_unrotated_coords[:, 1], camera_height)

    _yaw_angle = 360. - yaw_angle

    R = get_rotation_matrix(_yaw_angle)

    # Rotate the new coordinate
    rotated_coordinates = rotation_transform(global_unrotated_coords, R)

    return rotated_coordinates


def img_to_global_coord(image_coordinates: np.ndarray, camera_center: np.ndarray,
                        pixel_width: float, pixel_height: float, 
                        focal_length: float,
                        image_width: float, image_height: float,
                        camera_height:float,
                        yaw_angle: float, is_bbox: bool=True) -> np.ndarray:
    """_summary_

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
        np.ndarray: _description_
    """

    _image_coordinates = image_coordinates.copy()
    _camera_center = camera_center.copy()

    if np.ndim(image_coordinates) == 1:
        _image_coordinates = np.expand_dims(image_coordinates, axis=0)
    assert _image_coordinates.shape[1] == 2

    if np.ndim(_camera_center) == 1:
        _camera_center = np.expand_dims(_camera_center, axis=0)
    assert _camera_center.shape[1] == 2

    # Shift the origin to the image center: This corresponds to the camera center
    # in the global coordinates
    image_center = np.array([[float(image_width // 2), float(image_height//2)]])
    _image_coordinates -= image_center

    global_coordinates = find_global_coords(_image_coordinates, 
                                            yaw_angle, focal_length, 
                                            pixel_width, pixel_height, 
                                            camera_height)
    # Shift the origin back from the camera center
    global_coordinates += _camera_center

    if is_bbox:
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
                  camera_center: np.ndarray, 
                  pixel_width: float, pixel_height: float, 
                  focal_length: float,
                  image_width: int, image_height: int,
                  camera_height: float,
                  yaw_angle: float, pitch_angle: float, roll_angle: float):

    assert all([len(coord) == 2 for coord in [top_left, top_right, bottom_left, bottom_right, camera_center]])

    _top_right = np.array([top_right])
    _bottom_left = np.array([bottom_left])
    _top_left = np.array([top_left])
    _bottom_right = np.array([bottom_right])
    _camera_center = np.array([camera_center])

    image_coordinates = np.concatenate([_top_right, _bottom_left, _top_left, _bottom_right])
    # Change the origin to bottom-left of the image
    image_coordinates[:, 1] = image_height - image_coordinates[:, 1]

    bbox_global_coordinates = img_to_global_coord(
        image_coordinates, _camera_center, 
        pixel_width, pixel_height, focal_length, 
        image_width, image_height, camera_height, 
        yaw_angle, is_bbox=True
    )

    # Adjust for the roll and pitch
    mul = -1.
    if (360. - yaw_angle) > 180:
        mul = 1.
    # bbox_global_coordinates[:, 0] = pitch_correction(_camera_center, camera_height, bbox_global_coordinates, mul*pitch_angle)
    bbox_global_coordinates[:, 1] = roll_correction(_camera_center, camera_height, bbox_global_coordinates, -mul*roll_angle)

    # Unpack
    top_left = bbox_global_coordinates[2, :]
    top_right = bbox_global_coordinates[0, :]
    bottom_left = bbox_global_coordinates[1, :]
    bottom_right = bbox_global_coordinates[3, :]

    global_coordinates = BoxCoordinates(top_left, top_right, bottom_left, bottom_right)

    return global_coordinates


def select_best_bbox(bboxes, center_cooridnates, return_index=False):
    
    # Find the centroids
    centroids = []
    for bbox in bboxes:
        centroid_x = (bbox["top_left"][0] + bbox["top_right"][0]) / 2.
        centroid_y = (bbox["top_left"][1] + bbox["bottom_left"][1]) / 2.
        centroid = np.array([centroid_x, centroid_y])
        centroids.append(centroid)
    centroids = np.array(centroids)

    distances = np.sum((centroids - center_cooridnates)**2, axis=1)
    # Select the box with the minimum distance
    min_dist_bbox_idx = np.argmin(distances)

    min_dist_bbox = bboxes[min_dist_bbox_idx]

    if return_index:
        return min_dist_bbox, min_dist_bbox_idx
    return min_dist_bbox

class BBoxFilter:

    def __init__(self, images: List[Image]):
        self.images = images
        self.image_map = {image.id: image for image in images}
        self.total_bboxes = sum([len(image.bboxes) for image in self.images])

    def deduplicate_bboxes(self):

        comparisons = self.filter_images()
        self.filter_bounding_boxes(comparisons)

    def filter_images(self):

        image_ids = list(self.image_map.keys())
        comparisons = dict()
        # Find the overlap between FOVs of the images
        for i, image_id in enumerate(image_ids):
            image = self.image_map[image_id]
            comparisons[image_id] = []
            for j in range(i+1, len(image_ids)):
                compare_image_id = image_ids[j]
                compare_image = self.image_map[compare_image_id]
                fov_iou = bb_iou(image.fov, compare_image.fov)
                if fov_iou > FOV_IOU_THRESH:
                    comparisons[image_id].append(compare_image_id)

        return comparisons

    def filter_bounding_boxes(self, comparisons: Dict[str, List[str]]):

        # Only compare bounding boxes from images which have a significant overlap
        for image_id, image_ids_for_comparison in comparisons.items():
            for box in self.image_map[image_id].bboxes:
                compared = set()
                box_hash = generate_hash(box)
                for compare_image_id in image_ids_for_comparison:
                    boxes = self.image_map[compare_image_id].bboxes
                    for _box in boxes:
                        _box_hash = generate_hash(_box, box_hash)
                        if _box_hash in compared:
                            continue
                        compared.add(_box_hash)
                        iou = box.bb_iou(_box)
                        if iou > BBOX_OVERLAP_THRESH:
                            box.add_box(_box)
                            _box.add_box(box)
        self.select_best_bbox()

    def select_best_bbox(self):
        
        visited = set()
        for image in self.images:
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
                centers = np.array([self.image_map[_box.image_id].camera_location for _box in all_boxes])
                centroids = np.array([_box.global_centroid for _box in all_boxes])
                distances = ((centroids - centers[:, :2])**2).sum(axis=-1)
                min_idx = np.argmin(distances)
                all_boxes[min_idx].is_primary = True
        

class BBoxMapper():

    def __init__(self, images: List[Image]):
        """Class to map bounding box coordinates from image cordinates
           to global coordinates
        """
        self.images = images

    def map(self):
        """
        Maps all the bounding boxes to a global coordinate space
        """
        for image in self.images:

            camera_location = image.camera_location
            camera_x = camera_location[0]
            camera_y = camera_location[1]
            camera_height = camera_z = camera_location[2]

            camera_center = [camera_x, camera_y]
            
            global_coordinates = dict()
            for bbox in image.bboxes:
                
                top_left = bbox.local_coordinates.top_left
                bottom_left = bbox.local_coordinates.bottom_left
                top_right = bbox.local_coordinates.top_right
                bottom_right = bbox.local_coordinates.bottom_right
                
                # Transformation
                global_coordinates = bbox_to_global(
                    top_left, top_right, bottom_left, bottom_right, 
                    camera_center, image.pixel_width, image.pixel_height, 
                    image.focal_length, 
                    image.width, image.height, camera_height, 
                    image.yaw, image.pitch, image.roll
                )
                
                bbox.update_global_coordinates(global_coordinates)

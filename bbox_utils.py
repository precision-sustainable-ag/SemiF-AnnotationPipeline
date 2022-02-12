from typing import Tuple, List
import math
from xmlrpc.client import boolean
import numpy as np
from . import Reader

class BBox():

    def __init__(self, reader: Reader):

        self.config = reader.read()


def get_xy_rotation_matrix(angle: float) -> np.ndarray:
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


def find_world_coords(center_coords: np.ndarray, unrotated_coords: np.ndarray, yaw_angle: float, is_bbox: bool):

    world_center_coords = np.array(center_coords)
    if np.ndim(world_center_coords) == 1:
        world_center_coords = np.expand_dims(world_center_coords, axis=0)

    R = get_xy_rotation_matrix(yaw_angle)

    # Rotate the new coordinate
    rotated_coordinates = rotation_transform(unrotated_coords, R)

    # Invert the translation
    world_coordinates = rotated_coordinates + world_center_coords

    if is_bbox:
        _world_coordinates = world_coordinates.copy()
        mask = np.argsort(world_coordinates[:, 0])

        # Determine which is the top_left and bottom_left
        left_coordinates = world_coordinates[mask[:2], :]
        top_sort = np.argsort(left_coordinates[:, 1])
        bottom_left = left_coordinates[top_sort[0], :]
        top_left = left_coordinates[top_sort[1], :]
        
        # Determine which is the top_right and bottom_right
        right_coordinates = world_coordinates[mask[2:], :]
        top_sort = np.argsort(right_coordinates[:, 1])
        bottom_right = right_coordinates[top_sort[0], :]
        top_right = right_coordinates[top_sort[1], :]

        world_coordinates[0, :] = top_right
        world_coordinates[1, :] = bottom_left
        world_coordinates[2, :] = top_left
        world_coordinates[3, :] = bottom_right

    return world_coordinates


def img_to_world_coord(image_coordinates: np.ndarray, center_coordinates: np.ndarray,
                       pixel_width: float, pixel_height: float, 
                       yaw_angle: float, is_bbox: bool):
    

    _image_coordinates = image_coordinates.copy()
    _center_coordinates = center_coordinates.copy()
    if np.ndim(image_coordinates) == 1:
        _image_coordinates = np.expand_dims(image_coordinates, axis=0)
    assert _image_coordinates.shape[1] == 2

    if np.ndim(_center_coordinates) == 1:
        _center_coordinates = np.expand_dims(_center_coordinates, axis=0)
    assert _center_coordinates.shape[1] == 2

    scaled_coordinates = _image_coordinates.copy()
    scaled_coordinates[:, 0] = scaled_coordinates[0, :] * pixel_width
    scaled_coordinates[:, 1] = scaled_coordinates[0, :] * pixel_height

    # Unrotated coordinates wrt origin shifted to the camera location
    translated_coordinates = scaled_coordinates - _center_coordinates
    world_coordinates = find_world_coords(_center_coordinates, translated_coordinates, yaw_angle, is_bbox)

    return world_coordinates


def bbox_to_world(top_left: List, top_right: List, 
                  bottom_left: List, bottom_right: List, 
                  center_coordinates: List, 
                  pixel_width: float, pixel_height: float, 
                  yaw_angle: float):

    assert all([len(coord) == 2 for coord in [top_left, top_right, bottom_left, bottom_right, center_coordinates]])

    _top_right = np.array([top_right])
    _bottom_left = np.array([bottom_left])
    _top_left = np.array([top_left])
    _bottom_right = np.array([bottom_right])
    _center_coordinates = np.array([center_coordinates])

    image_coordinates = np.concatenate([_top_right, _bottom_left, _top_left, _bottom_right])

    bbox_world_coordinates = img_to_world_coord(
        image_coordinates, _center_coordinates, 
        pixel_width, pixel_height, yaw_angle
    )

    return bbox_world_coordinates


def select_best_bbox(
    bboxes: List[List[Tuple[float, float]]],
    center_cooridnates: Tuple, pixel_width: float, pixel_height: float, yaw_angle: float):
    
    # Convert the bounding boxes to world coordinates
    world_bboxes = []
    for bbox in bboxes:
        top_left = bbox[0]
        top_right = bbox[1]
        bottom_left = bbox[2]
        bottom_right = bbox[3]
        
        world_bbox = bbox_to_world(
            top_left, top_right, bottom_left, bottom_right, 
            center_cooridnates, pixel_width, pixel_height, yaw_angle
        )

        world_bboxes.append(world_bbox)

    # TODO: Logic for selecting the best bounding box
    # Should be based on the location of the bounding in real
    # world coordinates

    pass

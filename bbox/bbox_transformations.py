from typing import Tuple, List
import math
import numpy as np

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


def find_global_coords(center_coords: np.ndarray, 
                       unrotated_coords: np.ndarray, 
                       yaw_angle: float, 
                       is_bbox: bool=True) -> np.ndarray:
    """_summary_

    Args:
        center_coords (np.ndarray): Image center coordinates in global coordinate system
        unrotated_coords (np.ndarray): Unrotated coordinates in global coordinate system
        yaw_angle (float): Yaw angle in degrees
        is_bbox (bool, optional): Do the coordinates describe a boudig box? Defaults to True.

    Returns:
        np.ndarray: _description_
    """

    global_center_coords = np.array(center_coords)
    if np.ndim(global_center_coords) == 1:
        global_center_coords = np.expand_dims(global_center_coords, axis=0)

    R = get_xy_rotation_matrix(yaw_angle)

    # Rotate the new coordinate
    rotated_coordinates = rotation_transform(unrotated_coords, R)

    # Translate to image center in global coordinates
    global_coordinates = rotated_coordinates + global_center_coords

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


def img_to_global_coord(image_coordinates: np.ndarray, center_coordinates: np.ndarray,
                        pixel_width: float, pixel_height: float, 
                        image_width: int, image_height: int,
                        yaw_angle: float, is_bbox: bool=True) -> np.ndarray:
    """_summary_

    Args:
        image_coordinates (np.ndarray): Image coordinates
        center_coordinates (np.ndarray): Image center coordinates in global coordinate space
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
    _center_coordinates = center_coordinates.copy()

    if np.ndim(image_coordinates) == 1:
        _image_coordinates = np.expand_dims(image_coordinates, axis=0)
    assert _image_coordinates.shape[1] == 2

    if np.ndim(_center_coordinates) == 1:
        _center_coordinates = np.expand_dims(_center_coordinates, axis=0)
    assert _center_coordinates.shape[1] == 2

    image_center = np.array([[float(image_width // 2), float(image_height//2)]])

    # Change the origin to bottom-left of the image
    _image_coordinates[:, 0] = image_width - _image_coordinates[:, 0]

    # Shift the origin to the image center
    _image_coordinates -= image_center

    untranslated_global_coordinates = _image_coordinates.copy()
    untranslated_global_coordinates[:, 0] = untranslated_global_coordinates[0, :] * pixel_width
    untranslated_global_coordinates[:, 1] = untranslated_global_coordinates[0, :] * pixel_height

    global_coordinates = find_global_coords(center_coordinates, untranslated_global_coordinates, yaw_angle, is_bbox)

    return global_coordinates


def bbox_to_global(top_left: List, top_right: List, 
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

    bbox_global_coordinates = img_to_global_coord(
        image_coordinates, _center_coordinates, 
        pixel_width, pixel_height, yaw_angle, is_bbox=True
    )

    return bbox_global_coordinates


def select_best_bbox(
    bboxes: List[List[Tuple[float, float]]],
    center_cooridnates: Tuple, pixel_width: float, pixel_height: float, yaw_angle: float):
    
    # Convert the bounding boxes to global coordinates
    global_bboxes = []
    for bbox in bboxes:
        top_left = bbox[0]
        top_right = bbox[1]
        bottom_left = bbox[2]
        bottom_right = bbox[3]
        
        global_bbox = bbox_to_global(
            top_left, top_right, bottom_left, bottom_right, 
            center_cooridnates, pixel_width, pixel_height, yaw_angle
        )

        global_bboxes.append(global_bbox)

    # TODO: Logic for selecting the best bounding box
    # Should be based on the location of the bounding in
    # global coordinates

    pass

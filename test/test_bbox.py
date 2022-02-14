EPS = 1e-8

def test_rotation():
    import numpy as np
    import math
    from bbox.bbox_transformations import rotation_transform, get_xy_rotation_matrix

    coordinates = np.array([
        [1., 1.],
        [0., 1.],
        [1., 0.],
    ])

    R = get_xy_rotation_matrix(45.)

    sqrt2 = math.sqrt(2)
    expected_rotated_coordinates = np.array([
        [0., sqrt2],
        [-1./sqrt2, 1./sqrt2],
        [1./sqrt2, 1./sqrt2]
    ])

    rotated_coordinates = rotation_transform(coordinates, R)

    assert np.sum((expected_rotated_coordinates - rotated_coordinates)**2) < EPS

def test_find_global_coords():
    import numpy as np
    import math
    from bbox.bbox_transformations import find_global_coords

    yaw_angle = 45. # Degrees
    sqrt2 = math.sqrt(2)

    coordinates = np.array([
        [2., 2.],
        [1., 1.],
        [1., 2.],
        [2., 1.]
    ])
    center_coords = np.array([[3., 3.]])

    rotated_coords = np.array([
        [1/sqrt2, 3/sqrt2],
        [-1/sqrt2, 3/sqrt2],
        [0., 4/sqrt2],
        [0., 2/sqrt2]
        
    ])
    expected_coordinates = rotated_coords + center_coords

    global_coords = find_global_coords(center_coords, coordinates, yaw_angle, is_bbox=True)

    assert np.sum((expected_coordinates - global_coords)**2) < EPS

import numpy as np

"""
Transformations. Following frames are defined:

base: main frame where 3D annotations are done in, x-forward, y-left, z-up
upper_lidar: x-forward, y-left, z-up
lower_lidar: x-forward, y-left, z-up
laser: x-forward, y-left, z-up
"""


def _get_R_z(rot_z):
    cs, ss = np.cos(rot_z), np.sin(rot_z)
    return np.array([[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]], dtype=np.float32)


# laser to base
_rot_z_laser_to_base = np.pi / 120
_T_laser_to_base = np.array([0.019685, 0, -0.742092], dtype=np.float32).reshape(3, 1)
_R_laser_to_base = _get_R_z(_rot_z_laser_to_base)

# upper_lidar to base
_rot_z_upper_lidar_to_base = 0.085
_T_upper_lidar_to_base = np.array([0, 0, 0.33529], dtype=np.float32).reshape(3, 1)
_R_upper_lidar_to_base = _get_R_z(_rot_z_upper_lidar_to_base)

# lower_lidar to base
_rot_z_lower_lidar_to_base = 0.0
_T_lower_lidar_to_base = np.array([0, 0, -0.13511], dtype=np.float32).reshape(3, 1)
_R_lower_lidar_to_base = np.eye(3, dtype=np.float32)


"""
Transformation API
"""


def transform_pts_upper_velodyne_to_base(pts):
    """Transform points from upper velodyne frame to base frame

    Args:
        pts (np.array[3, N]): points (x, y, z)

    Returns:
        pts_trans (np.array[3, N])
    """
    return _R_upper_lidar_to_base @ pts + _T_upper_lidar_to_base


def transform_pts_lower_velodyne_to_base(pts):
    return _R_lower_lidar_to_base @ pts + _T_lower_lidar_to_base


def transform_pts_laser_to_base(pts):
    return _R_laser_to_base @ pts


def transform_pts_base_to_upper_velodyne(pts):
    return _R_upper_lidar_to_base.T @ (pts - _T_upper_lidar_to_base)


def transform_pts_base_to_lower_velodyne(pts):
    return _R_lower_lidar_to_base.T @ (pts - _T_lower_lidar_to_base)


def transform_pts_base_to_laser(pts):
    return _R_laser_to_base.T @ pts

def transform_laser_to_base(laser_data,):

    # Calculate angles corresponding to each range
    angles = np.linspace(-np.pi, np.pi, len(laser_data), dtype=float)

    # Calculate x and y coordinates
    x = laser_data * np.cos(angles)
    y = laser_data * np.sin(angles)

    # Create a 2xN matrix of points
    points = np.vstack((x, y))

    # Define the rotation matrix for the z-axis rotation
    rotation_matrix = np.array([
        [np.cos(_rot_z_laser_to_base), -np.sin(_rot_z_laser_to_base)],
        [np.sin(_rot_z_laser_to_base),  np.cos(_rot_z_laser_to_base)]
    ])

    # Apply the rotation and then the translation
    transformed_points = rotation_matrix @ points + _T_laser_to_base[:2]

    return transformed_points

def transform_box_3D_base_to_laser(boxes):
    """
    Transform bounding boxes from the base frame to the laser frame.

    Parameters:
    boxes (numpy.ndarray): A 7xN array where each column represents a bounding box.
                           The values inside each column are:
                           [x_center, y_center, z_center, length, width, height, rotation]

    Returns:
    numpy.ndarray: A 7xN array where each column represents the transformed bounding box.
                   The transformed bounding box values are:
                   [x_center_transformed, y_center_transformed, z_center_transformed, length, width, height, rotation_transformed]
    """
    # Check input shape
    assert boxes.shape[0] == 7, "Input boxes array must be of shape 7xN"

    # Extract the centers and rotation
    centers = boxes[:3, :]  # Shape [3, N]
    rotations = boxes[6, :] # Shape [N]

    # Apply the rotation transformation to the center coordinates using the transpose of the rotation matrix
    transformed_centers = _R_laser_to_base.T @ (centers - _T_laser_to_base ) # Shape [3, N]

    # Apply the inverse Z rotation to the existing rotation (base to laser)
    new_rotations = rotations - _rot_z_laser_to_base  # Shape [N]

    # Combine the transformed centers and rotations with the original box dimensions
    transformed_boxes = np.vstack([
        transformed_centers,          # Shape [3, N]
        boxes[3:6, :],                # Original dimensions [width, length, height], Shape [3, N]
        new_rotations[np.newaxis, :]  # Ensure shape [1, N] for vstack
    ])

    return transformed_boxes

def transform_box_3D_laser_to_base(boxes):
    """
    Transform bounding boxes from the laser frame to the base frame.

    Parameters:
    boxes (numpy.ndarray): A 7xN array where each column represents a bounding box.
                           The values inside each column are:
                           [x_center, y_center, z_center, length, width, height, rotation]

    Returns:
    numpy.ndarray: A 7xN array where each column represents the transformed bounding box.
                   The transformed bounding box values are:
                   [x_center_transformed, y_center_transformed, z_center_transformed, length, width, height, rotation_transformed]
    """
    # Check input shape
    assert boxes.shape[0] == 7, "Input boxes array must be of shape 7xN"

    # Extract the centers and rotation
    centers = boxes[:3, :]  # Shape [3, N]
    rotations = boxes[6, :] # Shape [N]

    # Apply the rotation transformation to the center coordinates
    transformed_centers = _R_laser_to_base @ centers + _T_laser_to_base  # Shape [3, N]

    # Apply the Z rotation to the existing rotation (laser to base)
    new_rotations = rotations + _rot_z_laser_to_base  # Shape [N]

    # Combine the transformed centers and rotations with the original box dimensions
    transformed_boxes = np.vstack([
        transformed_centers,          # Shape [3, N]
        boxes[3:6, :],                # Original dimensions [length, width, height], Shape [3, N]
        new_rotations[np.newaxis, :]  # Ensure shape [1, N] for vstack
    ])

    return transformed_boxes



def transform_pts_base_to_stitched_im(pts):
    """Project 3D points in base frame to the stitched image

    Args:
        pts (np.array[3, N]): points (x, y, z)

    Returns:
        pts_im (np.array[2, N])
        inbound_mask (np.array[N])
    """
    im_size = (480, 3760)

    # to image coordinate
    pts_rect = pts[[1, 2, 0], :]
    pts_rect[:2, :] *= -1

    # to pixel
    horizontal_theta = np.arctan2(pts_rect[0], pts_rect[2])
    horizontal_percent = horizontal_theta / (2 * np.pi) + 0.5
    x = im_size[1] * horizontal_percent
    y = (
        485.78 * pts_rect[1] / pts_rect[2] * np.cos(horizontal_theta)
        + 0.4375 * im_size[0]
    )
    # horizontal_theta = np.arctan(pts_rect[0, :] / pts_rect[2, :])
    # horizontal_theta += (pts_rect[2, :] < 0) * np.pi
    # horizontal_percent = horizontal_theta / (2 * np.pi)
    # x = ((horizontal_percent * im_size[1]) + 1880) % im_size[1]
    # y = (
    #     485.78 * (pts_rect[1, :] / ((1 / np.cos(horizontal_theta)) * pts_rect[2, :]))
    # ) + (0.4375 * im_size[0])

    # x is always in bound by cylindrical parametrization
    # y is always at the lower half of the image, since laser is lower than the camera
    # thus only one boundary needs to be checked
    inbound_mask = y < im_size[0]

    return np.stack((x, y), axis=0).astype(np.int32), inbound_mask


def transform_pts_laser_to_stitched_im(pts):
    pts_base = transform_pts_laser_to_base(pts)
    return transform_pts_base_to_stitched_im(pts_base)

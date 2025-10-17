import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy.interpolate import interp1d

# Constants for VLP-16 LiDAR
NUM_LINES = 16
VERTICAL_FOV_DEG = (-15, 15)  # VLP-16 field of view from -15 to 15 degrees
ANGLE_RESOLUTION_DEG = 0.2
POINTS_PER_LINE = int(360 / ANGLE_RESOLUTION_DEG)
TARGET_LINE_INDEX = 8  # 8th lowest line corresponds to -1 degree
MAX_DIST = 100  # Max distance for points out of range

# Directory paths
SiT_dataset_dir = '/scai_data/data01/daav/SiT'
overwrite = True  # Set this to True to overwrite existing files


def smooth1d(data, filter='savgol', window=21, polyorder=2, check_thres=False, thres=[-1.2, 1.5], mode='interp'):
    if check_thres:
        curr_nearest_valid = 0
        for idx in range(1, len(data)):
            if thres[0] < data[idx] < thres[1]:
                curr_nearest_valid = idx
            else:
                data[idx] = data[curr_nearest_valid]
    
    if filter == 'savgol':
        data_smoothed = signal.savgol_filter(data, window_length=window, polyorder=polyorder, mode=mode)
    elif filter == 'moving_average':
        ma_window = np.ones(window) / window
        data_smoothed = np.convolve(data, ma_window, mode='same')
    return data_smoothed

def smooth(nd_data, filter='savgol', window=21, polyorder=2, check_thres=False, thres=[-1.2, 1.5]):
    nd_data_smoothed = np.zeros_like(nd_data)
    for dim in range(np.shape(nd_data)[1]):
        nd_data_smoothed[:, dim] = smooth1d(
            nd_data[:, dim],
            filter=filter,
            window=window,
            polyorder=polyorder,
            check_thres=check_thres,
            thres=thres,
        )
    return nd_data_smoothed

def compute_motion_derivative(motion_stamped_dict, subset=None):
    ts = motion_stamped_dict.get("timestamp")
    dval_dt_dict = {"timestamp": ts}
    
    if subset is None:
        subset = motion_stamped_dict.keys()
    for val in subset:
        if val == "timestamp":
            continue
        else:
            dval_dt = np.gradient(motion_stamped_dict.get(val), ts, axis=0)
            dval_dt_dict.update({val: dval_dt})
    return dval_dt_dict

# Quaternion math functions
def quat_mul(quat0, quat1):
    x0, y0, z0, w0 = quat0
    x1, y1, z1, w1 = quat1
    return np.array([
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    ], dtype=np.float64)

def quat_norm(quat_):
    return quat_ / np.linalg.norm(quat_)

def quat_conjugate(quat_):
    x0, y0, z0, w0 = quat_
    return np.array([-x0, -y0, -z0, w0], dtype=np.float64)

def qv_mult(quat_, vec_):
    quat_ = quat_norm(quat_)
    temp = quat_mul(quat_, vec_)

    rotated_vec = quat_mul(temp, quat_conjugate(quat_))
    
    return rotated_vec[:3]

# Main velocity calculation
def calculate_local_x_velocity(position_g, quat_xyzw, timestamp):
    # Smooth position in the global frame
    print(f'Position shape is {position_g.shape}')
    smoothed_position_g = smooth(position_g, filter='savgol', window=21, polyorder=2)
    position_g = smoothed_position_g

    # Prepare state dict for derivative calculation
    state_pose_g = {
        "x": position_g[:, 0],
        "y": position_g[:, 1],
        "z": position_g[:, 2],
        "timestamp": timestamp,
    }
    # Calculate global velocity
    state_vel_g = compute_motion_derivative(state_pose_g, subset=["x", "y", "z"])
    xyz_vel_g = np.vstack((state_vel_g["x"], state_vel_g["y"], state_vel_g["z"])).T

    # Initialize velocity array to store transformed velocities in the local frame
    xyz_vel_local = np.zeros_like(xyz_vel_g)

    # Transform each global velocity component to the local frame using the inverse quaternion
    for idx in range(xyz_vel_g.shape[0]):
        vel = xyz_vel_g[idx, :]
        quat = quat_xyzw[idx, :]
        vel_ = np.zeros(4, dtype=np.float64)
        vel_[:3] = vel
        xyz_vel_local[idx, :] = qv_mult(quat_conjugate(quat), vel_)

    # Smooth the x-component of the local velocity (forward velocity in local frame)
    smoothed_x_vel = smooth1d(
        xyz_vel_local[:, 0],
        filter='savgol',
        window=21,
        polyorder=2,
        check_thres=True,
    )

    return smoothed_x_vel

# Example of updating the transform_dict with x_vel
def update_transform_dict_with_velocity(transform_dict, timestamp):
    # Calculate smoothed local x_vel
    position_global = transform_dict['position']
    quat_xyzw = transform_dict['orientation']
    smoothed_x_vel = calculate_local_x_velocity(position_global, quat_xyzw, timestamp)
    
    # Add x_vel to the transform_dict
    transform_dict.update({"x_vel": smoothed_x_vel})
    return transform_dict

# Helper function to load a transformation matrix from a text file
def load_transformation_matrix(filepath):
    try:
        with open(filepath, 'r') as f:
            data = f.read().strip().split(',')
            matrix = np.array(data, dtype=float).reshape(4, 4)
        return matrix
    except Exception as e:
        print(f"Error loading transformation matrix from {filepath}: {e}")
        return None

# Helper function to load 3D labels from a text file
def load_3d_labels(filepath):
    labels = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split()
                if parts[0] == 'Pedestrian':
                    track_id = int(parts[1].split(':')[1])
                    h, l, w = map(float, parts[2:5])
                    x, y, z = map(float, parts[5:8])
                    rot = float(parts[8])
                    labels.append([x, y, z, l, w, h, rot, track_id])
        return np.array(labels)
    except Exception as e:
        print(f"Error loading labels from {filepath}: {e}")
        return None

# Function to transform points and labels with a transformation matrix
def transform_points(points, transformation_matrix):
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = (transformation_matrix @ homogeneous_points.T).T[:, :3]
    return transformed_points

def transform_labels(labels, transformation_matrix):
    transformed_labels = []
    
    # Extract the rotation matrix and convert to a Rotation object
    rotation_matrix = transformation_matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    
    # Extract the yaw rotation from the rotation matrix
    _, _, transformed_yaw_base = rotation.as_euler('xyz', degrees=False)  # Extracts roll, pitch, and yaw (z rotation)

    for label in labels:
        # Transform the center point
        box_center = np.array([label[0], label[1], label[2], 1])
        transformed_center = (transformation_matrix @ box_center.T)[:3]

        # Extract the original yaw (z-rotation) angle from the label
        original_yaw = label[6]

        # Combine the original yaw with the extracted transformed yaw
        transformed_yaw = original_yaw + transformed_yaw_base

        # Create the transformed label
        transformed_label = [*transformed_center, *label[3:6], transformed_yaw, label[7]]  # Append track ID
        transformed_labels.append(transformed_label)

    return np.array(transformed_labels)


def load_pcd(pcd_file):
    """Load a PCD file and return points as a numpy array."""
    pcd = o3d.io.read_point_cloud(pcd_file)
    return np.asarray(pcd.points)

def get_lidar_lines(points):
    """Separate points into 16 vertical lines based on their vertical angles using vectorized operations."""
    vertical_angles = np.linspace(VERTICAL_FOV_DEG[0], VERTICAL_FOV_DEG[1], NUM_LINES)
    distances_xy = np.linalg.norm(points[:, :2], axis=1)
    mask_nonzero = distances_xy > 0

    vertical_angles_points = np.full(points.shape[0], np.nan)
    vertical_angles_points[mask_nonzero] = np.degrees(np.arctan2(points[mask_nonzero, 2], distances_xy[mask_nonzero]))
    line_indices = np.argmin(np.abs(vertical_angles_points[:, None] - vertical_angles), axis=1)

    lidar_lines = {i: points[line_indices == i] for i in range(NUM_LINES)}
    return lidar_lines

def assemble_angle_distance_data(line_points):
    """Assemble angle-distance data with interpolated points for missing data as necessary."""
    angle_distance_data = []
    angles = np.arctan2(line_points[:, 1], line_points[:, 0])
    distances = np.linalg.norm(line_points[:, :2], axis=1)
    sorted_indices = np.argsort(angles)
    angles = angles[sorted_indices]
    distances = distances[sorted_indices]

    expected_diff_rad = np.deg2rad(ANGLE_RESOLUTION_DEG)
    for i in range(len(angles) - 1):
        angle_distance_data.append((angles[i], distances[i]))
        angle_diff = angles[i + 1] - angles[i]
        if angle_diff > expected_diff_rad * 1.5:
            missing_points_count = int(round(angle_diff / expected_diff_rad)) - 1
            for j in range(missing_points_count):
                interpolated_angle = angles[i] + (j + 1) * expected_diff_rad
                angle_distance_data.append((interpolated_angle, MAX_DIST))
    
    angle_distance_data.append((angles[-1], distances[-1]))
    return angle_distance_data

def calculate_ranges(angle_distance_data):
    """Calculate the 2D ranges array for the specified line with POINTS_PER_LINE points from -pi to pi."""
    desired_angles = np.linspace(-np.pi, np.pi, POINTS_PER_LINE)
    data_angles = np.array([point[0] for point in angle_distance_data])
    data_distances = np.array([point[1] for point in angle_distance_data])
    interp_func = interp1d(data_angles, data_distances, kind='nearest', fill_value=MAX_DIST, bounds_error=False)
    ranges = interp_func(desired_angles)
    return ranges

def process_lidar_scan_2D(pcd_file):
    """Extracts and returns 2D lidar data from the specified PCD file."""
    points = load_pcd(pcd_file)
    lidar_lines = get_lidar_lines(points)
    target_line_points = lidar_lines[TARGET_LINE_INDEX]
    angle_distance_data = assemble_angle_distance_data(target_line_points)
    ranges = calculate_ranges(angle_distance_data)
    lidar_2D = {'bottom': {'ranges': ranges}}
    return lidar_2D

if __name__ == '__main__':
    # Iterate over each folder in the dataset directory
    for folder in os.listdir(SiT_dataset_dir):
        print(f'Processing folder {folder}')
        folder_path = os.path.join(SiT_dataset_dir, folder)
        if not os.path.isdir(folder_path) or folder.startswith('.'):
            continue

        # Check the number of frames based on ego_trajectory files
        ego_trajectory_dir = os.path.join(folder_path, 'ego_trajectory')
        if not os.path.exists(ego_trajectory_dir):
            print(f"Missing ego_trajectory directory in {folder_path}. Skipping folder.")
            continue

        frame_files = [f for f in os.listdir(ego_trajectory_dir) if f.endswith('.txt')]
        total_frames = len(frame_files)
        if total_frames == 0:
            print(f"No ego_trajectory .txt files found in {folder_path}. Skipping folder.")
            continue

        # Create the main processed directory
        processed_dir = os.path.join(SiT_dataset_dir, f'{folder}_processed')
        os.makedirs(processed_dir, exist_ok=True)

        # Create directories for results within the processed directory
        alg_res_dir = os.path.join(processed_dir, 'alg_res')
        lidars_dir = os.path.join(processed_dir, 'lidars')
        lidars_subdir = os.path.join(lidars_dir, folder)
        tracks_dir = os.path.join(alg_res_dir, 'tracks')
        detections_dir = os.path.join(alg_res_dir, 'detections')
        source_data_dir = os.path.join(processed_dir, 'source_data')
        timestamp_dir = os.path.join(source_data_dir, 'timestamp')
        tf_SiT_dir = os.path.join(source_data_dir, 'tf_SiT')
        lidars_2D_dir = os.path.join(processed_dir, 'lidars_2D', folder)

        os.makedirs(tracks_dir, exist_ok=True)
        os.makedirs(detections_dir, exist_ok=True)
        os.makedirs(lidars_subdir, exist_ok=True)
        os.makedirs(timestamp_dir, exist_ok=True)
        os.makedirs(tf_SiT_dir, exist_ok=True)
        os.makedirs(lidars_2D_dir, exist_ok=True)

        # Generate and save the timestamp array
        timestamps = np.linspace(0, total_frames / 10, num=total_frames, endpoint=False)
        timestamp_filename = os.path.join(timestamp_dir, f'{folder}_stamped.npy')
        np.save(timestamp_filename, timestamps)

        # Prepare arrays for positions and orientations
        positions = []
        orientations = []

        processed_frames = 0
        missing_files = 0
        error_count = 0
        missing_files_details = {
            'ego_trajectory': [],
            'label_3d': [],
            'pcd': []
        }

        # Dictionary to hold all transformed labels for the current folder
        out_det_all = {}

        # Iterate over the number of frames found
        for frame_counter in range(total_frames):
            ego_filepath = os.path.join(ego_trajectory_dir, f'{frame_counter}.txt')
            label_filepath = os.path.join(folder_path, 'label_3d', f'{frame_counter}.txt')
            pcd_filepath = os.path.join(folder_path, 'velo', 'concat', 'data', f'{frame_counter}.pcd')
            pcd_bottom_filepath = os.path.join(folder_path, 'velo', 'bottom', 'data', f'{frame_counter}.pcd')

            if not os.path.exists(ego_filepath):
                missing_files += 1
                missing_files_details['ego_trajectory'].append(ego_filepath)
                continue
            if not os.path.exists(label_filepath):
                missing_files += 1
                missing_files_details['label_3d'].append(label_filepath)
                continue
            if not os.path.exists(pcd_filepath):
                missing_files += 1
                missing_files_details['pcd'].append(pcd_filepath)
                continue

            # Load the transformation matrix from ego_trajectory
            transformation_matrix = load_transformation_matrix(ego_filepath)
            if transformation_matrix is None:
                error_count += 1
                continue

            # Extract position and quaternion from the transformation matrix
            position = transformation_matrix[:3, 3]  # x, y, z position
            rotation_matrix = transformation_matrix[:3, :3]
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()  # Quaternion as [x, y, z, w]

            positions.append(position)
            orientations.append(quaternion)

            # Load 3D labels
            labels = load_3d_labels(label_filepath)
            if labels is None or labels.size == 0:
                error_count += 1
                continue

            # Load point cloud data
            try:
                pc = load_pcd(pcd_filepath)
            except Exception as e:
                print(f"Error loading point cloud from {pcd_filepath}: {e}")
                error_count += 1
                continue

            # Transform point cloud and labels
            transformed_pc = transform_points(pc, transformation_matrix)
            # transformed_labels = transform_labels(labels, transformation_matrix)
            transformed_labels = labels # They are already transformed to global frame

            # Save lidar data as .npy in lidars/{folder_name}
            lidar_filename = os.path.join(lidars_subdir, f'{frame_counter:05d}.npy')
            if (not os.path.exists(lidar_filename)) or overwrite:
                with open(lidar_filename, "wb") as f:
                    np.save(f, transformed_pc)

            # Save transformed labels as a dictionary entry
            boxes_with_labels = transformed_labels
            out_det_all.update({frame_counter: boxes_with_labels})

            # Process and save 2D lidar data
            lidar_2D = process_lidar_scan_2D(pcd_bottom_filepath)
            lidar_2D_filename = os.path.join(lidars_2D_dir, f'{frame_counter:05d}.npy')
            if (not os.path.exists(lidar_2D_filename)) or overwrite:
                with open(lidar_2D_filename, "wb") as f:
                    np.save(f, lidar_2D)

            processed_frames += 1

        # Save the labels dictionary as .npy in alg_res/tracks
        labels_filename = os.path.join(tracks_dir, f'{folder}_gt.npy')
        if out_det_all:
            np.save(labels_filename, out_det_all)

        # Save the transform dictionary to the tf_SiT folder
        transform_dict = {
            'position': np.array(positions),
            'orientation': np.array(orientations),
            'timestamp': timestamps
        }
        transform_dict = update_transform_dict_with_velocity(transform_dict, timestamps)

        transform_filename = os.path.join(tf_SiT_dir, f'{folder}_tf_SiT_sampled.npy')
        np.save(transform_filename, transform_dict)

        # Print summary for the folder
        print(f"Summary for folder '{folder}':")
        print(f"  Total frames processed: {processed_frames}")
        print(f"  Missing files skipped: {missing_files}")
        if missing_files > 0:
            print(f"  Missing ego_trajectory files: {len(missing_files_details['ego_trajectory'])}")
            print(f"  Missing label_3d files: {len(missing_files_details['label_3d'])}")
            print(f"  Missing pcd files: {len(missing_files_details['pcd'])}")
            for file_type, file_list in missing_files_details.items():
                if file_list:
                    print(f"  Missing {file_type} files (first 5 shown): {file_list[:5]}")
        print(f"  Frames with errors: {error_count}")

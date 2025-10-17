import os
import json
import numpy as np
from crowdbot_data.crowdbot_data import CrowdBotDatabase
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


_pos_camera_to_base_chassis_link = np.array([-0.019685, 0, 0.742092], dtype=np.float32).reshape(3, 1)
folder = 'JRDB_whole'
config_path = './datasets_configs/data_path_JRDB.yaml'
JRDB_dataset_dir = '/scai_data/data01/daav/JRDB/train_dataset'

def _get_R_z(rot_z):
    cs, ss = np.cos(rot_z), np.sin(rot_z)
    return np.array([[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]], dtype=np.float32)

def get_yaw_from_quat(quat):
    scipy_rot = R.from_quat(quat)
    rot_zyx = scipy_rot.as_euler('zyx')
    return rot_zyx[0]

def transform_boxes_3D(boxes, pos, rot_z, is_quaternion=True):
    """
    Transform 3D bounding boxes.

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

    if is_quaternion:
        rot_z = get_yaw_from_quat(rot_z)
    _R_matrix = _get_R_z(rot_z)

    # Extract the centers and rotation
    centers = boxes[:3, :]  # Shape [3, N]
    rotations = boxes[6, :] # Shape [N]

    # Apply the rotation transformation to the center coordinates
    transformed_centers = _R_matrix @ centers + pos  # Shape [3, N]

    # Apply the Z rotation to the existing rotation (laser to base)
    new_rotations = rotations + rot_z  # Shape [N]

    # Combine the transformed centers and rotations with the original box dimensions
    transformed_boxes = np.vstack([
        transformed_centers,          # Shape [3, N]
        boxes[3:6, :],                # Original dimensions [length, width, height], Shape [3, N]
        new_rotations[np.newaxis, :]  # Ensure shape [1, N] for vstack
    ])

    return transformed_boxes

# Function to check if an array is sorted
def is_sorted(arr):
    return np.all(np.diff(arr) >= 0)

# Function to find closest indices using naive approach
def find_closest_indices_naive(timestamps_gt, timestamps_pose):
    closest_indices = []
    closest_times = []
    for pose_time in timestamps_pose:
        closest_time = np.min(np.abs(timestamps_gt - pose_time))
        closest_times.append(closest_time)
        closest_index = np.argmin(np.abs(timestamps_gt - pose_time))
        closest_indices.append(closest_index)
    print(max(closest_times))
    return closest_indices

# Function to find closest indices using two-pointer technique
def find_closest_indices_two_pointer(timestamps_gt, timestamps_pose):
    closest_indices = []
    i, j = 0, 0
    N, M = len(timestamps_gt), len(timestamps_pose)

    while j < M:
        # Move i to the closest element in timestamps_gt that is not greater than timestamps_pose[j]
        while i < N - 1 and timestamps_gt[i + 1] <= timestamps_pose[j]:
            i += 1
        # Check if the current or the next element is closer
        if i < N - 1 and abs(timestamps_gt[i + 1] - timestamps_pose[j]) < abs(timestamps_gt[i] - timestamps_pose[j]):
            closest_indices.append(i + 1)
        else:
            closest_indices.append(i)
        j += 1

    return closest_indices

# Main function to decide which approach to use
def find_closest_indices(timestamps_gt, timestamps_pose):
    timestamps_gt_array = np.array(timestamps_gt)
    timestamps_pose_array = np.array(timestamps_pose)
    
    if is_sorted(timestamps_gt_array) and is_sorted(timestamps_pose_array):
        return find_closest_indices_two_pointer(timestamps_gt_array, timestamps_pose_array)
    else:
        return find_closest_indices_naive(timestamps_gt_array, timestamps_pose_array)

def process_lidar_data(cb_data, input_dir, overwrite=False):
    seq_num = cb_data.nr_seqs()
    for seq_idx in tqdm(range(seq_num), desc="Processing sequences"):
        input_dir_labels = os.path.join(input_dir, 'labels/labels_3d')
        seq = cb_data.seqs[seq_idx]  # e.g., 'bytes-cafe-2019-02-07_0_all_transforms'
        seq_name = seq.replace('_all_transforms', '')  # Remove '_all_transforms' from the sequence name
        input_dir_timestamps = os.path.join(input_dir, 'timestamps', seq_name)
        file_name = seq_name + '.json'
        file_path = os.path.join(input_dir_labels, file_name)
        file_path_timestamps = os.path.join(input_dir_timestamps, 'frames_pc.json')
        
        pose_folder = 'tf_JRDB'
        pose_suffix = "_tfJRDB_sampled.npy"
        tf_dir = os.path.join(cb_data.source_data_dir, pose_folder)
        pose_stampe_path = os.path.join(
            tf_dir, seq + pose_suffix
        )
        lidar_pose_stamped = np.load(
            pose_stampe_path, allow_pickle=True
        ).item()
        timestamps_pose = lidar_pose_stamped['timestamp']
        with open(file_path_timestamps, 'r') as f_time:
            timestamps_gt = json.load(f_time)
            # Extract timestamps
            timestamps_gt = [entry['timestamp'] for entry in timestamps_gt['data']]
            timestamps_gt = np.array(timestamps_gt)

        closest_indices = find_closest_indices(timestamps_gt, timestamps_pose)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            out_det_all = {}
            frames_list = list(data['labels'].values())
            frames_closest_list = [frames_list[i] for i in closest_indices]

            for frame_counter, frame_data in enumerate(frames_closest_list):
                pos = lidar_pose_stamped['position'][frame_counter].reshape(3, 1)
                quat = lidar_pose_stamped['orientation'][frame_counter]

                # Assuming you have a list of label_ids corresponding to each det_data
                labels = [int(det_data['label_id'].split(':')[1]) for det_data in frame_data]
                boxes_ = []
                for det_data in frame_data:
                    box = det_data['box']
                    boxes_.append([box['cx'], box['cy'], box['cz'], box['l'], box['w'], box['h'], box['rot_z'],])
                
                boxes_ = np.array(boxes_)
                boxes_= transform_boxes_3D(boxes_.T, _pos_camera_to_base_chassis_link, 0, is_quaternion=False).T
                boxes_ = transform_boxes_3D(boxes_.T, pos, quat).T
                # Convert labels to a numpy array and reshape to match dimensions
                labels = np.array(labels).reshape(-1, 1)
                # Concatenate boxes_ and labels along the second axis (columns)
                boxes_with_labels = np.hstack((boxes_, labels))
                out_det_all.update({frame_counter: boxes_with_labels})
                frame_counter += 1
            
            output_file_name = file_name.replace('.json', '_all_transforms_gt.npy')
            dnpy_all_path = os.path.join(cb_data.trks_dir, output_file_name)
            
            if not os.path.exists(dnpy_all_path) or overwrite:
                np.save(dnpy_all_path, out_det_all)
            else:
                print(f"File {dnpy_all_path} already exists. Skipping sequence {seq}.")
        else:
            print(f"File {file_path} does not exist. Skipping sequence {seq}.")

if __name__ == '__main__':
    cb_data = CrowdBotDatabase(folder, config=config_path)
    process_lidar_data(cb_data, JRDB_dataset_dir, overwrite=True)
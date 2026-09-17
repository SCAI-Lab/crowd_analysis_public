import os
import json
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from qolo.core.crowdbot_data import CrowdBotDatabase


# JRDB camera -> base_chassis_link offset
_pos_camera_to_base_chassis_link = np.array(
    [-0.019685, 0.0, 0.742092], dtype=np.float32
).reshape(3, 1)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
folder = 'JRDB_test'   # e.g. 'JRDB_whole' or 'JRDB_test'
config_path = '/scai_data/data01/daav/JRDB/train_dataset_Crowdbot_format/config/data_path.yaml'
JRDB_dataset_dir = '/scai_data/data01/daav/JRDB/test_dataset'   # train_dataset or test_dataset


def _get_R_z(rot_z):
    cs, ss = np.cos(rot_z), np.sin(rot_z)
    return np.array([[cs, -ss, 0],
                     [ss,  cs, 0],
                     [0,   0,  1]], dtype=np.float32)


def get_yaw_from_quat(quat):
    scipy_rot = R.from_quat(quat)
    rot_zyx = scipy_rot.as_euler('zyx')
    return rot_zyx[0]


def transform_boxes_3D(boxes, pos, rot_z, is_quaternion=True):
    """
    boxes: shape (7, N)
    pos: shape (3,1)
    rot_z: quaternion or yaw
    returns: shape (7, N)
    """
    assert boxes.shape[0] == 7, "Input boxes array must be of shape 7xN"

    if is_quaternion:
        rot_z = get_yaw_from_quat(rot_z)

    _R_matrix = _get_R_z(rot_z)

    centers = boxes[:3, :]
    rotations = boxes[6, :]

    transformed_centers = _R_matrix @ centers + pos
    new_rotations = rotations + rot_z

    transformed_boxes = np.vstack([
        transformed_centers,
        boxes[3:6, :],
        new_rotations[np.newaxis, :]
    ])
    return transformed_boxes


def is_sorted(arr):
    return np.all(np.diff(arr) >= 0)


def find_closest_indices_naive(timestamps_gt, timestamps_pose):
    closest_indices = []
    closest_times = []
    for pose_time in timestamps_pose:
        closest_time = np.min(np.abs(timestamps_gt - pose_time))
        closest_times.append(closest_time)
        closest_index = np.argmin(np.abs(timestamps_gt - pose_time))
        closest_indices.append(closest_index)
    print("max timestamp diff:", max(closest_times))
    return closest_indices


def find_closest_indices_two_pointer(timestamps_gt, timestamps_pose):
    closest_indices = []
    i, j = 0, 0
    N, M = len(timestamps_gt), len(timestamps_pose)

    while j < M:
        while i < N - 1 and timestamps_gt[i + 1] <= timestamps_pose[j]:
            i += 1
        if i < N - 1 and abs(timestamps_gt[i + 1] - timestamps_pose[j]) < abs(timestamps_gt[i] - timestamps_pose[j]):
            closest_indices.append(i + 1)
        else:
            closest_indices.append(i)
        j += 1

    return closest_indices


def find_closest_indices(timestamps_gt, timestamps_pose):
    timestamps_gt_array = np.array(timestamps_gt)
    timestamps_pose_array = np.array(timestamps_pose)

    if is_sorted(timestamps_gt_array) and is_sorted(timestamps_pose_array):
        return find_closest_indices_two_pointer(timestamps_gt_array, timestamps_pose_array)
    else:
        return find_closest_indices_naive(timestamps_gt_array, timestamps_pose_array)


def get_jrdb_source_seq_name(processed_seq_name):
    """
    Processed train seqs are like:
      bytes-cafe-..._all_transforms
    Processed test seqs are like:
      gates-foyer-...
    Source labels/timestamps use the version without _all_transforms.
    """
    return processed_seq_name.replace('_all_transforms', '')


def sort_jrdb_frame_keys(frame_keys):
    """
    Sort frame keys robustly, e.g.:
      '000000.pcd', '000001.pcd', ...
    """
    def key_fn(x):
        stem = os.path.splitext(os.path.basename(x))[0]
        try:
            return int(stem)
        except ValueError:
            return stem
    return sorted(frame_keys, key=key_fn)


def process_lidar_data(cb_data, input_dir, overwrite=False, save_to_dets_dir=False):
    seq_num = cb_data.nr_seqs()

    def resolve_jrdb_labels_dir(input_dir):
        candidates = [
            os.path.join(input_dir, 'labels', 'labels_3d'),  # JRDB train
            os.path.join(input_dir, 'labels_3d'),            # JRDB test
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        raise FileNotFoundError(
            f"Could not find JRDB labels_3d directory in any of: {candidates}"
        )

    input_dir_labels = resolve_jrdb_labels_dir(input_dir)

    for seq_idx in tqdm(range(seq_num), desc="Processing sequences"):
        seq = cb_data.seqs[seq_idx]
        seq_name = get_jrdb_source_seq_name(seq)

        file_name = seq_name + '.json'
        file_path = os.path.join(input_dir_labels, file_name)

        file_path_timestamps = os.path.join(
            input_dir, 'timestamps', seq_name, 'frames_pc.json'
        )

        pose_folder = 'tf_JRDB'
        pose_suffix = "_tfJRDB_sampled.npy"
        tf_dir = os.path.join(cb_data.source_data_dir, pose_folder)
        pose_stampe_path = os.path.join(tf_dir, seq + pose_suffix)

        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping sequence {seq}.")
            continue

        if not os.path.exists(file_path_timestamps):
            print(f"File {file_path_timestamps} does not exist. Skipping sequence {seq}.")
            continue

        if not os.path.exists(pose_stampe_path):
            print(f"Pose file {pose_stampe_path} does not exist. Skipping sequence {seq}.")
            continue

        lidar_pose_stamped = np.load(pose_stampe_path, allow_pickle=True).item()
        timestamps_pose = lidar_pose_stamped['timestamp']

        with open(file_path_timestamps, 'r') as f_time:
            timestamps_gt_json = json.load(f_time)
            timestamps_gt = np.array(
                [entry['timestamp'] for entry in timestamps_gt_json['data']],
                dtype=np.float64
            )

        closest_indices = find_closest_indices(timestamps_gt, timestamps_pose)

        with open(file_path, 'r') as f:
            data = json.load(f)

        labels_dict = data['labels']
        frame_keys_sorted = sort_jrdb_frame_keys(list(labels_dict.keys()))

        if len(frame_keys_sorted) != len(timestamps_gt):
            print(
                f"Warning for {seq}: number of label frames ({len(frame_keys_sorted)}) "
                f"!= number of timestamps ({len(timestamps_gt)})"
            )

        out_det_all = {}

        for frame_counter, gt_idx in enumerate(closest_indices):
            if gt_idx >= len(frame_keys_sorted):
                out_det_all[frame_counter] = np.empty((0, 8), dtype=np.float32)
                continue

            frame_key = frame_keys_sorted[gt_idx]
            frame_data = labels_dict.get(frame_key, [])

            pos = lidar_pose_stamped['position'][frame_counter].reshape(3, 1)
            quat = lidar_pose_stamped['orientation'][frame_counter]

            if len(frame_data) == 0:
                out_det_all[frame_counter] = np.empty((0, 8), dtype=np.float32)
                continue

            labels = []
            boxes_ = []

            for det_data in frame_data:
                box = det_data['box']
                labels.append(int(det_data['label_id'].split(':')[1]))
                boxes_.append([
                    box['cx'], box['cy'], box['cz'],
                    box['l'], box['w'], box['h'],
                    box['rot_z'],
                ])

            boxes_ = np.asarray(boxes_, dtype=np.float32)

            # camera frame -> base_chassis_link frame
            boxes_ = transform_boxes_3D(
                boxes_.T,
                _pos_camera_to_base_chassis_link,
                0.0,
                is_quaternion=False
            ).T

            # base_chassis_link local -> global/odom frame
            boxes_ = transform_boxes_3D(
                boxes_.T,
                pos,
                quat,
                is_quaternion=True
            ).T

            labels = np.asarray(labels, dtype=np.int32).reshape(-1, 1)
            boxes_with_labels = np.hstack((boxes_, labels))

            out_det_all[frame_counter] = boxes_with_labels

        # IMPORTANT:
        # save name based on processed seq name, so both train and test work:
        #   train: bytes-cafe-..._all_transforms_gt.npy
        #   test : gates-foyer-..._gt.npy
        gt_file_name = seq + '_gt.npy'

        trk_gt_path = os.path.join(cb_data.trks_dir, gt_file_name)
        if (not os.path.exists(trk_gt_path)) or overwrite:
            np.save(trk_gt_path, out_det_all)
        else:
            print(f"File {trk_gt_path} already exists. Skipping sequence {seq}.")

        # Optional: also save to detections dir so cb_data[...].dets_gt is available
        if save_to_dets_dir:
            det_gt_path = os.path.join(cb_data.dets_dir, gt_file_name)
            if (not os.path.exists(det_gt_path)) or overwrite:
                np.save(det_gt_path, out_det_all)


if __name__ == '__main__':
    cb_data = CrowdBotDatabase(folder, config=config_path)
    process_lidar_data(cb_data, JRDB_dataset_dir, overwrite=True, save_to_dets_dir=False)
#!/usr/bin/env python3
# -*-coding:utf-8 -*-

import os
import argparse
import copy

import numpy as np
from tqdm import tqdm

from crowdbot_data.crowdbot_data import CrowdBotDatabase
from AB3DMOT_libs.model import AB3DMOT

from scipy.spatial.transform import Rotation as R

def get_yaw_from_quat(quat):
    scipy_rot = R.from_quat(quat)
    rot_zyx = scipy_rot.as_euler('zyx')
    return rot_zyx[0]

def get_quat_from_yaw(yaw):
    rot_euler = [yaw, 0, 0]
    scipy_rot = R.from_euler('zyx', rot_euler)
    return scipy_rot.as_quat()

def get_mat_from_yaw(yaw):
    rot_euler = [yaw, 0, 0]
    scipy_rot = R.from_euler('zyx', rot_euler)
    return scipy_rot.as_matrix()

def get_pc_tranform(pc, pos, quat):
    # pshape(pc)
    scipy_rot = R.from_quat(quat)
    rot_mat = scipy_rot.as_matrix() # 3x3
    # pshape(rot_mat)
    rot_pc = np.matmul(rot_mat, pc.T)  #  (3x3) (3xn)
    # pshape(rot_pc)
    return rot_pc.T + pos

def reorder(boxes):
    # from x, y, z,  l,  w, h, theta (lidar frame: x-forward, y-left, z-up)
    # to   h, w, l, -y, -z, x, theta (cam frame: z-forward, x-right, y-down)
    inds = [5, 4, 3, 1, 2, 0, 6]
    boxes = boxes[:, inds]
    boxes[:, 3] *= -1
    boxes[:, 4] *= -1
    return boxes


def reorder_back(boxes):
    # from h, w, l, -y, -z, x, theta, ID
    # to   x, y, z,  l,  w, h, theta, ID
    inds = [5, 3, 4, 2, 1, 0, 6, 7]
    boxes = boxes[:, inds]
    boxes[:, 1] *= -1
    boxes[:, 2] *= -1
    return boxes

# Step 1: Compute the distances for entries in trks_far and trks_2D_close relative to the wheelchair
def compute_distances(trks, pos):
    # Calculate the distance using x and y only
    distances = np.sqrt((trks[:, 0] - pos[0])**2 + (trks[:, 1] - pos[1])**2)
    return distances

# Step 2: Filter the entries based on the distance criteria
def filter_by_distance(trks, lower_bound, upper_bound, device_pos):
    distances = compute_distances(trks, device_pos)
    mask = (distances >= lower_bound) & (distances <= upper_bound)
    filtered_trks = trks[mask]
    return filtered_trks, distances

# Step 3: Extract the IDs from the filtered entries
def extract_ids(filtered_trks):
    ids = filtered_trks[:, 7].astype(int)  # Assuming ID is the 8th column and is integer
    return ids

# Step 4: Find the corresponding tracker objects based on these IDs
def find_trackers_by_ids(tracker_list, ids):
    return [tracker for tracker in tracker_list if tracker.id in ids]

# Function to update the tracker list without duplicates
def update_trackers(existing_trackers, new_trackers,):
    # Convert the existing list to a dictionary for efficient look-up
    tracker_dict = {tracker.id: tracker for tracker in existing_trackers}
        
    
    # Update the dictionary with new trackers (replace if already present and hits condition met)
    for tracker in new_trackers:
        if tracker.id in tracker_dict:
            if tracker.hits > tracker_dict[tracker.id].hits:
                tracker_dict[tracker.id] = copy.deepcopy(tracker)
        else:
            tracker_dict[tracker.id] = copy.deepcopy(tracker)
    
    # Convert the dictionary back to a list
    updated_trackers = list(tracker_dict.values())
    return updated_trackers

def safe_concatenate(arr1, arr2, trackers_close, trackers_far):
    if arr1.size == 0:
        return arr2
    elif arr2.size == 0:
        return arr1

    # Convert arrays to dictionaries using the last element (id) as the key
    dict1 = {int(row[-1]): row for row in arr1}
    dict2 = {int(row[-1]): row for row in arr2}

    # Convert trackers to dictionaries using .id as the key
    dict_close = {tracker.id: tracker for tracker in trackers_close}
    dict_far = {tracker.id: tracker for tracker in trackers_far}

    # Combine dictionaries and resolve conflicts based on .time_since_update
    combined_tracks = {}
    for track_id in set(dict1.keys()).union(set(dict2.keys())):
        if track_id in dict1 and track_id in dict2:
            # If track_id is in both trackers, choose based on time_since_update
            if dict_close[track_id].time_since_update < dict_far[track_id].time_since_update:
                combined_tracks[track_id] = dict1[track_id]
            else:
                combined_tracks[track_id] = dict2[track_id]
        elif track_id in dict1:
            combined_tracks[track_id] = dict1[track_id]
        elif track_id in dict2:
            combined_tracks[track_id] = dict2[track_id]

    # Convert back to array
    merged_array = np.array(list(combined_tracks.values()))
    return merged_array

class Settings:
    """
    Settings class to store script parameters.

    Attributes:
        dataset (str): Dataset that is being processed (default: 'JRDB')
        folder (str): Different subfolder in rosbag/ dir (default: '0424_mds_test').
        overwrite (bool): Whether to overwrite existing output (default: True).
        save_raw (bool): Whether to save raw data of detection results (default: False).
        min_conf (float): Minimum confidence threshold for 3D detections (default: 0.5).
        track_2D (bool): Flag indicating whether to perform 2D tracking (default: False).
        min_conf_2D (float): Minimum confidence threshold for 2D detections (default: 0.5).
    """

    def __init__(self, dataset, config_path, folder, overwrite=False, save_raw=False, min_conf=0.5,
                 track_2D=False, min_conf_2D=0.5,):
        self.dataset = dataset
        self.config_path = config_path
        self.folder = folder
        self.overwrite = overwrite
        self.save_raw = save_raw
        self.min_conf = min_conf
        self.track_2D = track_2D
        self.min_conf_2D = min_conf_2D

def parse_args():
    parser = argparse.ArgumentParser(description="Track lidar detections with AB3DMOT.")
    parser.add_argument("--dataset", required=True, choices=["Crowdbot", "JRDB", "JRDB_TEST", "Daav", "SiT", "SCAND"])
    parser.add_argument("--config", dest="config_path", required=True, help="Dataset path YAML.")
    parser.add_argument("--folder", action="append", required=True, help="Logical dataset folder; repeat as needed.")
    parser.add_argument("--track-2d", action="store_true")
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--min-conf-2d", type=float, default=0.5)
    parser.add_argument("--distance-split", type=float, default=5.0)
    parser.add_argument("--exchange-width", type=float, default=0.5)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dataset == "JRDB_TEST" and args.track_2d:
        parser.error("JRDB_TEST has no 2D lidar/detections; omit --track-2d.")
    return args


if __name__ == '__main__':
    cli_args = parse_args()
    tracking_distance_split = cli_args.distance_split
    tracking_info_exchange_width = cli_args.exchange_width

    for folder in cli_args.folder:
        args = Settings(
            dataset=cli_args.dataset,
            config_path=cli_args.config_path,
            folder=folder,
            track_2D=cli_args.track_2d,
            min_conf=cli_args.min_conf,
            min_conf_2D=cli_args.min_conf_2d,
            save_raw=cli_args.save_raw,
            overwrite=cli_args.overwrite,
        )

        if args.dataset == 'JRDB_TEST' and args.track_2D:
            raise ValueError("JRDB_TEST has no 2D lidar/detections. Use track_2D=False.")

        # Pose source
        if args.dataset in ['JRDB', 'JRDB_TEST']:
            pose_folder = 'tf_JRDB'
            pose_suffix = "_tfJRDB_sampled.npy"
        elif args.dataset == 'Crowdbot':
            pose_folder = 'tf_qolo'
            pose_suffix = "_tfqolo_sampled.npy"
        elif args.dataset == 'SiT':
            pose_folder = 'tf_SiT'
            pose_suffix = "_tf_SiT_sampled.npy"
        elif args.dataset == 'SCAND':
            pose_folder = 'tf_SCAND'
            pose_suffix = "_tfSCAND_sampled.npy"
        else:
            pose_folder = 'tf_daav'
            pose_suffix = "_tfdaav_sampled.npy"

        fake_2D_z = 0.9
        fake_2D_height = 1.8

        cb_data = CrowdBotDatabase(args.folder, config=args.config_path)

        counter = 0
        seq_num = cb_data.nr_seqs()

        ID_init_close = 0
        ID_init_far = 10000

        max_age = 7
        min_hits = 5
        thresh_2D_IoU = 0.3
        thresh_2D_dist = 0.5

        thresh_3D_IoU = 0.33
        thresh_3D_dist = 0.5

        for seq_idx in range(seq_num):
            seq = cb_data.seqs[seq_idx]

            counter += 1
            print("({}/{}): {} frames".format(counter, seq_num, cb_data.nr_frames(seq_idx)))

            if args.save_raw:
                trk_seq_dir = os.path.join(cb_data.trks_dir, seq)
            
            tnpy_all_path = os.path.join(cb_data.trks_dir, seq + '.npy')
            tnpy_all_path_2D = os.path.join(cb_data.trks_2D_dir, seq + '.npy')
            tnpy_all_merged_path = os.path.join(cb_data.trks_dir, seq + '_merged.npy')

            if not os.path.exists(tnpy_all_path) or args.overwrite:
                out_trk_all = dict()
                out_trk_merged_all = dict()

                log = os.path.join(cb_data.alg_res_dir, "logs", seq + '.txt')
                tracker = AB3DMOT(max_age=max_age, min_hits=min_hits, thres=[thresh_3D_IoU, thresh_3D_dist], metric=['iou_3d', 'dist_3d'], log=log)

                if args.track_2D:
                    out_trk_all_2D = dict()
                    tracker_2D = AB3DMOT(max_age=max_age, min_hits=min_hits, thres=[thresh_2D_IoU, thresh_2D_dist], metric=['iou_2d', 'dist_2d'], log=log,)
                    tracker_merged_close = AB3DMOT(max_age=max_age, min_hits=min_hits, thres=[thresh_2D_IoU, thresh_2D_dist], metric=['iou_2d', 'dist_2d'], log=log, ID_init=ID_init_close,)
                    tracker_merged_far = AB3DMOT(max_age=max_age, min_hits=min_hits, thres=[thresh_3D_IoU, thresh_3D_dist], metric=['iou_3d', 'dist_3d'], log=log, ID_init=ID_init_far)

                tf_dir = os.path.join(cb_data.source_data_dir, pose_folder)
                pose_stampe_path = os.path.join(tf_dir, seq + pose_suffix)
                lidar_pose_stamped = np.load(pose_stampe_path, allow_pickle=True).item()
                pbar = tqdm(total=cb_data.nr_frames(seq_idx))
                for fr_idx in range(cb_data.nr_frames(seq_idx)):
                    # if fr_idx < 3510:
                    #     continue
                    _, _, _, dets_gt, _, dets, dets_conf, _, dets_2D, dets_2D_conf, _, dets_far, dets_far_conf, dets_2D_close, dets_2D_conf_close, _ = cb_data[seq_idx, fr_idx]
                    dets = dets[dets_conf > args.min_conf]
                    pos = lidar_pose_stamped["position"][fr_idx, :]
                    if dets.size != 0:
                        dets = reorder(dets)
                    
                    trk_input = {"dets": dets, "info": np.zeros_like(dets)}
                    trks = tracker.track(trk_input, fr_idx, cb_data.seqs[seq_idx])[0][0]
                    trks = reorder_back(trks)
                    out_trk_all.update({fr_idx: trks})

                    if args.track_2D:
                        dets_2D = dets_2D[dets_2D_conf > args.min_conf_2D]
                        if dets_2D.size != 0:
                            x, y, length, width, theta = dets_2D[:, 0], dets_2D[:, 1], dets_2D[:, 2], dets_2D[:, 3], dets_2D[:, 4],
                            fake_z, fake_height = np.full_like(x, fake_2D_z), np.full_like(length, fake_2D_height)
                            dets_2D = np.stack((x, y, fake_z, length, width, fake_height, theta), axis=-1)
                            dets_2D = reorder(dets_2D)
                        trk_input_2D = {"dets": dets_2D, "info": np.zeros_like(dets_2D)}
                        trks_2D = tracker_2D.track(trk_input_2D, fr_idx, cb_data.seqs[seq_idx])[0][0]
                        trks_2D = reorder_back(trks_2D)
                        out_trk_all_2D.update({fr_idx: trks_2D})

                        dets_far = dets_far[dets_far_conf > args.min_conf]
                        if dets_far.size != 0:
                            dets_far = reorder(dets_far)
                        trk_input_far = {"dets": dets_far, "info": np.zeros_like(dets_far)}
                        trks_far = tracker_merged_far.track(trk_input_far, fr_idx, cb_data.seqs[seq_idx])[0][0]
                        trks_far = reorder_back(trks_far)

                        dets_2D_close = dets_2D_close[dets_2D_conf_close > args.min_conf_2D]
                        if dets_2D_close.size != 0:
                            x, y, length, width, theta = dets_2D_close[:, 0], dets_2D_close[:, 1], dets_2D_close[:, 2], dets_2D_close[:, 3], dets_2D_close[:, 4],
                            fake_z, fake_height = np.full_like(x, fake_2D_z), np.full_like(length, fake_2D_height)
                            dets_2D_close = np.stack((x, y, fake_z, length, width, fake_height, theta), axis=-1)
                            dets_2D_close = reorder(dets_2D_close)
                        trk_input_2D_close = {"dets": dets_2D_close, "info": np.zeros_like(dets_2D_close)}
                        trks_2D_close = tracker_merged_close.track(trk_input_2D_close, fr_idx, cb_data.seqs[seq_idx])[0][0]
                        trks_2D_close = reorder_back(trks_2D_close)

                        trks_merged = safe_concatenate(trks_2D_close, trks_far, trackers_close=tracker_merged_close.trackers, trackers_far=tracker_merged_far.trackers)

                        # Finding IDs in the distance range for trks_far
                        filtered_trks_far, distances_far = filter_by_distance(trks_far, lower_bound=tracking_distance_split-tracking_info_exchange_width, upper_bound=tracking_distance_split+tracking_info_exchange_width, device_pos=pos)
                        ids_far = extract_ids(filtered_trks_far)
                        ids_far = ids_far[ids_far >= ID_init_far]

                        # Finding IDs in the distance range for trks_2D_close
                        filtered_trks_2D_close, distances_2D_close = filter_by_distance(trks_2D_close, lower_bound=tracking_distance_split-tracking_info_exchange_width, upper_bound=tracking_distance_split+tracking_info_exchange_width, device_pos=pos)
                        ids_2D_close = extract_ids(filtered_trks_2D_close)
                        ids_2D_close = ids_2D_close[(ids_2D_close < ID_init_far)]

                        # Finding the tracker objects corresponding to these IDs
                        trackers_far = find_trackers_by_ids(tracker_merged_far.trackers, ids_far)
                        trackers_2D_close = find_trackers_by_ids(tracker_merged_close.trackers, ids_2D_close)

                        tracker_merged_far.trackers = update_trackers(tracker_merged_far.trackers, trackers_2D_close)
                        tracker_merged_close.trackers = update_trackers(tracker_merged_close.trackers, trackers_far,)

                        out_trk_merged_all.update({fr_idx: trks_merged})

                    pbar.update(1)

                    if args.save_raw:
                        f_path = os.path.join(
                            trk_seq_dir,
                            cb_data.frames[seq_idx][fr_idx].replace("npy", "txt"),
                        )
                        os.makedirs(trk_seq_dir, exist_ok=True)
                        np.savetxt(f_path, trks, delimiter=",")
                    
                pbar.close()
                np.save(tnpy_all_path, out_trk_all,)
                if args.track_2D:
                    np.save(tnpy_all_path_2D, out_trk_all_2D,)
                    np.save(tnpy_all_merged_path, out_trk_merged_all)
            else:
                print("{} tracking results already generated!!!".format(seq))
                print("Will not overwrite. If you want to overwrite, use flag --overwrite")
                continue

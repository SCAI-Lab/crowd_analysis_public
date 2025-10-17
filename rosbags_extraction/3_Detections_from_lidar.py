#!/usr/bin/env python3
import os
import sys
from lidar_det.DR_SPAAM_detector import DR_SPAAM_detector
import numpy as np
from tqdm import tqdm
from crowdbot_data.crowdbot_data import CrowdBotDatabase
from lidar_det.PersonMinkUnet_detector import DetectorWithClock
from scipy.spatial.transform import Rotation as R
from lidar_det.utils.utils_box3d import nms_3d_dist_gpu

from scipy.spatial.transform import Rotation as R
import open3d as o3d

import torch

print("Testing cuda installation...\n")

print(" -> Cuda available:         {}".format(torch.cuda.is_available()))
print(" -> Num. of cuda devices:   {}".format(torch.cuda.device_count()))
                                          
cuda_device_id = torch.cuda.current_device()
print(" -> Current cuda device:    {}".format(cuda_device_id))
print(" -> Name of cuda device:    {}".format(torch.cuda.get_device_name(cuda_device_id)))

print(" -> All cuda devices:\n")
for i in range(torch.cuda.device_count()) :
    print("    {}: {}".format(i, torch.cuda.get_device_name(i)))

print("\nAll done!")

def get_yaw_from_quat(quat):
    scipy_rot = R.from_quat(quat)
    rot_zyx = scipy_rot.as_euler('zyx')
    return rot_zyx[0]

def get_2D_transform(xy, boxes_2D, pos, quat):
    yaw_angle = get_yaw_from_quat(quat)
    rot_list = np.array(
        [
            [np.cos(yaw_angle), -np.sin(yaw_angle),],
            [np.sin(yaw_angle), np.cos(yaw_angle),],
        ]
    )  # (2, 2,)
    xy_rotated = rot_list @ xy.T  # 2x2 @ 2xN
    xy_rotated = xy_rotated.T
    # Apply the Z rotation to the existing rotation
    boxes_2D[:,-1] = boxes_2D[:,-1] + yaw_angle
    return xy_rotated + pos[:2], boxes_2D

def dets_2D_local_to_global(x_y, boxes_2D, pos, quat, dataset='JRDB', topic=None):
    if dataset == 'JRDB':
        rot_z_laser_to_base = np.pi/120
        cs, ss = np.cos(rot_z_laser_to_base), np.sin(rot_z_laser_to_base)
        R_laser_to_base = np.array([[cs, -ss], [ss, cs],], dtype=np.float32)
        x_y = R_laser_to_base @ x_y.T # 2x2 @ 2xN
        x_y = x_y.T
        boxes_2D[:,-1] = boxes_2D[:,-1] + rot_z_laser_to_base
    if dataset == 'Crowdbot' and (topic=='/front_lidar/scan' or topic=='/front_lidar/scan_modified'):
        trans_laserfront_to_base = [0.035, 0.0, 0.28]
        quat_laserfront_to_base = [0.0, 0.026176900686660683, 0.0, -0.9996573262225615]
        x_y, boxes_2D = get_2D_transform(x_y, boxes_2D, trans_laserfront_to_base, quat_laserfront_to_base)
        
    if dataset == 'Crowdbot' and (topic=='/rear_lidar/scan' or topic=='/rear_lidar/scan_modified'):
        trans_laserrear_to_base = [-0.516, 0.0, 0.164]
        quat_laserrear_to_base = [0.0, 0., 1.0, 0.]
        x_y, boxes_2D = get_2D_transform(x_y, boxes_2D, trans_laserrear_to_base, quat_laserrear_to_base)
        

    dets_2D_global, boxes_2D_global = get_2D_transform(x_y, boxes_2D, pos, quat)
    return dets_2D_global, boxes_2D_global

def detect_3D_with_plane_alignment(pc, pos, detector_3D, prev_plane_model=None):
    """
    Detect objects in a rotated LiDAR point cloud and transform detections back to the original frame.
    
    :param pc: Nx3 numpy array representing the point cloud.
    :param pos: Position of the robot (x, y, z) around which to fit the plane.
    :param detector_3D: A function that takes an Nx3 point cloud and returns boxes and scores.
                        - boxes: Dx7 array (x, y, z, length, width, height, rot_z).
                        - scores: D array with detection scores.
    :param prev_plane_model: Previous plane model (a, b, c, d) for filtering points.
    :return: boxes and scores transformed back to the original frame, and the current plane model.
    """
    distance_threshold = 0.04
    close_points_distance = 0.1
    z_min, z_max = -0.5, 0.5

    # Center the point cloud around the robot's position
    pc_centered = pc - pos

    # Filter points based on previous plane model or Z-range
    if prev_plane_model is not None:
        # Use previous plane model to filter points near the plane
        a, b, c, d = prev_plane_model
        distances = np.abs(a * pc_centered[:, 0] + b * pc_centered[:, 1] + c * pc_centered[:, 2] + d) / np.linalg.norm([a, b, c])
        pc_filtered = pc_centered[distances < close_points_distance]
    else:
        # Use Z-range filtering if no previous plane model is provided
        pc_filtered = pc_centered[(pc_centered[:, 2] >= z_min) & (pc_centered[:, 2] <= z_max)]

    # Convert the filtered points to an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_filtered)

    # Find the ground plane using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=11, num_iterations=1000)
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # Normalize the plane normal

    # Check if the plane's normal vector is valid
    if c < max(a, b):  # If Z-component is not dominant
        raise ValueError(f"Invalid plane detected: normal vector components are a={a}, b={b}, c={c}. Z-dominance violated.")

    # Target vector (z-axis) to align with
    target = np.array([0, 0, 1])

    # Compute rotation matrix to align normal with the z-axis
    rotation_axis = np.cross(normal, target)
    rotation_angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))  # Handle numerical issues
    if np.linalg.norm(rotation_axis) > 1e-6:  # Avoid division by zero for aligned vectors
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize rotation axis
        rotation_matrix = R.from_rotvec(rotation_axis * rotation_angle).as_matrix()
    else:
        rotation_matrix = np.eye(3)  # No rotation needed if already aligned

    # Rotate the point cloud
    rotated_pc = (rotation_matrix @ pc_centered.T).T

    # Run the detector on the rotated point cloud
    boxes, scores = detector_3D(rotated_pc.T)  # Pass as 3xN to detector

    # Rotate the detection boxes back to the original frame
    rotated_boxes = boxes.copy()
    centers = rotated_boxes[:, :3]  # x, y, z center coordinates
    rotated_centers = (rotation_matrix.T @ centers.T).T  # Inverse rotation for centers
    rotated_boxes[:, :3] = rotated_centers + pos  # Add the robot's position back

    # Adjust rotation angle in the boxes
    rotated_boxes[:, 6] = rotated_boxes[:, 6] - np.arctan2(rotation_axis[1], rotation_axis[0])

    # Return the detections and the current plane model
    return rotated_boxes, scores, plane_model

def detect_3D_without_plane_fitting(pc, pos, detector_3D):
    """
    Detect objects in a LiDAR point cloud centered around the origin,
    and transform detections back to the original frame.
    
    :param pc: Nx3 numpy array representing the point cloud.
    :param pos: Position of the robot (x, y, z) used to center the point cloud.
    :param detector_3D: A function that takes a 3xN point cloud and returns boxes and scores.
                        - boxes: Dx7 array (x, y, z, length, width, height, rot_z).
                        - scores: D array with detection scores.
    :return: boxes and scores transformed back to the original frame.
    """
    # Center the point cloud around the origin
    pc_centered = pc - pos

    # Transpose before passing to the detector
    boxes, scores = detector_3D(pc_centered.T)  # Pass as 3xN to the detector

    # Add the robot's position back to detections
    boxes[:, :3] += pos  # Restore the global position for the boxes

    return boxes, scores

def non_maximum_suppression(detections, nms_2D_dist_norm_thresh=0.4, indices_3D=None, l_ave=0.9, w_ave=0.5, h_ave=1.7):
    """
    Perform Non-Maximum Suppression (NMS) on Lidar detections, supporting both 2D (Nx6) and 3D (Nx8) box formats.
    Uses IoU-based NMS if detections are 3D, with optional confidence boosting for 3D detections.

    Parameters:
    - detections: np.ndarray, either Nx6 or Nx8:
        * Nx6: [x, y, l, w, rotation_z, score] (2D format with BEV orientation)
        * Nx8: [x, y, z, l, w, h, rotation_z, score] (3D format with BEV orientation)
    - nms_2D_dist_norm_thresh: float, 2D normalized distance threshold for NMS in BEV (default: 0.4).
    - indices_3D: int, the starting index of 3D detections in the array, optional.
    - l_ave: float, average box length for 3D detections (default: 0.9).
    - w_ave: float, average box width for 3D detections (default: 0.5).
    - h_ave: float, average box height for 3D detections (default: 1.7).

    Returns:
    - np.ndarray of suppressed detections.
    """
    if len(detections) == 0:
        return detections

    # Check for Nx6 or Nx8 format and handle accordingly
    is_nx6 = detections.shape[1] == 6
    if is_nx6:
        # Convert Nx6 to Nx8 by adding fake z and h dimensions
        detections_3d = np.zeros((detections.shape[0], 8))
        detections_3d[:, 0] = detections[:, 0]  # x
        detections_3d[:, 1] = detections[:, 1]  # y
        detections_3d[:, 2] = h_ave / 2  # fake z, centered at half of average height
        detections_3d[:, 3] = detections[:, 2]  # l
        detections_3d[:, 4] = detections[:, 3]  # w
        detections_3d[:, 5] = h_ave  # fake h
        detections_3d[:, 6] = detections[:, 4]  # rotation_z
        detections_3d[:, 7] = detections[:, 5]  # score
    elif detections.shape[1] == 8:
        # Already in Nx8 format, so use directly
        detections_3d = detections
    else:
        raise ValueError("Detections must be either Nx6 or Nx8 array.")

    scores = detections_3d[:, -1].copy()

    # Boost the confidence scores of 3D detections if indices_3D are provided to treat 3D detections with priority
    if indices_3D is not None:
        scores[indices_3D:] = 1

    # Sort detections by boosted scores in descending order
    # order = scores.argsort()[::-1]
    # detections_3d = detections_3d[order]

    # Prepare boxes and scores tensors for nms_3d_dist_gpu
    boxes = torch.tensor(detections_3d[:, :7], dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    # Move boxes and scores to GPU if available
    if torch.cuda.is_available():
        boxes = boxes.cuda()
        scores = scores.cuda()

    # Apply IoU-based NMS in BEV using nms_3d_dist_gpu
    inds = nms_3d_dist_gpu(boxes, scores, l_ave=l_ave, w_ave=w_ave, nms_thresh=nms_2D_dist_norm_thresh)

    # Convert indices back to CPU and then to NumPy array
    inds = inds.cpu().numpy()

    # Select the detections corresponding to the kept indices
    suppressed_detections = detections_3d[inds]

    # Remove fake z and h if the original format was Nx6
    if is_nx6:
        suppressed_detections = suppressed_detections[:, [0, 1, 3, 4, 6, 7]]  # Keep only [x, y, l, w, rotation_z, score]

    return suppressed_detections


def apply_nms_to_all_detections(out_det_2D_all, nms_2D_dist_norm_thresh=0.4, indices_3D=None,):
    """
    Apply NMS to all frames in the out_det_2D_all dictionary.

    Parameters:
    - out_det_2D_all: dict, keys are frame indices and values are detection arrays
                      shape (N, 5) for each frame, where N is the number of detections.
    - nms_2D_dist_norm_thresh: float, 2D normalized distance threshold for NMS in BEV (default: 0.4).
    - indices_3D: dict, keys are frame indices and values are starting indices of 3D detections, optional.

    Returns:
    - dict, keys are frame indices and values are detection arrays after NMS.
    """
    nms_results = {}
    for fr_idx, detections in out_det_2D_all.items():
        idx_3D = indices_3D[fr_idx] if indices_3D is not None else None
        nms_detections = non_maximum_suppression(detections, nms_2D_dist_norm_thresh, idx_3D,)
        nms_results[fr_idx] = nms_detections
    return nms_results


def filter_detections_within_distance(detections, pos, max_distance=7.0):
    """
    Filter detections to keep only those within a certain distance from the device in the x and y dimensions.

    Parameters:
    - detections: np.ndarray, shape (N, 5) or (N, 7) where N is the number of detections.
    - pos: np.ndarray, shape (3,), the position of the device.
    - max_distance: float, the maximum distance to filter detections.

    Returns:
    - below_distance: np.ndarray, filtered detections within the specified distance.
    - above_distance: np.ndarray, filtered detections beyond the specified distance.
    """
    if len(detections) == 0:
        return detections, np.array([])

    # Calculate the distances in the x and y dimensions
    distances = np.linalg.norm(detections[:, :2] - pos[:2], axis=1)
    below_distance = detections[distances <= max_distance]
    above_distance = detections[distances > max_distance]
    
    return below_distance, above_distance

def filter_detections(out_det_all, lidar_pose_stamped, max_distance=7.0):
    out_det_all_below = {}
    out_det_all_above = {}
    
    for fr_idx in out_det_all.keys():
        pos = lidar_pose_stamped['position'][fr_idx]
        
        # Filter detections
        detections = out_det_all[fr_idx]
        filtered_below, filtered_above = filter_detections_within_distance(detections, pos, max_distance)
        out_det_all_below[fr_idx] = filtered_below
        out_det_all_above[fr_idx] = filtered_above

    return out_det_all_below, out_det_all_above

def flatten_3d_to_2d(detections_3d):
    """
    Flatten 3D detections to 2D by ignoring the z-coordinate.

    Parameters:
    - detections_3d: np.ndarray, shape (N, 7) where N is the number of detections.

    Returns:
    - np.ndarray, shape (N, 5), flattened 2D detections.
    """
    detections_2d = np.zeros((detections_3d.shape[0], 6))
    detections_2d[:, :2] = detections_3d[:, :2]  # center_x, center_y
    detections_2d[:, 2:4] = detections_3d[:, 3:5]  # box_x, box_y
    detections_2d[:, 4] = detections_3d[:, 6]  # rotation z
    detections_2d[:, 5] = detections_3d[:, 7]  # score
    return detections_2d

def combine_filtered_detections(out_det_2D_all_below_5m, out_det_3D_all_below_5m):
    combined_detections = {}
    indices_3D = {}
    for fr_idx in out_det_2D_all_below_5m.keys():
        detections_2D = out_det_2D_all_below_5m[fr_idx]
        detections_3D = out_det_3D_all_below_5m[fr_idx]
        detections_3D_flattened = flatten_3d_to_2d(detections_3D)

        combined = np.vstack((detections_2D, detections_3D_flattened))
        combined_detections[fr_idx] = combined

        # Store the indices where 3D detections start
        indices_3D[fr_idx] = len(detections_2D)

    return combined_detections, indices_3D

class Settings:
    """
    Settings class to store script parameters.

    Attributes:
        dataset (str): Dataset that is being processed
        config_path (str): Path to the configuration file
        folder (str): Different subfolder in rosbag/ dir
        overwrite (bool): Whether to overwrite existing output
        save_raw (bool): Whether to save raw data of detection results
        model (str): Checkpoints filename
        save_thresh (float): Minimum confidence of detected boxes to save
        detect_2D (bool): Flag indicating whether to perform 2D detection
        model_2D (str): Checkpoints filename for the 2D detection model
        save_thresh_2D (float): Minimum confidence threshold for saving 2D detected boxes
    """

    def __init__(self, dataset, config_path, folder, overwrite=False, save_raw=False, model='./checkpoints/Person-MinkUNet-3D-JRDB-train-val-e40.pth', save_thresh=0.5,
                  detect_2D=False, model_2D='./checkpoints/DR-SPAAM-2D-JRDB-train-val-e20.pth', save_thresh_2D=0.5):
        self.dataset = dataset
        self.config_path = config_path
        self.folder = folder
        self.overwrite = overwrite
        self.save_raw = save_raw
        self.model = model
        self.save_thresh = save_thresh
        self.detect_2D = detect_2D
        self.model_2D = model_2D
        self.save_thresh_2D = save_thresh_2D

if __name__ == '__main__':
    # Instantiate the Settings object with custom or default values
    # folders = ['Cafeteria_1', 'Cafeteria_2', 'Cafeteria_3', 'Cafeteria_5', 'Cafeteria_6', 
    #           'Cafe_street_1-002', 'Cafe_street_2-001', 
    #           'Corridor_1', 'Corridor_10', 
    #           'Hallway_1', 'Hallway_2', 'Hallway_3', 'Hallway_4', 'Hallway_6', 'Hallway_7', 'Hallway_8', 'Hallway_9', 'Hallway_10', 'Hallway_11', 
    #           'Lobby_2', 'Lobby_3', 'Lobby_4', 'Lobby_5', 'Lobby_6', 'Lobby_7', 'Lobby_8', 
    #           'Corridor_2', 'Corridor_3', 'Corridor_5', 'Corridor_7', 'Corridor_8', 'Corridor_9','Corridor_11',  
    #           'Courtyard_1', 'Courtyard_2', 'Courtyard_4', 'Courtyard_5', 'Courtyard_6', 'Courtyard_8', 'Courtyard_9',
    #           'Outdoor_Alley_2', 'Outdoor_Alley_3', 
    #           'Subway_Entrance_2', 'Subway_Entrance_4', 
    #           'Three_way_Intersection_3', 'Three_way_Intersection_4', 'Three_way_Intersection_5', 'Three_way_Intersection_8', 
    #           'Crossroad_1-001',]
    folders = ['0325_rds_defaced', 
              '0325_shared_control_defaced', 
              '0327_shared_control_defaced', 
              '0410_mds_defaced', 
              '0410_rds_defaced', 
              '0410_shared_control_defaced', 
              '0424_mds_defaced', 
              '0424_rds_defaced', 
              '0424_shared_control_defaced', 
              '1203_manual_defaced', 
              '1203_shared_control_defaced']
    # folders = ['JRDB_whole',]
    nms_2D_dist_norm_thresh = 0.7  # Set the desired distance threshold (NMS 2D normalized threshold)
    max_distance = 5 # Set the desired distance partition (Tracking split)
    if len(sys.argv) > 1:
        subdir_arg = sys.argv[1]  # Get the single argument
        folders = [subdir_arg]
    
    for folder in folders:
        # args = Settings(dataset='SiT', config_path='./datasets_configs/data_path_SiT.yaml',
        #                 folder=folder, overwrite=True, detect_2D=True, save_thresh_2D=0.5, save_thresh=0.5)
        args = Settings(dataset='Crowdbot', config_path='./datasets_configs/data_path_Crowdbot.yaml',
                        folder=folder, overwrite=True, detect_2D=True, save_thresh_2D=0.5, save_thresh=0.5)
        # args = Settings(dataset='JRDB', config_path='./datasets_configs/data_path_JRDB.yaml',
        #                 folder=folder, overwrite=True, detect_2D=True, save_thresh_2D=0.5, save_thresh=0.5)


        assert args.dataset in ['JRDB', 'Crowdbot', 'SiT']

        # Create a CrowdBotDatabase instance with the specified folder and configuration path
        cb_data = CrowdBotDatabase(args.folder, config=args.config_path)

        seq_num = cb_data.nr_seqs()
        print("Starting detection from {} lidar sequences!".format(seq_num))

        counter = 0
        for seq_idx in range(seq_num):
            # Create a DetectorWithClock instance with the checkpoints path
            detector_3D = DetectorWithClock(args.model)
            plane_model = None

            # Source: lidar data in data/xxxx_processed/lidars
            lidar_file_dir = cb_data.lidar_dir

            seq = cb_data.seqs[seq_idx]
            if args.detect_2D:
                lidar_2D_file_dir = cb_data.lidar_2D_dir
                detectors_2D = []

                with open(os.path.join(lidar_2D_file_dir, seq, '00000.npy'), "rb") as f:
                    lasers = np.load(f, allow_pickle=True).item()
                for key in lasers.keys():
                    if args.dataset == 'Crowdbot' and key.find('front') != -1 and key.find('modified') != -1:
                        print('FRONT')
                        print(key)
                        dr_spaam_2D = DR_SPAAM_detector(ckpt_file=args.model_2D, cls_threshold=args.save_thresh_2D, laser_fov_deg=253.38, panoramic_scan=False, use_box=True)
                    elif args.dataset == 'Crowdbot' and key.find('rear') != -1  and key.find('modified') != -1:  
                        print('REAR')
                        print(key)
                        dr_spaam_2D = DR_SPAAM_detector(ckpt_file=args.model_2D, cls_threshold=args.save_thresh_2D, laser_fov_deg=236.18, panoramic_scan=False, use_box=True)
                    else:
                        dr_spaam_2D = DR_SPAAM_detector(ckpt_file=args.model_2D, cls_threshold=args.save_thresh_2D, laser_fov_deg=360, panoramic_scan=True, use_box=True)
                    dr_spaam_2D.prepare_model()
                    detectors_2D.append(dr_spaam_2D)

            counter += 1
            print("({}/{}): {}".format(counter, seq_num, seq))
            frames = [
                frame for frame in os.listdir(os.path.join(lidar_file_dir, seq))
                if not frame.startswith('.')
            ]
            frames.sort()

            # dets_dir = os.path.join(cb_data.dets_dir, 'new_dets')
            dets_dir = cb_data.dets_dir
            dnpy_all_path = os.path.join(dets_dir, seq + '.npy')
            dnpy_all_far_path = os.path.join(dets_dir, seq + '_far.npy')

            if args.detect_2D:
                dets_2D_dir = cb_data.dets_2D_dir
                dnpy_all_2D_path = os.path.join(dets_2D_dir, seq + '.npy')
                dnpy_all_2D_close_path = os.path.join(dets_2D_dir, seq + '_close.npy')
            

            # Create a directory for saving raw data if 'save_raw' is enabled
            if args.save_raw:
                det_seq_dir = os.path.join(dets_dir, seq)
            
            if args.dataset == 'JRDB':
                pose_folder = 'tf_JRDB'
                pose_suffix = "_tfJRDB_sampled.npy"
            elif args.dataset == 'Crowdbot':
                pose_folder = 'tf_qolo'
                pose_suffix = "_tfqolo_sampled.npy"
            elif args.dataset == 'SiT':
                pose_folder = 'tf_SiT'
                pose_suffix = "_tf_SiT_sampled.npy"
            else:
                raise RuntimeError
                
            tf_dir = os.path.join(cb_data.source_data_dir, pose_folder)
            pose_stampe_path = os.path.join(
                tf_dir, seq + pose_suffix
            )
            lidar_pose_stamped = np.load(
                pose_stampe_path, allow_pickle=True
            ).item()

            if os.path.exists(dnpy_all_path) and not args.overwrite:
                print("{} detection results already generated!!!".format(seq))
                print("Will not overwrite. If you want to overwrite, use flag --overwrite")
                continue
            else:

                # Create a subdirectory for each sequence if 'save_raw' is enabled
                if args.save_raw and (not os.path.exists(det_seq_dir)):
                    os.makedirs(det_seq_dir)

                out_det_all = dict()
                out_det_2D_all = dict()

                for fr_idx, frame in enumerate(tqdm(frames)):
                    # if fr_idx != 2240:
                    #     continue

                    pos = lidar_pose_stamped['position'][fr_idx]
                    quat = lidar_pose_stamped['orientation'][fr_idx]

                    with open(os.path.join(lidar_file_dir, seq, frame), "rb") as f:
                        pc = np.load(f)

                    if args.dataset == 'Crowdbot':
                        boxes, scores, plane_model = detect_3D_with_plane_alignment(pc, pos, detector_3D, plane_model)
                    else:
                        boxes, scores = detect_3D_without_plane_fitting(pc, pos, detector_3D)

                    cls_mask = scores > args.save_thresh
                    boxes_ = boxes[cls_mask]
                    scores_ = scores[cls_mask]

                    out_det = np.concatenate((boxes_, scores_[:, np.newaxis]), axis=1)
                    out_det_all.update({fr_idx: out_det})

                    if args.detect_2D:
                        out_det_2D_list = []
                        with open(os.path.join(lidar_2D_file_dir, seq, frame), "rb") as f:
                            laser = np.load(f, allow_pickle=True).item()
                            laser_keys = list(laser.keys())
                        for ind, detector_2D in enumerate(detectors_2D):
                            topic_name = laser_keys[ind]
                            # print(topic_name)
                            centers_2D, scores_2D, boxes_2D, _ = detector_2D.detect(laser[topic_name]['ranges'])

                            centers_2D, boxes_2D = dets_2D_local_to_global(centers_2D, boxes_2D, pos, quat, dataset=args.dataset, topic=topic_name)

                            out_det_2D_single = np.concatenate((centers_2D, boxes_2D, scores_2D[:, np.newaxis]), axis=1)
                            out_det_2D_list.append(out_det_2D_single)
                        out_det_2D = np.concatenate(out_det_2D_list)
                        out_det_2D_all.update({fr_idx: out_det_2D})
                    
                    if args.save_raw:
                        np.savetxt(
                            os.path.join(det_seq_dir, frame.replace("npy", "txt")),
                            out_det,
                            delimiter=",",
                        )

                np.save(
                    dnpy_all_path,
                    out_det_all,
                )

                if args.detect_2D:
                    out_det_2D_all_nms = apply_nms_to_all_detections(out_det_2D_all, nms_2D_dist_norm_thresh)
                    np.save(
                        dnpy_all_2D_path,
                        out_det_2D_all_nms,
                    )

                    # Apply filtering
                    out_det_2D_all_below_dist, _ = filter_detections(out_det_2D_all_nms, lidar_pose_stamped, max_distance)
                    out_det_3D_all_below_dist, out_det_3D_all_above_dist = filter_detections(out_det_all, lidar_pose_stamped, max_distance)
                    # Combine detections
                    out_det_combined_all_below_dist, indices_3D = combine_filtered_detections(out_det_2D_all_below_dist, out_det_3D_all_below_dist)

                    # Apply NMS
                    out_det_all_below_dist_nms = apply_nms_to_all_detections(out_det_combined_all_below_dist, nms_2D_dist_norm_thresh, indices_3D)

                    np.save(
                        dnpy_all_far_path,
                        out_det_3D_all_above_dist,
                    )

                    np.save(
                        dnpy_all_2D_close_path,
                        out_det_all_below_dist_nms,
                    )


                
                print(detector_3D)
                detector_3D.reset()
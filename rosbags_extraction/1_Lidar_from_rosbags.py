#!/usr/bin/env python3
import os
import numpy as np
import sys
import rospy
import rosbag
import tf2_py as tf2
# import ros_numpy
import sensor_msgs.point_cloud2 as pc2
import json

from crowdbot_data.crowdbot_data import bag_file_filter, processed_Crowdbot_bag_file_filter, CrowdBotDatabase

from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

JRDB_dataset_dir = '/scai_data/data01/daav/JRDB/train_dataset'

def get_starting_pos_offset_JRDB(bag, tf_buffer, first_timestamp):
    offset_trans = tf_buffer.lookup_transform_core(
                    "odom","base_chassis_link", first_timestamp
                )
    start_pos_offset = np.array([offset_trans.transform.translation.x,offset_trans.transform.translation.y, 0])
    return start_pos_offset

def get_tf_tree(bag):
    tf_buffer = tf2.BufferCore(rospy.Duration(1e9))
    for topic, msg, _ in bag.read_messages(topics=["/tf", "/tf_static"]):
        for msg_tf in msg.transforms:
            if topic == "/tf_static":
                tf_buffer.set_transform_static(msg_tf, "default_authority")
            else:
                tf_buffer.set_transform(msg_tf, "default_authority")

    return tf_buffer

def load_lidar(bag, topic, tf_buffer, target_frame, args):
    msgs, ts = [], []
    src_frame = None
    failed_counter = 0

    for _, msg, t in bag.read_messages(topics=[topic]):
        src_frame = msg.header.frame_id

        if target_frame != src_frame:
            try:
                trans = tf_buffer.lookup_transform_core(
                    target_frame, src_frame, msg.header.stamp
                )
                msg = do_transform_cloud(msg, trans)
            except tf2.ExtrapolationException as e:  # noqa
                # print(e)
                failed_counter += 1
                continue

        # pc_xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        pc_xyz = pc2.read_points(msg, skip_nans=True, field_names = ("x", "y", "z"))
        # print(next(pc_xyz))
        pc_xyz = np.fromiter(pc_xyz, dtype=np.dtype((float,3)))
        msgs.append(pc_xyz)
        ts.append(msg.header.stamp.to_sec())
        # ts.append(t.to_sec())

    ts = np.array(ts, dtype=np.float64)  # warning: float32 is not enough

    s = (
        "Summary\n"
        "topic: {}\n"
        "count: {}\n"
        "min timestamp: {}\n"
        "max timestamp: {}\n"
        "average time between frames: {}\n"
        "source frame: {}\n"
        "target frame: {}\n"
        "tf failed count: {}\n"
    ).format(
        topic,
        len(ts),
        ts[0],
        ts[-1],
        (ts[-1] - ts[0]) / len(ts),
        src_frame,
        target_frame,
        failed_counter,
    )

    print(s)

    return msgs, ts

def load_2D_lidar(bag, topic,):
    msgs, ts = [], []
    src_frame = None
    failed_counter = 0

    for _, msg, t in bag.read_messages(topics=[topic]):

        ranges = np.array(msg.ranges)
        ranges[~np.isfinite(ranges)] = msg.range_max
        intensities = np.array(msg.intensities)
        scan = {'ranges': ranges, 'intensities': intensities}
        msgs.append(scan)
        ts.append(msg.header.stamp.to_sec())
        # ts.append(t.to_sec())

    ts = np.array(ts, dtype=np.float64)  # warning: float32 is not enough

    s = (
        "Summary\n"
        "topic: {}\n"
        "count: {}\n"
        "min timestamp: {}\n"
        "max timestamp: {}\n"
        "average time between frames: {}\n"
        "source frame: {}\n"
        "tf failed count: {}\n"
    ).format(
        topic,
        len(ts),
        ts[0],
        ts[-1],
        (ts[-1] - ts[0]) / len(ts),
        src_frame,
        failed_counter,
    )

    print(s)

    return msgs, ts


def save_lidar(filename, pc, library="numpy", write_ascii=False, compressed=True, overwrite=False):
    # save with numpy
    if library == "numpy":
        filename = filename + ".npy"
        if (not os.path.exists(filename)) or overwrite:
            with open(filename, "wb") as f:
                np.save(f, pc)
        else:
            print('File {} already exists, not overwriting'.format(filename))

    # save with open3d
    elif library == "open3d":
        import open3d as o3d

        filename = filename + ".pcd"
        if (not os.path.exists(filename)) or overwrite:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            # open3d.io.write_point_cloud(filename, pointcloud, write_ascii=False, compressed=False, print_progress=False)
            o3d.io.write_point_cloud(
                filename, pcd, write_ascii=write_ascii, compressed=compressed
            )
        else:
            print('File {} already exists, not overwriting'.format(filename))


def extract_lidar_from_rosbag(bag_path, out_dirs, args, JRDB_gt_timestamps_dir=None, overwrite=False):
    """Extract and save combined laser scan from rosbag. Existing files will be overwritten"""

    if args.dataset == 'JRDB':
        target_frame = 'odom'
    elif args.dataset == 'Crowdbot':
        target_frame = 'odom'
    else:
        target_frame = 'camera_init'

    output_first_path = os.path.join(out_dirs[0], "00000.npy")
    if not overwrite and os.path.exists(output_first_path):
        print(f"Lidar for {bag_path} already exists, not overwriting")
        return

    with rosbag.Bag(bag_path) as bag:


        tf_buffer = get_tf_tree(bag)
        topic_3D_msgs_list = []
        nonground_3D_msgs_list = []
        topic_2D_msgs_list = []
        ts_list = []
        min_ts_list = []

        num_topics_3D = len(args.topics_3D)
        num_topics_nonground = 0
        num_topics_2D = 0

        for topic_3D in args.topics_3D: 
            topic_3D_msgs, topic_3D_ts = load_lidar(
                bag,
                topic_3D,
                tf_buffer,
                target_frame=target_frame,
                args=args,
            )
            topic_3D_msgs_list.append(topic_3D_msgs)
            ts_list.append(topic_3D_ts)
            min_ts_list.append(topic_3D_ts.min())

        if args.nonground_topics_3D != None:
            num_topics_nonground = len(args.nonground_topics_3D)
            for nonground_topic_3D in args.nonground_topics_3D:
                nonground_3D_msgs, nonground_3D_ts = load_lidar(
                    bag,
                    nonground_topic_3D,
                    tf_buffer,
                    target_frame=target_frame,
                    args=args,
                )
                nonground_3D_msgs_list.append(nonground_3D_msgs)
                ts_list.append(nonground_3D_ts)
                min_ts_list.append(nonground_3D_ts.min())
        
        if args.topics_2D != None:
            num_topics_2D = len(args.topics_2D)
            for topic_2D in args.topics_2D:
                laser_msgs, laser_ts = load_2D_lidar(
                    bag,
                    topic_2D,
                )
                topic_2D_msgs_list.append(laser_msgs)
                ts_list.append(laser_ts)
                min_ts_list.append(laser_ts.min())

    # sync lidar
    offset = min(min_ts_list)
    for ind, np_ts in enumerate(ts_list):
        ts_list[ind] = np_ts - offset
    
    lidar_t0s = []
    lidar_t1s = []
    lidar_dts = []
    for ind in range(num_topics_3D):
        lidar_t0, lidar_t1 = ts_list[ind].min(), ts_list[ind].max()
        lidar_dt = (lidar_t1 - lidar_t0) / float(len(ts_list[ind]))
        lidar_t0s.append(lidar_t0)
        lidar_t1s.append(lidar_t1)
        lidar_dts.append(lidar_dt)

    sync_dt = max(lidar_dts)  
    sync_t0 = max(lidar_t0s)
    sync_t1 = min(lidar_t1s)
    if args.dataset == 'JRDB':
        # Define the path to the JSON file
        frames_pc_path = os.path.join(JRDB_gt_timestamps_dir, 'frames_pc.json')

        # Load JSON data
        with open(frames_pc_path, 'r') as file:
            frames_data = json.load(file)

        # Step 2: Collect timestamps within the interval, adjusted by the offset
        sync_ts = []

        for frame in frames_data['data']:
            frame_timestamp = frame['timestamp'] - offset  # Adjust the frame's timestamp by the offset
            if sync_t0 <= frame_timestamp <= sync_t1:
                sync_ts.append(frame_timestamp)
                
        # Convert to numpy array for consistency
        sync_ts = np.array(sync_ts, dtype=np.float64)
    else:
        # Default behavior: sync to the slower one across all lidars
        sync_ts = np.arange(start=sync_t0, step=sync_dt, stop=sync_t1, dtype=np.float64)

    if args.dataset == 'JRDB':
            sync_start_time = sync_ts[0] + offset
            t0_secs = int(sync_start_time)
            t0_nsecs = int((sync_start_time - t0_secs) * 1e9)
            # Create a rospy.Time message
            sync_t0_msg = rospy.Time(t0_secs, t0_nsecs)
            JRDB_pos_offset = get_starting_pos_offset_JRDB(bag, tf_buffer, sync_t0_msg)
            topic_3D_msgs_list = [[pc_xyz - JRDB_pos_offset for pc_xyz in topic_3d_msgs] for topic_3d_msgs in topic_3D_msgs_list]

    def get_sync_inds(ts, sync_ts):
        d = np.abs(sync_ts.reshape(-1, 1) - ts.reshape(1, -1))
        return np.argmin(d, axis=1)

    sync_inds_list = []
    for np_ts in ts_list:
        sync_inds_list.append(get_sync_inds(np_ts, sync_ts))

    # write frame to file
    for frame_id, idx_tuple in enumerate(zip(*sync_inds_list)):
        file_path_3D = os.path.join(out_dirs[0], "{0:05d}".format(frame_id))

        individual_pc_list = [lidar_3D_msgs[idx_tuple[ind]] for ind, lidar_3D_msgs in enumerate(topic_3D_msgs_list)]
        pc = np.concatenate(individual_pc_list, axis=0)
        if args.compressed:
            save_lidar(file_path_3D, pc, library="open3d", overwrite=overwrite)
        else:
            save_lidar(file_path_3D, pc, overwrite=overwrite)

        if args.nonground_topics_3D != None:
            file_path_nonground = os.path.join(out_dirs[1], "{0:05d}".format(frame_id))
            individual_nonground_pc_list = [lidar_nonground_msgs[idx_tuple[num_topics_3D+ind]] for ind, lidar_nonground_msgs in enumerate(nonground_3D_msgs_list)]
            pc = np.concatenate(individual_nonground_pc_list, axis=0)
            if args.compressed:
                save_lidar(file_path_nonground, pc, library="open3d", overwrite=overwrite)
            else:
                save_lidar(file_path_nonground, pc, overwrite=overwrite)

        if args.topics_2D != None:
            file_path_2D = os.path.join(out_dirs[2], "{0:05d}".format(frame_id))
            lidar_2D_dict = {}
            for ind in range(num_topics_2D):
                lidar_2D_dict[args.topics_2D[ind]] = topic_2D_msgs_list[ind][idx_tuple[num_topics_3D+num_topics_nonground+ind]]
            # print('2D Saving ' + str(file_path_2D))
            save_lidar(file_path_2D, lidar_2D_dict, overwrite=overwrite)


    id_list, ts_list = [], []
    for frame_id, ts in enumerate(sync_ts):
        id_list.append(frame_id)
        ts_list.append(ts + offset)  # adding offset

    lidar_stamped_dict = {
        "timestamp": np.asarray(ts_list, dtype=np.float64),
        "id": np.array(frame_id),
    }

    s = (
        "Summary\n"
        "topic: synced frames\n"
        "count: {}\n"
        "average time between frames: {}\n"
    ).format(
        len(sync_ts),
        sync_dt,
    )
    print(s)

    return lidar_stamped_dict

class Settings:
    """
    Settings class to store script parameters.

    Attributes:
        dataset (str): Dataset that is being processed
        config_path (str): Path to the configuration file
        folder (str): Different subfolder in rosbag/ dir
        topics_3D (list of str): Topics for the 3D lidar
        topics_2D (list of str): Topics for the 2D lidar
        overwrite (bool): Overwrite existing rosbags
        compressed (bool): Save point cloud with compressed format
    """

    def __init__(self, dataset, config_path, folder, 
                 topics_3D,
                nonground_topics_3D=None, topics_2D=None, overwrite=False, compressed=False,):
        self.dataset = dataset
        self.config_path = config_path
        self.folder = folder
        self.topics_3D = topics_3D
        self.nonground_topics_3D = nonground_topics_3D
        self.topics_2D = topics_2D
        self.overwrite = overwrite
        self.compressed = compressed
        

if __name__ == '__main__':
    # Instantiate the Settings object with custom or default values
    # folders = ['0325_rds_defaced', 
    #           '0325_shared_control_defaced', 
    #           '0327_shared_control_defaced', 
    #           '0410_mds_defaced', 
    #           '0410_rds_defaced', 
    #           '0410_shared_control_defaced', 
    #           '0424_mds_defaced', 
    #           '0424_rds_defaced', 
    #           '0424_shared_control_defaced', 
    #           '1203_manual_defaced', 
    #           '1203_shared_control_defaced']
    folders = ['JRDB_whole',]
    if len(sys.argv) > 1:
        subdir_arg = sys.argv[1]  # Get the single argument
        folders = [subdir_arg]
        
    for folder in folders:
        # args = Settings(dataset='Crowdbot', folder=folder, config_path='./datasets_configs/data_path_Crowdbot.yaml',
        #                 topics_3D=['/front_lidar/vp_global', '/rear_lidar/vp_global'], topics_2D=['/front_lidar/scan_modified', '/rear_lidar/scan_modified', '/scan_multi'], overwrite=False)
        args = Settings(dataset='JRDB', folder=folder, config_path='./datasets_configs/data_path_JRDB.yaml',
                        topics_3D=['/upper_velodyne/velodyne_points', '/lower_velodyne/velodyne_points',], topics_2D=['/segway/scan_multi',], overwrite=True)

        assert args.dataset in ['JRDB', 'Crowdbot',]

        cb_data = CrowdBotDatabase(args.folder, config=args.config_path)

        JRDB_gt_timestamps_dir_base = os.path.join(JRDB_dataset_dir, 'timestamps')
        rosbag_dir = os.path.join(cb_data.bagbase_dir, args.folder)
        print(rosbag_dir)
        if args.dataset == 'Crowdbot':
            bag_files = list(filter(processed_Crowdbot_bag_file_filter, os.listdir(rosbag_dir)))
        elif args.dataset == 'JRDB':
            bag_files = list(filter(bag_file_filter, os.listdir(rosbag_dir)))
        else:
            bag_files = list(filter(bag_file_filter, os.listdir(rosbag_dir)))

        # destination: lidar data in xxxx_processed/lidars
        data_processed_dir = os.path.join(cb_data.outbase_dir, args.folder + "_processed")
        if not os.path.exists(data_processed_dir):
            os.makedirs(data_processed_dir)
        lidar_file_dir = cb_data.lidar_dir
        lidar_file_nonground_dir = cb_data.lidar_nonground_dir
        lidar_file_2D_dir = cb_data.lidar_2D_dir
        lidar_stamp_dir = os.path.join(cb_data.source_data_dir, "timestamp")
        if not os.path.exists(lidar_stamp_dir):
            os.makedirs(lidar_stamp_dir)

        print("Starting extracting lidar files from {} rosbags!".format(len(bag_files)))

        for idx, bf in enumerate(bag_files):
            print(bf)
            if bf.find('filtered') == -1 or bf.find('lidar_odom') != -1:
                bag_path = os.path.join(rosbag_dir, bf)
                bag_name = bf.split(".")[0]
                JRDB_gt_timestamps_dir = os.path.join(JRDB_gt_timestamps_dir_base, bag_name.replace('_all_transforms', ''))
                out_dir = os.path.join(lidar_file_dir, bag_name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                out_nonground_dir = os.path.join(lidar_file_nonground_dir, bag_name)
                if not os.path.exists(out_nonground_dir):
                    os.makedirs(out_nonground_dir)

                out_2D_dir = os.path.join(lidar_file_2D_dir, bag_name)
                if not os.path.exists(out_2D_dir):
                    os.makedirs(out_2D_dir)

                out_dirs = [out_dir, out_nonground_dir, out_2D_dir]
                print("({}/{}): {}".format(idx + 1, len(bag_files), bag_path))

                lidar_stamped_dict = extract_lidar_from_rosbag(bag_path, out_dirs, args, overwrite=args.overwrite, JRDB_gt_timestamps_dir = JRDB_gt_timestamps_dir)
                if lidar_stamped_dict is None:
                    continue
                
                print(
                    "lidar_stamped_dict with {} frames".format(
                        len(lidar_stamped_dict['timestamp'])
                    )
                )

                stamp_file_path = os.path.join(lidar_stamp_dir, bag_name + "_stamped.npy")
                np.save(stamp_file_path, lidar_stamped_dict)
#!/usr/bin/env python3
"""Extract synchronized lidar frames for the datasets used in the paper."""

import argparse
import io
import json
import os
import struct

import lzf
import numpy as np
import rospy
import rosbag
import tf2_py as tf2
import sensor_msgs.point_cloud2 as pc2

from crowdbot_data.crowdbot_data import (
    CrowdBotDatabase,
    bag_file_filter,
    processed_Crowdbot_bag_file_filter,
)

# sudo apt-get install ros-$ROS_DISTRO-tf2-sensor-msgs
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf.transformations import quaternion_matrix

try:
    import velodyne_decoder as vd
except ImportError:
    vd = None

def get_starting_pos_offset_JRDB(bag, tf_buffer, first_timestamp):
    offset_trans = tf_buffer.lookup_transform_core(
                    "odom","base_chassis_link", first_timestamp
                )
    start_pos_offset = np.array([offset_trans.transform.translation.x,offset_trans.transform.translation.y, 0])
    return start_pos_offset

def get_starting_pos_offset_SCAND(scand_odom, first_timestamp_sec):
    timestamps = scand_odom["timestamp"]
    closest_index = np.argmin(np.abs(timestamps - first_timestamp_sec))
    return scand_odom["position"][closest_index].copy()

def get_tf_tree(bag):
    tf_buffer = tf2.BufferCore(rospy.Duration(1e9))
    for topic, msg, _ in bag.read_messages(topics=["/tf", "/tf_static"]):
        for msg_tf in msg.transforms:
            if topic == "/tf_static":
                tf_buffer.set_transform_static(msg_tf, "default_authority")
            else:
                tf_buffer.set_transform(msg_tf, "default_authority")

    return tf_buffer

JRDB_TEST_UPPER_ROT_Z = 0.085
JRDB_TEST_UPPER_TRANS = np.array([0.0, 0.0, 0.33529], dtype=np.float64)

JRDB_TEST_LOWER_ROT_Z = 0.0
JRDB_TEST_LOWER_TRANS = np.array([0.0, 0.0, -0.13511], dtype=np.float64)

def get_R_z(rot_z):
    cs, ss = np.cos(rot_z), np.sin(rot_z)
    return np.array(
        [[cs, -ss, 0.0],
         [ss,  cs, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def apply_rigid_z_3xN(xyz_3xN, rot_z, trans_xyz):
    Rz = get_R_z(rot_z)
    t = np.asarray(trans_xyz, dtype=np.float64).reshape(3, 1)
    return (Rz @ xyz_3xN) + t

_PCD_TYPE_TO_NUMPY = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("U", 8): np.uint64,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
    ("I", 8): np.int64,
}


class PointCloud:
    def __init__(self, metadata, pc_data):
        self.metadata_keys = list(metadata.keys())
        for k, v in metadata.items():
            setattr(self, k, v)
        self.pc_data = pc_data
        self.check_sanity()

    def get_metadata(self):
        metadata = {}
        for k in self.metadata_keys:
            v = getattr(self, k)
            metadata[k] = list(v) if isinstance(v, list) else v
        return metadata

    def check_sanity(self):
        required = (
            "version", "fields", "size", "type", "count",
            "width", "height", "points", "viewpoint", "data"
        )
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            raise ValueError(f"Missing required metadata fields: {missing}")

        if not (
            len(self.fields) == len(self.size) ==
            len(self.type) == len(self.count)
        ):
            raise ValueError("fields, size, type, and count must have the same length")

        if self.width <= 0:
            raise ValueError("width must be > 0")
        if self.height <= 0:
            raise ValueError("height must be > 0")
        if self.points <= 0:
            raise ValueError("points must be > 0")
        if self.width * self.height != self.points:
            raise ValueError("width * height must equal points")
        if len(self.pc_data) != self.points:
            raise ValueError("pc_data length must equal points")


def _parse_header(lines):
    metadata = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(None, 1)
        if len(parts) != 2:
            continue

        key = parts[0].lower()
        value = parts[1].strip()

        if key == "version":
            metadata[key] = value
        elif key in ("fields", "type"):
            metadata[key] = value.split()
        elif key in ("size", "count"):
            metadata[key] = [int(x) for x in value.split()]
        elif key in ("width", "height", "points"):
            metadata[key] = int(value)
        elif key == "viewpoint":
            metadata[key] = [float(x) for x in value.split()]
        elif key == "data":
            metadata[key] = value.lower()

    if "fields" not in metadata:
        raise ValueError("PCD header missing FIELDS")
    if "size" not in metadata:
        raise ValueError("PCD header missing SIZE")
    if "type" not in metadata:
        raise ValueError("PCD header missing TYPE")
    if "width" not in metadata:
        raise ValueError("PCD header missing WIDTH")
    if "height" not in metadata:
        raise ValueError("PCD header missing HEIGHT")
    if "data" not in metadata:
        raise ValueError("PCD header missing DATA")

    if "count" not in metadata:
        metadata["count"] = [1] * len(metadata["fields"])
    if "viewpoint" not in metadata:
        metadata["viewpoint"] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if "version" not in metadata:
        metadata["version"] = ".7"
    if "points" not in metadata:
        metadata["points"] = metadata["width"] * metadata["height"]

    if not (
        len(metadata["fields"]) == len(metadata["size"]) ==
        len(metadata["type"]) == len(metadata["count"])
    ):
        raise ValueError("FIELDS, SIZE, TYPE, and COUNT must have the same length")

    return metadata


def _build_dtype(metadata):
    fieldnames = []
    typenames = []

    for f, c, t, s in zip(
        metadata["fields"], metadata["count"], metadata["type"], metadata["size"]
    ):
        np_type = _PCD_TYPE_TO_NUMPY[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            for i in range(c):
                fieldnames.append(f"{f}_{i:04d}")
                typenames.append(np_type)

    return np.dtype(list(zip(fieldnames, typenames)))


def _parse_ascii_pc_data(fileobj, dtype, metadata):
    payload = fileobj.read()
    if isinstance(payload, bytes):
        payload = payload.decode("ascii")
    payload = payload.strip()
    if not payload:
        return np.empty((0,), dtype=dtype)
    arr = np.loadtxt(io.StringIO(payload), dtype=dtype)
    return np.atleast_1d(arr)


def _parse_binary_pc_data(fileobj, dtype, metadata):
    rowstep = metadata["points"] * dtype.itemsize
    buf = fileobj.read(rowstep)
    if len(buf) != rowstep:
        raise ValueError(
            f"Binary data truncated: got {len(buf)} bytes, expected {rowstep}"
        )
    return np.frombuffer(buf, dtype=dtype, count=metadata["points"]).copy()


def _parse_binary_compressed_pc_data(fileobj, dtype, metadata):
    fmt = "<II"
    header = fileobj.read(struct.calcsize(fmt))
    if len(header) != struct.calcsize(fmt):
        raise ValueError("Truncated binary_compressed header")

    compressed_size, uncompressed_size = struct.unpack(fmt, header)
    compressed_data = fileobj.read(compressed_size)
    if len(compressed_data) != compressed_size:
        raise ValueError(
            f"Compressed payload truncated: got {len(compressed_data)} bytes, expected {compressed_size}"
        )

    buf = lzf.decompress(compressed_data, uncompressed_size)
    if buf is None or len(buf) != uncompressed_size:
        raise IOError("Error decompressing binary_compressed PCD data")

    pc_data = np.empty(metadata["width"], dtype=dtype)
    ix = 0
    for name in dtype.names:
        field_dtype = dtype.fields[name][0]
        nbytes = field_dtype.itemsize * metadata["width"]
        column = np.frombuffer(buf[ix:ix + nbytes], dtype=field_dtype, count=metadata["width"])
        pc_data[name] = column
        ix += nbytes

    return pc_data


def point_cloud_from_fileobj(fileobj):
    header = []
    while True:
        raw_line = fileobj.readline()
        if not raw_line:
            raise ValueError("Reached EOF before DATA line in PCD header")

        line = raw_line.decode("ascii").strip() if isinstance(raw_line, bytes) else raw_line.strip()
        header.append(line)

        if line.startswith("DATA"):
            metadata = _parse_header(header)
            dtype = _build_dtype(metadata)
            break

    if metadata["data"] == "ascii":
        pc_data = _parse_ascii_pc_data(fileobj, dtype, metadata)
    elif metadata["data"] == "binary":
        pc_data = _parse_binary_pc_data(fileobj, dtype, metadata)
    elif metadata["data"] == "binary_compressed":
        pc_data = _parse_binary_compressed_pc_data(fileobj, dtype, metadata)
    else:
        raise ValueError(f'Unsupported DATA field: {metadata["data"]}')

    return PointCloud(metadata, pc_data)


def point_cloud_from_path(fname):
    with open(fname, "rb") as f:
        return point_cloud_from_fileobj(f)

def load_pcd_xyz_3xN(pcd_path):
    pc_struct = point_cloud_from_path(pcd_path).pc_data
    xyz = np.array(
        [pc_struct["x"], pc_struct["y"], pc_struct["z"]],
        dtype=np.float64,
    )
    finite_mask = np.isfinite(xyz).all(axis=0)
    return xyz[:, finite_mask]


def load_merge_jrdb_test_frame_to_base(upper_pcd_path, lower_pcd_path):
    pc_u = load_pcd_xyz_3xN(upper_pcd_path)
    pc_l = load_pcd_xyz_3xN(lower_pcd_path)

    pc_u_base = apply_rigid_z_3xN(
        pc_u,
        JRDB_TEST_UPPER_ROT_Z,
        JRDB_TEST_UPPER_TRANS,
    )
    pc_l_base = apply_rigid_z_3xN(
        pc_l,
        JRDB_TEST_LOWER_ROT_Z,
        JRDB_TEST_LOWER_TRANS,
    )

    # return Nx3 to match the rest of this script
    pc_base = np.concatenate([pc_u_base, pc_l_base], axis=1).T
    return pc_base


def load_jrdb_test_frames_pc_json(frames_pc_path):
    with open(frames_pc_path, "r") as f:
        frames_data = json.load(f)

    frame_files = []
    frame_timestamps = []

    for idx, entry in enumerate(frames_data["data"]):
        if "timestamp" not in entry:
            continue
        frame_files.append(f"{idx:06d}.pcd")
        frame_timestamps.append(float(entry["timestamp"]))

    frame_timestamps = np.asarray(frame_timestamps, dtype=np.float64)

    if len(frame_timestamps) == 0:
        raise RuntimeError(f"No timestamps found in {frames_pc_path}")

    s = (
        "Summary\n"
        "topic: jrdb_test_frames_pc.json\n"
        "count: {}\n"
        "min timestamp: {}\n"
        "max timestamp: {}\n"
        "average time between frames: {}\n"
    ).format(
        len(frame_timestamps),
        frame_timestamps[0],
        frame_timestamps[-1],
        (frame_timestamps[-1] - frame_timestamps[0]) / len(frame_timestamps),
    )
    print(s)

    return frame_files, frame_timestamps


def load_jrdb_test_odom_csv(csv_path):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]

    timestamps = arr[:, 0].astype(np.float64) * 1e-9
    positions = arr[:, 1:4].astype(np.float64)
    quaternions = arr[:, 4:8].astype(np.float64)

    s = (
        "Summary\n"
        "topic: jrdb_test_odometry_csv\n"
        "count: {}\n"
        "min timestamp: {}\n"
        "max timestamp: {}\n"
        "average time between frames: {}\n"
        "source frame: odom\n"
        "child frame: base_chassis_link\n"
    ).format(
        len(timestamps),
        timestamps[0],
        timestamps[-1],
        (timestamps[-1] - timestamps[0]) / len(timestamps),
    )
    print(s)

    return {
        "timestamp": timestamps,
        "position": positions,
        "quaternion": quaternions,
        "frame_id": "odom",
        "child_frame_id": "base_chassis_link",
    }


def get_starting_pos_offset_JRDB_TEST(jrdb_test_odom, first_timestamp_sec):
    timestamps = jrdb_test_odom["timestamp"]
    closest_index = np.argmin(np.abs(timestamps - first_timestamp_sec))

    start_pos_offset = jrdb_test_odom["position"][closest_index].copy()
    start_pos_offset[2] = 0.0  # keep ground height convention same as JRDB train
    return start_pos_offset

SCAND_ROBOT_CONFIG = {
    "Jackal": {
        "odom_topic": "/jackal_velocity_controller/odom",
        "lidar_3d_candidates": ["/velodyne_points", "/velodyne_packets"],
        "lidar_2d_topic": "/velodyne_2dscan",
        # Body-to-lidar calibration used for the revision analysis.
        "body_to_lidar_translation": np.array([0.0, 0.0, 0.40], dtype=np.float64),
        "body_to_lidar_quaternion": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    },
    "Spot": {
        "odom_topic": "/odom",
        "lidar_3d_candidates": ["/velodyne_points", "/velodyne_packets"],
        "lidar_2d_topic": "/scan",
        # From the Spot TF you found: base_link -> velodyne
        "body_to_lidar_translation": np.array([0.0, 0.0, 0.86], dtype=np.float64),
        "body_to_lidar_quaternion": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    },
}


def infer_scand_robot_type_from_bag_path(bag_path):
    bag_name = os.path.basename(bag_path)
    stem = os.path.splitext(bag_name)[0]   # remove .bag
    parts = stem.split("_")

    # Expected examples:
    # A_Jackal_...
    # B_Spot_...
    # Z_Jackal_...
    if len(parts) < 2:
        raise ValueError(f"Could not infer SCAND robot type from bag name: {bag_name}")

    robot_token = parts[1]

    if robot_token == "Jackal":
        return "Jackal"
    if robot_token == "Spot":
        return "Spot"

    raise ValueError(f"Could not infer SCAND robot type from bag name: {bag_name}")


def require_velodyne_decoder():
    if vd is None:
        raise ImportError(
            "Processing /velodyne_packets requires velodyne-decoder. "
            "Install it with: pip install velodyne-decoder"
        )


def stamp_like_to_sec(stamp):
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return float(stamp.secs) + 1e-9 * float(stamp.nsecs)
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)
    return float(stamp)


def resolve_scand_topics(bag, robot_type):
    topic_info = bag.get_type_and_topic_info()[1]
    cfg = SCAND_ROBOT_CONFIG[robot_type]

    topic_3d = None
    topic_3d_type = None
    for candidate in cfg["lidar_3d_candidates"]:
        if candidate in topic_info:
            topic_3d = candidate
            topic_3d_type = topic_info[candidate].msg_type
            break

    if topic_3d is None:
        raise RuntimeError(
            f"SCAND {robot_type}: no 3D lidar topic found from candidates {cfg['lidar_3d_candidates']}"
        )

    odom_topic = cfg["odom_topic"]
    if odom_topic not in topic_info:
        raise RuntimeError(
            f"SCAND {robot_type}: required odom topic {odom_topic} not found in bag"
        )

    topic_2d = cfg["lidar_2d_topic"] if cfg["lidar_2d_topic"] in topic_info else None

    return {
        "robot_type": robot_type,
        "topic_3d": topic_3d,
        "topic_3d_type": topic_3d_type,
        "topic_2d": topic_2d,
        "odom_topic": odom_topic,
        "body_to_lidar_translation": cfg["body_to_lidar_translation"].copy(),
        "body_to_lidar_quaternion": cfg["body_to_lidar_quaternion"].copy(),
    }


def load_scand_odom(bag, odom_topic):
    ts = []
    positions = []
    quaternions = []
    src_frame = None
    child_frame = None

    for _, msg, _ in bag.read_messages(topics=[odom_topic]):
        src_frame = msg.header.frame_id
        child_frame = msg.child_frame_id

        ts.append(msg.header.stamp.to_sec())
        positions.append([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        quaternions.append([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])

    if len(ts) == 0:
        raise RuntimeError(f"No odometry messages found on topic {odom_topic}")

    ts = np.asarray(ts, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)

    s = (
        "Summary\n"
        "topic: {}\n"
        "count: {}\n"
        "min timestamp: {}\n"
        "max timestamp: {}\n"
        "average time between frames: {}\n"
        "source frame: {}\n"
        "child frame: {}\n"
    ).format(
        odom_topic,
        len(ts),
        ts[0],
        ts[-1],
        (ts[-1] - ts[0]) / len(ts),
        src_frame,
        child_frame,
    )
    print(s)

    return {
        "timestamp": ts,
        "position": positions,
        "quaternion": quaternions,
        "frame_id": src_frame,
        "child_frame_id": child_frame,
    }


def quat_xyzw_to_rotmat(quat_xyzw):
    return quaternion_matrix(quat_xyzw)[:3, :3]


def nearest_timestamp_index(sorted_ts, query_t):
    idx = np.searchsorted(sorted_ts, query_t, side="left")

    if idx <= 0:
        return 0
    if idx >= len(sorted_ts):
        return len(sorted_ts) - 1

    left_idx = idx - 1
    right_idx = idx

    if abs(query_t - sorted_ts[left_idx]) <= abs(sorted_ts[right_idx] - query_t):
        return left_idx
    return right_idx


def transform_points(points_xyz, rotation_matrix, translation_xyz):
    if points_xyz.shape[0] == 0:
        return points_xyz.copy()
    return (rotation_matrix @ points_xyz.T).T + translation_xyz.reshape(1, 3)


def load_lidar_scand_pointcloud2_global(
    bag,
    topic,
    odom_ts,
    odom_positions,
    odom_quaternions,
    body_to_lidar_translation,
    body_to_lidar_quaternion,
    target_frame="odom",
):
    msgs = []
    ts = []
    src_frame = None
    odom_sync_errors = []

    R_body_lidar = quat_xyzw_to_rotmat(body_to_lidar_quaternion)

    for _, msg, _ in bag.read_messages(topics=[topic]):
        src_frame = msg.header.frame_id
        stamp = msg.header.stamp.to_sec()

        pc_xyz = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
        pc_xyz = np.fromiter(pc_xyz, dtype=np.dtype((float, 3)))

        odom_idx = nearest_timestamp_index(odom_ts, stamp)
        odom_sync_errors.append(abs(odom_ts[odom_idx] - stamp))

        R_odom_body = quat_xyzw_to_rotmat(odom_quaternions[odom_idx])
        t_odom_body = odom_positions[odom_idx]

        pc_body = transform_points(pc_xyz, R_body_lidar, body_to_lidar_translation)
        pc_odom = transform_points(pc_body, R_odom_body, t_odom_body)

        msgs.append(pc_odom)
        ts.append(stamp)

    ts = np.asarray(ts, dtype=np.float64)

    s = (
        "Summary\n"
        "topic: {}\n"
        "count: {}\n"
        "min timestamp: {}\n"
        "max timestamp: {}\n"
        "average time between frames: {}\n"
        "source frame: {}\n"
        "target frame: {}\n"
        "mean abs odom sync error: {}\n"
        "max abs odom sync error: {}\n"
    ).format(
        topic,
        len(ts),
        ts[0],
        ts[-1],
        (ts[-1] - ts[0]) / len(ts),
        src_frame,
        target_frame,
        float(np.mean(odom_sync_errors)),
        float(np.max(odom_sync_errors)),
    )
    print(s)

    return msgs, ts


def load_lidar_scand_packets_global(
    bag,
    topic,
    odom_ts,
    odom_positions,
    odom_quaternions,
    body_to_lidar_translation,
    body_to_lidar_quaternion,
    target_frame="odom",
):
    require_velodyne_decoder()

    decoder = vd.ScanDecoder(vd.Config())

    msgs = []
    ts = []
    odom_sync_errors = []
    src_frame = None

    R_body_lidar = quat_xyzw_to_rotmat(body_to_lidar_quaternion)

    for _, msg, _ in bag.read_messages(topics=[topic]):
        src_frame = msg.header.frame_id
        stamp_sec = msg.header.stamp.to_sec()

        # Decode packets into points; ignore decoder's own stamp object
        _, points = decoder.decode_message(msg)

        if getattr(points, "dtype", None) is not None and points.dtype.names is not None:
            pc_xyz = np.column_stack(
                [points["x"], points["y"], points["z"]]
            ).astype(np.float64, copy=False)
        else:
            pc_xyz = np.asarray(points[:, :3], dtype=np.float64)

        odom_idx = nearest_timestamp_index(odom_ts, stamp_sec)
        odom_sync_errors.append(abs(odom_ts[odom_idx] - stamp_sec))

        R_odom_body = quat_xyzw_to_rotmat(odom_quaternions[odom_idx])
        t_odom_body = odom_positions[odom_idx]

        # velodyne -> body
        pc_body = transform_points(pc_xyz, R_body_lidar, body_to_lidar_translation)

        # body -> odom
        pc_odom = transform_points(pc_body, R_odom_body, t_odom_body)

        msgs.append(pc_odom)
        ts.append(stamp_sec)

    ts = np.asarray(ts, dtype=np.float64)

    s = (
        "Summary\n"
        "topic: {}\n"
        "count: {}\n"
        "min timestamp: {}\n"
        "max timestamp: {}\n"
        "average time between frames: {}\n"
        "source frame: {}\n"
        "target frame: {}\n"
        "mean abs odom sync error: {}\n"
        "max abs odom sync error: {}\n"
    ).format(
        topic,
        len(ts),
        ts[0],
        ts[-1],
        (ts[-1] - ts[0]) / len(ts),
        src_frame,
        target_frame,
        float(np.mean(odom_sync_errors)),
        float(np.max(odom_sync_errors)),
    )
    print(s)

    return msgs, ts


def load_lidar_scand_global(
    bag,
    topic,
    topic_type,
    odom_ts,
    odom_positions,
    odom_quaternions,
    body_to_lidar_translation,
    body_to_lidar_quaternion,
    target_frame="odom",
):
    if topic_type == "sensor_msgs/PointCloud2":
        return load_lidar_scand_pointcloud2_global(
            bag=bag,
            topic=topic,
            odom_ts=odom_ts,
            odom_positions=odom_positions,
            odom_quaternions=odom_quaternions,
            body_to_lidar_translation=body_to_lidar_translation,
            body_to_lidar_quaternion=body_to_lidar_quaternion,
            target_frame=target_frame,
        )

    if topic_type == "velodyne_msgs/VelodyneScan":
        return load_lidar_scand_packets_global(
            bag=bag,
            topic=topic,
            odom_ts=odom_ts,
            odom_positions=odom_positions,
            odom_quaternions=odom_quaternions,
            body_to_lidar_translation=body_to_lidar_translation,
            body_to_lidar_quaternion=body_to_lidar_quaternion,
            target_frame=target_frame,
        )

    raise NotImplementedError(
        f"Unsupported SCAND 3D lidar type {topic_type} on topic {topic}"
    )

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

def load_2D_lidar(bag, topic):
    msgs, ts = [], []
    src_frame = None

    for _, msg, _ in bag.read_messages(topics=[topic]):
        src_frame = msg.header.frame_id

        ranges = np.array(msg.ranges)
        ranges[~np.isfinite(ranges)] = msg.range_max
        intensities = np.array(msg.intensities)
        scan = {"ranges": ranges, "intensities": intensities}
        msgs.append(scan)
        ts.append(msg.header.stamp.to_sec())

    ts = np.array(ts, dtype=np.float64)

    s = (
        "Summary\n"
        "topic: {}\n"
        "count: {}\n"
        "min timestamp: {}\n"
        "max timestamp: {}\n"
        "average time between frames: {}\n"
        "source frame: {}\n"
    ).format(
        topic,
        len(ts),
        ts[0],
        ts[-1],
        (ts[-1] - ts[0]) / len(ts),
        src_frame,
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


def extract_lidar_from_rosbag(bag_path, out_dirs, args, det_gt_timestamp_folder=None, overwrite=False):
    """Extract and save combined laser scan from rosbag. Existing files will be overwritten"""

    if args.dataset in ["JRDB", "Crowdbot", "SCAND"]:
        target_frame = "odom"
    else:
        target_frame = "camera_init"

    output_first_path = os.path.join(out_dirs[0], "00000.npy")
    if not overwrite and os.path.exists(output_first_path):
        print(f"Lidar for {bag_path} already exists, not overwriting")
        return

    with rosbag.Bag(bag_path) as bag:
        tf_buffer = None
        scand_cfg = None
        scand_odom = None

        actual_topics_3D = list(args.topics_3D) if args.topics_3D is not None else []
        actual_topics_2D = list(args.topics_2D) if args.topics_2D is not None else None

        if args.dataset == "SCAND":
            robot_type = infer_scand_robot_type_from_bag_path(bag_path)
            scand_cfg = resolve_scand_topics(bag, robot_type)
            scand_odom = load_scand_odom(bag, scand_cfg["odom_topic"])

            actual_topics_3D = [scand_cfg["topic_3d"]]
            actual_topics_2D = [scand_cfg["topic_2d"]] if scand_cfg["topic_2d"] is not None else None
        else:
            tf_buffer = get_tf_tree(bag)

        topic_3D_msgs_list = []
        nonground_3D_msgs_list = []
        topic_2D_msgs_list = []
        ts_list = []
        min_ts_list = []

        num_topics_3D = len(actual_topics_3D)
        num_topics_nonground = 0
        num_topics_2D = 0 if actual_topics_2D is None else len(actual_topics_2D)

        for topic_3D in actual_topics_3D:
            if args.dataset == "SCAND":
                topic_3D_msgs, topic_3D_ts = load_lidar_scand_global(
                    bag=bag,
                    topic=topic_3D,
                    topic_type=scand_cfg["topic_3d_type"],
                    odom_ts=scand_odom["timestamp"],
                    odom_positions=scand_odom["position"],
                    odom_quaternions=scand_odom["quaternion"],
                    body_to_lidar_translation=scand_cfg["body_to_lidar_translation"],
                    body_to_lidar_quaternion=scand_cfg["body_to_lidar_quaternion"],
                    target_frame=target_frame,
                )
            else:
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

        if args.nonground_topics_3D is not None:
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

        if actual_topics_2D is not None:
            for topic_2D in actual_topics_2D:
                laser_msgs, laser_ts = load_2D_lidar(bag, topic_2D)
                topic_2D_msgs_list.append(laser_msgs)
                ts_list.append(laser_ts)
                min_ts_list.append(laser_ts.min())

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

    if args.dataset == "JRDB":
        frames_pc_path = os.path.join(det_gt_timestamp_folder, "frames_pc.json")
        with open(frames_pc_path, "r") as file:
            frames_data = json.load(file)

        sync_ts = []
        for frame in frames_data["data"]:
            frame_timestamp = frame["timestamp"] - offset
            if sync_t0 <= frame_timestamp <= sync_t1:
                sync_ts.append(frame_timestamp)

        sync_ts = np.array(sync_ts, dtype=np.float64)
    else:
        sync_ts = np.arange(start=sync_t0, step=sync_dt, stop=sync_t1, dtype=np.float64)

    if args.dataset == "JRDB":
        sync_start_time = sync_ts[0] + offset
        t0_secs = int(sync_start_time)
        t0_nsecs = int((sync_start_time - t0_secs) * 1e9)
        sync_t0_msg = rospy.Time(t0_secs, t0_nsecs)
        JRDB_pos_offset = get_starting_pos_offset_JRDB(bag, tf_buffer, sync_t0_msg)
        topic_3D_msgs_list = [
            [pc_xyz - JRDB_pos_offset for pc_xyz in topic_3d_msgs]
            for topic_3d_msgs in topic_3D_msgs_list
        ]
    elif args.dataset == "SCAND":
        sync_start_time = sync_ts[0] + offset  # absolute timestamp of first synced lidar frame
        SCAND_pos_offset = get_starting_pos_offset_SCAND(
            scand_odom,
            sync_start_time,
        )

        topic_3D_msgs_list = [
            [pc_xyz - SCAND_pos_offset for pc_xyz in topic_3d_msgs]
            for topic_3d_msgs in topic_3D_msgs_list
        ]

        if args.nonground_topics_3D is not None:
            nonground_3D_msgs_list = [
                [pc_xyz - SCAND_pos_offset for pc_xyz in topic_3d_msgs]
                for topic_3d_msgs in nonground_3D_msgs_list
            ]

    def get_sync_inds(ts, sync_ts):
        d = np.abs(sync_ts.reshape(-1, 1) - ts.reshape(1, -1))
        return np.argmin(d, axis=1)

    sync_inds_list = []
    for np_ts in ts_list:
        sync_inds_list.append(get_sync_inds(np_ts, sync_ts))

    for frame_id, idx_tuple in enumerate(zip(*sync_inds_list)):
        file_path_3D = os.path.join(out_dirs[0], "{0:05d}".format(frame_id))

        individual_pc_list = [
            lidar_3D_msgs[idx_tuple[ind]]
            for ind, lidar_3D_msgs in enumerate(topic_3D_msgs_list)
        ]
        pc = np.concatenate(individual_pc_list, axis=0)

        if args.compressed:
            save_lidar(file_path_3D, pc, library="open3d", overwrite=overwrite)
        else:
            save_lidar(file_path_3D, pc, overwrite=overwrite)

        if args.nonground_topics_3D is not None:
            file_path_nonground = os.path.join(out_dirs[1], "{0:05d}".format(frame_id))
            individual_nonground_pc_list = [
                lidar_nonground_msgs[idx_tuple[num_topics_3D + ind]]
                for ind, lidar_nonground_msgs in enumerate(nonground_3D_msgs_list)
            ]
            pc = np.concatenate(individual_nonground_pc_list, axis=0)
            if args.compressed:
                save_lidar(file_path_nonground, pc, library="open3d", overwrite=overwrite)
            else:
                save_lidar(file_path_nonground, pc, overwrite=overwrite)

        if actual_topics_2D is not None:
            file_path_2D = os.path.join(out_dirs[2], "{0:05d}".format(frame_id))
            lidar_2D_dict = {}
            for ind in range(num_topics_2D):
                lidar_2D_dict[actual_topics_2D[ind]] = topic_2D_msgs_list[ind][
                    idx_tuple[num_topics_3D + num_topics_nonground + ind]
                ]
            save_lidar(file_path_2D, lidar_2D_dict, overwrite=overwrite)

    id_list, ts_out_list = [], []
    for frame_id, ts in enumerate(sync_ts):
        id_list.append(frame_id)
        ts_out_list.append(ts + offset)

    lidar_stamped_dict = {
        "timestamp": np.asarray(ts_out_list, dtype=np.float64),
        "id": np.asarray(id_list, dtype=np.int32),
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

def extract_lidar_from_jrdb_test(seq_name, out_dirs, args, overwrite=False):
    """
    Extract JRDB test lidar frames from upper/lower PCD files + timestamps + odometry CSV.
    Saves the same processed output format as JRDB train:
      - lidars/<seq>/00000.npy ...
      - source_data/timestamp/<seq>_stamped.npy
    """

    output_first_path = os.path.join(out_dirs[0], "00000.npy")
    if not overwrite and os.path.exists(output_first_path):
        print(f"Lidar for {seq_name} already exists, not overwriting")
        return

    upper_seq_dir = os.path.join(
        args.jrdb_test_root, "pointclouds", "upper_velodyne", seq_name
    )
    lower_seq_dir = os.path.join(
        args.jrdb_test_root, "pointclouds", "lower_velodyne", seq_name
    )
    frames_pc_path = os.path.join(
        args.jrdb_test_root, "timestamps", seq_name, "frames_pc.json"
    )
    odom_csv_path = os.path.join(
        args.jrdb_test_odom_root, seq_name + ".csv"
    )

    if not os.path.isdir(upper_seq_dir):
        raise FileNotFoundError(f"Missing JRDB test upper lidar dir: {upper_seq_dir}")
    if not os.path.isdir(lower_seq_dir):
        raise FileNotFoundError(f"Missing JRDB test lower lidar dir: {lower_seq_dir}")
    if not os.path.isfile(frames_pc_path):
        raise FileNotFoundError(f"Missing JRDB test frames_pc.json: {frames_pc_path}")
    if not os.path.isfile(odom_csv_path):
        raise FileNotFoundError(f"Missing JRDB test odometry csv: {odom_csv_path}")

    frame_files, frame_timestamps = load_jrdb_test_frames_pc_json(frames_pc_path)
    jrdb_test_odom = load_jrdb_test_odom_csv(odom_csv_path)

    topic_3D_msgs = []
    topic_3D_ts = []
    odom_sync_errors = []

    for frame_file, stamp in zip(frame_files, frame_timestamps):
        upper_pcd = os.path.join(upper_seq_dir, frame_file)
        lower_pcd = os.path.join(lower_seq_dir, frame_file)

        if (not os.path.isfile(upper_pcd)) or (not os.path.isfile(lower_pcd)):
            continue

        pc_base = load_merge_jrdb_test_frame_to_base(upper_pcd, lower_pcd)

        odom_idx = nearest_timestamp_index(jrdb_test_odom["timestamp"], stamp)
        odom_sync_errors.append(abs(jrdb_test_odom["timestamp"][odom_idx] - stamp))

        R_odom_base = quat_xyzw_to_rotmat(jrdb_test_odom["quaternion"][odom_idx])
        t_odom_base = jrdb_test_odom["position"][odom_idx]

        pc_odom = transform_points(pc_base, R_odom_base, t_odom_base)

        topic_3D_msgs.append(pc_odom)
        topic_3D_ts.append(stamp)

    if len(topic_3D_msgs) == 0:
        raise RuntimeError(f"No valid JRDB test point clouds found for sequence {seq_name}")

    topic_3D_ts = np.asarray(topic_3D_ts, dtype=np.float64)

    s = (
        "Summary\n"
        "topic: jrdb_test_merged_pointclouds\n"
        "count: {}\n"
        "min timestamp: {}\n"
        "max timestamp: {}\n"
        "average time between frames: {}\n"
        "source frame: base_chassis_link\n"
        "target frame: odom\n"
        "mean abs odom sync error: {}\n"
        "max abs odom sync error: {}\n"
    ).format(
        len(topic_3D_ts),
        topic_3D_ts[0],
        topic_3D_ts[-1],
        (topic_3D_ts[-1] - topic_3D_ts[0]) / len(topic_3D_ts),
        float(np.mean(odom_sync_errors)),
        float(np.max(odom_sync_errors)),
    )
    print(s)

    # Match JRDB train convention: remove only initial XY odom offset, keep Z untouched
    JRDB_TEST_pos_offset = get_starting_pos_offset_JRDB_TEST(
        jrdb_test_odom,
        topic_3D_ts[0],
    )
    topic_3D_msgs = [
        pc_xyz - JRDB_TEST_pos_offset
        for pc_xyz in topic_3D_msgs
    ]

    # Save per-frame point clouds
    for frame_id, pc in enumerate(topic_3D_msgs):
        file_path_3D = os.path.join(out_dirs[0], "{0:05d}".format(frame_id))
        if args.compressed:
            save_lidar(file_path_3D, pc, library="open3d", overwrite=overwrite)
        else:
            save_lidar(file_path_3D, pc, overwrite=overwrite)

    id_list = np.arange(len(topic_3D_ts), dtype=np.int32)
    lidar_stamped_dict = {
        "timestamp": topic_3D_ts.astype(np.float64),
        "id": id_list,
    }

    s = (
        "Summary\n"
        "topic: synced frames\n"
        "count: {}\n"
        "average time between frames: {}\n"
    ).format(
        len(topic_3D_ts),
        (topic_3D_ts[-1] - topic_3D_ts[0]) / len(topic_3D_ts),
    )
    print(s)

    return lidar_stamped_dict

class Settings:
    """Runtime settings shared by the extraction functions."""

    def __init__(
        self,
        dataset,
        config_path,
        folder,
        topics_3D=None,
        nonground_topics_3D=None,
        topics_2D=None,
        overwrite=False,
        compressed=False,
        jrdb_test_root=None,
        jrdb_test_odom_root=None,
        jrdb_train_timestamps_root=None,
    ):
        self.dataset = dataset
        self.config_path = config_path
        self.folder = folder
        self.topics_3D = topics_3D
        self.nonground_topics_3D = nonground_topics_3D
        self.topics_2D = topics_2D
        self.overwrite = overwrite
        self.compressed = compressed
        self.jrdb_test_root = jrdb_test_root
        self.jrdb_test_odom_root = jrdb_test_odom_root
        self.jrdb_train_timestamps_root = jrdb_train_timestamps_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and synchronize lidar data for CrowdBot, JRDB, JRDB test, DAAV, or SCAND."
    )
    parser.add_argument("--dataset", required=True, choices=["Crowdbot", "JRDB", "JRDB_TEST", "Daav", "SCAND"])
    parser.add_argument("--config", dest="config_path", required=True, help="Dataset path YAML.")
    parser.add_argument("--folder", action="append", required=True, help="Logical output folder; repeat for multiple folders.")
    parser.add_argument("--topic-3d", dest="topics_3D", action="append", help="3D lidar topic; repeat as needed.")
    parser.add_argument("--nonground-topic-3d", dest="nonground_topics_3D", action="append")
    parser.add_argument("--topic-2d", dest="topics_2D", action="append", help="2D lidar topic; repeat as needed.")
    parser.add_argument("--jrdb-train-timestamps-root", help="JRDB train timestamps directory containing one folder per sequence.")
    parser.add_argument("--jrdb-test-root", help="JRDB test dataset root containing pointclouds/ and timestamps/.")
    parser.add_argument("--jrdb-test-odom-root", help="Directory containing SteamLO <sequence>.csv files.")
    parser.add_argument("--compressed", action="store_true", help="Write PCD files instead of NumPy arrays.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dataset == "JRDB" and not args.jrdb_train_timestamps_root:
        parser.error("--jrdb-train-timestamps-root is required for JRDB.")
    if args.dataset == "JRDB_TEST" and (not args.jrdb_test_root or not args.jrdb_test_odom_root):
        parser.error("JRDB_TEST requires --jrdb-test-root and --jrdb-test-odom-root.")
    if args.dataset not in {"SCAND", "JRDB_TEST"} and not args.topics_3D:
        parser.error("At least one --topic-3d is required for this dataset.")
    return args


def process_folder(args, folder):
        settings = Settings(
            dataset=args.dataset,
            config_path=args.config_path,
            folder=folder,
            topics_3D=args.topics_3D,
            nonground_topics_3D=args.nonground_topics_3D,
            topics_2D=args.topics_2D,
            overwrite=args.overwrite,
            compressed=args.compressed,
            jrdb_test_root=args.jrdb_test_root,
            jrdb_test_odom_root=args.jrdb_test_odom_root,
            jrdb_train_timestamps_root=args.jrdb_train_timestamps_root,
        )

        cb_data = CrowdBotDatabase(settings.folder, config=settings.config_path)

        lidar_file_dir = cb_data.lidar_dir
        lidar_file_nonground_dir = cb_data.lidar_nonground_dir
        lidar_file_2D_dir = cb_data.lidar_2D_dir
        lidar_stamp_dir = os.path.join(cb_data.source_data_dir, "timestamp")
        if not os.path.exists(lidar_stamp_dir):
            os.makedirs(lidar_stamp_dir)

        # -----------------------------
        # JRDB TEST branch (no rosbags)
        # -----------------------------
        if settings.dataset == 'JRDB_TEST':
            upper_root = os.path.join(settings.jrdb_test_root, 'pointclouds', 'upper_velodyne')
            seq_names = [
                d for d in os.listdir(upper_root)
                if os.path.isdir(os.path.join(upper_root, d)) and not d.startswith('.')
            ]
            seq_names.sort()

            print("Starting extracting lidar files from {} JRDB test sequences!".format(len(seq_names)))

            for idx, seq_name in enumerate(seq_names):
                out_dir = os.path.join(lidar_file_dir, seq_name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                out_nonground_dir = os.path.join(lidar_file_nonground_dir, seq_name)
                if not os.path.exists(out_nonground_dir):
                    os.makedirs(out_nonground_dir)

                out_2D_dir = os.path.join(lidar_file_2D_dir, seq_name)
                if not os.path.exists(out_2D_dir):
                    os.makedirs(out_2D_dir)

                out_dirs = [out_dir, out_nonground_dir, out_2D_dir]

                print("({}/{}): {}".format(idx + 1, len(seq_names), seq_name))

                lidar_stamped_dict = extract_lidar_from_jrdb_test(
                    seq_name,
                    out_dirs,
                    settings,
                    overwrite=settings.overwrite,
                )
                if lidar_stamped_dict is None:
                    continue

                print(
                    "lidar_stamped_dict with {} frames".format(
                        len(lidar_stamped_dict['timestamp'])
                    )
                )

                stamp_file_path = os.path.join(lidar_stamp_dir, seq_name + "_stamped.npy")
                np.save(stamp_file_path, lidar_stamped_dict)

            return

        # -----------------------------
        # Existing rosbag-based branch
        # -----------------------------
        rosbag_dir = os.path.join(cb_data.bagbase_dir, settings.folder)
        print(rosbag_dir)

        if settings.dataset == 'Crowdbot':
            bag_files = list(filter(processed_Crowdbot_bag_file_filter, os.listdir(rosbag_dir)))
        elif settings.dataset == 'JRDB':
            bag_files = list(filter(bag_file_filter, os.listdir(rosbag_dir)))
        else:
            bag_files = list(filter(bag_file_filter, os.listdir(rosbag_dir)))

        print("Starting extracting lidar files from {} rosbags!".format(len(bag_files)))

        for idx, bf in enumerate(bag_files):
            print(bf)
            if bf.find('filtered') == -1 or bf.find('lidar_odom') != -1:
                bag_path = os.path.join(rosbag_dir, bf)
                bag_name = bf.split(".")[0]
                det_gt_timestamp_folder = None
                if settings.dataset == "JRDB":
                    det_gt_timestamp_folder = os.path.join(
                        settings.jrdb_train_timestamps_root,
                        bag_name.replace('_all_transforms', ''),
                    )

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

                lidar_stamped_dict = extract_lidar_from_rosbag(
                    bag_path,
                    out_dirs,
                    settings,
                    overwrite=settings.overwrite,
                    det_gt_timestamp_folder=det_gt_timestamp_folder,
                )
                if lidar_stamped_dict is None:
                    continue

                print(
                    "lidar_stamped_dict with {} frames".format(
                        len(lidar_stamped_dict['timestamp'])
                    )
                )

                stamp_file_path = os.path.join(lidar_stamp_dir, bag_name + "_stamped.npy")
                np.save(stamp_file_path, lidar_stamped_dict)


def main():
    args = parse_args()
    for folder in args.folder:
        process_folder(args, folder)


if __name__ == "__main__":
    main()

import copy
import cv2
import json
import numpy as np
import os
from ._pypcd import point_cloud_from_path

# NOTE: Don't use open3d to load point cloud since it spams the console. Setting
# verbosity level does not solve the problem
# https://github.com/intel-isl/Open3D/issues/1921
# https://github.com/intel-isl/Open3D/issues/884


__all__ = ["JRDBHandleDet2D"]


class JRDBHandleDet2D:
    def __init__(self, data_dir, split, sequences=None, exclude_sequences=None, num_scans=10, scan_stride=1):

        self.__num_scans = num_scans
        self.__scan_stride = scan_stride

        data_dir = (
            os.path.join(data_dir, "train_dataset")
            if split == "train" or split == "val"
            else os.path.join(data_dir, "test_dataset")
        )

        self.data_dir = data_dir
        self.timestamp_dir = os.path.join(data_dir, "timestamps")
        self.pc_label_dir = os.path.join(data_dir, "labels", "labels_3d")

        sequence_names = (
            os.listdir(os.path.join(data_dir, "timestamps"))
            if sequences is None
            else sequences
        )

        # NOTE it is important to sort the return of os.listdir, since its order
        # changes for different file system.
        sequence_names.sort()

        if exclude_sequences is not None:
            sequence_names = [s for s in sequence_names if s not in exclude_sequences]

        self.sequence_names = sequence_names

        self.sequence_handle = []
        self._sequence_beginning_inds = [0]
        self.__flat_inds_sequence = []
        self.__flat_inds_frame = []
        for seq_idx, seq_name in enumerate(self.sequence_names):
            self.sequence_handle.append(_SequenceHandle2D(self.data_dir, seq_name))

            # build a flat index for all sequences and frames
            sequence_length = len(self.sequence_handle[-1])
            self.__flat_inds_sequence += sequence_length * [seq_idx]
            self.__flat_inds_frame += range(sequence_length)

            self._sequence_beginning_inds.append(
                self._sequence_beginning_inds[-1] + sequence_length
            )

    def __len__(self):
        return len(self.__flat_inds_frame)

    def __getitem__(self, idx):

        idx_sq = self.__flat_inds_sequence[idx]
        idx_fr = self.__flat_inds_frame[idx]

        frame_dict, pc_anns_url, = self.sequence_handle[idx_sq][idx_fr]

        laser_data = self._load_consecutive_lasers(frame_dict["laser_frame"]["url"])

        frame_dict.update(
            {
                "frame_id": int(frame_dict["frame_id"]),
                "sequence": self.sequence_handle[idx_sq].sequence,
                "first_frame": idx_fr == 0,
                "dataset_idx": idx,
                "laser_data": laser_data,
                "laser_grid": np.linspace(
                    -np.pi, np.pi, laser_data.shape[1], dtype=np.float32
                ),
                "laser_z": -0.5 * np.ones(laser_data.shape[1], dtype=np.float32),
            }
        )

        if pc_anns_url is not None:
            frame_dict["label_str"] = self.load_label(pc_anns_url)

        return frame_dict

    @staticmethod
    def box_is_on_ground(jrdb_ann_dict):
        bottom_h = float(jrdb_ann_dict["box"]["cz"]) - 0.5 * float(
            jrdb_ann_dict["box"]["h"]
        )

        return bottom_h < -0.69  # value found by examining dataset

    @property
    def sequence_beginning_inds(self):
        return copy.deepcopy(self._sequence_beginning_inds)

    def _load_pointcloud(self, url):
        """Load a point cloud given file url.

        Returns:
            pc (np.ndarray[3, N]):
        """
        # pcd_load =
        # o3d.io.read_point_cloud(os.path.join(self.data_dir, url), format='pcd')
        # return np.asarray(pcd_load.points, dtype=np.float32)
        pc = point_cloud_from_path(os.path.join(self.data_dir, url)).pc_data
        # NOTE: redundent copy, ok for now
        pc = np.array([pc["x"], pc["y"], pc["z"]], dtype=np.float32)
        return pc
    
    def load_label(self, url):
        with open(url, "r") as f:
            s = f.read()
        return s
    
    def _load_consecutive_lasers(self, url):
        """Load current and previous consecutive laser scans.

        Args:
            url (str): file url of the current scan

        Returns:
            pc (np.ndarray[self.num_scan, N]): Forward in time with increasing
                row index, i.e. the latest scan is pc[-1]
        """
        fpath = os.path.dirname(url)
        current_frame_idx = int(os.path.basename(url).split(".")[0])
        frames_list = []
        for del_idx in reversed(range(self.__num_scans)):
            frame_idx = max(0, current_frame_idx - del_idx * self.__scan_stride)
            url = os.path.join(fpath, str(frame_idx).zfill(6) + ".txt")
            frames_list.append(self._load_laser(url))

        return np.stack(frames_list, axis=0)

    def _load_laser(self, url):
        """Load a laser given file url.

        Returns:
            pc (np.ndarray[N, ]):
        """
        return np.loadtxt(os.path.join(self.data_dir, url), dtype=np.float32)


class _SequenceHandle2D:
    def __init__(self, data_dir, sequence):
        self.sequence = sequence
        self._frames = []
        # load frames of the sequence
        timestamp_dir = os.path.join(data_dir, "timestamps")
        fname = os.path.join(timestamp_dir, self.sequence, "frames_pc_im_laser.json")
        with open(fname, "r") as f:
            """
            list[dict]. Each dict has following keys:
                pc_frame: dict with keys frame_id, pointclouds, laser, timestamp
                im_frame: same as above
                laser_frame: dict with keys url, name, timestamp
                frame_id: same as pc_frame["frame_id"]
                timestamp: same as pc_frame["timestamp"]
            """
            frames = json.load(f)["data"]
        # labels
        label_dir = os.path.join(data_dir, "labels_kitti", sequence)
        if os.path.exists(label_dir):
            labeled_frames = [f.split(".")[0] for f in os.listdir(label_dir)]


        # find out which frames has 3D annotation
        for frame in frames:
            pc_frame = frame["pc_frame"]["pointclouds"][1]["url"].split("/")[-1].split(".")[0]
            if pc_frame in labeled_frames:
                self._frames.append(frame)
        
        self._label_dir = label_dir
        self._load_labels = os.path.exists(label_dir)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        # NOTE It's important to use a copy as the return dict, otherwise the
        # original dict in the data handle will be corrupted
        frame = copy.deepcopy(self._frames[idx])

        pc_frame = os.path.basename(frame["pc_frame"]["pointclouds"][0]["url"]).split('.')[0] # Remove the .pcd while working with kitti labels
        pc_anns_url = (
            os.path.join(self._label_dir, pc_frame + ".txt") if self._load_labels else None
        )

        return frame, pc_anns_url,

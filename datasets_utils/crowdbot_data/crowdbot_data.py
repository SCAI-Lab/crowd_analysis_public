#!/usr/bin/env python3

import os
import yaml
import numpy as np
from pathlib import Path

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
CROWDBOT_EVAL_TOOLKIT_DIR = os.path.abspath(os.path.join(curr_dir_path, "..", ".."))


def read_yaml(yaml_file):
    with open(yaml_file, encoding='utf-8') as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data


class CrowdbotExpParam:
    """Class for extracting experiment parameter according to date and type"""

    def __init__(self, file, encoding="utf-8"):
        self.file = file
        self.data = read_yaml(self.file)

    def get_params(self, date, control_type):
        return self.data[date][control_type]


class CrowdBotData(object):
    """Class for extracting experiment parameter according to date and type"""

    DEFAULT_CONFIG_PATH = os.path.join(CROWDBOT_EVAL_TOOLKIT_DIR, 'data/data_path.yaml')

    def __init__(self, config=DEFAULT_CONFIG_PATH):
        self.config = config
        data_config = read_yaml(self.config)
        self.bagbase_dir = data_config['bagbase_dir']
        self.outbase_dir = data_config['outbase_dir']

    def write_yaml(self, data):
        """
        :param yaml_path
        :param data
        :param encoding
        """
        with open(self.config, 'w', encoding='utf-8') as f:
            yaml.dump(data, stream=f, allow_unicode=True)


class CrowdBotDatabase(CrowdBotData):
    def __init__(self, classdir, config=None):
        if config is None:
            super(CrowdBotDatabase, self).__init__()
        else:
            super(CrowdBotDatabase, self).__init__(config)
        
        data_config = read_yaml(self.config)
        self.bagbase_dir = data_config['bagbase_dir']
        self.outbase_dir = data_config['outbase_dir']

        # Store the classdir list or convert it to a list if it's a single item
        self.classdirs = classdir if isinstance(classdir, list) else [classdir]

        # Initialize original single directory attributes as None
        self.lidar_dir = None
        self.lidar_nonground_dir = None
        self.lidar_2D_dir = None
        self.alg_res_dir = None
        self.dets_dir = None
        self.dets_2D_dir = None
        self.trks_dir = None
        self.trks_2D_dir = None
        self.source_data_dir = None
        self.ped_data_dir = None
        self.metrics_dir = None
        self.media_dir = None

        # Initialize lists for each directory-related attribute
        self.lidar_dirs = []
        self.lidar_nonground_dirs = []
        self.lidar_2D_dirs = []
        self.alg_res_dirs = []
        self.dets_dirs = []
        self.dets_2D_dirs = []
        self.trks_dirs = []
        self.trks_2D_dirs = []
        self.source_data_dirs = []
        self.ped_data_dirs = []
        self.metrics_dirs = []
        self.media_dirs = []

        # Initialize sequences and frames
        self.seqs = []
        self.frames = []
        self.frames_bottom = []
        self.frames_top = []

        # Process each directory in the classdir list
        for dir_item in self.classdirs:
            # Check if dir_item is a directory and not a hidden item
            if os.path.basename(dir_item).startswith('.'):
                continue
            self.process_directory(dir_item)

    def process_directory(self, classdir):
        data_processed = classdir + "_processed"
        data_processed_dir = os.path.join(self.outbase_dir, data_processed)
        if not os.path.exists(data_processed_dir):
            os.makedirs(data_processed_dir)

        # Base directories
        lidar_dir = os.path.join(data_processed_dir, "lidars")
        if not os.path.exists(lidar_dir):
            os.makedirs(lidar_dir)
        
        lidar_nonground_dir = os.path.join(data_processed_dir, "lidars_nonground")
        if not os.path.exists(lidar_nonground_dir):
            os.makedirs(lidar_nonground_dir)
        
        lidar_2D_dir = os.path.join(data_processed_dir, "lidars_2D")
        if not os.path.exists(lidar_2D_dir):
            os.makedirs(lidar_2D_dir)
        
        alg_res_dir = os.path.join(data_processed_dir, "alg_res")
        if not os.path.exists(alg_res_dir):
            os.makedirs(alg_res_dir)
        
        dets_dir = os.path.join(alg_res_dir, "detections")
        if not os.path.exists(dets_dir):
            os.makedirs(dets_dir)
        
        dets_2D_dir = os.path.join(alg_res_dir, "detections_2D")
        if not os.path.exists(dets_2D_dir):
            os.makedirs(dets_2D_dir)
        
        trks_dir = os.path.join(alg_res_dir, "tracks")
        if not os.path.exists(trks_dir):
            os.makedirs(trks_dir)
        
        trks_2D_dir = os.path.join(alg_res_dir, "tracks_2D")
        if not os.path.exists(trks_2D_dir):
            os.makedirs(trks_2D_dir)

        source_data_dir = os.path.join(data_processed_dir, "source_data")
        if not os.path.exists(source_data_dir):
            os.makedirs(source_data_dir)
        
        ped_data_dir = os.path.join(data_processed_dir, "ped_data")
        if not os.path.exists(ped_data_dir):
            os.makedirs(ped_data_dir)
        
        metrics_dir = os.path.join(data_processed_dir, "metrics")
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        
        media_dir = os.path.join(data_processed_dir, "media")
        if not os.path.exists(media_dir):
            os.makedirs(media_dir)

        # Collect sequences and frames
        seqs = [
            f
            for f in os.listdir(lidar_dir)
            if (os.path.isdir(os.path.join(lidar_dir, f)) and f[0] != '.')
        ]
        seqs.sort()

        for seq in seqs:
            self.seqs.append(seq)
            frs = [
                fr for fr in os.listdir(os.path.join(lidar_dir, seq))
                if not (fr.startswith('.') or fr.endswith('_top.npy') or fr.endswith('_bottom.npy'))
            ]
            frs_bottom = [
                fr for fr in os.listdir(os.path.join(lidar_dir, seq))
                if not fr.startswith('.') and fr.endswith('_bottom.npy')
            ]
            frs_top = [
                fr for fr in os.listdir(os.path.join(lidar_dir, seq))
                if not fr.startswith('.') and fr.endswith('_top.npy')
            ]
            frs.sort()
            frs_bottom.sort()
            frs_top.sort()
            self.frames.append(frs)
            self.frames_bottom.append(frs_bottom)
            self.frames_top.append(frs_top)
            
            # Append directories corresponding to the current sequence
            self.lidar_dirs.append(lidar_dir)
            self.lidar_nonground_dirs.append(lidar_nonground_dir)
            self.lidar_2D_dirs.append(lidar_2D_dir)
            self.alg_res_dirs.append(alg_res_dir)
            self.dets_dirs.append(dets_dir)
            self.dets_2D_dirs.append(dets_2D_dir)
            self.trks_dirs.append(trks_dir)
            self.trks_2D_dirs.append(trks_2D_dir)
            self.source_data_dirs.append(source_data_dir)
            self.ped_data_dirs.append(ped_data_dir)
            self.metrics_dirs.append(metrics_dir)
            self.media_dirs.append(media_dir)

        if len(self.classdirs) == 1:
            self.lidar_dir = lidar_dir
            self.lidar_nonground_dir = lidar_nonground_dir
            self.lidar_2D_dir = lidar_2D_dir
            self.alg_res_dir = alg_res_dir
            self.dets_dir = dets_dir
            self.dets_2D_dir = dets_2D_dir
            self.trks_dir = trks_dir
            self.trks_2D_dir = trks_2D_dir
            self.source_data_dir = source_data_dir
            self.ped_data_dir = ped_data_dir
            self.metrics_dir = metrics_dir
            self.media_dir = media_dir

    def nr_seqs(self):
        return len(self.seqs)

    def nr_frames(self, sq_idx):
        return len(self.frames[sq_idx])

    def __getitem__(self, sq_fr_idx):
        sq_idx, fr_idx = sq_fr_idx

        assert sq_idx < self.nr_seqs(), (
            "Sequence index out of range. "
            f"Requested {sq_idx}, maximum {self.nr_seqs()}."
        )
        assert fr_idx < self.nr_frames(sq_idx), (
            "Frame index out of range. "
            f"Requested {fr_idx}, maximum {self.nr_frames(sq_idx)}."
        )

        seq = self.seqs[sq_idx]
        fr = self.frames[sq_idx][fr_idx]

        l_path = os.path.join(self.lidar_dirs[sq_idx], seq, fr)
        lidar = np.load(l_path) if os.path.isfile(l_path) else None
        if lidar is not None:
            lidar = lidar.T

        l_nonground_path = os.path.join(self.lidar_nonground_dirs[sq_idx], seq, fr)
        lidar_nonground = np.load(l_nonground_path) if os.path.isfile(l_nonground_path) else None
        if lidar_nonground is not None:
            lidar_nonground = lidar_nonground.T

        l_2D_path = os.path.join(self.lidar_2D_dirs[sq_idx], seq, fr)
        lidar_2D = np.load(l_2D_path, allow_pickle=True).item() if os.path.isfile(l_2D_path) else None

        dnpy_all_path = os.path.join(self.dets_dirs[sq_idx], seq + ".npy")
        dnpy_gt_path = os.path.join(self.dets_dirs[sq_idx], seq + "_gt.npy")
        dnpy_all_far_path = os.path.join(self.dets_dirs[sq_idx], seq + "_far.npy")
        dnpy_2D_all_path = os.path.join(self.dets_2D_dirs[sq_idx], seq + ".npy")
        dnpy_2D_all_close_path = os.path.join(self.dets_2D_dirs[sq_idx], seq + "_close.npy")
        tnpy_all_path = os.path.join(self.trks_dirs[sq_idx], seq + ".npy")
        tnpy_gt_path = os.path.join(self.trks_dirs[sq_idx], seq + "_gt.npy")
        tnpy_all_merged_path = os.path.join(self.trks_dirs[sq_idx], seq + "_merged.npy")
        tnpy_2D_all_path = os.path.join(self.trks_2D_dirs[sq_idx], seq + ".npy")

        if os.path.exists(dnpy_all_path):
            with open(dnpy_all_path, "rb") as dnpy_all:
                det_all = np.load(dnpy_all, allow_pickle=True).item()
            dets_ = det_all[fr_idx]
            if dets_.size == 0:
                dets, dets_conf = np.empty((0,)), np.empty((0,))
            else:
                dets, dets_conf = dets_[:, :-1], dets_[:, -1]
        else:
            dets, dets_conf = None, None

        if os.path.exists(dnpy_gt_path):
            with open(dnpy_gt_path, "rb") as dnpy_all:
                det_all = np.load(dnpy_all, allow_pickle=True).item()
            dets_ = det_all[fr_idx]
            if dets_.size == 0:
                dets_gt = np.empty((0,)) 
            else:
                dets_gt= dets_
        else:
            dets_gt = None

        if os.path.exists(dnpy_all_far_path):
            with open(dnpy_all_far_path, "rb") as dnpy_far_all:
                det_all = np.load(dnpy_far_all, allow_pickle=True).item()
            dets_far_ = det_all[fr_idx]
            if dets_far_.size == 0:
                dets_far, dets_far_conf = np.empty((0,)), np.empty((0,))
            else:
                dets_far, dets_far_conf = dets_far_[:, :-1], dets_far_[:, -1]
        else:
            dets_far, dets_far_conf = None, None

        if os.path.exists(dnpy_2D_all_path):
            with open(dnpy_2D_all_path, "rb") as dnpy_2D_all:
                det_2D_all = np.load(dnpy_2D_all, allow_pickle=True).item()
            dets_2D_ = det_2D_all[fr_idx]
            if dets_2D_.size == 0:
                dets_2D, dets_2D_conf = np.empty((0,)), np.empty((0,))
            else:
                dets_2D, dets_2D_conf = dets_2D_[:, :-1], dets_2D_[:, -1]
        else:
            dets_2D, dets_2D_conf = None, None

        if os.path.exists(dnpy_2D_all_close_path):
            with open(dnpy_2D_all_close_path, "rb") as dnpy_2D_close_all:
                det_2D_close_all = np.load(dnpy_2D_close_all, allow_pickle=True).item()
            dets_2D_close_ = det_2D_close_all[fr_idx]
            if dets_2D_close_.size == 0:
                dets_2D_close, dets_2D_conf_close = np.empty((0,)), np.empty((0,))
            else:
                dets_2D_close, dets_2D_conf_close = dets_2D_close_[:, :-1], dets_2D_close_[:, -1]
        else:
            dets_2D_close, dets_2D_conf_close = None, None

        if os.path.exists(tnpy_all_path):
            with open(tnpy_all_path, "rb") as tnpy_all:
                trk_all = np.load(tnpy_all, allow_pickle=True).item()
            trks = trk_all[fr_idx]
        else:
            trks = None

        if os.path.exists(tnpy_gt_path):
            with open(tnpy_gt_path, "rb") as tnpy_all:
                trk_all = np.load(tnpy_all, allow_pickle=True).item()
            trks_gt = trk_all[fr_idx]
        else:
            trks_gt = None

        if os.path.exists(tnpy_all_merged_path):
            with open(tnpy_all_merged_path, "rb") as tnpy_merged_all:
                trk_merged_all = np.load(tnpy_merged_all, allow_pickle=True).item()
            trks_merged = trk_merged_all[fr_idx]
        else:
            trks_merged = None

        if os.path.exists(tnpy_2D_all_path):
            with open(tnpy_2D_all_path, "rb") as tnpy_2D_all:
                trk_2D_all = np.load(tnpy_2D_all, allow_pickle=True).item()
            trks_2D = trk_2D_all[fr_idx]
        else:
            trks_2D = None

        return lidar, lidar_nonground, lidar_2D, dets_gt, trks_gt, dets, dets_conf, trks, dets_2D, dets_2D_conf, trks_2D, dets_far, dets_far_conf, dets_2D_close, dets_2D_conf_close, trks_merged

    def get_all_tracks_for_sequence(self, seq_idx):
        assert seq_idx < self.nr_seqs(), (
            "Sequence index out of range. "
            f"Requested {seq_idx}, maximum {self.nr_seqs()}."
        )

        seq = self.seqs[seq_idx]
        num_frames = self.nr_frames(seq_idx)

        # Initialize lists to store tracks for all frames
        trks_all = []
        trks_gt_all = []
        trks_merged_all = []
        trks_2D_all = []

        tnpy_all_path = os.path.join(self.trks_dirs[seq_idx], seq + ".npy")
        tnpy_gt_path = os.path.join(self.trks_dirs[seq_idx], seq + "_gt.npy")
        tnpy_all_merged_path = os.path.join(self.trks_dirs[seq_idx], seq + "_merged.npy")
        tnpy_2D_all_path = os.path.join(self.trks_2D_dirs[seq_idx], seq + ".npy")

        # Load tracks for the entire sequence
        if os.path.exists(tnpy_all_path):
            with open(tnpy_all_path, "rb") as tnpy_all:
                trks_all = np.load(tnpy_all, allow_pickle=True).item()
        else:
            trks_all = [None] * num_frames

        if os.path.exists(tnpy_gt_path):
            with open(tnpy_gt_path, "rb") as tnpy_gt:
                trks_gt_all = np.load(tnpy_gt, allow_pickle=True).item()
        else:
            trks_gt_all = [None] * num_frames

        if os.path.exists(tnpy_all_merged_path):
            with open(tnpy_all_merged_path, "rb") as tnpy_merged:
                trks_merged_all = np.load(tnpy_merged, allow_pickle=True).item()
        else:
            trks_merged_all = [None] * num_frames

        if os.path.exists(tnpy_2D_all_path):
            with open(tnpy_2D_all_path, "rb") as tnpy_2D:
                trks_2D_all = np.load(tnpy_2D, allow_pickle=True).item()
        else:
            trks_2D_all = [None] * num_frames

        return trks_all, trks_gt_all, trks_merged_all, trks_2D_all
    
    def get_lidar_data(self, seq_idx, fr_idx):
        assert seq_idx < self.nr_seqs(), (
            "Sequence index out of range. "
            f"Requested {seq_idx}, maximum {self.nr_seqs()}."
        )
        assert fr_idx < self.nr_frames(seq_idx), (
            "Frame index out of range. "
            f"Requested {fr_idx}, maximum {self.nr_frames(seq_idx)}."
        )

        seq = self.seqs[seq_idx]
        fr = self.frames[seq_idx][fr_idx]

        l_path = os.path.join(self.lidar_dirs[seq_idx], seq, fr)
        lidar = np.load(l_path) if os.path.isfile(l_path) else None
        if lidar is not None:
            lidar = lidar.T

        l_nonground_path = os.path.join(self.lidar_nonground_dirs[seq_idx], seq, fr)
        lidar_nonground = np.load(l_nonground_path) if os.path.isfile(l_nonground_path) else None
        if lidar_nonground is not None:
            lidar_nonground = lidar_nonground.T

        l_2D_path = os.path.join(self.lidar_2D_dirs[seq_idx], seq, fr)
        lidar_2D = np.load(l_2D_path, allow_pickle=True).item() if os.path.isfile(l_2D_path) else None

        return lidar, lidar_nonground, lidar_2D
    
    def get_tracks_and_3d_lidars_data(self, seq_idx, fr_idx):
        assert seq_idx < self.nr_seqs(), (
            "Sequence index out of range. "
            f"Requested {seq_idx}, maximum {self.nr_seqs()}."
        )
        assert fr_idx < self.nr_frames(seq_idx), (
            "Frame index out of range. "
            f"Requested {fr_idx}, maximum {self.nr_frames(seq_idx)}."
        )

        seq = self.seqs[seq_idx]
        fr = self.frames[seq_idx][fr_idx]
        fr_bottom = self.frames_bottom[seq_idx][fr_idx]
        fr_top = self.frames_top[seq_idx][fr_idx]

        l_path = os.path.join(self.lidar_dirs[seq_idx], seq, fr)
        lidar = np.load(l_path) if os.path.isfile(l_path) else None
        if lidar is not None:
            lidar = lidar.T
        
        l_path_bottom = os.path.join(self.lidar_dirs[seq_idx], seq, fr_bottom)
        lidar_bottom = np.load(l_path_bottom) if os.path.isfile(l_path_bottom) else None
        if lidar_bottom is not None:
            lidar_bottom = lidar_bottom.T

        l_path_top = os.path.join(self.lidar_dirs[seq_idx], seq, fr_top)
        lidar_top = np.load(l_path_top) if os.path.isfile(l_path_top) else None
        if lidar_top is not None:
            lidar_top = lidar_top.T

        tnpy_all_path = os.path.join(self.trks_dirs[seq_idx], seq + ".npy")
        tnpy_gt_path = os.path.join(self.trks_dirs[seq_idx], seq + "_gt.npy")
        tnpy_all_merged_path = os.path.join(self.trks_dirs[seq_idx], seq + "_merged.npy")
        tnpy_2D_all_path = os.path.join(self.trks_2D_dirs[seq_idx], seq + ".npy")

        if os.path.exists(tnpy_all_path):
            with open(tnpy_all_path, "rb") as tnpy_all:
                trk_all = np.load(tnpy_all, allow_pickle=True).item()
            trks = trk_all[fr_idx]
        else:
            trks = None

        if os.path.exists(tnpy_gt_path):
            with open(tnpy_gt_path, "rb") as tnpy_all:
                trk_all = np.load(tnpy_all, allow_pickle=True).item()
            trks_gt = trk_all[fr_idx]
        else:
            trks_gt = None

        if os.path.exists(tnpy_all_merged_path):
            with open(tnpy_all_merged_path, "rb") as tnpy_merged_all:
                trk_merged_all = np.load(tnpy_merged_all, allow_pickle=True).item()
            trks_merged = trk_merged_all[fr_idx]
        else:
            trks_merged = None

        if os.path.exists(tnpy_2D_all_path):
            with open(tnpy_2D_all_path, "rb") as tnpy_2D_all:
                trk_2D_all = np.load(tnpy_2D_all, allow_pickle=True).item()
            trks_2D = trk_2D_all[fr_idx]
        else:
            trks_2D = None
        
        return lidar, lidar_top, lidar_bottom, trks_gt, trks_merged, trks, trks_2D

        



# filter the files with specific extensions
def bag_file_filter(f):
    if f[-4:] in [".bag"]:
        return True
    else:
        return False
    
# filter the files with specific extensions
def processed_Crowdbot_bag_file_filter(f):
    if f[-23:] in ["filtered_lidar_odom.bag"]:
        return True
    else:
        return False

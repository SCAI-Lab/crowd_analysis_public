import numpy as np
import torch
import sys
import os
import glob
from lidar_det.model.nets.dr_spaam import DrSpaam
import lidar_det.utils.utils_dr_spaam as u
from tqdm import tqdm

class DR_SPAAM_detector():
    
    def __init__(self, laser_fov_deg = 360, ct_stride = 1, ct_window_width = 1.0, ct_window_depth = 0.5, ct_padding_val = 29.99, num_cutout_pts = 56, cls_threshold=0.5, panoramic_scan=False, use_box=False, ckpt_file = 'ckpt_jrdb_ann_ft_dr_spaam_e20.pth'):
        self.laser_fov_deg = laser_fov_deg
        self.ct_stride = ct_stride
        self.ct_window_width = ct_window_width 
        self.ct_window_depth = ct_window_depth
        self.ct_padding_val = ct_padding_val
        self.num_cutout_pts = num_cutout_pts
        self.cls_threshold = cls_threshold
        self.panoramic_scan = panoramic_scan
        self.use_box = use_box
        self.ckpt_file = ckpt_file
        self.gpu_state = torch.cuda.is_available()
        self.scan_phi = None

    def prepare_model(self):
        ckpt = torch.load(self.ckpt_file)
        self.model = DrSpaam(num_pts=self.num_cutout_pts, panoramic_scan=self.panoramic_scan, use_box=self.use_box)
        self.model.load_state_dict(ckpt["model_state"])
        if self.gpu_state:
            self.model = self.model.cuda()
        
        self.half_fov_rad = 0.5 * np.deg2rad(self.laser_fov_deg)

    def detect(self, scan,):
        if self.scan_phi is None:
            self.scan_phi = np.linspace(-self.half_fov_rad, self.half_fov_rad, len(scan), dtype=float)
        ct = u.scans_to_cutout(
            scan[None, ...],
            self.scan_phi,
            stride=self.ct_stride,
            centered=True,
            fixed=True,
            window_width=self.ct_window_width,
            window_depth=self.ct_window_depth,
            num_cutout_pts=self.num_cutout_pts,
            padding_val=self.ct_padding_val,
            area_mode=True,
            )
         
        ct = torch.from_numpy(ct).float() # Convert to torch.tensor
        if self.gpu_state:
            ct = ct.cuda()

        with torch.no_grad():

            if self.use_box:
                pred_cls, pred_reg, pred_box, _ = self.model(ct.unsqueeze(dim=0), inference=True)
            else:
                pred_cls, pred_reg, _ = self.model(ct.unsqueeze(dim=0), inference=True)
        

        pred_cls = torch.sigmoid(pred_cls[0]).data.cpu().numpy()
        pred_reg = pred_reg[0].data.cpu().numpy()

        if self.use_box:
            pred_box = pred_box[0].data.cpu().numpy()
            JRDB_mean_length = 0.9
            JRDB_mean_width = 0.5
            lengths = np.exp(pred_box[:, 0])*JRDB_mean_length
            widths = np.exp(pred_box[:, 1])*JRDB_mean_width
            sinr = pred_box[:, 2]
            cosr = pred_box[:, 3]
            rots = np.arctan2(sinr, cosr)
            pred_box = np.stack((lengths, widths, rots), axis=-1)


        cls_threshold = self.cls_threshold

        pred_cls_scan = pred_cls
        pred_reg_scan = pred_reg
        if self.use_box:
            dets_xy, dets_cls, dets_box, instance_mask = u.nms_predicted_center(
                        scan[:: self.ct_stride],
                        self.scan_phi[:: self.ct_stride],
                        pred_cls_scan[:, 0],
                        pred_reg_scan,
                        pred_box=pred_box
                    )
        else:
            dets_xy, dets_cls, instance_mask = u.nms_predicted_center(
                        scan[:: self.ct_stride],
                        self.scan_phi[:: self.ct_stride],
                        pred_cls_scan[:, 0],
                        pred_reg_scan,
                    )
        
        cls_mask = dets_cls > cls_threshold
        dets_cls = dets_cls[cls_mask]
        dets_xy = dets_xy[cls_mask]
        if self.use_box:
            dets_box = dets_box[cls_mask]
        
        if self.use_box:
            return dets_xy, dets_cls, dets_box, instance_mask
        else:
            return dets_xy, dets_cls, instance_mask
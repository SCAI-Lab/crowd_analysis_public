import numpy as np
import torch
from torch.utils.data import Dataset

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

import lidar_det.utils.jrdb_transforms as jt
import lidar_det.utils.utils_box3d as ub3d

from .utils import collate_sparse_tensors, boxes_to_target

# from .utils import get_prediction_target
import lidar_det.utils.utils_dr_spaam as u_spaam

__all__ = [
    "JRDBDet3D",
    "JRDBDet2D",
    "NuScenesDet3D",
]


class _DatasetBase(Dataset):
    def __init__(self, data_dir, split, cfg):
        vs = cfg["voxel_size"]
        voxel_size = (
            np.array(vs, dtype=np.float32)
            if isinstance(vs, list)
            else np.array([vs, vs, vs], dtype=np.float32)
        )
        self._voxel_size = voxel_size.reshape(3, 1)
        self._voxel_offset = np.array([1e5, 1e5, 1e4], dtype=np.int32).reshape(3, 1)
        self._num_points = cfg["num_points"]
        self._na = cfg["num_anchors"]
        self._no = cfg["num_ori_bins"]
        self._canonical = cfg["canonical"]
        self._included_classes = cfg["included_classes"]
        self._additional_features = cfg["additional_features"]
        self._nsweeps = cfg["nsweeps"]
        self._augmentation = cfg["augmentation"]

        self.__training = "train" in split  # loss will be computed
        self.__split = split
        self.__handle = self._get_handle(data_dir, split)

    def _get_handle(self, data_dir, split):
        raise NotImplementedError

    def _get_data(self, data_dict, training=True):
        raise NotImplementedError

    def _do_augmentation(self, pc, boxes):
        # random scale
        scale_factor = np.random.uniform(0.95, 1.05)
        pc *= scale_factor

        # random rotation
        theta = np.random.uniform(0, 2 * np.pi)
        rot_mat = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        pc = rot_mat @ pc

        if boxes is not None and len(boxes) > 0:
            boxes[:, :6] *= scale_factor
            boxes[:, :3] = boxes[:, :3] @ rot_mat.T
            boxes[:, 6] += theta

        return pc, boxes

    @property
    def split(self):
        return self.__split  # used by trainer.py

    def __len__(self):
        return len(self.__handle)

    def __getitem__(self, idx):
        data_dict = self.__handle[idx]

        pc, boxes_gt, boxes_gt_cls, pc_offset, addi_feats = self._get_data(data_dict)

        if self.__training and self._augmentation:
            pc, boxes_gt = self._do_augmentation(pc, boxes_gt)

        # voxel coordinate
        pc_voxel = np.round(pc / self._voxel_size) + self._voxel_offset
        pc_voxel = pc_voxel.T
        _, inds, inverse_map = sparse_quantize(
            pc_voxel, return_index=True, return_inverse=True,
        )  # NOTE all this does is find indices of non-duplicating elements

        # print("Dataset len(inds), inds[0], inds[-1]")
        # print(len(inds), inds[0], inds[-1], type(inds), type(inverse_map))
   
        # for nuScenes with multisweep, only do prediction for keyframe voxels
        if "pc_dt" in data_dict:
            pc_dt = data_dict["pc_dt"]
            pc_kfmask = pc_dt == pc_dt.min()
            net_input_kfmask = pc_kfmask[inds]
            net_input_kfmask[inverse_map[pc_kfmask]] = 1
            # print("pc_kfmask", pc_kfmask.shape, pc_kfmask.sum())
            # print("net_input_kfmask", net_input_kfmask.shape, net_input_kfmask.sum())
        else:
            pc_kfmask = None
            net_input_kfmask = None

        # upper cap on memory consumption
        if self.__training and len(inds) > self._num_points:
            kept_inds = np.random.choice(len(inds), self._num_points, replace=False)
            inds = inds[kept_inds]
            if net_input_kfmask is not None:
                net_input_kfmask = net_input_kfmask[kept_inds]

        input_feat = (
            pc.T[inds]
            if addi_feats is None
            else np.concatenate((pc.T[inds], addi_feats.T[inds]), axis=1)
        )  # (N, C)
        net_input = SparseTensor(input_feat, pc_voxel[inds])

        # print("Coordinates shape:")
        # print(pc_voxel[inds].shape)
        # print(pc_voxel[0], pc_voxel[1], pc_voxel[2])
        # print(net_input.C.shape, net_input.F.shape)
        
        if net_input_kfmask is not None:
            net_input_kfmask = torch.from_numpy(net_input_kfmask).bool()

        params_dict = {
            "ave_lwh": self._ave_lwh,
            "canonical": self._canonical,
            "voxel_offset": self._voxel_offset,
            "voxel_size": self._voxel_size,
            "class_mapping": self._inds_to_cls,
            "dist_thresh": self._dist_thresh,
        }

        data_dict.update(
            {
                "net_input": net_input,
                "net_input_kfmask": net_input_kfmask,
                # "inverse_map": inverse_map,
                "points": pc,  # (3, N)
                "points_offset": pc_offset,  # (3,)
                "points_kfmask": pc_kfmask,  # (N, )
                "num_voxels": len(inds),
                "additional_features": addi_feats,  # (C, N) or None
                "boxes_gt": boxes_gt,  # (B, 7) or None
                "boxes_gt_cls": boxes_gt_cls,  # (B,) or None
                "params": params_dict,
            }
        )

        if not self.__training:
            return data_dict

        # assigning target for each class independently
        N = len(inds)
        A = self._na
        S = self._nc
        btmp = boxes_to_target(np.ones((1, 7)), self._ave_lwh[0], A, self._no)
        C = btmp.shape[-1]
        closest_box_inds = -1 * np.ones((N, S), dtype=np.int32)
        boxes_matched = np.zeros((N, S, 7), dtype=np.float32)
        boxes_encoded = np.zeros((N, A, S, C), dtype=np.float32)
        if boxes_gt is not None:
            for icls in range(self._nc):
                cmask = boxes_gt_cls == icls
                boxes_gt_c = boxes_gt[cmask]
                if len(boxes_gt_c) == 0:
                    continue
                closest_box_inds_c, _ = ub3d.find_closest_boxes(pc, boxes_gt_c)
                closest_box_inds_c = closest_box_inds_c[inds]
                boxes_matched_c = boxes_gt_c[closest_box_inds_c]
                closest_box_inds[:, icls] = closest_box_inds_c
                boxes_matched[:, icls, :] = boxes_matched_c
                boxes_encoded[:, :, icls, :] = boxes_to_target(
                    boxes_matched_c, self._ave_lwh[icls], A, self._no
                )

        boxes_matched = torch.from_numpy(boxes_matched)
        boxes_encoded = torch.from_numpy(boxes_encoded)
        # boxes_cls = (
        #     torch.from_numpy(boxes_gt_cls[closest_box_inds])
        #     if boxes_gt_cls is not None
        #     else None
        # )

        data_dict.update(
            {
                "boxes_matched": boxes_matched,  # (N, S, 7)
                "boxes_encoded": boxes_encoded,  # (N, A, S, C)
                # "boxes_cls": boxes_cls,  # (N,)
                "closest_box_inds": closest_box_inds,  # (N, S)
            }
        )

        return data_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, v in batch[0].items():
            if isinstance(v, SparseTensor):
                rtn_dict[k] = collate_sparse_tensors([sample[k] for sample in batch])
            elif isinstance(v, torch.Tensor):
                rtn_dict[k] = torch.cat([sample[k] for sample in batch], dim=0)
            elif k == "params":
                if k not in rtn_dict:
                    rtn_dict[k] = v
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict


class JRDBDet3D(_DatasetBase):
    def __init__(self, *args, **kwargs):
        super(JRDBDet3D, self).__init__(*args, **kwargs)
        self._ave_lwh = [(0.9, 0.5, 1.7)]
        self._dist_thresh = [(0.5, 0.7)]
        self._nc = 1
        self._inds_to_cls = ["pedestrian"]  # not used

    def _get_handle(self, data_dir, split):
        from .handles.jrdb_handle import JRDBHandleDet3D

        jrdb_val_seq = [
            "clark-center-2019-02-28_1",
            "gates-ai-lab-2019-02-08_0",
            "huang-2-2019-01-25_0",
            "meyer-green-2019-03-16_0",
            "nvidia-aud-2019-04-18_0",
            "tressider-2019-03-16_1",
            "tressider-2019-04-26_2",
        ]

        if split == "train":
            return JRDBHandleDet3D(data_dir, "train", exclude_sequences=jrdb_val_seq)
        elif split == "val":
            return JRDBHandleDet3D(data_dir, "train", sequences=jrdb_val_seq)
        elif split == "train_val":
            return JRDBHandleDet3D(data_dir, "train")
        elif split == "test":
            return JRDBHandleDet3D(data_dir, "test")
        else:
            raise RuntimeError(f"Invalid split: {split}")

    def _get_data(self, data_dict):
        # point cloud in base frame
        pc_upper = data_dict["pc_upper"]
        pc_lower = data_dict["pc_lower"]
        pc_upper = jt.transform_pts_upper_velodyne_to_base(pc_upper)
        pc_lower = jt.transform_pts_lower_velodyne_to_base(pc_lower)
        pc = np.concatenate([pc_upper, pc_lower], axis=1)  # (3, N)
        pc_offset = np.zeros(3, dtype=np.float32)

        if "label_str" not in data_dict.keys():
            return pc, None, None, pc_offset, None

        # bounding box in base frame
        boxes, _ = ub3d.string_to_boxes(data_dict["label_str"])

        # filter out corrupted annotations with negative dimension
        valid_mask = (boxes[:, 3:6] > 0.0).min(axis=1).astype(bool)
        boxes = boxes[valid_mask]
        boxes_cls = np.zeros(len(boxes), dtype=np.int32)

        return pc, boxes, boxes_cls, pc_offset, None
    
class NuScenesDet3D(_DatasetBase):
    def __init__(self, *args, **kwargs):
        super(NuScenesDet3D, self).__init__(*args, **kwargs)
        self._ave_lwh = [
            (0.50, 2.53, 0.98),
            (1.70, 0.60, 1.28),
            (11.23, 2.93, 3.47),
            (4.62, 1.95, 1.73),
            (6.37, 2.85, 3.19),
            (2.11, 0.77, 1.47),
            (0.73, 0.67, 1.77),
            (0.41, 0.41, 1.07),
            (12.29, 2.90, 3.87),
            (6.93, 2.51, 2.84),
        ]  # from nusc.list_category()
        self._dist_thresh = [
            (0.6, 2.63),
            (0.7, 1.8),
            (3.03, 11.33),
            (2.05, 4.72),
            (2.95, 6.47),
            (0.87, 2.21),
            (0.77, 0.83),
            (0.51, 0.71),
            (3.0, 12.39),
            (2.61, 7.03),
        ]
        self._nc = 10

        self._cls_mapping = {
            "animal": "void",
            "human.pedestrian.personal_mobility": "void",
            "human.pedestrian.stroller": "void",
            "human.pedestrian.wheelchair": "void",
            "movable_object.debris": "void",
            "movable_object.pushable_pullable": "void",
            "static_object.bicycle_rack": "void",
            "vehicle.emergency.ambulance": "void",
            "vehicle.emergency.police": "void",
            "movable_object.barrier": "barrier",
            "vehicle.bicycle": "bicycle",
            "vehicle.bus.bendy": "bus",
            "vehicle.bus.rigid": "bus",
            "vehicle.car": "car",
            "vehicle.construction": "construction_vehicle",
            "vehicle.motorcycle": "motorcycle",
            "human.pedestrian.adult": "pedestrian",
            "human.pedestrian.child": "pedestrian",
            "human.pedestrian.construction_worker": "pedestrian",
            "human.pedestrian.police_officer": "pedestrian",
            "movable_object.trafficcone": "traffic_cone",
            "vehicle.trailer": "trailer",
            "vehicle.truck": "truck",
        }

        self._cls_to_inds = {
            "void": -1,
            "barrier": 0,
            "bicycle": 1,
            "bus": 2,
            "car": 3,
            "construction_vehicle": 4,
            "motorcycle": 5,
            "pedestrian": 6,
            "traffic_cone": 7,
            "trailer": 8,
            "truck": 9,
        }

        self._inds_to_cls = [
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
        ]
        for i, c in enumerate(self._inds_to_cls):
            assert self._cls_to_inds[c] == i

        # customized classes
        nc = len(self._included_classes)
        if nc > 0:
            cls_to_inds = {"void": -1}
            inds_to_cls = []
            dist_thresh = []
            ave_lwh = []

            for i, c in enumerate(self._included_classes):
                cls_to_inds[c] = i
                inds_to_cls.append(c)
                idx = self._cls_to_inds[c]
                dist_thresh.append(self._dist_thresh[idx])
                ave_lwh.append(self._ave_lwh[idx])

            for k, c in self._cls_mapping.items():
                if c not in self._included_classes:
                    self._cls_mapping[k] = "void"

            self._nc = nc
            self._cls_to_inds = cls_to_inds
            self._inds_to_cls = inds_to_cls
            self._dist_thresh = dist_thresh
            self._ave_lwh = ave_lwh

    def _get_handle(self, data_dir, split):
        from .handles.nuscenes_handle import NuScenesHandle

        # return NuScenesHandle(data_dir, split, mini=True, nsweeps=self._nsweeps)
        return NuScenesHandle(data_dir, split, mini=False, nsweeps=self._nsweeps)

    def _get_data(self, data_dict):
        # point cloud in global frame
        pc = data_dict["pc"].points[:3]  # (3, N)

        # center point cloud
        pc_mean = pc.mean(axis=1, keepdims=True)
        pc -= pc_mean

        # additional features
        addi_feats = []
        if "intensity" in self._additional_features:
            intensity = (data_dict["pc"].points[3] / 255.0) - 0.5
            addi_feats.append(intensity)
        if "pc_dt" in data_dict and "time" in self._additional_features:
            addi_feats.append(data_dict["pc_dt"])
        addi_feats = np.stack(addi_feats, axis=0) if len(addi_feats) > 0 else None

        if len(data_dict["anns"]) == 0:
            return pc, None, None, pc_mean, addi_feats

        boxes = []
        boxes_cls = []
        for ann in data_dict["anns"]:
            cls_str = self._cls_mapping[ann["category_name"]]
            if cls_str != "void":
                box, _ = ub3d.box_from_nuscenes(ann)
                boxes.append(box)
                boxes_cls.append(self._cls_to_inds[cls_str])

        boxes = np.array(boxes, dtype=np.float32)
        boxes_cls = np.array(boxes_cls, dtype=np.int32)

        if boxes.shape[0] > 0:
            boxes[:, :3] = boxes[:, :3] - pc_mean.T

        return pc, boxes, boxes_cls, pc_mean, addi_feats
    
class _DatasetBase2D(Dataset):
    def __init__(self, data_dir, split, cfg):
        self._augmentation = cfg["augmentation"]

        self.__training = "train" in split  # loss will be computed
        self.__split = split
        self.__handle = self._get_handle(data_dir, split)
        self._cutout_kwargs = cfg["cutout_kwargs"]
        self._pl_correction_level = cfg["pl_correction_level"]
        self._mixup_alpha = cfg["mixup_alpha"] if "mixup_alpha" in cfg else 0.0

    def _get_handle(self, data_dir, split):
        raise NotImplementedError

    def _get_data(self, data_dict, training=True):
        raise NotImplementedError

    def _do_augmentation(self, laser_data, boxes, scan_rphi):
        # Random scale
        scale_factor = np.random.uniform(0.95, 1.05)
        laser_data *= scale_factor

        # Random rotation constrained by lidar resolution
        resolution = np.diff(scan_rphi)[0]  # Assuming uniform resolution
        max_index_shift = int(2 * np.pi / resolution)
        index_shift = np.random.randint(0, max_index_shift)
        
        # Create rotated laser_data with shifted indices
        laser_data_rot = np.roll(laser_data, shift=index_shift, axis=1)
        
        if boxes is not None and len(boxes) > 0:
            # Apply scale to boxes
            boxes[:, :6] *= scale_factor
            
            # Rotate boxes
            theta = index_shift * resolution
            rot_mat = np.array(
                [
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            boxes[:, :3] = boxes[:, :3] @ rot_mat.T
            boxes[:, 6] += theta

        return laser_data_rot, boxes

    @property
    def split(self):
        return self.__split  # used by trainer.py

    def __len__(self):
        return len(self.__handle)

    def __getitem__(self, idx):
        data_dict = self.__handle[idx]

        laser_data, scan_rphi, boxes_gt, boxes_gt_cls, = self._get_data(data_dict)

        if self.__training and self._augmentation:
            laser_data, boxes_gt = self._do_augmentation(laser_data, boxes_gt, scan_rphi)

        params_dict = {
            "ave_lwh": self._ave_lwh,
        }

        data_dict.update(
            {
                "laser_data": laser_data,
                "boxes_gt": boxes_gt,  # (B, 7) or None
                "boxes_gt_cls": boxes_gt_cls,  # (B,) or None
                "params": params_dict,
            }
        )

        data_dict["net_input"] = u_spaam.scans_to_cutout(
            data_dict["laser_data"],
            data_dict["laser_grid"],
            stride=1,
            **self._cutout_kwargs,
        )

        # if not self.__training:
        #     return data_dict

        dets_center2D_rphi = np.stack(u_spaam.xy_to_rphi(boxes_gt[:,0], boxes_gt[:,1]), axis=0)
        dets_dimensions_rotation_2D = boxes_gt[:,[3,4,6]].T # Length, Width, Rotation

        # regression target
        target_cls, target_reg, target_dim, target_theta, anns_valid_mask = self._get_regression_target(
            scan_rphi,
            dets_center2D_rphi,
            dets_dimensions_rotation_2D,
            person_radius_small=0.5,
            person_radius_large=0.8,
            min_close_points=5,
            ave_lw=self._ave_lwh[0][:2]
        )

        data_dict.update(
            {
                "target_reg": target_reg,  # (N, 2)
                "target_dim": target_dim,  # (N, 2)
                "target_theta": target_theta,  # (N, 2)
                "target_cls": target_cls,  # (N, 1)
                "anns_valid_mask": anns_valid_mask,  # (M,)
            }
        )


        return data_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in [
                "target_cls",
                "target_reg",
                "target_dim",
                "target_theta",
                "net_input",
                "target_cls_mixup",
                "input_mixup",
                "laser_data",
                "anns_valid_mask",
            ]:
                rtn_dict[k] = np.array([sample[k] for sample in batch])
            else:
                rtn_dict[k] = [sample[k] for sample in batch]

        return rtn_dict
    
    def _get_regression_target(
        self, scan_rphi, dets_rphi, dets_dimensions_rotation, person_radius_small, person_radius_large, min_close_points, ave_lw,
    ):
        """Generate classification and regression label.

        Args:
            scan_rphi (np.ndarray[2, N]): Scan points in polar coordinate
            dets_rphi (np.ndarray[2, M]): Annotated person centers in polar coordinate
            person_radius_small (float): Points less than this distance away
                from an annotation is assigned to that annotation and marked as fg.
            person_radius_large (float): Points with no annotation smaller
                than this distance is marked as bg.
            min_close_points (int): Annotations with supportive points fewer than this
                value is marked as invalid. Supportive points are those within the small
                radius.

        Returns:
            target_cls (np.ndarray[N]): Classification label, 1=fg, 0=bg, -1=ignore
            target_reg (np.ndarray[N, 2]): Regression label
            anns_valid_mask (np.ndarray[M])
        """
        N = scan_rphi.shape[1]
        # no annotation in this frame
        if len(dets_rphi) == 0:
            return np.zeros(N, dtype=np.int64), np.zeros((N, 2), dtype=np.float32), []

        scan_xy = np.stack(u_spaam.rphi_to_xy(scan_rphi[0], scan_rphi[1]), axis=0)
        dets_xy = np.stack(u_spaam.rphi_to_xy(dets_rphi[0], dets_rphi[1]), axis=0)

        dist_scan_dets = np.hypot(
            scan_xy[0].reshape(1, -1) - dets_xy[0].reshape(-1, 1),
            scan_xy[1].reshape(1, -1) - dets_xy[1].reshape(-1, 1),
        )  # (M, N) pairwise distance between scan and detections

        # mark out annotations that has too few scan points
        anns_valid_mask = (
            np.sum(dist_scan_dets < person_radius_small, axis=1) > min_close_points
        )  # (M, )

        # for each point, find the distance to its closest annotation
        argmin_dist_scan_dets = np.argmin(dist_scan_dets, axis=0)  # (N, )
        min_dist_scan_dets = dist_scan_dets[argmin_dist_scan_dets, np.arange(N)]

        # points within small radius, whose corresponding annotation is valid, is marked
        # as foreground
        target_cls = -1 * np.ones(N, dtype=np.int64)
        fg_mask = np.logical_and(
            anns_valid_mask[argmin_dist_scan_dets], min_dist_scan_dets < person_radius_small
        )
        target_cls[fg_mask] = 1
        target_cls[min_dist_scan_dets > person_radius_large] = 0

        # regression target
        dets_matched_rphi = dets_rphi[:, argmin_dist_scan_dets] # 2xN
        target_reg = np.stack(
            u_spaam.global_to_canonical(
                scan_rphi[0], scan_rphi[1], dets_matched_rphi[0], dets_matched_rphi[1]
            ),
            axis=1,
        ) # Nx2

        dets_matched_dimensions_rotation = dets_dimensions_rotation[:, argmin_dist_scan_dets] # 3xN
        target_dim = dets_matched_dimensions_rotation[:2] / np.array(ave_lw, dtype=np.float32).reshape(2, 1) # 2xN
        target_dim = np.log(target_dim).T # Nx2
        target_theta = np.stack((np.sin(dets_matched_dimensions_rotation[2]), np.cos(dets_matched_dimensions_rotation[2])), axis=1) # Nx2
        
        

        return target_cls, target_reg, target_dim, target_theta, anns_valid_mask
    
    def _get_sample_with_mixup(self, idx):
        # randomly find another sample
        mixup_idx = idx
        while mixup_idx == idx:
            mixup_idx = int(np.random.randint(0, len(self.__handle), 1)[0])

        data_dict_0 = self._get_sample(idx)
        data_dict_1 = self._get_sample(mixup_idx)

        input_mixup, target_cls_mixup = self._mixup_samples(
            data_dict_0["net_input"],
            data_dict_0["target_cls"],
            data_dict_1["net_input"],
            data_dict_1["target_cls"],
            self._mixup_alpha,
        )

        data_dict_0["input_mixup"] = input_mixup
        data_dict_0["target_cls_mixup"] = target_cls_mixup

        return data_dict_0
    
    def _mixup_samples(self, x0, t0, x1, t1, alpha):
        lam = np.random.beta(alpha, alpha, 1)
        x01 = x0 * lam + x1 * (1.0 - lam)
        t01 = t0 * lam + t1 * (1.0 - lam)

        # only mixup points that are valid for classification in both scans
        invalid_mask = np.logical_or(t0 < 0, t1 < 0)
        t01[invalid_mask] = -2

        return x01, t01
    
class JRDBDet2D(_DatasetBase2D):
    def __init__(self, *args, **kwargs):
        super(JRDBDet2D, self).__init__(*args, **kwargs)
        self._ave_lwh = [(0.9, 0.5, 1.7)]

    def _get_handle(self, data_dir, split):
        from .handles.jrdb_handle_2D import JRDBHandleDet2D

        jrdb_val_seq = [
            "clark-center-2019-02-28_1",
            "gates-ai-lab-2019-02-08_0",
            "huang-2-2019-01-25_0",
            "meyer-green-2019-03-16_0",
            "nvidia-aud-2019-04-18_0",
            "tressider-2019-03-16_1",
            "tressider-2019-04-26_2",
        ]

        if split == "train":
            return JRDBHandleDet2D(data_dir, "train", exclude_sequences=jrdb_val_seq)
        elif split == "val":
            return JRDBHandleDet2D(data_dir, "train", sequences=jrdb_val_seq)
        elif split == "train_val":
            return JRDBHandleDet2D(data_dir, "train")
        elif split == "test":
            return JRDBHandleDet2D(data_dir, "test")
        else:
            raise RuntimeError(f"Invalid split: {split}")

    def _get_data(self, data_dict):
        # DROW defines laser frame as x-forward, y-right, z-downward
        # JRDB defines laser frame as x-forward, y-left, z-upward
        # Use DROW frame for DR-SPAAM or DROW3

        # equivalent of flipping y axis (inversing laser phi angle)
        laser_data = data_dict["laser_data"][:, ::-1]
        scan_rphi = np.stack(
            (laser_data[-1], data_dict["laser_grid"]), axis=0
        )

        if "label_str" not in data_dict.keys():
            return laser_data, scan_rphi, None, None,

        # bounding box in laser frame
        boxes, _ = ub3d.string_to_boxes(data_dict["label_str"]) # Nx7
        # filter out corrupted annotations with negative dimension
        valid_mask = (boxes[:, 3:6] > 0.0).min(axis=1).astype(bool)
        boxes = boxes[valid_mask]
        boxes_cls = np.zeros(len(boxes), dtype=np.int32)
        boxes = jt.transform_box_3D_base_to_laser(boxes.T) # 7xN
        boxes[1] = -boxes[1] # to DROW frame - y flip
        boxes[2] = -boxes[2] # to DROW frame - z flip
        boxes[6] = -boxes[6] # to DROW frame - rotation flip
        boxes = boxes.T # Nx7

        return laser_data, scan_rphi, boxes, boxes_cls,



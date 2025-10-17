import numpy as np
import os
import torch
import torch.nn.functional as F
import lidar_det.utils.jrdb_transforms as jt
import lidar_det.utils.utils_dr_spaam as u
from .model_fn import write_jrdb_results, eval_jrdb
# import dr_spaam.utils.precision_recall as pru
# from dr_spaam.utils.plotting import plot_one_frame


def _sample_or_repeat(population, n):
    """Select n sample from population, without replacement if population size
    greater than n, otherwise with replacement.

    Only work for population of 1D tensor (N,)
    """
    N = len(population)
    if N == n:
        return population
    elif N > n:
        return population[torch.randperm(N, device=population.device)[:n]]
    else:
        return population[torch.randint(N, (n,), device=population.device)]


def _balanced_sampling_reweighting(target_cls, goal_fg_ratio=0.4):
    # target_cls is 1D tensor (N, )
    N = target_cls.shape[0]
    goal_fg_num = int(N * goal_fg_ratio)
    goal_bg_num = int(N * (1.0 - goal_fg_ratio))

    inds = torch.arange(N, device=target_cls.device)
    fg_inds = inds[target_cls > 0]
    bg_inds = inds[target_cls == 0]

    if len(fg_inds) > 0:
        fg_inds = _sample_or_repeat(fg_inds, goal_fg_num)
        bg_inds = _sample_or_repeat(bg_inds, goal_bg_num)
        sample_inds = torch.cat((fg_inds, bg_inds))
    else:
        sample_inds = _sample_or_repeat(bg_inds, N)

    weights = torch.zeros(N, device=target_cls.device).float()
    weights.index_add_(0, sample_inds, torch.ones_like(sample_inds).float())

    return weights


def _model_fn(model, batch_dict, max_num_pts=1e6, cls_loss_weight=1.0):
    tb_dict, rtn_dict = {}, {}

    net_input = batch_dict["net_input"]

    # # train only on part of scan, if the GPU cannot fit the whole scan
    # num_pts = target_cls.shape[1]
    # if model.training and num_pts > max_num_pts:
    #     idx0 = np.random.randint(0, num_pts - max_num_pts)
    #     idx1 = idx0 + max_num_pts
    #     target_cls = target_cls[:, idx0:idx1]
    #     target_reg = target_reg[:, idx0:idx1, :]
    #     net_input = net_input[:, idx0:idx1, :, :]
    #     N = max_num_pts
    B, N = net_input.shape[0], net_input.shape[1]
    # to gpu
    net_input = torch.from_numpy(net_input).cuda(non_blocking=True).float()

    # forward pass
    rtn_tuple = model(net_input)

    pred_cls, pred_reg, pred_box, pred_sim = rtn_tuple
    pred_dim = pred_box[..., :2]
    pred_theta = pred_box[..., 2:]
    rtn_dict["pred_cls"] = pred_cls.view(B, N)
    rtn_dict["pred_reg"] = pred_reg.view(B, N, 2)
    rtn_dict["pred_box"] = pred_box.view(B, N, 4)
    rtn_dict["pred_dim"] = pred_dim.view(B, N, 2)
    rtn_dict["pred_theta"] = pred_theta.view(B, N, 2)
    rtn_dict["pred_sim"] = pred_sim

    # no label for test set, inference only
    if "target_cls" not in batch_dict.keys():
        tb_dict = {}
        return 0.0, tb_dict, rtn_dict

    target_cls, target_reg, target_dim, target_theta = batch_dict["target_cls"], batch_dict["target_reg"], batch_dict["target_dim"], batch_dict["target_theta"] 

    target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).float()
    target_reg = torch.from_numpy(target_reg).cuda(non_blocking=True).float()
    target_dim = torch.from_numpy(target_dim).cuda(non_blocking=True).float()
    target_theta = torch.from_numpy(target_theta).cuda(non_blocking=True).float()

    target_cls = target_cls.view(B * N)
    pred_cls = pred_cls.view(B * N)

    # number of valid points
    valid_mask = target_cls >= 0
    valid_ratio = torch.sum(valid_mask).item() / (B * N)
    # assert valid_ratio > 0, "No valid points in this batch."
    tb_dict["valid_ratio"] = valid_ratio

    # cls loss
    cls_loss = (
        model.cls_loss(pred_cls[valid_mask], target_cls[valid_mask], reduction="mean")
        * cls_loss_weight
    )
    total_loss = cls_loss
    tb_dict["cls_loss"] = cls_loss.item()

    # number fg points
    # NOTE supervise regression for both close and far neighbor points
    fg_mask = torch.logical_or(target_cls == 1, target_cls == -1)
    fg_ratio = torch.sum(fg_mask).item() / (B * N)
    tb_dict["fg_ratio"] = fg_ratio

    # reg loss
    if fg_ratio > 0.0:
        target_reg = target_reg.view(B * N, -1)
        pred_reg = pred_reg.view(B * N, -1)
        reg_loss = F.mse_loss(pred_reg[fg_mask], target_reg[fg_mask], reduction="none")
        reg_loss = torch.sqrt(torch.sum(reg_loss, dim=1)).mean()
        total_loss = total_loss + reg_loss
        tb_dict["reg_loss"] = reg_loss.item()

    # dim loss
    if fg_ratio > 0.0:
        target_dim = target_dim.view(B * N, -1)
        pred_dim = pred_dim.view(B * N, -1)
        dim_loss = F.mse_loss(pred_dim[fg_mask], target_dim[fg_mask], reduction="none")
        dim_loss = torch.sqrt(torch.sum(dim_loss, dim=1)).mean()
        total_loss = total_loss + dim_loss
        tb_dict["dim_loss"] = dim_loss.item()

    # theta loss
    if fg_ratio > 0.0:
        target_theta = target_theta.view(B * N, -1)
        pred_theta = pred_theta.view(B * N, -1)
        theta_loss = F.mse_loss(pred_theta[fg_mask], target_theta[fg_mask], reduction="none")
        theta_loss = torch.sqrt(torch.sum(theta_loss, dim=1)).mean()
        total_loss = total_loss + theta_loss
        tb_dict["theta_loss"] = theta_loss.item()

    # # regularization loss for spatial attention
    # if spatial_drow:
    #     # shannon entropy
    #     att_loss = (-torch.log(pred_sim + 1e-5) * pred_sim).sum(dim=2).mean()
    #     tb_dict['att_loss'] = att_loss.item()
    #     total_loss = total_loss + att_loss

    return total_loss, tb_dict, rtn_dict


def _model_fn_mixup(model, batch_dict, max_num_pts=1e6, cls_loss_weight=1.0):
    # mixup regularization for robust training against label noise
    # https://arxiv.org/pdf/1710.09412.pdf

    tb_dict, rtn_dict = {}, {}

    net_input = batch_dict["input_mixup"]
    target_cls = batch_dict["target_cls_mixup"]

    B, N = target_cls.shape

    # train only on part of scan, if the GPU cannot fit the whole scan
    num_pts = target_cls.shape[1]
    if model.training and num_pts > max_num_pts:
        idx0 = np.random.randint(0, num_pts - max_num_pts)
        idx1 = idx0 + max_num_pts
        target_cls = target_cls[:, idx0:idx1]
        net_input = net_input[:, idx0:idx1, :, :]
        N = max_num_pts

    # to gpu
    net_input = torch.from_numpy(net_input).cuda(non_blocking=True).float()
    target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).float()

    # forward pass
    rtn_tuple = model(net_input)

    # so this function can be used for both DROW and DR-SPAAM
    if len(rtn_tuple) == 2:
        pred_cls, pred_reg = rtn_tuple
    elif len(rtn_tuple) == 3:
        pred_cls, pred_reg, pred_sim = rtn_tuple
        rtn_dict["pred_sim"] = pred_sim

    target_cls = target_cls.view(B * N)
    pred_cls = pred_cls.view(B * N)

    # number of valid points
    valid_mask = target_cls >= 0
    valid_ratio = torch.sum(valid_mask).item() / (B * N)
    # assert valid_ratio > 0, "No valid points in this batch."
    tb_dict["valid_ratio_mixup"] = valid_ratio

    # cls loss
    cls_loss = (
        model.cls_loss(pred_cls[valid_mask], target_cls[valid_mask], reduction="mean")
        * cls_loss_weight
    )
    total_loss = cls_loss
    tb_dict["cls_loss_mixup"] = cls_loss.item()

    return total_loss, tb_dict, rtn_dict


def _model_eval_fn(model, batch_dict, full_eval=False, plotting=False, output_dir=None, nuscenes=False, visible_2D = False,):
    _, tb_dict, rtn_dict = _model_fn(model, batch_dict) # Laser frame (base_link) and DROW coordinate system

    if not full_eval or output_dir is None:
        return tb_dict, {}

    pred_cls = torch.sigmoid(rtn_dict["pred_cls"]).data.cpu().numpy()
    pred_reg = rtn_dict["pred_reg"].data.cpu().numpy()
    pred_box = rtn_dict["pred_box"].data.cpu().numpy()
    JRDB_mean_length = 0.9
    JRDB_mean_width = 0.5
    lengths = np.exp(pred_box[:, :, 0])*JRDB_mean_length
    widths = np.exp(pred_box[:, :, 1])*JRDB_mean_width
    sinr = pred_box[:, :, 2]
    cosr = pred_box[:, :, 3]
    rots = np.arctan2(sinr, cosr)
    pred_box = np.stack((lengths, widths, rots), axis=-1)

    # postprocess network prediction to get detection
    scans = batch_dict["laser_data"]
    scan_phi = batch_dict["laser_grid"]
    fake_2D_z = -0.9
    fake_2D_height = 1.8
    eval_dict = {}
    for ib in range(len(scans)):
        dets_xy, dets_cls, dets_box, _ = u.nms_predicted_center(
                        scans[ib][-1],
                        scan_phi[ib],
                        pred_cls[ib],
                        pred_reg[ib],
                        pred_box=pred_box[ib]
                    )
        x, y, length, width, theta = dets_xy[:, 0], dets_xy[:, 1], dets_box[:, 0], dets_box[:, 1], dets_box[:, 2],
        fake_z, fake_height = np.full_like(x, fake_2D_z), np.full_like(x, fake_2D_height)
        boxes_nms = np.stack((x, y, fake_z, length, width, fake_height, theta), axis=-1)
        boxes_nms[:,1] = -boxes_nms[:,1] # DROW to JRDB - y flip
        boxes_nms[:,2] = -boxes_nms[:,2] # DROW to JRDB - z flip
        boxes_nms[:,6] = -boxes_nms[:,6] # DROW to JRDB - rotation flip
        boxes_nms = jt.transform_box_3D_laser_to_base(boxes_nms.T).T # Nx7
        scores_nms = dets_cls

        boxes_gt = batch_dict["boxes_gt"][ib]
        boxes_gt[:,1] = -boxes_gt[:,1] # DROW to JRDB - y flip
        boxes_gt[:,2] = -boxes_gt[:,2] # DROW to JRDB - z flip
        boxes_gt[:,6] = -boxes_gt[:,6] # DROW to JRDB - rotation flip
        boxes_gt = jt.transform_box_3D_laser_to_base(boxes_gt.T).T # Nx7
        if visible_2D:
            ann_mask_2D = batch_dict["anns_valid_mask"][ib]
            boxes_gt = boxes_gt[ann_mask_2D]
        batch_dict["boxes_gt"][ib] = boxes_gt

        score_thresh = 0.5
        write_jrdb_results(
            boxes_nms,
            scores_nms,
            score_thresh,
            batch_dict,
            ib,
            output_dir=output_dir,
            plotting=plotting,
            lidar_type='2D'
        )


    return tb_dict, eval_dict


def _model_eval_collate_fn(tb_dict_list,
    eval_dict_list,
    output_dir,
    full_eval=False,
    rm_files=False,
    nuscenes=False,
    labels_kitti_path = None,
    visible_2D = False,):
    # tb_dict should only contain scalar values, collate them into an array
    # and take their mean as the value of the epoch
    epoch_tb_dict = {}
    for batch_tb_dict in tb_dict_list:
        for k, v in batch_tb_dict.items():
            epoch_tb_dict.setdefault(k, []).append(v)
    for k, v in epoch_tb_dict.items():
        epoch_tb_dict[k] = np.array(v).mean()

    if not full_eval:
        return epoch_tb_dict

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        if labels_kitti_path is None:
            labels_kitti_path = "./data/JRDB/train_dataset/labels_kitti"
        ap_dict, ret_dict = eval_jrdb(
            gt_dir=labels_kitti_path,
            det_dir=os.path.join(output_dir, "detections"),
            rm_det_files=rm_files,
            visible_2D=visible_2D,
        )  # NOTE hard-coded path

        for k, v in ap_dict.items():
            epoch_tb_dict[f"ap_{k}"] = v
        for k, v in ret_dict.items():
            epoch_tb_dict[k] = v
    except Exception as e:
        print(e)
        print("Evaluation failed")

    return epoch_tb_dict


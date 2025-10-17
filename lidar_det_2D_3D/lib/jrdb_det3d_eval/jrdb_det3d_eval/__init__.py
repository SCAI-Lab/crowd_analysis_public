import os
import warnings
import numpy as np
import shutil

from .kitti_common import get_label_annos
from .eval import get_official_eval_result


def _eval_seq(gt_annos, det_annos):
    with warnings.catch_warnings():
        # so numba compile warning does not pollute logs
        warnings.simplefilter("ignore")
        result_str, result_dict = get_official_eval_result(gt_annos, det_annos, 1, metric='iou3d')

    # print(result_str)
    for k, v in result_dict.items():
        if k[:3] != 'ALL' and k != 'MAX_3D_DISTANCES':
            print(k, f'{v:.4f}')
    seq_ap = result_dict["Pedestrian_3d/moderate_mAP/Overlap_0.30"]

    return seq_ap, result_dict


def eval_jrdb(gt_dir, det_dir, rm_det_files=False, visible_2D=False):
    # gt_sequences = sorted(os.listdir(gt_dir))
    det_sequences = sorted(os.listdir(det_dir))
    # assert gt_sequences == det_sequences
    ann_mask_2D_dir = None
    if visible_2D:
        ann_mask_2D_dir = '/home/jupyter-dominik/Person_MinkUNet_adapt/logs/annotations_mask_2D'

    ap_dict = {}
    result_dict_list = []
    ret_dict = {}

    # per sequence eval
    result_dict = {}
    seq_ap, seq_len = [], []
    for idx, seq in enumerate(det_sequences):
        print(f"({idx + 1}/{len(det_sequences)}) Evaluating {seq}")
        ann_mask_2D_seq_dir = None
        if ann_mask_2D_dir is not None:
            ann_mask_2D_seq_dir = os.path.join(ann_mask_2D_dir, seq)

        gt_annos = get_label_annos(os.path.join(gt_dir, seq), ann_mask_2D_folder=ann_mask_2D_seq_dir)
        det_annos = get_label_annos(os.path.join(det_dir, seq))

        ap_dict[seq], result_dict  = _eval_seq(gt_annos, det_annos)
        result_dict_list.append(result_dict)
        print(f"{seq}, AP={ap_dict[seq]:.4f}, len={len(gt_annos)}")

        seq_ap.append(ap_dict[seq])
        seq_len.append(len(gt_annos))

    # NOTE Jointly evaluating all sequences crashes, don't know why. Use average
    # AP of all sequences instead.
    print("Evaluating whole set")
    seq_ap = np.array(seq_ap)
    seq_len = np.array(seq_len)
    ap_dict["all"] = np.sum(seq_ap * (seq_len / seq_len.sum()))
    print(f"Whole set, AP={ap_dict['all']:.4f}, len={seq_len.sum()}")

    for k in result_dict.keys():
        if k != 'MAX_3D_DISTANCES':
            ret_dict[k] = tuple(d[k] for d in result_dict_list)
        else:
            ret_dict[k] = result_dict[k]

    for k, metrics_tuple in ret_dict.items():  
        if k != 'MAX_3D_DISTANCES':      
            metrics_tuple = np.array(metrics_tuple)
            if k[:3] != 'ALL':
                ret_dict[k] = np.sum(metrics_tuple * (seq_len / seq_len.sum()))
            else:
                # shortest_thresh_len_seq = np.count_nonzero(metrics_tuple[0]) # Take only Precision/Recall thresholds that can be achieved
                # for seq in range(len(metrics_tuple)):
                #     thresh_len_seq = np.count_nonzero(metrics_tuple[seq])
                #     if thresh_len_seq < shortest_thresh_len_seq:
                #         shortest_thresh_len_seq = thresh_len_seq 
                # ret_all_np = np.zeros(shortest_thresh_len_seq)
                # for thresh_ind in range(shortest_thresh_len_seq):
                #     ret_all_np[thresh_ind] = np.sum(metrics_tuple[:, thresh_ind] * (seq_len / seq_len.sum()))
                num_thresholds = metrics_tuple.shape[1]
                ret_all_np = []
                for thresh_ind in range(num_thresholds):
                    non_zero_indices = np.nonzero(metrics_tuple[:, thresh_ind])[0]
                    if len(non_zero_indices) >= 5:
                        values = metrics_tuple[non_zero_indices, thresh_ind]
                        lengths = seq_len[non_zero_indices]
                        weighted_average = np.sum(values * (lengths / lengths.sum()))
                        ret_all_np.append(weighted_average)
                ret_dict[k] = np.array(ret_all_np)
                # print(shortest_thresh_len_seq)
                # print(ret_dict[k])



    if rm_det_files:
        shutil.rmtree(det_dir)

    return ap_dict, ret_dict

import io as sysio

import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval

# from iou3d import boxes_iou3d_gpu
# import torch


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 1 / (num_sample_pts - 1.0)
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds

# @numba.jit
# def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41, score_jump_thresh=0.05):
#     scores.sort()
#     scores = scores[::-1]
#     current_recall = 1 / (num_sample_pts - 1.0)
#     thresholds = []
#     last_score = scores[0]
#     last_index = 0
#     num_gts_per_threshold = []
#     max_r = float(scores.shape[0])/float(num_gt)
#     print("MAX RECALL FOR ALL IS")
#     print(max_r)
#     for i, score in enumerate(scores):
#         l_recall = (i + 1) / num_gt
#         if i < (len(scores) - 1):
#             r_recall = (i + 2) / num_gt
#         else:
#             r_recall = l_recall

#         # Check for significant confidence drop
#         if last_score - score > score_jump_thresh:
#             thresholds.append(score)
#             last_score = score
#             num_gts_per_threshold.append(i-last_index)
#             last_index = i

#         if (((r_recall - current_recall) < (current_recall - l_recall))
#                 and (i < (len(scores) - 1))):
#             continue
#         # recall = l_recall
#         thresholds.append(score)
#         current_recall += 1 / (num_sample_pts - 1.0)
#         last_score = score
#         num_gts_per_threshold.append(i-last_index)
#         last_index = i
#     # print(num_gts_per_threshold)
    
#     return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty, MAX_3D_DISTS):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]

    MIN_NUM_POINTS = 10
    MAX_3D_DIST = MAX_3D_DISTS[difficulty]

    gt_dists = np.hypot(gt_anno["location"][:, 0], gt_anno["location"][:, 2])
    dt_dists = np.hypot(dt_anno["location"][:, 0], dt_anno["location"][:, 2])

    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1

        if gt_anno["num_points"][i] < 0:
            valid_class = -1

        ignore = gt_anno["num_points"][i] < MIN_NUM_POINTS or gt_dists[i] > MAX_3D_DIST

        # if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
        #         or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
        #         or (height <= MIN_HEIGHT[difficulty])):
        #     # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
        #     ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        # height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        # if height < MIN_HEIGHT[difficulty]:
        #     ignored_dt.append(1)
        if dt_dists[i] > MAX_3D_DIST:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[i, j]    # @CHANGED
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        # print('gt_nums = ' + str(gt_nums[i]))
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[gt_num:gt_num + gt_nums[i], dt_num:dt_num + dt_nums[i]] 
            # Change order of indices as it was wrong (gt,dt) instead of (dt,gt)

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64) # Part of the frames, shape (gt_boxes_all_part, dt_boxes_all_part) - Pairwise even in-between frames
            # gt_boxes_torch = torch.from_numpy(gt_boxes).cuda().float()
            # dt_boxes_torch = torch.from_numpy(dt_boxes).cuda().float()
            # overlap_part = boxes_iou3d_gpu(gt_boxes_torch, dt_boxes_torch)
            # overlap_part = overlap_part.data.cpu().numpy()
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty, max_3d_dists):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty, max_3d_dists)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               max_3d_dists,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=100):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos), "gt and det number mismatch "
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    # @CHANGED
    rets = calculate_iou_partly(gt_annos, dt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_gt_num, total_dt_num = rets # wrong return order
    # print("SHAPES gt_annos, dt_annos, overlaps, total_dt_num, total_gt_num")
    # print(len(gt_annos), len(dt_annos), len(overlaps), total_dt_num.shape, total_gt_num.shape)
    # print("gt_annos[0][name], gt_annos[-1][name], overlaps[0].shape, overlaps[-1].shape")
    # print(gt_annos[0]["name"], gt_annos[-1]["name"], overlaps[0].shape, overlaps[-1].shape)
    N_SAMPLE_PTS = 101
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    confidences = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty, max_3d_dists)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)): # For each frame
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0,
                        compute_fp=False)
                    _, _, _, _, thresholds = rets
                    thresholdss += thresholds.tolist() # Add a list of scores for frame i that matched detections in that frame
                thresholdss = np.array(thresholdss) # Numpy array of scores for all detections that matched for all frames
                # print(thresholdss.shape, thresholdss[0], thresholdss[1], thresholdss[-1])
                thresholds = get_thresholds(thresholdss, total_num_valid_gt, num_sample_pts=N_SAMPLE_PTS) # Score thresholds on p-r curve
                # print('tp, fp, fn from first run')
                # print(tp, fp, fn)
                # print(len(thresholds), thresholds)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    # print('GOING INTO FUSED')
                    # print(total_gt_num.shape, total_gt_num)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                # print('tp, fp, fn')
                # print(pr[:,0], pr[:,1], pr[:,2])
                # print(thresholds)
                for i in range(len(thresholds)):
                    if pr[i, 0] >= 50 and (pr[i, 1] + pr[i, 2]) >= 50: # Minimum thresholds of samples for calculation
                        confidences[m, l, k, i] = thresholds[i]
                        recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                        precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                        if compute_aos:
                            aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                    else:
                        pass
                # print(precision.shape)
                # for i in range(len(thresholds)):
                #     precision[m, l, k, i] = np.max(
                #         precision[m, l, k, i:], axis=-1)
                #     recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                #     if compute_aos:
                #         aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
        "confidences": confidences,
    }
    return ret_dict

def get_final_precision(prec):
    n_class = prec.shape[0]
    n_difficulty = prec.shape[1]
    n_overlap = prec.shape[2]
    n_points = prec.shape[3]
    sums = np.zeros((n_class, n_difficulty, n_overlap))
    length = np.zeros((n_class, n_difficulty, n_overlap))
    for i in range(n_class):
        for j in range(n_difficulty):
            for k in range(n_overlap): 
                for l in range(1, n_points):
                    if prec[i, j, k, l] != 0:
                        sums[i, j, k] += prec[i, j, k, l]
                        length[i, j, k] += 1
                    # else:
                    #     print("ZERO DETECTED")
                    #     print("n_point = " + str(l))
        

    # for i in range(1, prec.shape[-1]):
    #     sums = sums + prec[..., i]
    return sums / length * 100

def get_final_recall(rec):
    n_class = rec.shape[0]
    n_difficulty = rec.shape[1]
    n_overlap = rec.shape[2]
    n_points = rec.shape[3]
    sums = np.zeros((n_class, n_difficulty, n_overlap))
    length = np.zeros((n_class, n_difficulty, n_overlap))
    for i in range(n_class):
        for j in range(n_difficulty):
            for k in range(n_overlap): 
                for l in range(1, n_points):
                    if rec[i, j, k, l] != 0:
                        sums[i, j, k] += rec[i, j, k, l]
                        length[i, j, k] += 1

    # sums = 0
    # for i in range(1, rec.shape[-1]):
    #     sums = sums + rec[..., i]
    return sums / length * 100


# def get_mAP_R40(prec):
#     sums = 0
#     for i in range(1, prec.shape[-1]):
#         sums = sums + prec[..., i]
#     return sums / 40 * 100

def get_mAP_Rtrapz(prec, rec):
    ap = np.trapz(prec, rec, dx=0.01)
    return ap * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            metric='iou3d'):
############## Original
    # # min_overlaps: [num_minoverlap, metric, num_class]
    # difficultys = [0, 1, 2]
    # ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
    #                  min_overlaps, compute_aos)
    # # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    # mAP_bbox = get_mAP(ret["precision"])
    # mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    # if PR_detail_dict is not None:
    #     PR_detail_dict['bbox'] = ret['precision']

    # mAP_aos = mAP_aos_R40 = None
    # if compute_aos:
    #     mAP_aos = get_mAP(ret["orientation"])
    #     mAP_aos_R40 = get_mAP_R40(ret["orientation"])

    #     if PR_detail_dict is not None:
    #         PR_detail_dict['aos'] = ret['orientation']

    # ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
    #                  min_overlaps)
    # mAP_bev = get_mAP(ret["precision"])
    # mAP_bev_R40 = get_mAP_R40(ret["precision"])

    # if PR_detail_dict is not None:
    #     PR_detail_dict['bev'] = ret['precision']

    # ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
    #                  min_overlaps)
    # mAP_3d = get_mAP(ret["precision"])
    # mAP_3d_R40 = get_mAP_R40(ret["precision"])
    # if PR_detail_dict is not None:
    #     PR_detail_dict['3d'] = ret['precision']
    # return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40
############ @CHANGED
    difficultys = [0, 1, 2]
    max_3d_dists = [5, 10, 20] # Easy, Moderate, Hard
    # mAP_bbox = np.zeros((1, len(difficultys), min_overlaps.shape[0]), dtype=np.float32)
    # mAP_bbox_R40 = np.zeros((1, len(difficultys), min_overlaps.shape[0]), dtype=np.float32)
    # mAP_aos = mAP_aos_R40 = None
    # mAP_bev = np.zeros((1, len(difficultys), min_overlaps.shape[0]), dtype=np.float32)
    # mAP_bev_R40 = np.zeros((1, len(difficultys), min_overlaps.shape[0]), dtype=np.float32)
    # prec_3d = np.zeros((1, len(difficultys), min_overlaps.shape[0]), dtype=np.float32)
    # rec_3d = np.zeros((1, len(difficultys), min_overlaps.shape[0]), dtype=np.float32)
    if metric == 'iou2d_wh':
        metric_idx = 0
    elif metric == 'iou2d_lw':
        metric_idx = 1
    elif metric == 'iou3d':
        metric_idx = 2
    else:
        raise RuntimeError

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, max_3d_dists, metric_idx, min_overlaps, compute_aos=compute_aos)
    all_prec_3d, all_recall_3d, all_confidences = ret["precision"], ret["recall"], ret["confidences"]
    # prec_3d = get_final_precision(ret["precision"])
    # rec_3d = get_final_recall(ret["recall"])
    # mAP_3d_R40 = get_mAP_R40(ret["precision"])
    mAP_3d = get_mAP_Rtrapz(ret["precision"], ret["recall"])
    return mAP_3d, all_prec_3d, all_recall_3d, all_confidences, max_3d_dists
################


def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges,
                       compute_aos):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def get_official_eval_result(gt_annos, dt_annos, current_classes, metric='iou3d'):
    # @CHANGED
    overlap_0_7 = np.array([[0.7, 0.7, 0.5, 0.7, 0.5, 0.7],  # Metric (3) and Class (6)
                            [0.7, 0.7, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.7, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], 
                            [0.5, 0.5, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.5, 0.25, 0.5, 0.25, 0.5]])
    overlap_0_3 = np.array([[0.5, 0.3, 0.5, 0.7, 0.5, 0.5], 
                            [0.3, 0.3, 0.25, 0.5, 0.25, 0.5],
                            [0.3, 0.3, 0.25, 0.5, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5, overlap_0_3], axis=0)  # [3, 3, 6]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'Truck'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    # @CHANGED
    # for anno in dt_annos:
    #     if anno['alpha'].shape[0] != 0:
    #         if anno['alpha'][0] != -10:
    #             compute_aos = True
    #         break
    mAP3d, all_prec_3d, all_rec_3d, all_confidences, max_3d_dists = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, metric=metric)

    ret_dict = {}
    
    ret_dict['MAX_3D_DISTANCES'] = max_3d_dists
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            # result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
            #                      f"{mAPbbox[j, 1, i]:.4f}, "
            #                      f"{mAPbbox[j, 2, i]:.4f}"))
            # result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
            #                      f"{mAPbev[j, 1, i]:.4f}, "
            #                      f"{mAPbev[j, 2, i]:.4f}"))
            # result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
            #                      f"{mAP3d[j, 1, i]:.4f}, "
            #                      f"{mAP3d[j, 2, i]:.4f}"))
            

            # if compute_aos:
                # result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
                #                      f"{mAPaos[j, 1, i]:.2f}, "
                #                      f"{mAPaos[j, 2, i]:.2f}"))
                # if i == 0:
                   # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
                   # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
                   # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]

            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            # result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
            #                      f"{mAPbbox_R40[j, 1, i]:.4f}, "
            #                      f"{mAPbbox_R40[j, 2, i]:.4f}"))
            # result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
            #                      f"{mAPbev_R40[j, 1, i]:.4f}, "
            #                      f"{mAPbev_R40[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
                                 f"{mAP3d[j, 1, i]:.4f}, "
                                 f"{mAP3d[j, 2, i]:.4f}"))
            # if compute_aos:
            #     result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
            #                          f"{mAPaos_R40[j, 1, i]:.2f}, "
            #                          f"{mAPaos_R40[j, 2, i]:.2f}"))
            #     if i == 0:
            #        ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
            #        ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
            #        ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]
            
            
            # ret_dict[f'{class_to_name[curcls]}_3d/easy_mAP/Overlap_{min_overlaps[i,0,j]:.2f}'] = mAP3d_R40[j, 0, i]
            ret_dict[f'{class_to_name[curcls]}_3d/moderate_mAP/Overlap_{min_overlaps[i,1,j]:.2f}'] = mAP3d[j, 1, i]
            # ret_dict[f'{class_to_name[curcls]}_3d/hard_mAP/Overlap_{min_overlaps[i,2,j]:.2f}'] = mAP3d_R40[j, 2, i]
            # ret_dict[f'{class_to_name[curcls]}_3d/easy_Precision/Overlap_{min_overlaps[i,0,j]:.2f}'] = prec_3d[j, 0, i]
            # ret_dict[f'{class_to_name[curcls]}_3d/moderate_Precision/Overlap_{min_overlaps[i,1,j]:.2f}'] = prec_3d[j, 1, i]
            # ret_dict[f'{class_to_name[curcls]}_3d/hard_Precision/Overlap_{min_overlaps[i,2,j]:.2f}'] = prec_3d[j, 2, i]
            # ret_dict[f'{class_to_name[curcls]}_3d/easy_Recall/Overlap_{min_overlaps[i,0,j]:.2f}'] = rec_3d[j, 0, i]
            # ret_dict[f'{class_to_name[curcls]}_3d/moderate_Recall/Overlap_{min_overlaps[i,1,j]:.2f}'] = rec_3d[j, 1, i]
            # ret_dict[f'{class_to_name[curcls]}_3d/hard_Recall/Overlap_{min_overlaps[i,2,j]:.2f}'] = rec_3d[j, 2, i]
            # ret_dict[f'{class_to_name[curcls]}_bev/easy_R40'] = mAPbev_R40[j, 0, i]
            # ret_dict[f'{class_to_name[curcls]}_bev/moderate_R40'] = mAPbev_R40[j, 1, i]
            # ret_dict[f'{class_to_name[curcls]}_bev/hard_R40'] = mAPbev_R40[j, 2, i]
            # ret_dict[f'{class_to_name[curcls]}_image/easy_R40'] = mAPbbox_R40[j, 0, i]
            # ret_dict[f'{class_to_name[curcls]}_image/moderate_R40'] = mAPbbox_R40[j, 1, i]
            # ret_dict[f'{class_to_name[curcls]}_image/hard_R40'] = mAPbbox_R40[j, 2, i]
            if i == 2:
                ret_dict[f'ALL_{class_to_name[curcls]}_3d/easy_Recall/Overlap_{min_overlaps[i,1,j]:.2f}'] = all_rec_3d[j, 0, i, :]
                ret_dict[f'ALL_{class_to_name[curcls]}_3d/easy_Precision/Overlap_{min_overlaps[i,1,j]:.2f}'] = all_prec_3d[j, 0, i, :]
                ret_dict[f'ALL_{class_to_name[curcls]}_3d/easy_Confidences/Overlap_{min_overlaps[i,1,j]:.2f}'] = all_confidences[j, 0, i, :]

                ret_dict[f'ALL_{class_to_name[curcls]}_3d/moderate_Recall/Overlap_{min_overlaps[i,1,j]:.2f}'] = all_rec_3d[j, 1, i, :]
                ret_dict[f'ALL_{class_to_name[curcls]}_3d/moderate_Precision/Overlap_{min_overlaps[i,1,j]:.2f}'] = all_prec_3d[j, 1, i, :]
                ret_dict[f'ALL_{class_to_name[curcls]}_3d/moderate_Confidences/Overlap_{min_overlaps[i,1,j]:.2f}'] = all_confidences[j, 1, i, :]

                ret_dict[f'ALL_{class_to_name[curcls]}_3d/hard_Recall/Overlap_{min_overlaps[i,1,j]:.2f}'] = all_rec_3d[j, 2, i, :]
                ret_dict[f'ALL_{class_to_name[curcls]}_3d/hard_Precision/Overlap_{min_overlaps[i,1,j]:.2f}'] = all_prec_3d[j, 2, i, :]
                ret_dict[f'ALL_{class_to_name[curcls]}_3d/hard_Confidences/Overlap_{min_overlaps[i,1,j]:.2f}'] = all_confidences[j, 2, i, :]



    return result, ret_dict


def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, "
                             f"{mAPbbox[j, 1]:.2f}, "
                             f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, "
                                 f"{mAPaos[j, 1]:.2f}, "
                                 f"{mAPaos[j, 2]:.2f}"))
    return result

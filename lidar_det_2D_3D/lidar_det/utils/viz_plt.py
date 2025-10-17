import numpy as np
import matplotlib.pyplot as plt

import lidar_det.utils.utils_box3d as ub3d
import lidar_det.utils.jrdb_transforms as jt


def plot_bev(
    pc=None,
    laser_data=None,
    pts_color=None,
    pts_scale=0.01,
    title=None,
    fig=None,
    ax=None,
    xlim=(-10, 10),
    ylim=(-10, 10),
    boxes=None,
    boxes_cls=None,
    scores=None,
    score_thresh=0.0,
    boxes_gt=None,
    boxes_gt_cls=None,
):
    """Plot BEV of LiDAR points

    Args:
        pc (array[3, N]): xyz
        pts_color (array[N, 3] or tuple(3))
        boxes (array[B, 7])
        scores (array[B]): Used to color code box
        score_threh: Box with lower scores are not plotted
        boxes_gt (array[B, 7])

    Returns:
        fig, ax
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

        if title is not None:
            ax.set_title(f"{title}_thresh={score_thresh:.2f}")

    if laser_data is not None:
        # Invert laser data (flip y-axis)
        laser_data = laser_data[::-1]
        laser_xy = jt.transform_laser_to_base(laser_data)

        ax.scatter(laser_xy[0], laser_xy[1], color='blue', s=5, label='Laser Data')


    # lidar
    if pc is not None:
        if pts_color is None:
            s = np.hypot(pc[0], pc[1])
            pts_color = plt.cm.jet(np.clip(s / 30, 0, 1))

        ax.scatter(pc[0], pc[1], color=pts_color, s=pts_scale, label='Lidar Points')

    # boxes
    if boxes is not None and len(boxes) > 0:
        if scores is not None:
            # print("SCORES SHAPE = " + str(len(scores)))
            # print("SCORES MAX = " + str(max(scores)))
            boxes = boxes[scores >= score_thresh]
            scores = scores[scores >= score_thresh]
            # print("BOXES LEN ABOVE THRESHOLD = " + str(len(boxes)))
            # plot low confidence boxes first (on bottom layer)
            s_argsort = scores.argsort()
            boxes = boxes[s_argsort]
            scores = scores[s_argsort]

        # color coded classes
        boxes_color = get_boxes_color(boxes, boxes_cls, (0.0, 1.0, 0.0), scores)
        corners = ub3d.boxes_to_corners(boxes)
        for corner, c in zip(corners, boxes_color):
            # c = 'red'
            inds = [0, 3, 2, 1]
            ax.plot(corner[0, inds], corner[1, inds], linestyle="-", color=c, label='Predicted Boxes')
            ax.plot(corner[0, :2], corner[1, :2], linestyle="--", color=c)

    if boxes_gt is not None and len(boxes_gt) > 0:
        # print("BOXES GT LEN = " + str(len(boxes_gt)))
        boxes_gt_color = get_boxes_color(boxes_gt, boxes_gt_cls, (1.0, 0.0, 0.0))
        corners = ub3d.boxes_to_corners(boxes_gt)
        for corner, c in zip(corners, boxes_gt_color):
            # c = 'blue'
            inds = [0, 3, 2, 1]
            ax.plot(
                corner[0, inds],
                corner[1, inds],
                linestyle="dotted",
                color=c,
                linewidth=2.0,
                label='Ground Truth Boxes'
            )
            ax.plot(
                corner[0, :2], corner[1, :2], linestyle="--", color=c, linewidth=2.0
            )
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    return fig, ax

def plot_p_r(prec, rec, confidences, title_fontsz=20, label_fontsz=18, tick_fontsz=14):
    """
    Plot a precision-recall curve with color-coded points based on confidences.

    Args:
    prec (numpy.ndarray): An array containing precision values.
    rec (numpy.ndarray): An array containing recall values (assumed to be sorted).
    confidences (numpy.ndarray): An array containing the corresponding confidences.

    Returns:
    fig (matplotlib.figure.Figure): The figure object.
    ax (matplotlib.axes.Axes): The axis object.
    """
    # Check that all input arrays have the same length
    assert len(prec) == len(rec) == len(confidences), "Input arrays must have the same length"

    # Ensure the input arrays are NumPy arrays
    assert isinstance(prec, np.ndarray), "Input 'prec' must be a NumPy array"
    assert isinstance(rec, np.ndarray), "Input 'rec' must be a NumPy array"
    assert isinstance(confidences, np.ndarray), "Input 'confidences' must be a NumPy array"

    # Create a figure and an axis with a specified size of (10, 10)
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = np.linspace(1, 0, len(confidences))
    # Create the precision-recall curve
    scatter = ax.scatter(rec, prec, c=colors, marker='o', label='Precision-Recall Points', cmap='viridis_r')
    
    # Connect the points with a line
    ax.plot(rec, prec, linestyle='-', color='gray', alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.set_xlabel('Recall',  fontsize=label_fontsz)
    ax.set_ylabel('Precision',  fontsize=label_fontsz)
    ax.set_title('Precision-Recall Curve', fontsize=title_fontsz)
    ax.tick_params(axis='x', labelsize=tick_fontsz)
    ax.tick_params(axis='y', labelsize=tick_fontsz)

    # Create a custom colorbar with labeled tick locations
    cbar = plt.colorbar(scatter, ax=ax,)
    cbar.set_label('Confidences', fontsize=label_fontsz)
    cbar.set_ticks(colors)
    cbar.set_ticklabels([f'{c:.2f}' for c in confidences])

    return fig, ax

def plot_multiple_p_r(prec_list, rec_list, confidences_list, max_3d_dists, title_fontsz=20, label_fontsz=18, tick_fontsz=14):
    """
    Plot multiple precision-recall curves with color-coded points based on confidences.

    Args:
    prec_list (list of numpy.ndarray): A list of arrays containing precision values for each curve.
    rec_list (list of numpy.ndarray): A list of arrays containing recall values for each curve.
    confidences_list (list of numpy.ndarray): A list of arrays containing the corresponding confidences for each curve.
    max_3d_dists (list of scalars): A list of maximum distances for each precision-recall curve.

    Returns:
    fig (matplotlib.figure.Figure): The figure object.
    ax (matplotlib.axes.Axes): The axis object.
    """
    # Check that all input lists have the same length
    print(len(prec_list), len(rec_list),  len(confidences_list), len(max_3d_dists))
    assert len(prec_list) == len(rec_list) == len(confidences_list) == len(max_3d_dists), "Input lists must have the same length"

    # Ensure the input arrays are NumPy arrays
    for prec, rec, confidences in zip(prec_list, rec_list, confidences_list):
        assert isinstance(prec, np.ndarray), "Precision values must be NumPy arrays"
        assert isinstance(rec, np.ndarray), "Recall values must be NumPy arrays"
        assert isinstance(confidences, np.ndarray), "Confidence values must be NumPy arrays"

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(10, 10))

    markers = ['o', 's', '^', 'v', 'D', 'p', 'H', 'X', '8', '*', '+', '<', '>']  # List of popular scatter markers
    c_max, c_min = 0, 1
    for confidences in confidences_list:
        max_confidence, min_confidence = np.max(np.array(confidences)), np.min(np.array(confidences))
        if max_confidence > c_max:
            c_max = max_confidence
        if min_confidence < c_min:
            c_min = min_confidence

    for i, (prec, rec, confidences, max_dist) in enumerate(zip(prec_list, rec_list, confidences_list, max_3d_dists)):
        # Create the precision-recall curve for each dataset with a unique marker
        scatter = ax.scatter(rec, prec, c=confidences, marker=markers[i], label=f'Max Dist: {max_dist:.2f} m', alpha=0.7, cmap='viridis_r')
        scatter.set_clim(c_min, c_max)

        # Connect the points with a line
        ax.plot(rec, prec, linestyle='-', color='gray', alpha=0.5)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.set_xlabel('Recall',  fontsize=label_fontsz)
    ax.set_ylabel('Precision',  fontsize=label_fontsz)
    ax.set_title('3D Lidar Precision-Recall Curves on JRDB', fontsize=title_fontsz)
    ax.tick_params(axis='x', labelsize=tick_fontsz)
    ax.tick_params(axis='y', labelsize=tick_fontsz)
    ax.legend(loc='lower left', fontsize = tick_fontsz)

    # Add a colorbar to the right of the plot
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidences', fontsize=label_fontsz)
    ticks = np.linspace(min_confidence, max_confidence, 21)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:.2f}' for t in ticks])

    return fig, ax


def get_boxes_color(boxes, boxes_cls, default_color, alphas=None):
    B = len(boxes)
    if boxes_cls is not None:
        if isinstance(boxes_cls, (int, float)):
            boxes_cls = boxes_cls * np.ones(B)
        boxes_color = plt.cm.prism(boxes_cls / 10)
        if alphas is not None:
            boxes_color[:, 3] = alphas
    else:
        boxes_color = np.tile(np.array(default_color), (B, 1))
        if alphas is not None:
            boxes_color = np.concatenate((boxes_color, alphas.reshape(B, 1)), axis=1)

    return boxes_color

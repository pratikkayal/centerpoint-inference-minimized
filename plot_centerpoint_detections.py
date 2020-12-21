
import glob
import pdb
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from argoverse.utils.pkl_utils import load_pkl_dictionary





def render(
    box,
    axis: Axes,
    view: np.ndarray = np.eye(3),
    normalize: bool = False,
    colors: Tuple = ("b", "r", "k"),
    linewidth: float = 2,
) -> None:
    """
    Renders the box in the provided Matplotlib axis.
    :param axis: Axis onto which the box should be drawn.
    :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
        back and sides.
    :param linewidth: Width in pixel of the box sides.
    """
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            axis.plot(
                [prev[0], corner[0]],
                [prev[1], corner[1]],
                color=color,
                linewidth=linewidth,
            )
            prev = corner

    # Draw the sides
    for i in range(4):
        axis.plot(
            [corners.T[i][0], corners.T[i + 4][0]],
            [corners.T[i][1], corners.T[i + 4][1]],
            color=colors[2],
            linewidth=linewidth,
        )

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0])
    draw_rect(corners.T[4:], colors[1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    axis.plot(
        [center_bottom[0], center_bottom_forward[0]],
        [center_bottom[1], center_bottom_forward[1]],
        color=colors[0],
        linewidth=linewidth,
    )


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(
        self,
        center: List[float],
        size: List[float],
        orientation: Quaternion,
        label: int = np.nan,
        score: float = np.nan,
        velocity: Tuple = (np.nan, np.nan, np.nan),
        name: str = None,
        token: str = None,
    ):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        # print(center.shape)
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token


def _second_det_to_nusc_box(detection):
    """ """
    box3d = detection["box3d_lidar"]
    scores = detection["scores"]
    labels = detection["label_preds"]
    box3d[:, -1] = -box3d[:, -1] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(
            list(box3d[i, :3]),
            list(box3d[i, 3:6]),
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list




def visual(points, gt_anno, det, i, eval_range=35, conf_th=0.5):
    """ """
    _, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=200)
    points = remove_close(points, radius=3)
    points = view_points(points[:3, :], np.eye(4), normalize=False)

    dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    boxes_gt = _second_det_to_nusc_box(gt_anno)
    boxes_est = _second_det_to_nusc_box(det)

    # Show GT boxes.
    for box in boxes_gt:
        render_nuscenes_box(
            box, ax, view=np.eye(4), colors=("r", "r", "r"), linewidth=2
        )

    # Show EST boxes.
    for box in boxes_est:
        if box.score >= conf_th:
            render_nuscenes_box(
                box, ax, view=np.eye(4), colors=("b", "b", "b"), linewidth=1
            )

    axes_limit = (
        eval_range + 3
    )  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    plt.axis("off")

    plt.savefig("demo/file%02d.png" % i)
    plt.close()

def read_file(path, tries=2, num_point_feature=4):
    points = None
    try_cnt = 0
    while points is None and try_cnt < tries:
        try_cnt += 1
        try:
            points = np.fromfile(path, dtype=np.float32)
            s = points.shape[0]
            if s % 5 != 0:
                points = points[: s - (s % 5)]
            points = points.reshape(-1, 5)[:, :num_point_feature]
        except Exception:
            points = None

    return points

    
def main():
    """ """
    pkl_fpath = "/Users/jlambert/Downloads/prediction.pkl"
    pkl_data = load_pkl_dictionary(pkl_fpath)
    
    keys = list(pkl_data.keys()) # get the tokens
    token = keys[0]
    
    pkl_data[token]['box3d_lidar']
    pkl_data[token]['scores']
    pkl_data[token]['label_preds']
    pkl_data[token]['metadata']
    
    pdb.set_trace()
    
    lidar_fpath = glob.glob('/Users/jlambert/Downloads/n015*229.pcd.bin')[0]
    from centerpoint.utils.loading import read_file
    points = read_file(lidar_fpath)
    
    from nuscenes_2a1710d55ac747339eae4502565b956b_python import annos
    
    visual(points, gt_anno, dets=pkl_data, i=0, eval_range=35, conf_th=0.5)
    
    
    

if __name__ == "__main__":
    main()

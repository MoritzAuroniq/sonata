"""
ransac/steps/step5_bounding_boxes.py

Step 5 — Per-cluster upright bounding boxes + structured Detection records.
"""

import open3d as o3d
import numpy as np
from typing import List, Tuple

from .. import config
from ..types import Detection


_UP_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _upright_obb(cluster):
    """
    Upright OBB: rotation only around the up axis. The box can rotate to
    fit the object's footprint, but cannot tilt.
    """
    pts = np.asarray(cluster.points)
    up = _UP_AXIS_INDEX[config.UP_AXIS]
    floor_axes = [i for i in (0, 1, 2) if i != up]

    floor_pts = pts[:, floor_axes]                   # (N, 2)
    up_vals   = pts[:, up]                           # (N,)

    # PCA on the floor-projected points to find the best rotation
    centred = floor_pts - floor_pts.mean(axis=0)
    cov = np.cov(centred.T)
    _, eigvecs = np.linalg.eigh(cov)
    primary = eigvecs[:, -1]                          # largest eigenvalue
    theta = float(np.arctan2(primary[1], primary[0]))

    # Rotate points so principal axis aligns with first floor axis
    R2 = np.array([[ np.cos(-theta), -np.sin(-theta)],
                   [ np.sin(-theta),  np.cos(-theta)]])
    rotated = centred @ R2.T

    min2 = rotated.min(axis=0)
    max2 = rotated.max(axis=0)
    size2 = max2 - min2
    centre2_rot = (max2 + min2) / 2

    # Rotate the centre back
    Rback = np.array([[ np.cos(theta), -np.sin(theta)],
                      [ np.sin(theta),  np.cos(theta)]])
    centre2 = Rback @ centre2_rot + floor_pts.mean(axis=0)

    # Build a 3D box
    centre_3d = np.zeros(3)
    centre_3d[floor_axes[0]] = centre2[0]
    centre_3d[floor_axes[1]] = centre2[1]
    centre_3d[up]            = (up_vals.max() + up_vals.min()) / 2

    extent_3d = np.zeros(3)
    extent_3d[floor_axes[0]] = size2[0]
    extent_3d[floor_axes[1]] = size2[1]
    extent_3d[up]            = up_vals.max() - up_vals.min()

    # Rotation matrix — only around the up axis
    c, s = np.cos(theta), np.sin(theta)
    if config.UP_AXIS == "Y":
        R = np.array([[ c, 0, -s],
                      [ 0, 1,  0],
                      [ s, 0,  c]])
    elif config.UP_AXIS == "Z":
        R = np.array([[ c, -s, 0],
                      [ s,  c, 0],
                      [ 0,  0, 1]])
    else:  # X-up
        R = np.array([[1, 0,  0],
                      [0, c, -s],
                      [0, s,  c]])

    obb = o3d.geometry.OrientedBoundingBox(centre_3d, R, extent_3d)
    return obb, float(np.degrees(theta))


def compute_boxes(clusters, room_cloud) -> Tuple[List, List[Detection]]:
    print(f"[Step 5] USE_ORIENTED_BBOX = {config.USE_ORIENTED_BBOX}")
    print(f"[Step 5] UP_AXIS = {config.UP_AXIS}")
    print(f"[Step 5] Drawing bounding boxes for {len(clusters)} clusters")

    room_bb  = room_cloud.get_axis_aligned_bounding_box()
    room_min = room_bb.get_min_bound() - config.ROOM_PADDING
    room_max = room_bb.get_max_bound() + config.ROOM_PADDING

    boxes, detections = [], []
    object_id = 0

    for i, cluster in enumerate(clusters):
        box_type = "AABB"
        theta_deg = 0.0

        if config.USE_ORIENTED_BBOX:
            try:
                box, theta_deg = _upright_obb(cluster)
                size = box.extent
                box_type = "upright-OBB"
            except Exception as e:
                print(f"  Cluster {i}: upright-OBB failed ({e}) — falling back to AABB")
                box = cluster.get_axis_aligned_bounding_box()
                size = box.get_extent()
        else:
            box = cluster.get_axis_aligned_bounding_box()
            size = box.get_extent()

        ctr = box.get_center()

        too_small = min(size) < config.MIN_CLUSTER_DIM
        too_wide  = max(size[0], size[2]) > config.MAX_CLUSTER_DIM
        too_tall  = size[1] > config.MAX_CLUSTER_HEIGHT
        outside   = (
            ctr[0] < room_min[0] or ctr[0] > room_max[0] or
            ctr[1] < room_min[1] or ctr[1] > room_max[1] or
            ctr[2] < room_min[2] or ctr[2] > room_max[2]
        )

        if too_small or too_wide or too_tall or outside:
            reasons = []
            if too_small: reasons.append("too small")
            if too_wide:  reasons.append("too wide")
            if too_tall:  reasons.append("too tall")
            if outside:   reasons.append("outside room")
            print(f"  Cluster {i}: skipped ({', '.join(reasons)}) [{box_type}]")
            continue

        box.color = (1, 0, 0)
        boxes.append(box)

        det = Detection(
            object_id    = f"OBJ-{object_id:03d}",
            centre_xyz_m = (float(ctr[0]), float(ctr[1]), float(ctr[2])),
            size_wdh_m   = (float(size[0]), float(size[1]), float(size[2])),
            theta_deg    = theta_deg,
            point_count  = len(cluster.points),
        )
        detections.append(det)

        print(f"  {det.object_id} [{box_type}]: "
              f"xyz=({ctr[0]:+.2f},{ctr[1]:+.2f},{ctr[2]:+.2f}) m | "
              f"size {size[0]:.2f}x{size[1]:.2f}x{size[2]:.2f} m | "
              f"θ={theta_deg:+.0f}° | {len(cluster.points):,} pts")

        object_id += 1

    return boxes, detections


def make_position_markers(detections):
    markers = []
    for det in detections:
        cx, cy, cz = det.centre_xyz_m
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.25, origin=[cx, cy, cz]
        )
        markers.append(frame)
    return markers
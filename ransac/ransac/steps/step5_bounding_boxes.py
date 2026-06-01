"""
ransac/steps/step5_bounding_boxes.py

Step 5 — Per-cluster bounding boxes + structured Detection records.
Boxes are constrained to stand upright (rotation only around the up axis).
"""

import open3d as o3d
import numpy as np
from typing import List, Tuple

from .. import config
from ..types import Detection


# Map up-axis name to index
_UP_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _upright_obb(cluster: o3d.geometry.PointCloud) -> o3d.geometry.OrientedBoundingBox:
    """
    Compute an oriented bounding box that is constrained to be upright:
    rotation only around the up axis, no tilt.

    1. Take the cluster's XZ (floor-plane) points.
    2. Fit a 2D oriented rectangle to them (find the rotation that minimises area).
    3. Use the full Y range as the box height.
    """
    pts = np.asarray(cluster.points)
    up = _UP_AXIS_INDEX[config.UP_AXIS]

    # Split into up-axis and floor-plane axes
    up_vals = pts[:, up]
    floor_axes = [i for i in (0, 1, 2) if i != up]
    floor_pts = pts[:, floor_axes]  # shape (N, 2)

    # ── Find the best rotation in the floor plane ─────────────────
    # Use PCA on the 2D points — the principal direction is the best alignment
    centred = floor_pts - floor_pts.mean(axis=0)
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvec with largest eigenvalue = primary axis
    primary = eigvecs[:, -1]
    theta = np.arctan2(primary[1], primary[0])  # angle from floor-axis-0

    # Rotate points so primary axis aligns with X — then take AABB in that frame
    R2 = np.array([[ np.cos(-theta), -np.sin(-theta)],
                   [ np.sin(-theta),  np.cos(-theta)]])
    rotated = centred @ R2.T

    min2 = rotated.min(axis=0)
    max2 = rotated.max(axis=0)
    size2 = max2 - min2
    centre2_rot = (max2 + min2) / 2

    # Rotate the rectangle centre back to original floor frame
    Rback = np.array([[ np.cos(theta), -np.sin(theta)],
                      [ np.sin(theta),  np.cos(theta)]])
    centre2 = Rback @ centre2_rot + floor_pts.mean(axis=0)

    # ── Build a 3D OBB from this rectangle ────────────────────────
    centre_3d = np.zeros(3)
    centre_3d[floor_axes[0]] = centre2[0]
    centre_3d[floor_axes[1]] = centre2[1]
    centre_3d[up]           = (up_vals.max() + up_vals.min()) / 2

    extent_3d = np.zeros(3)
    extent_3d[floor_axes[0]] = size2[0]
    extent_3d[floor_axes[1]] = size2[1]
    extent_3d[up]           = up_vals.max() - up_vals.min()

    # Rotation matrix: rotation only around the up axis
    R = np.eye(3)
    c, s = np.cos(theta), np.sin(theta)
    if config.UP_AXIS == "Y":
        # Rotation in XZ plane
        R = np.array([[ c, 0, -s],
                      [ 0, 1,  0],
                      [ s, 0,  c]])
    elif config.UP_AXIS == "Z":
        R = np.array([[ c, -s, 0],
                      [ s,  c, 0],
                      [ 0,  0, 1]])
    elif config.UP_AXIS == "X":
        R = np.array([[1, 0,  0],
                      [0, c, -s],
                      [0, s,  c]])

    obb = o3d.geometry.OrientedBoundingBox(centre_3d, R, extent_3d)
    return obb, float(np.degrees(theta))


def compute_boxes(
    clusters:   List[o3d.geometry.PointCloud],
    room_cloud: o3d.geometry.PointCloud
) -> Tuple[List, List[Detection]]:
    print(f"[Step 5] Drawing upright bounding boxes for {len(clusters)} clusters")

    room_bb  = room_cloud.get_axis_aligned_bounding_box()
    room_min = room_bb.get_min_bound() - config.ROOM_PADDING
    room_max = room_bb.get_max_bound() + config.ROOM_PADDING

    boxes:      List = []
    detections: List[Detection] = []
    object_id = 0

    for i, cluster in enumerate(clusters):
        if not config.USE_ORIENTED_BBOX:
            box  = cluster.get_axis_aligned_bounding_box()
            size = box.get_extent()
            theta_deg = 0.0
        else:
            try:
                box, theta_deg = _upright_obb(cluster)
                size = box.extent
            except Exception:
                box  = cluster.get_axis_aligned_bounding_box()
                size = box.get_extent()
                theta_deg = 0.0

        ctr = box.get_center()

        # Filters
        too_small  = min(size) < config.MIN_CLUSTER_DIM
        too_wide   = max(size[0], size[2]) > config.MAX_CLUSTER_DIM
        too_tall   = size[1] > config.MAX_CLUSTER_HEIGHT
        outside    = (
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
            print(f"  Cluster {i}: skipped ({', '.join(reasons)})")
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

        print(f"  {det.object_id}: "
              f"xyz=({ctr[0]:+.2f},{ctr[1]:+.2f},{ctr[2]:+.2f}) m | "
              f"size {size[0]:.2f}x{size[1]:.2f}x{size[2]:.2f} m | "
              f"θ={theta_deg:+.0f}° | {len(cluster.points):,} pts")

        object_id += 1

    return boxes, detections


def make_position_markers(detections: List[Detection]) -> List[o3d.geometry.TriangleMesh]:
    """Small RGB axis triads at each detected object's centre."""
    markers = []
    for det in detections:
        cx, cy, cz = det.centre_xyz_m
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.25, origin=[cx, cy, cz]
        )
        markers.append(frame)
    return markers
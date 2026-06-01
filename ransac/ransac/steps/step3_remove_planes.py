"""
ransac/steps/step3_remove_planes.py

Step 3 — RANSAC plane removal (peel off floor, ceiling, walls).
Returns the remaining object points, the removed planes, and plane metadata.
"""

import open3d as o3d
import numpy as np
from typing import List, Tuple
from .. import config


# Index of the "up" axis in the (x, y, z) tuple
_UP_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def remove_planes(
    pcd: o3d.geometry.PointCloud
) -> Tuple[
    o3d.geometry.PointCloud,
    List[o3d.geometry.PointCloud],
    List[Tuple[str, float]]
]:
    up_idx = _UP_AXIS_INDEX[config.UP_AXIS]

    remaining    = pcd
    plane_clouds = []
    plane_info   = []  # parallel list: (kind, mean_up_value)

    print(f"[Step 3] Starting with {len(remaining.points):,} points")

    for i in range(config.PLANE_MAX_COUNT):
        if len(remaining.points) < config.PLANE_MIN_REMAINING:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=config.PLANE_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=config.PLANE_RANSAC_ITERATIONS,
        )
        # Normal is [a, b, c]. The up-axis component tells us horizontal vs vertical.
        normal = plane_model[:3]
        kind = ("horizontal" if abs(normal[up_idx]) > config.HORIZONTAL_NORMAL_THRESH
                else "vertical")

        plane_cloud = remaining.select_by_index(inliers)
        remaining   = remaining.select_by_index(inliers, invert=True)

        mean_up = float(np.asarray(plane_cloud.points)[:, up_idx].mean())
        plane_info.append((kind, mean_up))

        plane_cloud.paint_uniform_color([1, 0, 0] if kind == "horizontal" else [0, 0, 1])
        plane_clouds.append(plane_cloud)

        print(f"  Plane {i+1}: {len(inliers):,} pts | "
              f"n=({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f}) | "
              f"{kind} | mean {config.UP_AXIS}={mean_up:+.2f}m")

    remaining.paint_uniform_color([0, 0.8, 0])
    print(f"[Step 3] Remaining objects: {len(remaining.points):,} points")
    return remaining, plane_clouds, plane_info


def drop_ceiling(
    plane_clouds: List[o3d.geometry.PointCloud],
    plane_info:   List[Tuple[str, float]]
) -> List[o3d.geometry.PointCloud]:
    """Return plane_clouds with the highest horizontal plane removed (the ceiling)."""
    horizontals = [(i, h) for i, (k, h) in enumerate(plane_info) if k == "horizontal"]
    if not horizontals:
        return plane_clouds
    ceiling_idx = max(horizontals, key=lambda x: x[1])[0]
    print(f"[Step 6] Ceiling identified at plane index {ceiling_idx} "
          f"({config.UP_AXIS}={plane_info[ceiling_idx][1]:+.2f}m) — removing")
    return [p for i, p in enumerate(plane_clouds) if i != ceiling_idx]


def get_floor(plane_clouds, plane_info):
    """Return the lowest horizontal plane (the floor)."""
    horizontals = [(i, h) for i, (k, h) in enumerate(plane_info) if k == "horizontal"]
    if not horizontals:
        return None, None
    floor_idx = min(horizontals, key=lambda x: x[1])[0]
    floor_y   = plane_info[floor_idx][1]
    print(f"[Step 7] Floor identified at plane index {floor_idx} "
          f"({config.UP_AXIS}={floor_y:+.2f}m)")
    return plane_clouds[floor_idx], floor_y

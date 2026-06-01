"""
ransac/steps/step1_load.py

Step 1 — Load the PLY file, report basic stats, auto-detect up axis.
"""

from pathlib import Path
from typing import Union
import numpy as np
import open3d as o3d


def load(pcd_path: Union[str, Path]) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if len(pcd.points) == 0:
        raise ValueError(f"No points loaded from {pcd_path} — file missing or unreadable")
    print(f"[Step 1] Loaded {len(pcd.points):,} points from {Path(pcd_path).name}")
    return pcd


def detect_up_axis(pcd: o3d.geometry.PointCloud) -> str:
    """
    Auto-detect which axis points "up" using two independent heuristics
    and a consensus check.

      Heuristic A — bounding box: the axis with the smallest extent is
                    almost always the up axis (rooms are wider than tall).
      Heuristic B — plane normal: RANSAC the biggest plane, take its normal,
                    the dominant component is the up axis.

    If they agree, return that axis. If they disagree, trust the plane
    normal (stronger geometric evidence).
    """
    pts = np.asarray(pcd.points)

    # ── Heuristic A — bounding box extents ────────────────────────
    extents = pts.max(axis=0) - pts.min(axis=0)
    bb_idx = int(np.argmin(extents))

    # ── Heuristic B — largest plane normal ────────────────────────
    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.05, ransac_n=3, num_iterations=500
    )
    normal = np.abs(plane_model[:3])
    plane_idx = int(np.argmax(normal))

    # ── Decide ────────────────────────────────────────────────────
    bb_axis    = "XYZ"[bb_idx]
    plane_axis = "XYZ"[plane_idx]

    if bb_idx == plane_idx:
        axis = bb_axis
        print(f"[Auto] Up axis = {axis} "
              f"(extents {extents[0]:.1f},{extents[1]:.1f},{extents[2]:.1f} m · "
              f"plane normal aligns with {plane_axis})")
    else:
        axis = plane_axis
        print(f"[Auto] Up axis = {axis} "
              f"(BB suggested {bb_axis}, plane normal stronger evidence → {plane_axis})")

    return axis
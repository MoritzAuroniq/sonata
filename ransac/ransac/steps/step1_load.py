"""
ransac/steps/step1_load.py

Step 1 — Load the PLY file and report basic stats.
"""

from pathlib import Path
import open3d as o3d


def load(pcd_path: str | Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if len(pcd.points) == 0:
        raise ValueError(f"No points loaded from {pcd_path} — file missing or unreadable")
    print(f"[Step 1] Loaded {len(pcd.points):,} points from {Path(pcd_path).name}")
    return pcd

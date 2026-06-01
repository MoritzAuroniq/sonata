"""
ransac/steps/step1_5_denoise.py

Step 1.5 — Statistical outlier removal (drop floating noise).
"""

import open3d as o3d
from .. import config


def denoise(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=config.SOR_NEIGHBOURS,
        std_ratio=config.SOR_STD_RATIO,
    )
    removed = len(pcd.points) - len(pcd_clean.points)
    print(f"[Step 1.5] Removed {removed:,} noise points "
          f"→ {len(pcd_clean.points):,} remain")
    return pcd_clean

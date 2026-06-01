"""
ransac/steps/step2_downsample.py

Step 2 — Voxel downsampling.
"""

import open3d as o3d
from .. import config


def downsample(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd_down = pcd.voxel_down_sample(voxel_size=config.VOXEL_SIZE)
    print(f"[Step 2] Downsampled {len(pcd.points):,} → "
          f"{len(pcd_down.points):,} points (voxel = {config.VOXEL_SIZE*100:.0f} cm)")
    return pcd_down

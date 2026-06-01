"""
ransac/steps/step4_cluster.py

Step 4 — DBSCAN clustering of the remaining object points.
"""

import open3d as o3d
import numpy as np
from typing import List
from .. import config


def cluster(pcd: o3d.geometry.PointCloud) -> List[o3d.geometry.PointCloud]:
    labels = np.array(pcd.cluster_dbscan(
        eps=config.DBSCAN_EPS,
        min_points=config.DBSCAN_MIN_POINTS,
        print_progress=False,
    ))
    if len(labels) == 0:
        print("[Step 4] No points to cluster")
        return []

    n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
    n_noise    = int((labels == -1).sum())
    print(f"[Step 4] Found {n_clusters} clusters, {n_noise:,} noise points")

    clusters = []
    rng = np.random.default_rng(42)
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        cluster_cloud = pcd.select_by_index(idx)
        cluster_cloud.paint_uniform_color(rng.uniform(0.3, 1.0, 3))
        clusters.append(cluster_cloud)
    return clusters

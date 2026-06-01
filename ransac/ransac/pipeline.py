"""
ransac/pipeline.py

Orchestrates the full Phase 2 pipeline. Returns Detection records.
"""

from pathlib import Path
from typing import List, Tuple, Union

import open3d as o3d

from . import config
from . import steps
from .types import Detection
from .steps.step1_load           import detect_up_axis
from .steps.step3_remove_planes  import get_floor
from .steps.step5_bounding_boxes import filter_on_floor


def run(pcd_path: Union[str, Path] = None) -> Tuple[List[Detection], dict]:
    pcd_path = pcd_path or config.DEFAULT_PLY

    # ── Load + auto-detect up axis ────────────────────────────────
    pcd = steps.load(pcd_path)
    if config.UP_AXIS == "AUTO":
        config.UP_AXIS = detect_up_axis(pcd)

    # ── Main pipeline ─────────────────────────────────────────────
    pcd_clean                    = steps.denoise(pcd)
    pcd_down                     = steps.downsample(pcd_clean)
    objects, planes, plane_info  = steps.remove_planes(pcd_down)
    clusters                     = steps.cluster(objects)
    boxes, detections            = steps.compute_boxes(clusters, pcd_down)
    markers                      = steps.make_position_markers(detections)

    # ── Step 7 — floor-only filter ────────────────────────────────
    floor_cloud, floor_height = get_floor(planes, plane_info)

    if floor_cloud is not None:
        # Floor in red
        floor_cloud_red = o3d.geometry.PointCloud(floor_cloud)
        floor_cloud_red.paint_uniform_color([1, 0, 0])

        # Keep only ground-level objects
        floor_dets, floor_boxes = filter_on_floor(detections, boxes, floor_height)

        # Ground-level clusters in green
        floor_clusters = []
        for cluster, det in zip(clusters, detections):
            if det in floor_dets:
                green = o3d.geometry.PointCloud(cluster)
                green.paint_uniform_color([0, 0.8, 0])
                floor_clusters.append(green)
    else:
        floor_cloud_red = None
        floor_dets, floor_boxes, floor_clusters = [], [], []

    # ── Pack everything for the viewer ────────────────────────────
    artefacts = {
        "raw":              [pcd],
        "denoised":         [pcd_clean],
        "downsampled":      [pcd_down],
        "planes":           planes,
        "objects":          [objects],
        "clusters":         clusters,
        "boxes":            boxes,
        "markers":          markers,
        # Step 7 artefacts
        "floor":            [floor_cloud_red] if floor_cloud_red is not None else [],
        "floor_clusters":   floor_clusters,
        "floor_boxes":      floor_boxes,
        "floor_detections": floor_dets,
    }
    return detections, artefacts
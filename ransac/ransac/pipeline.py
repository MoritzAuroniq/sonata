"""
ransac/pipeline.py

Orchestrates the full Phase 2 pipeline. Returns Detection records.
Pure logic — no rendering — so it can be reused in batch / API contexts.
"""

from pathlib import Path
from typing import List, Tuple

from . import config
from . import steps
from .types import Detection


def run(pcd_path: str | Path = None) -> Tuple[List[Detection], dict]:
    """
    Run the full Phase 2 pipeline on a PLY file.

    Returns:
        detections: list of Detection records (the actual output)
        artefacts:  dict of intermediate geometries for visualisation
    """
    pcd_path = pcd_path or config.DEFAULT_PLY

    pcd                          = steps.load(pcd_path)
    pcd_clean                    = steps.denoise(pcd)
    pcd_down                     = steps.downsample(pcd_clean)
    objects, planes, plane_info  = steps.remove_planes(pcd_down)
    clusters                     = steps.cluster(objects)
    boxes, detections            = steps.compute_boxes(clusters, pcd_down)
    markers                      = steps.make_position_markers(detections)
    planes_no_ceiling            = steps.drop_ceiling(planes, plane_info)

    artefacts = {
        "raw":              [pcd],
        "denoised":         [pcd_clean],
        "downsampled":      [pcd_down],
        "planes":           planes,
        "objects":          [objects],
        "clusters":         clusters,
        "boxes":            boxes,
        "markers":          markers,
        "planes_no_ceiling": planes_no_ceiling,
    }
    return detections, artefacts

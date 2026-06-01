"""
ransac/pipeline.py

Orchestrates the full Phase 2 pipeline. Returns Detection records.
"""

from pathlib import Path
from typing import List, Tuple, Union

from . import config
from . import steps
from .types import Detection
from .steps.step1_load import detect_up_axis


def run(pcd_path: Union[str, Path] = None) -> Tuple[List[Detection], dict]:
    pcd_path = pcd_path or config.DEFAULT_PLY

    # ── Load + auto-detect up axis ────────────────────────────────
    pcd = steps.load(pcd_path)
    # config.UP_AXIS = detect_up_axis(pcd)        # override per scan

    if config.UP_AXIS == "AUTO":
        config.UP_AXIS = detect_up_axis(pcd)
    else:
        print(f"[Auto] Up axis forced from config: {config.UP_AXIS}")

    # ── Rest of the pipeline ──────────────────────────────────────
    pcd_clean                    = steps.denoise(pcd)
    pcd_down                     = steps.downsample(pcd_clean)
    objects, planes, plane_info  = steps.remove_planes(pcd_down)
    clusters                     = steps.cluster(objects)
    boxes, detections            = steps.compute_boxes(clusters, pcd_down)
    markers                      = steps.make_position_markers(detections)
    planes_no_ceiling            = steps.drop_ceiling(planes, plane_info)

    artefacts = {
        "raw":               [pcd],
        "denoised":          [pcd_clean],
        "downsampled":       [pcd_down],
        "planes":            planes,
        "objects":           [objects],
        "clusters":          clusters,
        "boxes":             boxes,
        "markers":           markers,
        "planes_no_ceiling": planes_no_ceiling,
    }
    return detections, artefacts
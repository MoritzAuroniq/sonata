"""
ransac/visualise.py

Visualisation helpers for the pipeline.
Kept separate from the pipeline logic so headless / batch runs skip these.
"""

import open3d as o3d
from typing import List


def show(geometries: List, title: str = "Phase 2") -> None:
    """Open an interactive viewer with the given geometries."""
    o3d.visualization.draw_geometries(geometries, window_name=title)

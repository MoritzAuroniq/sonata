"""
ransac/visualise.py

Visualisation helpers.
- show()       — basic viewer (no labels)
- show_labeled() — GUI viewer with 3D text labels above each detected object
"""

from typing import List, Optional
import open3d as o3d
import numpy as np


def show(geometries: List, title: str = "Phase 2") -> None:
    """Basic Open3D viewer — no labels."""
    o3d.visualization.draw_geometries(geometries, window_name=title)


def show_labeled(
    geometries: List,
    detections: Optional[list] = None,
    title: str = "Phase 2 — labeled",
    label_size: int = 18,
) -> None:
    """
    GUI viewer with floating 3D text labels above each detection.

    detections: list of Detection objects (from ransac.types).
                Each gets a label showing its object_id at the centre top.
    """
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()

    win = app.create_window(title, 1280, 800)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)
    win.add_child(scene)

    # ── Add geometries ────────────────────────────────────────────
    mat_points = rendering.MaterialRecord()
    mat_points.shader = "defaultUnlit"
    mat_points.point_size = 3.0

    mat_lines = rendering.MaterialRecord()
    mat_lines.shader = "unlitLine"
    mat_lines.line_width = 2.0

    for i, geom in enumerate(geometries):
        name = f"geom_{i}"
        if isinstance(geom, o3d.geometry.PointCloud):
            scene.scene.add_geometry(name, geom, mat_points)
        elif isinstance(geom, (o3d.geometry.OrientedBoundingBox,
                                o3d.geometry.AxisAlignedBoundingBox)):
            # Convert box to a LineSet so we can control its appearance
            lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(geom) \
                if isinstance(geom, o3d.geometry.OrientedBoundingBox) \
                else o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(geom)
            lineset.paint_uniform_color([1, 0, 0])
            scene.scene.add_geometry(name, lineset, mat_lines)
        elif isinstance(geom, o3d.geometry.TriangleMesh):
            scene.scene.add_geometry(name, geom, mat_points)
        else:
            try:
                scene.scene.add_geometry(name, geom, mat_points)
            except Exception:
                pass

    # ── Add 3D text labels ────────────────────────────────────────
    if detections:
        for det in detections:
            cx, cy, cz = det.centre_xyz_m
            w, h, d   = det.size_wdh_m

            # Float the label a little above the top of the box
            # (assumes Y is up; the small offset is in metres)
            label_pos = [cx, cy + h / 2 + 0.15, cz]

            text = f"{det.object_id}\n{w:.2f}×{d:.2f} m  θ={det.theta_deg:+.0f}°"
            label = scene.add_3d_label(label_pos, text)
            label.color = gui.Color(0.2, 0.9, 0.3)    # green
            label.scale = 0.5

    # ── Camera ────────────────────────────────────────────────────
    bounds = scene.scene.bounding_box
    scene.setup_camera(60, bounds, bounds.get_center())

    app.run()
"""
Path A — Classical point cloud pipeline.
Tune parameters at the top. Final viewer shows all stages side by side.
"""

import open3d as o3d
import numpy as np


# ══════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS
# ══════════════════════════════════════════════════════════════════

PLY_PATH = "/home/neura_ai/Downloads/meeting_room.ply"

# Step 1.5 — Noise removal (statistical outlier)
SOR_NEIGHBOURS = 20                  #     | neighbours considered per point
SOR_STD_RATIO  = 2.0                 #     | lower = stricter, more points removed

# Step 2 — Downsample
VOXEL_SIZE = 0.05                    # m   | smaller = more detail, slower

# Step 3 — Plane removal (RANSAC)
PLANE_DISTANCE_THRESHOLD = 0.03      # m   | how close to count as "on plane"
PLANE_RANSAC_ITERATIONS  = 1000      #     | candidates tried per pass
PLANE_MAX_COUNT          = 4         #     | how many planes to peel off
HORIZONTAL_NORMAL_THRESH = 0.9       #     | |normal.y| above this = horizontal
PLANE_MIN_REMAINING      = 1000      #     | stop if fewer points left

# Step 4 — Clustering (DBSCAN)
DBSCAN_EPS        = 0.20             # m   | max distance between neighbours
DBSCAN_MIN_POINTS = 50               #     | minimum points to form a cluster

# Step 5 — Bounding box filters
MIN_CLUSTER_DIM    = 0.15            # m   | discard clusters smaller than this
MAX_CLUSTER_DIM    = 5.0             # m   | discard clusters wider than this
MAX_CLUSTER_HEIGHT = 3.0             # m   | discard clusters taller than this
ROOM_PADDING = 0.5                   # m   | how far outside the room we tolerate

# Display — side-by-side layout
GAP_BETWEEN_STAGES = 5.0             # m   | space between stages in viewer


# ══════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════

def step1_load(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"[Step 1] Loaded {len(pcd.points):,} points")
    return pcd

def step1_5_denoise(pcd):
    pcd_clean, kept = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NEIGHBOURS,
        std_ratio=SOR_STD_RATIO,
    )
    removed = len(pcd.points) - len(pcd_clean.points)
    print(f"[Step 1.5] Removed {removed:,} noise points  → {len(pcd_clean.points):,} remain")
    return pcd_clean


def step2_downsample(pcd):
    pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"[Step 2] Downsampled {len(pcd.points):,} → {len(pcd_down.points):,} points")
    return pcd_down


def step3_remove_planes(pcd):
    remaining    = pcd
    plane_clouds = []

    print(f"[Step 3] Starting with {len(remaining.points):,} points")

    for i in range(PLANE_MAX_COUNT):
        if len(remaining.points) < PLANE_MIN_REMAINING:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=PLANE_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=PLANE_RANSAC_ITERATIONS,
        )
        a, b, c, d = plane_model
        kind = "horizontal" if abs(b) > HORIZONTAL_NORMAL_THRESH else "vertical"

        plane_cloud = remaining.select_by_index(inliers)
        remaining   = remaining.select_by_index(inliers, invert=True)

        plane_cloud.paint_uniform_color([1, 0, 0] if kind == "horizontal" else [0, 0, 1])
        plane_clouds.append(plane_cloud)

        print(f"  Plane {i+1}: {len(inliers):,} pts | n=({a:.2f},{b:.2f},{c:.2f}) | {kind}")

    remaining.paint_uniform_color([0, 0.8, 0])
    print(f"[Step 3] Remaining objects: {len(remaining.points):,} points")
    return remaining, plane_clouds


def step4_cluster(pcd):
    labels = np.array(pcd.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=False,
    ))
    n_clusters = labels.max() + 1
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


def step5_bounding_boxes(clusters, room_cloud):
    print(f"[Step 5] Drawing bounding boxes for {len(clusters)} clusters")

    # Define the room's bounding box (with a small padding tolerance)
    room_bb = room_cloud.get_axis_aligned_bounding_box()
    room_min = room_bb.get_min_bound() - ROOM_PADDING
    room_max = room_bb.get_max_bound() + ROOM_PADDING

    boxes = []
    for i, cluster in enumerate(clusters):
        box  = cluster.get_axis_aligned_bounding_box()
        size = box.get_extent()
        ctr  = box.get_center()

        # Size filter
        too_small  = min(size) < MIN_CLUSTER_DIM
        too_wide   = max(size[0], size[2]) > MAX_CLUSTER_DIM
        too_tall   = size[1] > MAX_CLUSTER_HEIGHT

        # Position filter — centre must lie inside the room
        outside_room = (
            ctr[0] < room_min[0] or ctr[0] > room_max[0] or
            ctr[1] < room_min[1] or ctr[1] > room_max[1] or
            ctr[2] < room_min[2] or ctr[2] > room_max[2]
        )

        if too_small or too_wide or too_tall or outside_room:
            reason = []
            if too_small:    reason.append("too small")
            if too_wide:     reason.append("too wide")
            if too_tall:     reason.append("too tall")
            if outside_room: reason.append("outside room")
            print(f"  Object {i}: skipped ({', '.join(reason)})  "
                  f"size {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m")
            continue

        box.color = (1, 0, 0)
        boxes.append(box)
        print(f"  Object {i}: kept  size {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m")

    return boxes


# ══════════════════════════════════════════════════════════════════
# SIDE-BY-SIDE VIEWER
# ══════════════════════════════════════════════════════════════════

def show_all_stages(stages):
    """
    stages: list of (label, geometry_list).
    Lays each stage out along the X axis with a gap between them.
    """
    from copy import deepcopy

    laid_out = []
    offset_x = 0.0

    for label, geoms in stages:
        if not geoms:
            continue

        # Find the X extent of this stage's geometries (use the first one)
        ref = geoms[0]
        bb = ref.get_axis_aligned_bounding_box()
        min_b = bb.get_min_bound()
        max_b = bb.get_max_bound()
        width = max_b[0] - min_b[0]

        # Shift every geometry in this stage by current offset
        for g in geoms:
            g_copy = deepcopy(g)
            g_copy.translate((offset_x - min_b[0], 0, 0))
            laid_out.append(g_copy)

        print(f"  Stage placed: {label} at X offset {offset_x:.2f} m")
        offset_x += width + GAP_BETWEEN_STAGES

    o3d.visualization.draw_geometries(
        laid_out,
        window_name="All stages — raw | downsampled | planes removed | clustered+boxes"
    )


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pcd                  = step1_load(PLY_PATH)
    pcd_clean            = step1_5_denoise(pcd)
    pcd_down             = step2_downsample(pcd_clean)
    objects, planes      = step3_remove_planes(pcd_down)
    clusters             = step4_cluster(objects)
    boxes                = step5_bounding_boxes(clusters, pcd_down)

    # Step 1 — raw scan
    # o3d.visualization.draw_geometries([pcd], window_name="Step 1 — raw scan")

    # Step 1.5 — denoised
    # o3d.visualization.draw_geometries([pcd_clean], window_name="Step 1.5 — denoised")

    # Step 2 — downsampled
    # o3d.visualization.draw_geometries([pcd_down], window_name="Step 2 — downsampled")

    # Step 3 — planes removed
    # o3d.visualization.draw_geometries(planes + [objects], window_name="Step 3 — planes removed")

    # Step 4 — clusters
    o3d.visualization.draw_geometries(clusters, window_name="Step 4 — clusters")

    # Step 5 — clusters + bounding boxes
    o3d.visualization.draw_geometries(clusters + boxes, window_name="Step 5 — clusters with bounding boxes")
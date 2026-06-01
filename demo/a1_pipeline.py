"""
Path A — Classical point cloud pipeline.
Tune parameters at the top.
"""

import open3d as o3d
import numpy as np
from copy import deepcopy


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
ROOM_PADDING       = 0.5             # m   | how far outside the room we tolerate

# Bounding box style
USE_ORIENTED_BBOX  = True            #     | True = OBB (rotates to fit), False = AABB
LABEL_TEXT_SIZE    = 0.12            # m   | size of floating label text


# ══════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════

def step1_load(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"[Step 1] Loaded {len(pcd.points):,} points")
    return pcd


def step1_5_denoise(pcd):
    pcd_clean, _ = pcd.remove_statistical_outlier(
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
    plane_info   = []   # parallel list: (kind, mean_y)

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

        mean_y = np.asarray(plane_cloud.points)[:, 1].mean()
        plane_info.append((kind, mean_y))

        plane_cloud.paint_uniform_color([1, 0, 0] if kind == "horizontal" else [0, 0, 1])
        plane_clouds.append(plane_cloud)

        print(f"  Plane {i+1}: {len(inliers):,} pts | n=({a:.2f},{b:.2f},{c:.2f}) | "
              f"{kind} | mean Y={mean_y:+.2f}m")

    remaining.paint_uniform_color([0, 0.8, 0])
    print(f"[Step 3] Remaining objects: {len(remaining.points):,} points")
    return remaining, plane_clouds, plane_info


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
    """
    Returns:
        boxes      — list of bounding box geometries (for rendering)
        detections — list of dicts with structured info per object
    """
    print(f"[Step 5] Drawing bounding boxes for {len(clusters)} clusters")

    room_bb  = room_cloud.get_axis_aligned_bounding_box()
    room_min = room_bb.get_min_bound() - ROOM_PADDING
    room_max = room_bb.get_max_bound() + ROOM_PADDING

    boxes      = []
    detections = []
    object_id  = 0

    for i, cluster in enumerate(clusters):
        # Decide which kind of box to compute
        if USE_ORIENTED_BBOX:
            try:
                box = cluster.get_oriented_bounding_box()
            except RuntimeError:
                box = cluster.get_axis_aligned_bounding_box()
        else:
            box = cluster.get_axis_aligned_bounding_box()

        # Extract dimensions and centre
        ctr  = box.get_center()
        size = box.extent if USE_ORIENTED_BBOX else box.get_extent()

        # Size + position filters
        too_small  = min(size) < MIN_CLUSTER_DIM
        too_wide   = max(size[0], size[2]) > MAX_CLUSTER_DIM
        too_tall   = size[1] > MAX_CLUSTER_HEIGHT
        outside    = (
            ctr[0] < room_min[0] or ctr[0] > room_max[0] or
            ctr[1] < room_min[1] or ctr[1] > room_max[1] or
            ctr[2] < room_min[2] or ctr[2] > room_max[2]
        )

        if too_small or too_wide or too_tall or outside:
            reasons = []
            if too_small: reasons.append("too small")
            if too_wide:  reasons.append("too wide")
            if too_tall:  reasons.append("too tall")
            if outside:   reasons.append("outside room")
            print(f"  Cluster {i}: skipped ({', '.join(reasons)})")
            continue

        # Extract orientation angle around Y (rotation in the floor plane)
        if USE_ORIENTED_BBOX:
            R = box.R
            theta_deg = float(np.degrees(np.arctan2(R[2, 0], R[0, 0])))
        else:
            theta_deg = 0.0

        # Colour the box red (OBB and AABB both support .color)
        box.color = (1, 0, 0)
        boxes.append(box)

        detection = {
            "object_id": f"OBJ-{object_id:03d}",
            "centre_xyz_m": (float(ctr[0]), float(ctr[1]), float(ctr[2])),
            "size_wdh_m":   (float(size[0]), float(size[1]), float(size[2])),
            "theta_deg":    theta_deg,
            "point_count":  len(cluster.points),
        }
        detections.append(detection)

        print(f"  {detection['object_id']}: "
              f"xyz=({ctr[0]:+.2f}, {ctr[1]:+.2f}, {ctr[2]:+.2f}) m  "
              f"| size {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m  "
              f"| θ={theta_deg:+.0f}°  | {len(cluster.points):,} pts")

        object_id += 1

    return boxes, detections


def make_text_labels(detections):
    """
    Build 3D text geometries floating above each detected object.
    Open3D's text3d primitive requires the legacy API and is not available
    on all builds — so we fall back to a small coordinate frame marker.
    """
    labels = []
    for det in detections:
        cx, cy, cz = det["centre_xyz_m"]
        size = det["size_wdh_m"]

        # Place a small coordinate frame at the centre of each object
        # to mark its position and orientation visually
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.25, origin=[cx, cy, cz]
        )
        labels.append(frame)

        # Try to add 3D text (works in newer Open3D builds)
        try:
            label_text = f"{det['object_id']}"
            text_mesh = o3d.t.geometry.TriangleMesh.create_text(
                label_text, depth=0.0
            ).to_legacy()
            # Scale and position above the object
            text_mesh.scale(LABEL_TEXT_SIZE / 10.0, center=(0, 0, 0))
            text_mesh.translate((cx - 0.3, cy + size[1] / 2 + 0.2, cz))
            text_mesh.paint_uniform_color([0.1, 0.1, 0.1])
            labels.append(text_mesh)
        except Exception:
            pass  # legacy build without text support — coordinate frames are enough

    return labels


def remove_ceiling(plane_clouds, plane_info):
    """Drop the highest horizontal plane from the list."""
    horizontals = [(i, h) for i, (k, h) in enumerate(plane_info) if k == "horizontal"]
    if not horizontals:
        return plane_clouds
    ceiling_idx = max(horizontals, key=lambda x: x[1])[0]
    print(f"[Step 6] Ceiling identified at plane index {ceiling_idx} "
          f"(Y={plane_info[ceiling_idx][1]:+.2f}m) — removing")
    return [p for i, p in enumerate(plane_clouds) if i != ceiling_idx]


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pcd                          = step1_load(PLY_PATH)
    pcd_clean                    = step1_5_denoise(pcd)
    pcd_down                     = step2_downsample(pcd_clean)
    objects, planes, plane_info  = step3_remove_planes(pcd_down)
    clusters                     = step4_cluster(objects)
    boxes, detections            = step5_bounding_boxes(clusters, pcd_down)
    labels                       = make_text_labels(detections)
    planes_no_ceiling            = remove_ceiling(planes, plane_info)

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"DETECTED {len(detections)} OBJECTS")
    print("=" * 70)
    print(f"{'ID':<10}{'X':>8}{'Y':>8}{'Z':>8}    {'W':>6}{'H':>6}{'D':>6}    {'θ':>6}")
    print("-" * 70)
    for d in detections:
        x, y, z = d["centre_xyz_m"]
        w, h, dp = d["size_wdh_m"]
        print(f"{d['object_id']:<10}{x:>+8.2f}{y:>+8.2f}{z:>+8.2f}    "
              f"{w:>6.2f}{h:>6.2f}{dp:>6.2f}    {d['theta_deg']:>+6.0f}°")
    print("=" * 70 + "\n")

    # Step 1 — raw scan
    # o3d.visualization.draw_geometries([pcd], window_name="Step 1 — raw scan")

    # Step 1.5 — denoised
    # o3d.visualization.draw_geometries([pcd_clean], window_name="Step 1.5 — denoised")

    # Step 2 — downsampled
    # o3d.visualization.draw_geometries([pcd_down], window_name="Step 2 — downsampled")

    # Step 3 — planes removed
    # o3d.visualization.draw_geometries(planes + [objects], window_name="Step 3 — planes removed")

    # Step 4 — clusters
    # o3d.visualization.draw_geometries(clusters, window_name="Step 4 — clusters")

    # Step 5 — clusters + oriented bounding boxes + position markers
    o3d.visualization.draw_geometries(
        clusters + boxes + labels,
        window_name="Step 5 — objects with OBB + position markers"
    )

    # Step 6 — full scene without ceiling
    o3d.visualization.draw_geometries(
        planes_no_ceiling + clusters + boxes + labels,
        window_name="Step 6 — scene without ceiling"
    )
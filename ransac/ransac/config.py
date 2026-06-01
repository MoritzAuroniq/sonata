"""
ransac/config.py

All tunable parameters in one place. Import from here instead of hard-coding values.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

# Default input path (override per-run on the CLI)
DEFAULT_PLY = DATA_DIR / "meeting_room.ply"


# ── Step 1.5 — Noise removal (statistical outlier) ────────────────
SOR_NEIGHBOURS = 20                  #     | neighbours considered per point
SOR_STD_RATIO  = 1.5                 #     | lower = stricter

# ── Step 2 — Downsample ───────────────────────────────────────────
VOXEL_SIZE = 0.05                    # m   | smaller = more detail, slower

# ── Step 3 — Plane removal (RANSAC) ───────────────────────────────
PLANE_DISTANCE_THRESHOLD = 0.02      # m   | how close to count as "on plane"
PLANE_RANSAC_ITERATIONS  = 1000      #     | candidates tried per pass
PLANE_MAX_COUNT          = 6         #     | how many planes to peel off
HORIZONTAL_NORMAL_THRESH = 0.9       #     | |normal.y| above this = horizontal
PLANE_MIN_REMAINING      = 1000      #     | stop if fewer points left

# ── Step 4 — Clustering (DBSCAN) ──────────────────────────────────
DBSCAN_EPS        = 0.20             # m   | max distance between neighbours
DBSCAN_MIN_POINTS = 50               #     | minimum points to form a cluster

# ── Step 5 — Bounding box filters ─────────────────────────────────
MIN_CLUSTER_DIM    = 0.15            # m   | discard clusters smaller than this
MAX_CLUSTER_DIM    = 5.0             # m   | discard clusters wider than this
MAX_CLUSTER_HEIGHT = 3.0             # m   | discard clusters taller than this
ROOM_PADDING       = 0.5             # m   | how far outside the room we tolerate

USE_ORIENTED_BBOX  = True            #     | True = OBB, False = AABB

# ── Up-axis convention ────────────────────────────────────────────
# Your meeting room scan has Y as the up axis (floor lies in XZ).
# Change to "Z" if your data is Z-up (standard for most LiDAR).
UP_AXIS = "AUTO"     # was "Y"

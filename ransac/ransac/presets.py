"""
ransac/presets.py

Parameter presets per environment type.

The values in config.py are the DEFAULT preset — they work for typical
indoor scans (meeting rooms, offices). The presets below override
specific parameters for tighter or larger spaces.

Apply via:  python scripts/run.py --preset tight
            python scripts/run.py --preset large
"""

PRESETS = {

    # ────────────────────────────────────────────────────────────
    # DEFAULT — mirrors config.py
    # Typical indoor scans: meeting rooms, offices, labs.
    # ────────────────────────────────────────────────────────────
    "default": {
        "SOR_STD_RATIO":            1.5,
        "VOXEL_SIZE":               0.05,
        "PLANE_DISTANCE_THRESHOLD": 0.02,
        "PLANE_MAX_COUNT":          6,
        "DBSCAN_EPS":               0.20,
        "DBSCAN_MIN_POINTS":        50,
        "MIN_CLUSTER_DIM":          0.15,
        "MAX_CLUSTER_DIM":          5.0,
        "MAX_CLUSTER_HEIGHT":       3.0,
    },

    # ────────────────────────────────────────────────────────────
    # TIGHT — small, crowded rooms (kitchens, server rooms, cells)
    # Objects are close together → smaller eps so they don't merge.
    # Smaller min cluster size so we don't drop small items.
    # ────────────────────────────────────────────────────────────
    "tight": {
        "SOR_STD_RATIO":            1.5,
        "VOXEL_SIZE":               0.03,    # finer detail
        "PLANE_DISTANCE_THRESHOLD": 0.02,
        "PLANE_MAX_COUNT":          6,
        "DBSCAN_EPS":               0.10,    # was 0.20 — keep close objects separate
        "DBSCAN_MIN_POINTS":        30,      # was 50  — accept smaller objects
        "MIN_CLUSTER_DIM":          0.08,    # was 0.15
        "MAX_CLUSTER_DIM":          3.0,     # was 5.0
        "MAX_CLUSTER_HEIGHT":       2.5,     # was 3.0
    },

    # ────────────────────────────────────────────────────────────
    # LARGE — warehouses, factory halls, big open spaces
    # Bigger objects, rougher floors, scan gaps between surfaces.
    # ────────────────────────────────────────────────────────────
    "large": {
        "SOR_STD_RATIO":            2.0,     # gentler — fewer real points dropped
        "VOXEL_SIZE":               0.08,    # coarser — faster on huge clouds
        "PLANE_DISTANCE_THRESHOLD": 0.05,    # rougher floors / walls
        "PLANE_MAX_COUNT":          4,       # fewer planes (no internal walls)
        "DBSCAN_EPS":               0.40,    # bridge gaps in big objects
        "DBSCAN_MIN_POINTS":        150,     # higher bar for what counts as an object
        "MIN_CLUSTER_DIM":          0.40,    # ignore small clutter
        "MAX_CLUSTER_DIM":          12.0,    # allow large machines
        "MAX_CLUSTER_HEIGHT":       6.0,     # allow tall racks / machines
    },
}


def apply_preset(name: str, config_module) -> None:
    """
    Apply a preset by overwriting attributes on the config module.
    Raises ValueError if the preset name is unknown.
    """
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {name!r}. Available: {list(PRESETS.keys())}"
        )
    overrides = PRESETS[name]
    for key, val in overrides.items():
        setattr(config_module, key, val)
    print(f"[Preset] Applied '{name}': {len(overrides)} parameter(s) overridden")
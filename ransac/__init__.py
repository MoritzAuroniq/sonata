"""Phase 2 — Classical point cloud pipeline."""

import numpy as np
import open3d as o3d
import random

# Make every run reproducible — same input + same config → same output
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
o3d.utility.random.seed(SEED)
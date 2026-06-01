# Phase 2 — Classical Point Cloud Pipeline

Modular implementation of the Path A pipeline.

## Project structure

```
ransac/
├── ransac/                          ← the package
│   ├── __init__.py
│   ├── config.py                    ← all tunable parameters
│   ├── types.py                     ← Detection dataclass + IO helpers
│   ├── pipeline.py                  ← orchestrator (calls all steps in order)
│   ├── visualise.py                 ← Open3D viewer wrappers
│   └── steps/
│       ├── __init__.py              ← re-exports all step functions
│       ├── step1_load.py
│       ├── step1_5_denoise.py
│       ├── step2_downsample.py
│       ├── step3_remove_planes.py
│       ├── step4_cluster.py
│       └── step5_bounding_boxes.py
├── scripts/
│   └── run.py                       ← CLI entry point
├── data/                            ← put your PLY files here
└── outputs/                         ← detections.json saved here
```

## Run

From the project root (`ransac/`):

```bash
# Default — uses config.DEFAULT_PLY
python scripts/run.py

# Custom file
python scripts/run.py data/my_scan.ply

# Batch / headless — no Open3D windows
python scripts/run.py data/my_scan.ply --no-viz

# Save JSON output
python scripts/run.py --save
```

## Tuning

Edit `ransac/config.py` — every tunable parameter is there with a comment
explaining what it controls. Change one at a time.

## Use as a library

The pipeline can be imported and called directly:

```python
from ransac import pipeline

detections, artefacts = pipeline.run("my_scan.ply")
for det in detections:
    print(det.object_id, det.centre_xyz_m, det.theta_deg)
```

## Adding new steps

1. Create `ransac/steps/stepN_your_step.py` with a single public function
2. Add `from .stepN_your_step import your_func` to `ransac/steps/__init__.py`
3. Call `steps.your_func(...)` from `ransac/pipeline.py`

"""
scripts/run.py

CLI entry point. Run from project root:

    python scripts/run.py                              # uses DEFAULT_PLY
    python scripts/run.py /path/to/scan.ply            # custom file
    python scripts/run.py /path/to/scan.ply --no-viz   # batch mode (no windows)
"""

import sys
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

# Make 'ransac' importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ransac import pipeline, config
from ransac.types import save_detections, print_detections_table
from ransac.visualise import show


def snapshot_config(run_dir: Path) -> dict:
    """Capture all UPPERCASE constants from config.py as a dict and save them."""
    params = {
        name: getattr(config, name)
        for name in dir(config)
        if name.isupper() and not name.startswith("_")
    }
    # Convert any Path objects to strings for JSON
    params_serialisable = {k: (str(v) if isinstance(v, Path) else v)
                           for k, v in params.items()}

    with open(run_dir / "params.json", "w") as f:
        json.dump(params_serialisable, f, indent=2, default=str)

    # Also keep a copy of the actual config.py for reference
    shutil.copy(Path(config.__file__), run_dir / "config_snapshot.py")
    return params


def make_run_dir(input_ply: Path) -> Path:
    """Create outputs/<timestamp>_<input_stem>/ for this run."""
    stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = Path(input_ply).stem
    run_dir   = config.OUTPUTS_DIR / f"{stamp}_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    ap = argparse.ArgumentParser(description="Phase 2 — classical PCL pipeline")
    ap.add_argument("ply_path", nargs="?", default=None,
                    help="Path to input PLY (defaults to config.DEFAULT_PLY)")
    ap.add_argument("--no-viz", action="store_true",
                    help="Skip Open3D windows (batch / headless mode)")
    ap.add_argument("--no-save", action="store_true",
                    help="Do not save any output (default: save everything)")
    args = ap.parse_args()

    input_path = Path(args.ply_path) if args.ply_path else config.DEFAULT_PLY

    # ── Always create a run folder unless user opts out ──────────
    if not args.no_save:
        run_dir = make_run_dir(input_path)
        print(f"\n[Run] Output folder: {run_dir}\n")

    # ── Run the pipeline ─────────────────────────────────────────
    detections, artefacts = pipeline.run(input_path)
    print_detections_table(detections)

    # ── Save outputs ─────────────────────────────────────────────
    if not args.no_save:
        # 1. JSON detections
        save_detections(detections, run_dir / "detections.json")

        # 2. Parameters used
        snapshot_config(run_dir)

        # 3. Plain-text summary that mirrors what was printed
        with open(run_dir / "summary.txt", "w") as f:
            f.write(f"Input: {input_path}\n")
            f.write(f"Run:   {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"Detected {len(detections)} objects\n\n")
            f.write(f"{'ID':<10}{'X':>8}{'Y':>8}{'Z':>8}    "
                    f"{'W':>6}{'H':>6}{'D':>6}    {'theta':>6}\n")
            f.write("-" * 70 + "\n")
            for d in detections:
                x, y, z = d.centre_xyz_m
                w, h, dp = d.size_wdh_m
                f.write(f"{d.object_id:<10}{x:>+8.2f}{y:>+8.2f}{z:>+8.2f}    "
                        f"{w:>6.2f}{h:>6.2f}{dp:>6.2f}    {d.theta_deg:>+6.0f}\n")

        print(f"[Run] All outputs saved to {run_dir}")

    # ── Visualisation ────────────────────────────────────────────
    from ransac.visualise import show, show_labeled
    
    if not args.no_viz:
        show_labeled(
            artefacts["clusters"] + artefacts["boxes"] + artefacts["markers"],
            detections=detections,
            title="Step 5 — labeled detections"
        )

        show_labeled(
            artefacts["planes"]
            + artefacts["clusters"]
            + artefacts["boxes"]
            + artefacts["markers"],
            detections=detections,
            title="Step 6 — full scene with labels"
        )


if __name__ == "__main__":
    main()
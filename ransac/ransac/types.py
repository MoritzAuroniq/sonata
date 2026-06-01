"""
ransac/types.py

Shared data types used across pipeline steps.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, List
import json
from pathlib import Path


@dataclass
class Detection:
    """One detected object after the full Phase 2 pipeline."""
    object_id:    str
    centre_xyz_m: Tuple[float, float, float]
    size_wdh_m:   Tuple[float, float, float]
    theta_deg:    float
    point_count:  int

    def to_dict(self) -> dict:
        return asdict(self)


def save_detections(detections: List[Detection], output_path: Path) -> None:
    """Write a list of detections to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([d.to_dict() for d in detections], f, indent=2)
    print(f"[Output] Saved {len(detections)} detections to {output_path}")


def print_detections_table(detections: List[Detection]) -> None:
    """Pretty-print all detections in a summary table."""
    print("\n" + "=" * 70)
    print(f"DETECTED {len(detections)} OBJECTS")
    print("=" * 70)
    print(f"{'ID':<10}{'X':>8}{'Y':>8}{'Z':>8}    "
          f"{'W':>6}{'H':>6}{'D':>6}    {'θ':>6}")
    print("-" * 70)
    for d in detections:
        x, y, z = d.centre_xyz_m
        w, h, dp = d.size_wdh_m
        print(f"{d.object_id:<10}{x:>+8.2f}{y:>+8.2f}{z:>+8.2f}    "
              f"{w:>6.2f}{h:>6.2f}{dp:>6.2f}    {d.theta_deg:>+6.0f}°")
    print("=" * 70 + "\n")

# burst_paths.py
"""Paths for Rank-Surprise burst timing CSVs under each run directory."""
from __future__ import annotations

from pathlib import Path

BURST_TIMINGS_SUBDIR = "burst_timings"


def burst_timings_dir(run_dir: Path, *, flat: bool = False) -> Path:
    """Return the CSV destination, optionally using the run directory directly."""
    run_dir = Path(run_dir)
    return run_dir if flat else run_dir / BURST_TIMINGS_SUBDIR


def burst_network_csv_path(run_dir: Path, side_tag: str, *, flat: bool = False) -> Path:
    """side_tag: combined | left | right → network_LR.csv / network_bursts_L.csv / network_bursts_R.csv."""
    d = burst_timings_dir(run_dir, flat=flat)
    if side_tag == "combined":
        return d / "network_LR.csv"
    if side_tag == "left":
        return d / "network_bursts_L.csv"
    if side_tag == "right":
        return d / "network_bursts_R.csv"
    raise ValueError(f"side_tag must be combined|left|right, got {side_tag!r}")


def burst_unit_csv_path(run_dir: Path, side_tag: str, *, flat: bool = False) -> Path:
    """combined → unit_LR.csv; left/right → unit_bursts_L.csv / unit_bursts_R.csv."""
    d = burst_timings_dir(run_dir, flat=flat)
    if side_tag == "combined":
        return d / "unit_LR.csv"
    if side_tag == "left":
        return d / "unit_bursts_L.csv"
    if side_tag == "right":
        return d / "unit_bursts_R.csv"
    raise ValueError(f"side_tag must be combined|left|right, got {side_tag!r}")


def burst_region_csv_path(run_dir: Path, side_tag: str, *, flat: bool = False) -> Path:
    """combined → region_LR.csv; left/right → region_bursts_L.csv / region_bursts_R.csv."""
    d = burst_timings_dir(run_dir, flat=flat)
    if side_tag == "combined":
        return d / "region_LR.csv"
    if side_tag == "left":
        return d / "region_bursts_L.csv"
    if side_tag == "right":
        return d / "region_bursts_R.csv"
    raise ValueError(f"side_tag must be combined|left|right, got {side_tag!r}")

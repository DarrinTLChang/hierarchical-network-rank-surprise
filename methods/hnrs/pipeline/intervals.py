from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowIndex:
    """Search index for inclusive membership in possibly overlapping windows."""

    starts: np.ndarray
    prefix_max_ends: np.ndarray


def prepare_window_index(windows) -> WindowIndex:
    valid: list[tuple[float, float]] = []
    for start, end in windows:
        start = float(start)
        end = float(end)
        if np.isfinite(start) and np.isfinite(end) and end >= start:
            valid.append((start, end))
    valid.sort(key=lambda pair: pair[0])
    starts = np.asarray([pair[0] for pair in valid], dtype=float)
    ends = np.asarray([pair[1] for pair in valid], dtype=float)
    return WindowIndex(starts, np.maximum.accumulate(ends) if ends.size else ends)


def time_in_window_index(time_ms: float, index: WindowIndex) -> bool:
    if index.starts.size == 0 or not np.isfinite(time_ms):
        return False
    pos = int(np.searchsorted(index.starts, float(time_ms), side="right")) - 1
    return pos >= 0 and float(time_ms) <= float(index.prefix_max_ends[pos])


def slice_sorted_inclusive(
    values: np.ndarray, start: float, end: float,
) -> np.ndarray:
    """Return sorted values in the inclusive interval ``start <= value <= end``."""
    left = int(np.searchsorted(values, start, side="left"))
    right = int(np.searchsorted(values, end, side="right"))
    return values[left:right]


def merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping or touching windows while preserving numeric units."""
    if not windows:
        return []
    ordered = sorted(windows, key=lambda pair: pair[0])
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged:
            merged.append([start, end])
        elif start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]

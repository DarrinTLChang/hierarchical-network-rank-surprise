"""Validated analyzed observation intervals, independent of spike activity."""
from __future__ import annotations

import numpy as np


def validate_observation_bounds_s(bounds_s) -> tuple[float, float]:
    try:
        start_s, end_s = map(float, bounds_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Observation bounds must be a (start_s, end_s) pair") from exc
    bounds_ms = np.asarray([start_s, end_s], dtype=float) * 1000.0
    if not np.all(np.isfinite(bounds_ms)) or end_s <= start_s:
        raise ValueError("Observation bounds must be finite with end > start")
    return start_s, end_s


def stage1_observation_bounds_ms(
    spike_struct, *, explicit_bounds_s: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], str]:
    if explicit_bounds_s is not None:
        start_s, end_s = validate_observation_bounds_s(explicit_bounds_s)
        source = "explicit_stage1_observation_bounds_s"
    else:
        channels = np.ravel(spike_struct)
        if not channels.size:
            raise ValueError("Stage-1 observation bounds require metadata or an explicit interval")
        declared = [hasattr(ch, "analyzed_start_s") or hasattr(ch, "analyzed_end_s") for ch in channels]
        if any(declared):
            if not all(declared):
                raise ValueError("Conflicting channel observation metadata: analyzed bounds are missing on some channels")
            try:
                intervals = np.asarray([validate_observation_bounds_s(
                    (ch.analyzed_start_s, ch.analyzed_end_s)) for ch in channels])
                durations = np.asarray([float(ch.dataSegmentLength) for ch in channels])
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("Invalid or incomplete analyzed observation metadata") from exc
            if not np.allclose(durations, intervals[:, 1] - intervals[:, 0], rtol=1e-10, atol=1e-9):
                raise ValueError("Declared duration conflicts with analyzed start/end bounds")
            source = ";".join(sorted({str(getattr(ch, "duration_source", "loader.analyzed_bounds")) for ch in channels}))
        else:
            try:
                durations = np.asarray([float(ch.dataSegmentLength) for ch in channels])
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Missing Stage-1 observation duration; supply explicit bounds. "
                    "First/last spikes do not establish the observation interval."
                ) from exc
            if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
                raise ValueError("All declared observation durations must be finite and positive")
            intervals = np.column_stack((np.zeros(durations.size), durations))
            source = "loader.dataSegmentLength_zero_based"
        if not np.allclose(intervals, intervals[0], rtol=1e-10, atol=1e-9):
            raise ValueError(
                "Conflicting channel observation intervals; supply a verified common Stage-1 interval"
            )
        start_s, end_s = map(float, intervals[0])
    bounds_ms = np.asarray([start_s, end_s], dtype=float) * 1000.0
    if not np.all(np.isfinite(bounds_ms)) or bounds_ms[1] <= bounds_ms[0]:
        raise ValueError("Stage-1 observation bounds must be finite with end > start")
    return (float(bounds_ms[0]), float(bounds_ms[1])), source

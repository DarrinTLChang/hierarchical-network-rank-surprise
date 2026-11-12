# detection.py
#
# All burst-detection algorithms:
#   - detect_bursts()     : maxISI / alpha-meanISI threshold method
#   - RS_detect_burst()   : Rank Surprise method
#
# Rank-surprise algorithm adapted from Subhasis Ray's Matlab -> Python port.
# Reference: Gourevitch, B. & Eggermont, J. J. A nonparametric approach for
# detection of bursts in spike trains. J Neurosci Methods 160, 349-358 (2007).
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3, or (at your option) any
# later version.  When redistributing the code preserve the reference
# and the credits.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from operator import itemgetter
from scipy.stats import norm
from config import *

from .burst_csv import (
    NETWORK_BURST_COLUMNS,
    REGION_BURST_COLUMNS,
    UNIT_BURST_COLUMNS,
    write_burst_csv,
)
from .burst_paths import burst_network_csv_path, burst_region_csv_path, burst_unit_csv_path
from .intervals import merge_windows, prepare_window_index, slice_sorted_inclusive


# =====================================================================
# maxISI / alpha-meanISI burst detection
# =====================================================================

def detect_bursts(
    spikes_ms: np.ndarray,
    elec: str,
    cl: int,
    STATS,
    override_thr: float | None = None,
    override_min_spikes: int | None = None,
    ibi_merge_factor=1.5,
):
    if spikes_ms.size < 2:
        return []
    row = STATS.loc[(elec, cl)]
    thr = float(override_thr) if override_thr is not None else float(row["thr"])
    fr_hz = float(row["FR_Hz"])

    isis = np.diff(spikes_ms)
    split = np.where(isis > thr)[0] + 1
    segs = np.split(spikes_ms, split)

    bursts_indiv = []
    for seg in segs:
        min_spikes = (
            int(override_min_spikes)
            if override_min_spikes is not None
            else MIN_SPIKES_IN_BURST
        )

        if seg.size < min_spikes:
            continue
        dur = seg[-1] - seg[0]
        # Legacy threshold detector; RS uses the stage-specific duration settings.
        bursts_indiv.append(
            dict(
                Start_ms=float(seg[0]),
                End_ms=float(seg[-1]),
                Duration_ms=float(dur),
                Num_Spikes=int(seg.size),
                thr=thr,
                FR_Hz=fr_hz,
            )
        )
    if len(bursts_indiv) < 2:
        return bursts_indiv

    bursts_merged = [bursts_indiv[0].copy()]

    for b in bursts_indiv[1:]:
        prev = bursts_merged[-1]
        ibi = float(b["Start_ms"] - prev["End_ms"])

        if ibi <= ibi_merge_factor * thr:
            prev["End_ms"] = max(prev["End_ms"], float(b["End_ms"]))
            prev["Num_Spikes"] = int(prev["Num_Spikes"]) + int(b["Num_Spikes"])
            prev["Duration_ms"] = float(prev["End_ms"] - prev["Start_ms"])
        else:
            bursts_merged.append(b.copy())

    return bursts_merged


# =====================================================================
# Rank Surprise burst detection
# =====================================================================

def _ranks_against_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Map each value to a (possibly fractional) rank against a reference sample.

    Rank definition matches the tie-handling used by `_val2rank` (mean rank for ties),
    but ranks are computed relative to `reference` instead of within `values`.

    Returns ranks in [1, len(reference)].
    """
    v = np.asarray(values, dtype=float)
    r = np.asarray(reference, dtype=float)
    if r.ndim != 1 or r.size == 0 or not np.all(np.isfinite(r)):
        raise ValueError("Rank reference must be a nonempty finite one-dimensional sample")
    if not np.all(np.isfinite(v)):
        raise ValueError("Values ranked against the reference must be finite")
    r = np.sort(r)

    # left = count(< x), right = count(<= x)
    left = np.searchsorted(r, v, side="left").astype(float)
    right = np.searchsorted(r, v, side="right").astype(float)
    # mean rank for ties in 1..N indexing
    return np.clip((left + 1.0 + right) / 2.0, 1.0, float(r.size))


def _build_offset_reference_train_ms(
    source_trains_ms: dict[str, np.ndarray],
    *,
    record_len_ms: float | None = None,
    seed: int,
    max_offset_ms: float | None,
    observation_bounds_ms: tuple[float, float] | None = None,
    context: str = "",
) -> np.ndarray:
    """
    Build a reference event train by applying a random constant offset per source train,
    then merging all sources.

    Each source has its own stable seed, so adding unrelated sources does not
    change its offset. Wrapping uses the declared observation interval.
    """
    if not source_trains_ms:
        return np.asarray([], dtype=float)
    bounds = observation_bounds_ms
    if bounds is None and record_len_ms is not None:
        bounds = (0.0, record_len_ms)
    lo, hi = _validated_observation_bounds_ms(bounds)
    rec = hi - lo
    max_off = float(max_offset_ms) if max_offset_ms is not None else rec
    if not np.isfinite(max_off) or max_off < 0:
        raise ValueError("Offset limit must be finite and nonnegative")
    merged: list[np.ndarray] = []
    for k in sorted(source_trains_ms.keys()):
        t = np.asarray(source_trains_ms[k], dtype=float)
        if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(t < lo) or np.any(t > hi):
            raise ValueError(f"Offset source {k!r} must lie within the observation interval")
        if t.size == 0:
            continue
        seed_text = json.dumps([int(seed), str(context), str(k)], separators=(",", ":"))
        source_seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(source_seed)
        delta = float(rng.uniform(0.0, max_off)) if max_off > 0 else 0.0
        tt = lo + np.mod(t - lo + delta, rec)
        merged.append(tt)

    if not merged:
        return np.asarray([], dtype=float)
    out = np.sort(np.concatenate(merged).astype(float, copy=False))
    return out


class ShuffleUnavailableError(ValueError):
    """An otherwise valid interval cannot supply the requested shuffle null."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _validated_observation_bounds_ms(bounds) -> tuple[float, float]:
    if bounds is None:
        raise ValueError("WIN-SHUFF requires explicit observation_bounds_ms=(start, end)")
    try:
        start, end = map(float, bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_bounds_ms must contain two finite numbers") from exc
    if not (np.isfinite(start) and np.isfinite(end) and end > start
            and np.isfinite(end - start)):
        raise ValueError("observation_bounds_ms must be finite with end > start")
    return start, end


def win_shuff_spike_times_ms(
    spiketimes_ms: np.ndarray,
    *,
    window_ms: float,
    bin_ms: float,
    rng: np.random.Generator,
    observation_bounds_ms: tuple[float, float] | None = None,
    automatic: bool = False,
    diagnostics: dict | None = None,
) -> np.ndarray:
    """
    One WIN-SHUFF surrogate (Stella et al., eNeuro 2022): exclusive windows of length
    ``window_ms``; within each window, spikes are binned into ``bin_ms`` bins, bin
    occupancy counts are permuted across bins, then spike times are redrawn i.i.d.
    uniform within each bin. Same spike count as input.

    Bounds describe observed time, including silence. Internal windows are half-open;
    the observation's final endpoint belongs to the final window/remainder. Automatic
    mode partitions the interval into exactly ten windows of ten bins. Custom mode
    shuffles only full windows and retains its explicitly reported partial remainder.
    """
    from .detector_profile import resolve_win_shuff_params_ms

    t0, t1 = _validated_observation_bounds_ms(observation_bounds_ms)
    W, b = resolve_win_shuff_params_ms(
        (t1 - t0) / 1000.0, auto_from_recording=automatic,
        window_ms=None if automatic else window_ms,
        bin_ms=None if automatic else bin_ms,
    )
    t = np.asarray(spiketimes_ms, dtype=float).ravel()
    if np.any(~np.isfinite(t)):
        raise ValueError("WIN-SHUFF spike times must be finite")
    if np.any((t < t0) | (t > t1)):
        raise ValueError("WIN-SHUFF spike times lie outside observation_bounds_ms")
    if np.any(t[1:] < t[:-1]):
        t = np.sort(t)
    n_sub = 10 if automatic else int(round(W / b))
    if automatic:
        edges = t0 + (t1 - t0) * (np.arange(11, dtype=float) / 10.0)
        edges[0], edges[-1] = t0, t1
        n_windows = 10
    else:
        ratio = (t1 - t0) / W
        if not np.isfinite(ratio):
            raise ValueError("Custom shuffle window count is not representable")
        n_windows = int(np.floor(ratio))
        # Correct only floating-point roundoff at an exact full-window endpoint.
        nearest = int(round(ratio))
        # Time-origin magnitude must not erase a real, representable tail.
        tolerance = 8 * np.finfo(float).eps * max(t1 - t0, W)
        if nearest > n_windows and abs(t0 + nearest * W - t1) <= tolerance:
            n_windows = nearest
        edges = t0 + np.arange(n_windows + 1, dtype=float) * W
        if n_windows and abs(edges[-1] - t1) <= tolerance:
            edges[-1] = t1
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Shuffle window boundaries are not numerically representable")
    remainder_start = float(edges[-1])
    has_remainder = remainder_start < t1
    processed_count = int(np.searchsorted(t, remainder_start, side="left")) if has_remainder else int(t.size)
    info = {
        "window_ms": W, "bin_ms": b, "window_count": n_windows,
        "bins_per_window": n_sub, "input_spike_count": int(t.size),
        "shuffled_spike_count": processed_count,
        "remainder_duration_ms": float(t1 - remainder_start),
        "remainder_spike_count": int(t.size - processed_count),
        "shuffled_end_ms": remainder_start,
    }
    if diagnostics is not None:
        diagnostics.update(info)
    if not n_windows:
        raise ShuffleUnavailableError("segment_shorter_than_custom_window")
    if not t.size:
        return t.copy()
    if not processed_count:
        raise ShuffleUnavailableError("no_spikes_in_complete_windows")

    out_list: list[np.ndarray] = []
    for window_index in range(n_windows):
        lo, hi = float(edges[window_index]), float(edges[window_index + 1])
        final_window = window_index == n_windows - 1 and not has_remainder
        a = int(np.searchsorted(t, lo, side="left"))
        z = int(np.searchsorted(t, hi, side="right" if final_window else "left"))
        seg = t[a:z]
        if automatic:
            bin_edges = lo + (hi - lo) * (np.arange(11, dtype=float) / 10.0)
        else:
            bin_edges = lo + np.arange(n_sub + 1, dtype=float) * b
        bin_edges[0], bin_edges[-1] = lo, hi
        if np.any(~np.isfinite(bin_edges)) or np.any(np.diff(bin_edges) <= 0):
            raise ValueError("Shuffle bin boundaries are not numerically representable")
        indexes = np.searchsorted(bin_edges, seg, side="right") - 1
        indexes = np.minimum(indexes, n_sub - 1)
        counts = np.bincount(indexes, minlength=n_sub)
        perm = rng.permutation(n_sub)
        counts_shuf = counts[perm]
        new_spikes: list[float] = []
        for j in range(n_sub):
            c = int(counts_shuf[j])
            if c <= 0:
                continue
            left = float(bin_edges[j])
            right = float(bin_edges[j + 1])
            draws = np.asarray(rng.uniform(left, right, size=c), dtype=float)
            # Floating-point uniform sampling can round upward onto an internal edge.
            draws = np.minimum(draws, np.nextafter(right, left))
            new_spikes.extend(draws.tolist())
        if new_spikes:
            out_list.append(np.asarray(new_spikes, dtype=float))
    if has_remainder:
        out_list.append(t[processed_count:].copy())
    out = np.sort(np.concatenate(out_list))
    if out.size != t.size or np.any(~np.isfinite(out)) or np.any((out < t0) | (out > t1)):
        raise RuntimeError("WIN-SHUFF failed spike-count or observation-bound conservation")
    return out


def _stage1_chunk_windows_ms(
    spk_ms: np.ndarray,
    *,
    observation_bounds_ms: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """Build Stage-1 chunk windows in ms for one unit spike train."""
    if observation_bounds_ms is not None:
        lo, hi = _validated_observation_bounds_ms(observation_bounds_ms)
        mode = str(RS_STAGE1_SEGMENT_MODE).strip().lower()
        if mode not in ("nonoverlap", "sliding", "custom"):
            raise ValueError(f"Invalid Stage-1 segment mode: {mode!r}")
        if mode == "custom":
            windows = []
            last_end = -float("inf")
            for pair in getattr(__import__("config"), "RS_STAGE1_CUSTOM_WINDOWS_S", []):
                try:
                    start, end = (float(value) * 1000.0 for value in pair)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid custom analysis interval: {pair!r}") from exc
                if not (np.isfinite(start) and np.isfinite(end) and end > start):
                    raise ValueError(f"Invalid custom analysis interval: {pair!r}")
                if start < last_end:
                    raise ValueError("Custom Stage-1 analysis intervals must be ordered and nonoverlapping")
                last_end = end
                start, end = max(start, lo), min(end, hi)
                if end > start:
                    windows.append((start, end))
            return windows
        length = float(RS_STAGE1_SEGMENT_LEN_S) * 1000.0
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Stage-1 segment length must be finite and positive")
        overlap = float(RS_STAGE1_SEGMENT_OVERLAP_FRACTION) if mode == "sliding" else 0.0
        if not np.isfinite(overlap) or not 0 <= overlap < 1:
            raise ValueError("Stage-1 sliding overlap must satisfy 0 <= overlap < 1")
        stride = length * (1.0 - overlap)
        windows = []
        index = 0
        previous_start = -float("inf")
        while True:
            start = lo + index * stride
            if start >= hi:
                break
            if not np.isfinite(start) or start <= previous_start:
                raise ValueError("Stage-1 segment boundaries are not numerically representable")
            end = min(start + length, hi)
            if end <= start:
                raise ValueError("Stage-1 segment boundaries are not numerically representable")
            windows.append((float(start), float(end)))
            if end >= hi:
                break
            previous_start = start
            index += 1
        return windows

    # Preserve the historical no-bounds classic-RS path.
    t = np.asarray(spk_ms, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return []
    if np.any(t[1:] < t[:-1]):
        t = np.sort(t)
    lo = float(t[0])
    hi = float(t[-1])
    if hi <= lo:
        return [(lo, hi)]

    L = max(float(RS_STAGE1_SEGMENT_LEN_S) * 1000.0, 1e-6)
    mode = str(RS_STAGE1_SEGMENT_MODE).strip().lower()
    if mode not in ("nonoverlap", "sliding", "custom"):
        mode = "nonoverlap"

    if mode == "custom":
        wins: list[tuple[float, float]] = []
        custom_s = list(getattr(__import__("config"), "RS_STAGE1_CUSTOM_WINDOWS_S", []))
        last_end = -float("inf")
        for pair in custom_s:
            try:
                s_s, e_s = pair
                s = float(s_s) * 1000.0
                e = float(e_s) * 1000.0
            except Exception:
                continue
            if not (np.isfinite(s) and np.isfinite(e)) or e <= s:
                continue
            if s < 0:
                s = 0.0
            if s < last_end - 1e-9:
                # skip overlapping custom windows
                continue
            wins.append((s, e))
            last_end = e
        return wins

    if mode == "sliding":
        ov = float(np.clip(RS_STAGE1_SEGMENT_OVERLAP_FRACTION, 0.0, 0.99))
        stride = max(L * (1.0 - ov), 1e-6)
    else:
        stride = L

    wins: list[tuple[float, float]] = []
    start = lo
    while start <= hi + 1e-9:
        end = min(start + L, hi)
        if end > start:
            wins.append((float(start), float(end)))
        if end >= hi:
            break
        start += stride
    return wins


def _interval_iou_ms(a0: float, a1: float, b0: float, b1: float) -> float:
    """IoU for 1D time intervals in ms."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return float(inter / union)


def _dedup_stage1_bursts_for_sliding(bursts: list[dict]) -> list[dict]:
    """Remove near-duplicate bursts from overlapping sliding chunks."""
    if not bursts:
        return []
    out: list[dict] = []
    iou_min = float(np.clip(RS_STAGE1_POSTHOC_DEDUP_IOU_MIN, 0.0, 1.0))
    for b in sorted(bursts, key=lambda x: (float(x["Start_ms"]), float(x["End_ms"]))):
        s = float(b["Start_ms"]); e = float(b["End_ms"])
        if e <= s:
            continue
        keep = True
        for prev in out:
            ps = float(prev["Start_ms"]); pe = float(prev["End_ms"])
            iou = _interval_iou_ms(s, e, ps, pe)
            if iou >= iou_min:
                # Keep the stronger candidate by RS, otherwise keep the longer.
                rs_b = float(b.get("RS", 0.0))
                rs_p = float(prev.get("RS", 0.0))
                if (rs_b > rs_p) or (abs(rs_b - rs_p) < 1e-12 and (e - s) > (pe - ps)):
                    prev.update(b)
                keep = False
                break
        if keep:
            out.append(dict(b))
    return out


def _merge_stage1_bursts_posthoc(bursts: list[dict], original_spikes_ms: np.ndarray) -> list[dict]:
    """Merge intervals, then count original spike events in their final bounds."""
    spikes = np.asarray(original_spikes_ms, dtype=float)
    if spikes.ndim != 1 or not np.all(np.isfinite(spikes)) or np.any(np.diff(spikes) < 0):
        raise ValueError("Merged-burst recount requires a finite sorted original spike train")
    if not bursts:
        return []
    gap = float(RS_STAGE1_POSTHOC_MERGE_GAP_MS)
    rows = sorted(
        [dict(b) for b in bursts if float(b.get("End_ms", np.nan)) > float(b.get("Start_ms", np.nan))],
        key=lambda x: (float(x["Start_ms"]), float(x["End_ms"]))
    )
    if not rows:
        return []
    merged: list[dict] = [rows[0]]
    for b in rows[1:]:
        cur = merged[-1]
        if float(b["Start_ms"]) <= float(cur["End_ms"]) + gap:
            cur["End_ms"] = max(float(cur["End_ms"]), float(b["End_ms"]))
            cur["Duration_ms"] = float(cur["End_ms"] - cur["Start_ms"])
            cur["RS"] = float(max(float(cur.get("RS", 0.0)), float(b.get("RS", 0.0))))
            cur["RS_aggregation"] = "max_constituent"
        else:
            merged.append(dict(b))
    for burst in merged:
        start, end = float(burst["Start_ms"]), float(burst["End_ms"])
        burst["Num_Spikes"] = int(np.searchsorted(spikes, end, side="right")
                                  - np.searchsorted(spikes, start, side="left"))
        burst["Duration_ms"] = end - start
        burst.setdefault("RS_aggregation", "single_interval")
    return merged


def _detect_stage1_unit_bursts(
    spk: np.ndarray, elec: str, cl: int, *,
    observation_bounds_ms: tuple[float, float] | None = None,
    observation_source: str = "unspecified",
    diagnostics: list[dict] | None = None,
    side_tag: str = "",
) -> list[dict]:
    """Stage-1 RS bursts for one unit, optionally using local segmentation."""
    t = np.asarray(spk, dtype=float).ravel()
    t = t[np.isfinite(t)]
    shuffle_enabled = bool(RS_WIN_SHUFF_STAGE1_ENABLE)
    bounded_path = shuffle_enabled or observation_bounds_ms is not None
    if t.size < 2 and not bounded_path:
        return []
    if np.any(t[1:] < t[:-1]):
        t = np.sort(t)

    def _detect_one_chunk(chunk_spk: np.ndarray, ref_isis=None) -> list[dict]:
        start_idx, length_spikes, RSvals = RS_detect_burst(
            chunk_spk,
            limit=RS_Limit_stage1,
            RSalpha=RS_alpha_stage1,
            min_spikes=MIN_SPIKES_IN_BURST,
            RS_Percentile_Limit=RS_Percentile_Limit_stage1,
            reference_isis=ref_isis,
        )
        out_rows: list[dict] = []
        for s, L, rs in zip(start_idx, length_spikes, RSvals):
            s = int(s); e = int(s + L - 1)
            if not (0 <= s < chunk_spk.size and 0 <= e < chunk_spk.size):
                raise RuntimeError("RS returned a burst outside the observed spike array")
            dur_ms = float(chunk_spk[e] - chunk_spk[s])
            min_unit_duration_ms = float(
                globals().get(
                    "MIN_UNIT_BURST_DURATION_MS",
                    globals().get("MIN_BURST_DURATION", 0.0),
                )
            )
            if dur_ms < min_unit_duration_ms:
                continue
            max_unit_duration_ms = globals().get(
                "MAX_UNIT_BURST_DURATION_MS", None
            )
            if (
                max_unit_duration_ms is not None
                and dur_ms > float(max_unit_duration_ms)
            ):
                continue
            out_rows.append({
                "Electrode": elec,
                "Cluster": int(cl),
                "Start_ms": float(chunk_spk[s]),
                "End_ms": float(chunk_spk[e]),
                "Duration_ms": dur_ms,
                "Num_Spikes": int(L),
                "RS": float(rs),
                "RS_aggregation": "single_interval",
                "Method": "RankSurprise",
            })
        return out_rows

    if bounded_path:
        from .detector_profile import resolve_win_shuff_params_ms

        sink = diagnostics if diagnostics is not None else []
        automatic = bool(RS_WIN_SHUFF_AUTO_FROM_RECORDING) if shuffle_enabled else False
        base = {
            "side": str(side_tag), "electrode": str(elec), "cluster": int(cl),
            "mode": ("automatic" if automatic else "custom") if shuffle_enabled else "classic",
            "sizing_policy_version": "b1-v2", "observation_source": str(observation_source),
            "seed_policy": "sha256-unit-and-interval-v1",
            "segment_start_ms": None, "segment_end_ms": None,
            "window_ms": None, "bin_ms": None, "window_count": 0,
            "bins_per_window": 0, "input_spike_count": 0,
            "shuffled_spike_count": 0, "remainder_duration_ms": 0.0,
            "remainder_spike_count": 0, "seed": None, "eligible": False,
            "status": "failed", "reason": "",
        }
        try:
            bounds = _validated_observation_bounds_ms(observation_bounds_ms)
            base.update(segment_start_ms=bounds[0], segment_end_ms=bounds[1])
            t = t[(t >= bounds[0]) & (t <= bounds[1])]
            # Validate only active parameters, even when this unit has no spikes.
            if shuffle_enabled and not automatic:
                resolve_win_shuff_params_ms(
                    (bounds[1] - bounds[0]) / 1000.0, auto_from_recording=False,
                    window_ms=RS_WIN_SHUFF_WINDOW_MS, bin_ms=RS_WIN_SHUFF_BIN_MS,
                )
            wins = (_stage1_chunk_windows_ms(t, observation_bounds_ms=bounds)
                    if bool(RS_STAGE1_LOCAL_SEGMENT_ENABLE) else [bounds])
        except Exception as exc:
            sink.append(dict(base, reason="invalid_settings_or_observation_bounds", error=str(exc)))
            raise

        if not wins:
            sink.append(dict(base, status="skipped", reason="no_selected_intervals"))
            return []
        min_spk = max(2, int(MIN_SPIKES_IN_BURST))
        if bool(RS_STAGE1_LOCAL_SEGMENT_ENABLE):
            min_spk = max(min_spk, int(RS_STAGE1_SEGMENT_MIN_SPIKES))
        raw_rows = []
        for w0, w1 in wins:
            left = int(np.searchsorted(t, w0, side="left"))
            right = int(np.searchsorted(t, w1, side="right" if w1 == bounds[1] else "left"))
            chunk = t[left:right]
            row = dict(base, segment_start_ms=w0, segment_end_ms=w1,
                       input_spike_count=int(chunk.size), eligible=bool(chunk.size >= min_spk))
            try:
                if shuffle_enabled:
                    W, b = resolve_win_shuff_params_ms(
                        (w1 - w0) / 1000.0, auto_from_recording=automatic,
                        window_ms=None if automatic else RS_WIN_SHUFF_WINDOW_MS,
                        bin_ms=None if automatic else RS_WIN_SHUFF_BIN_MS,
                    )
                    row.update(window_ms=W, bin_ms=b, bins_per_window=int(round(W / b)))
                if chunk.size < min_spk:
                    row.update(status="skipped", reason="insufficient_spikes")
                    sink.append(row)
                    continue
                ref_isis = None
                if shuffle_enabled:
                    identity = f"{int(RS_WIN_SHUFF_SEED)}|{elec}|{int(cl)}|{float(w0).hex()}|{float(w1).hex()}"
                    seed = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")
                    row["seed"] = seed
                    surrogate = win_shuff_spike_times_ms(
                        chunk, window_ms=W, bin_ms=b, rng=np.random.default_rng(seed),
                        observation_bounds_ms=(w0, w1), automatic=automatic, diagnostics=row,
                    )
                    ref_isis = np.diff(surrogate)
                    if ref_isis.size != chunk.size - 1 or not np.all(np.isfinite(ref_isis)):
                        raise RuntimeError("Requested WIN-SHUFF did not supply a valid reference")
                raw_rows.extend(_detect_one_chunk(chunk, ref_isis))
                row.update(status="ok", reason="")
            except ShuffleUnavailableError as exc:
                row.update(status="skipped", reason=exc.reason)
            except Exception as exc:
                row.update(status="failed", reason="reference_or_detection_failure", error=str(exc))
                sink.append(row)
                raise
            sink.append(row)
        if bool(RS_STAGE1_LOCAL_SEGMENT_ENABLE):
            if str(RS_STAGE1_SEGMENT_MODE).strip().lower() == "sliding":
                raw_rows = _dedup_stage1_bursts_for_sliding(raw_rows)
            if bool(RS_STAGE1_POSTHOC_MERGE_ENABLE):
                raw_rows = _merge_stage1_bursts_posthoc(raw_rows, t)
        return raw_rows

    if not bool(RS_STAGE1_LOCAL_SEGMENT_ENABLE):
        return _detect_one_chunk(t)

    wins = _stage1_chunk_windows_ms(t)
    raw_rows: list[dict] = []
    min_spk = int(max(2, RS_STAGE1_SEGMENT_MIN_SPIKES))
    for i, (w0, w1) in enumerate(wins):
        chunk = slice_sorted_inclusive(t, w0, w1)
        if chunk.size < min_spk:
            continue
        raw_rows.extend(_detect_one_chunk(chunk))

    if str(RS_STAGE1_SEGMENT_MODE).strip().lower() == "sliding":
        raw_rows = _dedup_stage1_bursts_for_sliding(raw_rows)
    if bool(RS_STAGE1_POSTHOC_MERGE_ENABLE):
        raw_rows = _merge_stage1_bursts_posthoc(raw_rows, t)
    return raw_rows


def RS_detect_burst(
    spiketimes,
    limit=None,
    RSalpha=-np.log(0.01),
    min_spikes=3,
    RS_Percentile_Limit=75,
    *,
    baseline_spiketimes=None,
    reference_isis: np.ndarray | None = None,
):
    """Detect bursts in spiketimes using Rank-Surprise method.

    Return a tuple (start, length, RS) where
      start:  spike number of burst start for each burst detected
      length: burst length for each burst detected (in spikes)
      RS:     rank surprise value for each burst detected.

    Observed times must be a finite, sorted one-dimensional array; duplicates are
    allowed. Observations are never filtered or reordered here, so output indices
    always address the caller's array. An explicitly requested reference must be
    usable; invalid/empty references never fall back to classic RS.
    """
    from operator import index
    from .rank_probability import rank_sum_logcdf

    observed = np.asarray(spiketimes, dtype=float)
    if observed.ndim != 1 or not np.all(np.isfinite(observed)):
        raise ValueError("Observed spike times must be a finite one-dimensional array")
    if np.any(observed[1:] < observed[:-1]):
        raise ValueError("Observed spike times must be sorted; indices refer to the original array")
    try:
        l_min = index(min_spikes)
    except TypeError as exc:
        raise ValueError("min_spikes must be an integer of at least two") from exc
    if l_min < 2:
        raise ValueError("min_spikes must be an integer of at least two")
    RSalpha = float(RSalpha)
    if np.isnan(RSalpha) or RSalpha < 0:
        raise ValueError("RSalpha must be nonnegative")
    if reference_isis is not None and baseline_spiketimes is not None:
        raise ValueError("Supply reference_isis or baseline_spiketimes, not both")

    with np.errstate(over="ignore", invalid="ignore"):
        ISI = np.diff(observed)
    if not np.all(np.isfinite(ISI)):
        raise ValueError("Observed inter-spike intervals must be finite")
    observed_isi_count = int(ISI.size)
    reference = None
    if reference_isis is not None:
        reference = np.asarray(reference_isis, dtype=float)
        if reference.ndim != 1 or reference.size == 0:
            raise ValueError("Requested reference_isis must be nonempty and one-dimensional")
        if not np.all(np.isfinite(reference)) or np.any(reference < 0):
            raise ValueError("Requested reference_isis must contain finite nonnegative intervals")
    elif baseline_spiketimes is not None:
        baseline = np.asarray(baseline_spiketimes, dtype=float)
        if baseline.ndim != 1 or baseline.size < 2 or not np.all(np.isfinite(baseline)):
            raise ValueError("Requested baseline_spiketimes must contain at least two finite times")
        # Reference order has no relation to the returned observed-array indices.
        with np.errstate(over="ignore", invalid="ignore"):
            reference = np.diff(np.sort(baseline))
        if not np.all(np.isfinite(reference)):
            raise ValueError("Requested baseline inter-spike intervals must be finite")

    empty_result = (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))
    if observed.size < l_min:
        return empty_result
    reference_count = int(reference.size) if reference is not None else observed_isi_count
    if limit is None:
        percentile = float(RS_Percentile_Limit)
        if not np.isfinite(percentile) or not 0 <= percentile <= 100:
            raise ValueError("RS_Percentile_Limit must be between zero and 100")
        limit = float(np.percentile(reference if reference is not None else ISI, percentile))
    else:
        limit = float(limit)
        if np.isnan(limit) or limit < 0:
            raise ValueError("The ISI limit must be nonnegative")
    R = _ranks_against_reference(ISI, reference) if reference is not None else _val2rank(ISI)

    ISI_limit = np.diff(np.where(ISI < limit, 1, 0))
    begin_int = np.nonzero(ISI_limit == 1)[0] + 1
    if ISI[0] < limit:
        begin_int = np.r_[0, begin_int]
    end_int = np.nonzero(ISI_limit == -1)[0]
    if len(end_int) < len(begin_int):
        end_int = np.r_[end_int, observed_isi_count - 1]
    length_int = end_int - begin_int + 1

    archive_burst_RS = []
    archive_burst_length = []
    archive_burst_start = []

    for n_j, p_j in zip(begin_int, length_int):
        subseq_RS = []
        for i in range(p_j - (l_min - 1) + 1):
            for q in range(l_min - 1, p_j - i + 1):
                u = np.sum(R[n_j + i:n_j + i + q])
                u = int(np.floor(u))
                RS = -rank_sum_logcdf(u, int(q), reference_count)
                if RS >= RSalpha:
                    subseq_RS.append((np.r_[RS, i, q]))

        if len(subseq_RS) > 0:
            subseq_RS = sorted(subseq_RS, key=itemgetter(0), reverse=True)
            while len(subseq_RS) > 0:
                current_burst = subseq_RS[0]
                archive_burst_RS.append(current_burst[0])
                archive_burst_length.append(current_burst[2] + 1)
                archive_burst_start.append(n_j + current_burst[1])
                subseq_RS = [row for row in subseq_RS[1:]
                             if ((row[1] + row[2] - 1) < current_burst[1]) or
                             (row[1] > (current_burst[1] + current_burst[2] - 1))]

    ind_sort = np.argsort(archive_burst_start)
    archive_burst_start = np.take(archive_burst_start, ind_sort).astype(int)
    archive_burst_RS = np.take(archive_burst_RS, ind_sort)
    archive_burst_length = np.take(archive_burst_length, ind_sort).astype(int)
    if (np.any(archive_burst_start < 0)
            or np.any(archive_burst_length < l_min)
            or np.any(archive_burst_start + archive_burst_length > observed.size)
            or np.any(~np.isfinite(archive_burst_RS))):
        raise RuntimeError("Rank-Surprise returned invalid observed-array bounds or scores")
    return (archive_burst_start, archive_burst_length, archive_burst_RS)


def _val2rank(values):
    """Convert values to ranks, with mean of ranks for tied values."""
    values = np.asarray(values, dtype=float)
    lp = values.size
    if lp == 0:
        return np.empty(values.shape, dtype=float)

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_ranks = np.empty(lp, dtype=float)

    i = 0
    while i < lp:
        j = i + 1
        while j < lp and sorted_values[j] == sorted_values[i]:
            j += 1
        # Average the 1-based ranks occupied by the complete tie group,
        # matching MATLAB tied-rank behavior used by cotterill_rs.m.
        sorted_ranks[i:j] = (i + 1 + j) / 2.0
        i = j

    ranks = np.empty(lp, dtype=float)
    ranks[order] = sorted_ranks
    return ranks


# =====================================================================
# Unified span computation (used for CSV and plotting)
# =====================================================================

def _ch_id(b: dict, channel_mode: str = "elec_cluster") -> str:
    """Get channel ID from burst dict."""
    elec = str(b.get("Electrode", ""))
    if channel_mode == "elec":
        return elec
    cl = int(b.get("Cluster", -1))
    return f"{elec}|{cl}"


def compute_spans_and_bars(
    windows_ms: list[tuple[float, float]],
    bursts: list[dict],
    *,
    min_unique_channels: int = 1,
    min_unique_regions: int = 1,
    channel_mode: str = "elec_cluster",
    region_fn=None,
    use_onset_plus_length: bool = False,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Compute spans for each window and merged bars for plotting.

    If use_onset_plus_length is False: span = (median(starts), median(ends)).
    If use_onset_plus_length is True: span = (w0, w0 + median(burst lengths)).

    Returns:
        spans_ms: List of (span_start_ms, span_end_ms) per window (for CSV)
        bars_s: List of merged (start_s, end_s) bars (for plotting)
    """
    spans_ms: list[tuple[float, float]] = []
    bars_s: list[tuple[float, float]] = []
    valid_bursts: list[tuple[float, float, dict]] = []
    for burst in bursts:
        start = float(burst.get("Start_ms", np.nan))
        end = float(burst.get("End_ms", np.nan))
        if np.isfinite(start) and np.isfinite(end) and end > start:
            valid_bursts.append((start, end, burst))
    valid_bursts.sort(key=lambda item: item[0])
    onsets = np.asarray([item[0] for item in valid_bursts], dtype=float)

    for w0, w1 in windows_ms:
        if not (np.isfinite(w0) and np.isfinite(w1)) or w1 < w0:
            spans_ms.append((float("nan"), float("nan")))
            continue

        overlapping = []
        chans = set()
        regs = set()

        left = int(np.searchsorted(onsets, w0, side="left"))
        right = int(np.searchsorted(onsets, w1, side="right"))
        for s, e, b in valid_bursts[left:right]:
            overlapping.append((s, e))
            chans.add(_ch_id(b, channel_mode))
            if region_fn is not None:
                try:
                    reg = region_fn(b)
                except Exception:
                    reg = b.get("Region", b.get("Electrode", ""))
            else:
                reg = b.get("Region", b.get("Electrode", ""))
            regs.add(str(reg))

        if (
            not overlapping
            or len(chans) < min_unique_channels
            or len(regs) < int(max(1, min_unique_regions))
        ):
            spans_ms.append((float("nan"), float("nan")))
            continue

        if use_onset_plus_length:
            lengths = [e - s for (s, e) in overlapping]
            median_length_ms = float(np.median(lengths))
            span_start = float(w0)
            span_end = w0 + median_length_ms
        else:
            starts = [x[0] for x in overlapping]
            ends = [x[1] for x in overlapping]
            span_start = float(np.median(starts))
            span_end = float(np.median(ends))

        if span_end <= span_start:
            spans_ms.append((float("nan"), float("nan")))
            continue

        spans_ms.append((span_start, span_end))
        bars_s.append((span_start / 1000.0, span_end / 1000.0))
    
    # Merge overlapping bars
    bars_s = merge_windows(bars_s)
    
    return spans_ms, bars_s


def _filter_network_support_events(
    windows_ms: list[tuple[float, float]],
    spans_ms: list[tuple[float, float]],
    rs_values: list[float],
    event_counts: list[int],
) -> tuple[list[tuple[float, float]], list[dict], list[tuple[float, float]]]:
    """Validate network cores and apply final support-duration gates only."""
    accepted_windows: list[tuple[float, float]] = []
    accepted_spans: list[tuple[float, float]] = []
    network_bursts: list[dict] = []
    min_support_ms = float(
        globals().get(
            "MIN_NETWORK_BURST_SUPPORT_DURATION_MS",
            globals().get("MIN_BURST_DURATION", 0.0),
        )
    )
    max_support_value = globals().get(
        "MAX_NETWORK_BURST_SUPPORT_DURATION_MS",
        globals().get("MAX_BURST_DURATION", None),
    )
    max_support_ms = (
        None if max_support_value is None else float(max_support_value)
    )

    for window, span, rs_value, num_events in zip(
        windows_ms,
        spans_ms,
        rs_values,
        event_counts,
    ):
        core_start, core_end = float(window[0]), float(window[1])
        if not (np.isfinite(core_start) and np.isfinite(core_end)):
            continue
        if core_end < core_start:
            continue

        support_start, support_end = float(span[0]), float(span[1])
        if not (np.isfinite(support_start) and np.isfinite(support_end)):
            continue
        support_duration_ms = support_end - support_start
        if support_duration_ms < min_support_ms:
            continue
        if max_support_ms is not None and support_duration_ms > max_support_ms:
            continue

        accepted_windows.append((core_start, core_end))
        accepted_spans.append((support_start, support_end))
        network_bursts.append(
            {
                "rs_core_start_ms": core_start,
                "rs_core_end_ms": core_end,
                "burst_start_ms": support_start,
                "burst_end_ms": support_end,
                "burst_duration_ms": support_duration_ms,
                "num_onsets": int(num_events),
                "RS": float(rs_value),
            }
        )

    bars_s = merge_windows(
        [
            (support_start / 1000.0, support_end / 1000.0)
            for support_start, support_end in accepted_spans
        ]
    )
    return accepted_windows, network_bursts, bars_s


# =====================================================================
# Multi-stage burst detection pipeline
# =====================================================================

def _coverage_length_ms(intervals: list[tuple[float, float]]) -> float:
    valid = sorted((float(a), float(b)) for a, b in intervals
                   if np.isfinite(a) and np.isfinite(b) and b > a)
    total = 0.0
    end = -float("inf")
    for start, stop in valid:
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return float(total)


def summarize_stage1_diagnostics(rows: list[dict]) -> dict:
    """Summarize per-unit unique coverage; overlapping segments count once."""
    from .detector_profile import DETECTOR_IMPLEMENTATION_VERSION
    def state(selected):
        if any(r.get("status") == "failed" for r in selected):
            return "failed", "stage1_failure"
        custom = [r for r in selected if r.get("mode") == "custom" and r.get("window_ms") is not None]
        eligible = [r for r in selected if r.get("eligible")]
        if eligible and custom and not any(r.get("status") == "ok" for r in selected) and all(
            float(r["segment_end_ms"]) - float(r["segment_start_ms"]) < float(r["window_ms"])
            for r in custom
        ):
            return "failed", "incompatible_custom_window"
        if any(r.get("eligible") and r.get("status") == "skipped" for r in selected):
            return "incomplete", "eligible_intervals_omitted"
        if any(r.get("status") == "ok" for r in selected):
            return "complete", ""
        return "no_analyzable_data", "no_eligible_input"

    status, reason = state(rows)
    sides = sorted({str(r.get("side", "")) for r in rows})
    per_side = {}
    for side in sides:
        selected = [r for r in rows if str(r.get("side", "")) == side]
        side_status, side_reason = state(selected)
        per_side[side] = {
            "status": side_status, "reason": side_reason,
            "ok_intervals": sum(r.get("status") == "ok" for r in selected),
            "skipped_intervals": sum(r.get("status") == "skipped" for r in selected),
            "failed_intervals": sum(r.get("status") == "failed" for r in selected),
        }
    coverage = []
    keys = sorted({(str(r.get("side", "")), str(r.get("electrode", "")), int(r.get("cluster", -1))) for r in rows})
    for side, electrode, cluster in keys:
        selected = [r for r in rows if (str(r.get("side", "")), str(r.get("electrode", "")), int(r.get("cluster", -1))) == (side, electrode, cluster)]
        def intervals(predicate):
            return [(r["segment_start_ms"], r["segment_end_ms"]) for r in selected
                    if predicate(r) and r.get("segment_start_ms") is not None and r.get("segment_end_ms") is not None]
        evaluated = intervals(lambda r: r.get("status") == "ok")
        omitted = intervals(lambda r: r.get("eligible") and r.get("status") != "ok")
        shuffled = [(r["segment_start_ms"], r["shuffled_end_ms"]) for r in selected
                    if r.get("status") == "ok" and r.get("shuffled_end_ms") is not None]
        remainder = [(r["shuffled_end_ms"], r["segment_end_ms"]) for r in selected
                     if r.get("status") == "ok" and r.get("shuffled_end_ms") is not None]
        evaluated_ms = _coverage_length_ms(evaluated)
        shuffled_ms = _coverage_length_ms(shuffled)
        coverage.append({
            "side": side, "electrode": electrode, "cluster": cluster,
            "requested_time_ms": _coverage_length_ms(intervals(lambda r: r.get("reason") != "no_selected_intervals")),
            "evaluated_time_ms": evaluated_ms,
            "omitted_eligible_time_ms": max(0.0, _coverage_length_ms(evaluated + omitted) - evaluated_ms),
            "insufficient_spike_time_ms": _coverage_length_ms(intervals(lambda r: r.get("reason") == "insufficient_spikes")),
            "shuffled_time_ms": shuffled_ms,
            "retained_remainder_time_ms": _coverage_length_ms(remainder),
            "unchanged_only_time_ms": max(0.0, _coverage_length_ms(shuffled + remainder) - shuffled_ms),
        })
    return {
        "schema_version": 1, "implementation_version": "b1-v2",
        "detector_implementation_version": DETECTOR_IMPLEMENTATION_VERSION,
        "status": status, "reason": reason, "per_side": per_side,
        "coverage_per_unit": coverage,
        "stage_status": {side: {stage: info["status"] for stage in ("unit", "region", "network")}
                         for side, info in per_side.items()},
        "coverage_note": "Time coverage is unique per unit and side; overlapping segments and combined/hemisphere passes must not be summed as recording duration.",
    }


def _write_stage1_analysis_status(run_dir: Path, rows: list[dict], status: dict) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    columns = ["side", "electrode", "cluster", "mode", "sizing_policy_version",
               "observation_source", "segment_start_ms", "segment_end_ms",
               "window_ms", "bin_ms", "window_count", "bins_per_window",
               "input_spike_count", "shuffled_spike_count", "remainder_duration_ms",
               "remainder_spike_count", "shuffled_end_ms", "seed", "seed_policy",
               "eligible", "status", "reason", "error"]
    frame = pd.DataFrame(rows).reindex(columns=columns)
    # Mixed integer/None columns otherwise pass through float and lose seed bits.
    frame["seed"] = [str(row["seed"]) if row.get("seed") is not None else "" for row in rows]
    frame.to_csv(run_dir / "stage1_shuffle_diagnostics.csv", index=False)
    (run_dir / "analysis_status.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class BurstResults:
    all_bursts: list[dict[str, Any]] = field(default_factory=list)
    all_bursts_L: list[dict[str, Any]] = field(default_factory=list)
    all_bursts_R: list[dict[str, Any]] = field(default_factory=list)
    stage1_unit_bursts: list[dict[str, Any]] = field(default_factory=list)
    stage1_unit_bursts_L: list[dict[str, Any]] = field(default_factory=list)
    stage1_unit_bursts_R: list[dict[str, Any]] = field(default_factory=list)
    stage2_region_bursts: list[dict[str, Any]] = field(default_factory=list)
    stage2_region_bursts_L: list[dict[str, Any]] = field(default_factory=list)
    stage2_region_bursts_R: list[dict[str, Any]] = field(default_factory=list)
    stage3_network_bursts: list[dict[str, Any]] = field(default_factory=list)
    stage3_network_bursts_L: list[dict[str, Any]] = field(default_factory=list)
    stage3_network_bursts_R: list[dict[str, Any]] = field(default_factory=list)
    region_windows: dict = field(default_factory=dict)
    region_windows_L: dict = field(default_factory=dict)
    region_windows_R: dict = field(default_factory=dict)
    network_windows: list = field(default_factory=list)
    network_windows_L: list = field(default_factory=list)
    network_windows_R: list = field(default_factory=list)
    network_bars_L: list[tuple[float, float]] = field(default_factory=list)
    network_bars_R: list[tuple[float, float]] = field(default_factory=list)
    region_bars_by_region_L: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    region_bars_by_region_R: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    stage1_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    analysis_status: dict[str, Any] = field(default_factory=dict)


def _rs_burst_detection_impl(
    spike_struct,
    spike_struct_L,
    spike_struct_R,
    STATS: pd.DataFrame,
    run_dir: Path,
    iter_units_fn,
    infer_region_fn,
    burst_in_window_fn,
    flat_output: bool = False,
    *,
    observation_bounds_ms: tuple[float, float] | None = None,
    observation_source: str = "unspecified",
    _result: BurstResults | None = None,
) -> BurstResults:
    """Run the full burst-detection pipeline (unit -> region -> network).

    Burst CSVs normally use ``run_dir / burst_timings /``; flat batch output writes
    the same short names directly under ``run_dir``.

    Parameters
    ----------
    run_dir : Path
        Run output directory (e.g. main ``run_dir``).
    flat_output : bool
        Write burst CSVs directly under ``run_dir`` instead of ``burst_timings/``.
    iter_units_fn : callable
        ``iter_units_from_stats(spike_struct, STATS)`` — yields (elec, cl, spk).
    infer_region_fn : callable
        ``infer_region(electrode_name)`` — returns region string.
    burst_in_window_fn : callable
        ``burst_in_window(t_ms, windows)`` — returns bool.
    """
    s_list = [
        (spike_struct, "combined"),
        (spike_struct_L, "left"),
        (spike_struct_R, "right"),
    ]

    def _write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
        write_burst_csv(rows, path, columns)

    res = _result if _result is not None else BurstResults()

    if maxisi_toggle or alpha_meanisi_toggle:
        for elec, cl, spk in iter_units_fn(spike_struct, STATS):
            detect_bursts(spk, elec, cl, STATS, ibi_merge_factor=ibi_merge_factor)
        return res

    if not rankSurprise_toggle:
        return res

    from .utils import calculation_region
    if observation_bounds_ms is None and np.ravel(spike_struct).size:
        from .observation import stage1_observation_bounds_ms
        observation_bounds_ms, observation_source = stage1_observation_bounds_ms(spike_struct)

    if bool(RS_WIN_SHUFF_STAGE1_ENABLE) or observation_bounds_ms is not None:
        bounds = _validated_observation_bounds_ms(observation_bounds_ms)
        if bool(RS_WIN_SHUFF_STAGE1_ENABLE):
            from .detector_profile import resolve_win_shuff_params_ms
            automatic = bool(RS_WIN_SHUFF_AUTO_FROM_RECORDING)
            resolve_win_shuff_params_ms(
                (bounds[1] - bounds[0]) / 1000.0, auto_from_recording=automatic,
                window_ms=None if automatic else RS_WIN_SHUFF_WINDOW_MS,
                bin_ms=None if automatic else RS_WIN_SHUFF_BIN_MS,
            )

    all_bursts_list: list[list[dict]] = []
    stage1_unit_bursts_list: list[list[dict]] = []
    stage2_region_bursts_list: list[list[dict]] = []
    stage3_network_bursts_list: list[list[dict]] = []
    region_window_list: list[dict] = []
    network_window_list: list[list] = []
    network_bars_temp: dict[str, list[tuple[float, float]]] = {}
    region_bars_temp: dict[str, dict[str, list[tuple[float, float]]]] = {}

    for spike_struct_side, side_tag in s_list:
        # Stage 1: unit-level bursts
        unit_bursts: list[dict] = []
        for elec, cl, spk in iter_units_fn(spike_struct_side, STATS):
            unit_bursts.extend(_detect_stage1_unit_bursts(
                spk, str(elec), int(cl), observation_bounds_ms=observation_bounds_ms,
                observation_source=observation_source, diagnostics=res.stage1_diagnostics,
                side_tag=side_tag,
            ))
        side_summary = summarize_stage1_diagnostics(
            [r for r in res.stage1_diagnostics if r.get("side") == side_tag]
        )
        if side_summary["status"] == "failed":
            raise ValueError(f"Stage-1 {side_tag} analysis failed: {side_summary['reason']}")

        # Keep an untouched Stage-1 snapshot for dedicated plotting.
        stage1_unit_bursts = [dict(b) for b in unit_bursts]
        all_bursts = unit_bursts
        # RS unit-level CSVs: unified burst fields
        unit_csv_rows = [
            {
                "Electrode": b["Electrode"],
                "Cluster": b["Cluster"],
                "burst_start_ms": b["Start_ms"],
                "burst_end_ms": b["End_ms"],
                "burst_duration_ms": b["Duration_ms"],
                "num_onsets": b["Num_Spikes"],
                "RS": b["RS"],
                "RS_aggregation": b.get("RS_aggregation", "single_interval"),
                "Method": b["Method"],
            }
            for b in unit_bursts
        ]
        _write_csv(
            unit_csv_rows,
            burst_unit_csv_path(run_dir, side_tag, flat=flat_output),
            UNIT_BURST_COLUMNS,
        )

        # Stage 1.5: region-level bursts
        region_windows_by_region: dict[str, list[tuple[float, float]]] = {}

        if RS_region_burst_toggle:
            region_onsets: dict[str, list[float]] = {}
            for b in unit_bursts:
                reg = calculation_region(str(b["Electrode"]))
                b["_Region"] = reg
                region_onsets.setdefault(reg, []).append(float(b["Start_ms"]))

            region_bursts: list[dict] = []
            for reg, onsets in region_onsets.items():
                onsets_ms = np.asarray(onsets, dtype=float)
                onsets_ms = onsets_ms[np.isfinite(onsets_ms)]
                onsets_ms = np.sort(onsets_ms)

                if onsets_ms.size < MIN_SPIKES_IN_BURST:
                    continue

                unit_bursts_in_reg = [b for b in unit_bursts if b.get("_Region") == reg]
                # Build a per-(electrode,cluster) onset train and apply a seeded random offset per source
                # to form an independence-preserving reference baseline for region-level RS.
                baseline_onsets_ms = None
                if RS_OFFSET_NULL_ENABLE:
                    by_unit: dict[str, np.ndarray] = {}
                    for bb in unit_bursts_in_reg:
                        key = f"{bb.get('Electrode')}|{int(bb.get('Cluster', -1))}"
                        by_unit.setdefault(key, []).append(float(bb["Start_ms"]))
                    by_unit_arr: dict[str, np.ndarray] = {
                        k: np.sort(np.asarray(v, dtype=float)) for k, v in by_unit.items()
                    }
                    baseline_onsets_ms = _build_offset_reference_train_ms(
                        by_unit_arr,
                        observation_bounds_ms=observation_bounds_ms,
                        context=f"region:{reg}",
                        seed=int(RS_OFFSET_NULL_SEED),
                        max_offset_ms=RS_OFFSET_NULL_MAX_OFFSET_MS,
                    )

                start_idx_r, length_r, RSvals_r = RS_detect_burst(
                    onsets_ms,
                    limit=RS_Limit_region,
                    RSalpha=RS_alpha_region,
                    min_spikes=MIN_SPIKES_IN_BURST,
                    RS_Percentile_Limit=RS_Percentile_Limit_region,
                    baseline_spiketimes=baseline_onsets_ms,
                )

                windows: list[tuple[float, float]] = []
                rs_vals_for_windows: list[float] = []
                num_events_for_windows: list[int] = []
                for s, L, rs in zip(start_idx_r, length_r, RSvals_r):
                    s = int(s); e = int(s + L - 1)
                    t0 = float(onsets_ms[s]); t1 = float(onsets_ms[e])
                    if t1 < t0:
                        continue
                    windows.append((t0, t1))
                    rs_vals_for_windows.append(float(rs))
                    num_events_for_windows.append(int(L))

                if windows:
                    spans_ms, _ = compute_spans_and_bars(
                        windows, unit_bursts_in_reg,
                        min_unique_channels=MIN_UNIQUE_CHANNELS_REGION,
                        channel_mode="elec_cluster",
                        use_onset_plus_length=REGION_SPAN_ONSET_PLUS_LENGTH,
                    )
                    
                    filtered_windows: list[tuple[float, float]] = []
                    accepted_spans: list[tuple[float, float]] = []
                    min_region_support_ms = float(
                        globals().get(
                            "MIN_REGION_BURST_SUPPORT_DURATION_MS",
                            globals().get("MIN_BURST_DURATION", 0.0),
                        )
                    )
                    for w, span, rs_val, num_ev in zip(
                        windows,
                        spans_ms,
                        rs_vals_for_windows,
                        num_events_for_windows,
                    ):
                        if not (np.isfinite(span[0]) and np.isfinite(span[1])):
                            continue
                        span_start = float(span[0])
                        span_end = float(span[1])
                        span_dur = span_end - span_start
                        if span_dur < min_region_support_ms:
                            continue
                        filtered_windows.append(w)
                        accepted_spans.append((span_start, span_end))
                        region_bursts.append({
                            "Region": reg,
                            "rs_core_start_ms": w[0],
                            "rs_core_end_ms": w[1],
                            "burst_start_ms": span_start,
                            "burst_end_ms": span_end,
                            "burst_duration_ms": span_dur,
                            "num_onsets": num_ev,
                            "RS": rs_val,
                            "Method": "RS_region_onsets",
                        })
                    
                    if filtered_windows:
                        region_windows_by_region[reg] = filtered_windows
                        bars_s = merge_windows(
                            [
                                (span_start / 1000.0, span_end / 1000.0)
                                for span_start, span_end in accepted_spans
                            ]
                        )
                        if side_tag in ("left", "right"):
                            if side_tag not in region_bars_temp:
                                region_bars_temp[side_tag] = {}
                            region_bars_temp[side_tag][reg] = bars_s

            region_window_indexes = {
                reg: prepare_window_index(wins)
                for reg, wins in region_windows_by_region.items()
            }
            all_bursts_region: list[dict] = []
            for b in unit_bursts:
                reg = b.get("_Region") or calculation_region(str(b["Electrode"]))
                index = region_window_indexes.get(reg)
                flag = index is not None and burst_in_window_fn(float(b["Start_ms"]), index)
                b["IsRegionBurst"] = bool(flag)
                if flag:
                    all_bursts_region.append(b)

            _write_csv(
                region_bursts,
                burst_region_csv_path(run_dir, side_tag, flat=flat_output),
                REGION_BURST_COLUMNS,
            )

            all_bursts = all_bursts_region

        stage2_region_bursts = [dict(b) for b in all_bursts]
        region_window_list.append(region_windows_by_region)

        # Stage 2: network-level bursts
        network_windows: list[tuple[float, float]] = []

        if RS_NETWORK_ONSETS_TOGGLE:
            onset_source = all_bursts
            onsets_ms = np.sort(np.asarray([b["Start_ms"] for b in onset_source], float))

            network_bursts: list[dict] = []
            if onsets_ms.size >= MIN_SPIKES_IN_BURST:
                baseline_onsets_ms = None
                if RS_OFFSET_NULL_ENABLE:
                    # Build per-region onset trains (sources) and apply a seeded random offset per region.
                    by_region: dict[str, list[float]] = {}
                    for bb in onset_source:
                        reg = calculation_region(str(bb.get("Electrode", "")))
                        by_region.setdefault(reg, []).append(float(bb["Start_ms"]))
                    by_region_arr: dict[str, np.ndarray] = {
                        k: np.sort(np.asarray(v, dtype=float)) for k, v in by_region.items()
                    }
                    baseline_onsets_ms = _build_offset_reference_train_ms(
                        by_region_arr,
                        observation_bounds_ms=observation_bounds_ms,
                        context="network",
                        seed=int(RS_OFFSET_NULL_SEED),
                        max_offset_ms=RS_OFFSET_NULL_MAX_OFFSET_MS,
                    )

                start_idx2, length2, RSvals2 = RS_detect_burst(
                    onsets_ms,
                    limit=RS_Limit_network,
                    RSalpha=RS_alpha_network,
                    min_spikes=MIN_SPIKES_IN_BURST,
                    RS_Percentile_Limit=RS_Percentile_Limit_network,
                    baseline_spiketimes=baseline_onsets_ms,
                )

                windows: list[tuple[float, float]] = []
                rs_vals_for_windows: list[float] = []
                num_events_for_windows: list[int] = []
                for s2, L2, rs2 in zip(start_idx2, length2, RSvals2):
                    s2 = int(s2); e2 = int(s2 + L2 - 1)
                    t0 = float(onsets_ms[s2]); t1 = float(onsets_ms[e2])
                    if t1 < t0:
                        continue
                    windows.append((t0, t1))
                    rs_vals_for_windows.append(float(rs2))
                    num_events_for_windows.append(int(L2))

                if windows:
                    spans_ms, _ = compute_spans_and_bars(
                        windows, all_bursts,
                        min_unique_channels=MIN_UNIQUE_CHANNELS_NETWORK,
                        min_unique_regions=MIN_UNIQUE_REGIONS_NETWORK,
                        channel_mode="elec_cluster",
                        region_fn=lambda b: infer_region_fn(str(b.get("Electrode", ""))),
                        use_onset_plus_length=NETWORK_SPAN_ONSET_PLUS_LENGTH,
                    )
                    (
                        accepted_windows,
                        network_bursts,
                        bars_s,
                    ) = _filter_network_support_events(
                        windows,
                        spans_ms,
                        rs_vals_for_windows,
                        num_events_for_windows,
                    )
                    network_windows.extend(accepted_windows)
                    network_bars_temp[side_tag] = bars_s

            network_window_index = prepare_window_index(network_windows)
            for b in all_bursts:
                b["IsNetworkBurst"] = burst_in_window_fn(float(b["Start_ms"]), network_window_index)

            _write_csv(
                network_bursts,
                burst_network_csv_path(run_dir, side_tag, flat=flat_output),
                NETWORK_BURST_COLUMNS,
            )

            if plot_network_bursts:
                all_bursts = [b for b in all_bursts if b.get("IsNetworkBurst")]

        stage3_network_bursts = [dict(b) for b in all_bursts if b.get("IsNetworkBurst")]
        network_window_list.append(network_windows)
        stage1_unit_bursts_list.append(stage1_unit_bursts)
        stage2_region_bursts_list.append(stage2_region_bursts)
        stage3_network_bursts_list.append(stage3_network_bursts)
        all_bursts_list.append(all_bursts)

    res.all_bursts = all_bursts_list[0]
    res.all_bursts_L = all_bursts_list[1]
    res.all_bursts_R = all_bursts_list[2]
    res.stage1_unit_bursts = stage1_unit_bursts_list[0]
    res.stage1_unit_bursts_L = stage1_unit_bursts_list[1]
    res.stage1_unit_bursts_R = stage1_unit_bursts_list[2]
    res.stage2_region_bursts = stage2_region_bursts_list[0]
    res.stage2_region_bursts_L = stage2_region_bursts_list[1]
    res.stage2_region_bursts_R = stage2_region_bursts_list[2]
    res.stage3_network_bursts = stage3_network_bursts_list[0]
    res.stage3_network_bursts_L = stage3_network_bursts_list[1]
    res.stage3_network_bursts_R = stage3_network_bursts_list[2]
    res.region_windows = region_window_list[0]
    res.region_windows_L = region_window_list[1]
    res.region_windows_R = region_window_list[2]
    res.network_windows = network_window_list[0]
    res.network_windows_L = network_window_list[1]
    res.network_windows_R = network_window_list[2]
    
    # Assign pre-computed bars
    res.network_bars_L = network_bars_temp.get("left", [])
    res.network_bars_R = network_bars_temp.get("right", [])
    res.region_bars_by_region_L = region_bars_temp.get("left", {})
    res.region_bars_by_region_R = region_bars_temp.get("right", {})

    print(f"Burst counts: combined={len(res.all_bursts)}, L={len(res.all_bursts_L)}, R={len(res.all_bursts_R)}")

    return res


def rs_burst_detection(
    spike_struct, spike_struct_L, spike_struct_R, STATS: pd.DataFrame,
    run_dir: Path, iter_units_fn, infer_region_fn, burst_in_window_fn,
    flat_output: bool = False, *,
    observation_bounds_ms: tuple[float, float] | None = None,
    observation_source: str = "unspecified",
) -> BurstResults:
    """Run the cascade and persist the coverage/status of its Stage-1 inputs."""
    res = BurstResults()
    try:
        _rs_burst_detection_impl(
            spike_struct, spike_struct_L, spike_struct_R, STATS, run_dir,
            iter_units_fn, infer_region_fn, burst_in_window_fn, flat_output,
            observation_bounds_ms=observation_bounds_ms,
            observation_source=observation_source, _result=res,
        )
    except Exception as exc:
        failure_summary = summarize_stage1_diagnostics(res.stage1_diagnostics)
        if not any(r.get("status") == "failed" for r in res.stage1_diagnostics):
            res.stage1_diagnostics.append({
                "side": "", "electrode": "", "cluster": -1,
                "observation_source": str(observation_source),
                "status": "failed", "reason": "analysis_failure", "error": str(exc),
                "eligible": False,
            })
        res.analysis_status = summarize_stage1_diagnostics(res.stage1_diagnostics)
        res.analysis_status.update(status="failed", error=str(exc))
        if failure_summary["status"] == "failed":
            res.analysis_status["reason"] = failure_summary["reason"]
        _write_stage1_analysis_status(run_dir, res.stage1_diagnostics, res.analysis_status)
        raise
    if not rankSurprise_toggle:
        return res
    res.analysis_status = summarize_stage1_diagnostics(res.stage1_diagnostics)
    if not bool(RS_WIN_SHUFF_STAGE1_ENABLE) and not res.stage1_diagnostics:
        # Existing no-bounds classic runs do not use the new shuffle diagnostics.
        res.analysis_status.update(status="complete", reason="shuffle_disabled")
    for side in ("combined", "left", "right"):
        res.analysis_status["per_side"].setdefault(side, {
            "status": "complete" if not bool(RS_WIN_SHUFF_STAGE1_ENABLE) and not res.stage1_diagnostics else "no_analyzable_data",
            "reason": "shuffle_disabled" if not bool(RS_WIN_SHUFF_STAGE1_ENABLE) and not res.stage1_diagnostics else "no_eligible_input",
            "ok_intervals": 0, "skipped_intervals": 0, "failed_intervals": 0,
        })
        res.analysis_status["stage_status"].setdefault(side, {
            stage: res.analysis_status["per_side"][side]["status"]
            for stage in ("unit", "region", "network")
        })
    _write_stage1_analysis_status(run_dir, res.stage1_diagnostics, res.analysis_status)
    return res

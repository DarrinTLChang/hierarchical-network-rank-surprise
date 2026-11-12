# utils.py
from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from config import *
from .detection import *
from .intervals import WindowIndex, time_in_window_index
from .stats import snr_fails_filter
from .detector_profile import WIN_SHUFF_IMPLEMENTATION_VERSION


def _format_number(value: float) -> str:
    number = float(value)
    if np.isfinite(number) and abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


def _rs_tag_head(params: dict) -> str:
    alphas = (
        int(round(100 * float(params["RS_alpha_cluster"]))),
        int(round(100 * float(params["RS_alpha_region"]))),
        int(round(100 * float(params["RS_alpha_network"]))),
    )
    limits = (
        int(params["RS_limit_cluster"]),
        int(params["RS_limit_region"]),
        int(params["RS_limit_network"]),
    )
    return f"({alphas[0]},{alphas[1]},{alphas[2]})_({limits[0]},{limits[1]},{limits[2]})"


def _rs_minimums_tag(params: dict) -> str:
    max_duration = (
        f"__maxDur={params['MAX_BURST_DURATION']}ms"
        if params.get("MAX_BURST_DURATION") is not None
        else ""
    )
    return (
        f"minSpk={params['MIN_SPIKES_IN_BURST']}"
        f"__FR={_format_number(float(params['FR_MIN_HZ']))}hz"
        f"__minDur={params['MIN_BURST_DURATION']}ms"
        f"{max_duration}"
        f"__minCh={params['MIN_UNIQUE_CHANNELS_NETWORK']}"
        f"__minReg={params['MIN_UNIQUE_REGIONS_NETWORK']}"
    )


def display_region_name(region: str) -> str:
    """Map a raw region code to a presentation label (slashes, casing, etc.)."""
    label = str(region)
    anatomical = anatomical_region_label(label)
    side = label[len(anatomical) + 1:] if anatomical != label else ""
    m = re.fullmatch(r"shank(\d+)", anatomical, flags=re.IGNORECASE)
    if m:
        display = f"Shank {m.group(1)}"
    else:
        display = REGION_DISPLAY.get(anatomical, anatomical)
    return f"{display} ({side})" if side else display


def dataset_labels(mat_path: str) -> tuple[str, str]:
    p = Path(mat_path)
    patient = p.parent.name or "unknown"
    m = re.search(r"_p(\d+)", p.stem)
    period = f"Period {m.group(1)}" if m else "Period ?"
    return patient, period


def dataset_labels_any(spec: str) -> tuple[str, str]:
    """Like dataset_labels but supports ``kilosort:`` and ``kilosortset:`` entries (see kilosort_loader)."""
    from .kilosort_loader import dataset_labels_kilosort, parse_dataset_spec

    kind, path = parse_dataset_spec(spec)
    if kind == "kilosort":
        patient, period = dataset_labels_kilosort(path)
    elif kind == "kilosortset":
        first = path.split(";", 1)[0].strip()
        patient, period = dataset_labels_kilosort(first)
    else:
        return dataset_labels(path)

    # Compact label for titles/prints: first + last underscore token
    # (e.g. M415_gate..._imec1 → M415_imec1; legacy rat m360_shank0_imec0 → m360_imec0).
    parts = patient.split("_")
    if len(parts) >= 3:
        patient = f"{parts[0]}_{parts[-1]}"
    return patient, period


def compute_recording_duration_s(spike_struct) -> float:
    from .observation import stage1_observation_bounds_ms
    (start, end), _ = stage1_observation_bounds_ms(spike_struct)
    return (end - start) / 1000.0


def apply_observation_interval(spike_struct, bounds_ms, source):
    """Copy channels and restrict all downstream calculations to [start, end]."""
    from copy import copy
    from .observation import validate_observation_bounds_s
    start_s, end_s = validate_observation_bounds_s(np.asarray(bounds_ms) / 1000.0)
    start_ms, end_ms = map(float, bounds_ms)
    channels = []
    for original in np.ravel(spike_struct):
        ch = copy(original)
        raw = getattr(ch, "time", [])
        # scipy squeeze_me flattens a single MATLAB cell into a numeric vector.
        arr = np.asarray(raw)
        trains = [arr] if arr.dtype != object else list(np.ravel(arr))
        ch.time = np.empty(len(trains), dtype=object)
        for i, train in enumerate(trains):
            times = np.asarray(train, dtype=float).reshape(-1)
            ch.time[i] = times[(times >= start_ms) & (times <= end_ms)]
        ch.analyzed_start_s, ch.analyzed_end_s = start_s, end_s
        ch.dataSegmentLength = end_s - start_s
        ch.duration_source = str(source)
        channels.append(ch)
    return np.asarray(channels, dtype=object)


def parse_electrode(elec: str, combine_numbered_regions: bool | None = None) -> dict:
    """Parse electrode string. If combine_numbered_regions is None, uses config COMBINE_NUMBERED_REGIONS."""
    if combine_numbered_regions is None:
        combine_numbered_regions = COMBINE_NUMBERED_REGIONS
    base = str(elec).replace("_CommonFiltered", "")
    parts = base.split("_")

    head_raw = parts[0] if parts else base
    head = head_raw[5:] if head_raw.lower().startswith("micro") else head_raw

    side = parts[1] if len(parts) > 1 and parts[1] in ("L", "R") else ""
    idx = parts[2] if len(parts) > 2 and parts[2].isdigit() else (parts[2] if len(parts) > 2 else "")

    region = re.sub(r"\d+$", "", head) if combine_numbered_regions else head

    return {"base": base, "head": head, "side": side, "idx": idx, "region": region}


def infer_shank(elec: str) -> str | None:
    """
    Parse shank id from synthetic Kilosort electrode names (``rat_*_...``).

    1. ``shankN`` / ``ShankN`` anywhere in the string (covers rat legacy paths and M415-style gates).
    2. Else M401-style ``SNN_g`` (e.g. ``S13_g0`` → ``shank13``). Precedence: (1) wins if both match.
    """
    s = str(elec)
    m = re.search(r"(shank\d+)", s, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"(?<![A-Za-z])S(\d+)_g", s, re.IGNORECASE)
    if m2:
        return f"shank{int(m2.group(1))}"
    return None


def infer_depth_um(elec: str) -> float | None:
    """Parse depth tag from synthetic Kilosort names like ``..._d1234um_CommonFiltered``."""
    m = re.search(r"_d(-?\d+)um", str(elec), re.IGNORECASE)
    return float(m.group(1)) if m else None


def infer_region(elec: str, combine_numbered_regions: bool | None = None) -> str:
    """Return region for electrode. If combine_numbered_regions is None, uses config COMBINE_NUMBERED_REGIONS (True = GPi1+GPi2→GPi).

    When ``KILOSORT_COLOR_BY_SHANK`` and the name looks like ``rat_*_shankN_*``, returns ``shankN`` so plots match human region grouping.
    """
    if KILOSORT_COLOR_BY_SHANK and str(elec).startswith("rat_"):
        sk = infer_shank(elec)
        if sk is not None:
            return sk
    return parse_electrode(elec, combine_numbered_regions=combine_numbered_regions)["region"]


def calculation_region(elec: str) -> str:
    """Return a regional calculation identity, retaining numbering and hemisphere.

    Contacts and clusters with the same anatomical head and side share one region.
    Kilosort's ``rat_*`` names retain their existing shank mapping; their embedded
    ``_L_`` is routing metadata and is not interpreted as anatomical laterality.
    ``infer_region`` remains the separate anatomical network-eligibility policy.
    """
    name = str(elec)
    if name.startswith("rat_"):
        return infer_region(name)
    parsed = parse_electrode(name, combine_numbered_regions=False)
    if not parsed["head"].strip() or parsed["side"] not in ("L", "R"):
        raise ValueError(
            f"Cannot determine a regional calculation identity for electrode {name!r}: "
            "a nonempty region and explicit _L_ or _R_ hemisphere are required"
        )
    return f"{parsed['head']}_{parsed['side']}"


def anatomical_region_label(region: str) -> str:
    """Remove only a calculation key's hemisphere suffix for labels and colors.

    Numbered anatomical instances remain distinct. This presentation helper does
    not define the separate network minimum-anatomical-region eligibility gate.
    """
    return re.sub(r"_(?:L|R)$", "", str(region))


def color_for_plot_group(name: str) -> str:
    """Line or bar color for a brain-region or shank plot group."""
    name = anatomical_region_label(name)
    if name in SHANK_COLORS:
        return SHANK_COLORS[name]
    if str(name).lower().startswith("shank") and str(name)[5:].isdigit():
        return "rgb(140, 140, 160)"
    return REGION_COLORS.get(
        name, REGION_COLORS.get(re.sub(r"\d+$", "", name), "rgb(30,30,200)"),
    )


def color_for_nonburst(elec: str) -> str:
    return color_for_plot_group(calculation_region(elec))


def build_sorted_spike_cache(spike_struct) -> dict[tuple[str, int], np.ndarray]:
    """Convert and sort every unit once for reuse within a dataset run."""
    cache: dict[tuple[str, int], np.ndarray] = {}
    for ch in np.ravel(spike_struct):
        elec = str(ch.electrode)
        for cl, arr in enumerate(np.ravel(ch.time)):
            spk = np.asarray(arr).ravel().astype(float)
            cache[(elec, int(cl))] = np.sort(spk)
    return cache


def iter_units_from_stats(spike_struct, STATS, *, spike_cache=None):
    """Yields (elec, cl, spikes_ms_sorted) for each unit in STATS.index."""
    ch_by_elec = {str(ch.electrode): ch for ch in np.ravel(spike_struct)}

    for (elec, cl) in STATS.index:
        elec = str(elec)
        cl = int(cl)

        ch = ch_by_elec.get(elec)
        if ch is None:
            continue

        key = (elec, cl)
        if spike_cache is not None and key in spike_cache:
            spk = spike_cache[key]
            spk = spk[np.isfinite(spk)]
        else:
            try:
                arr = ch.time[cl]
            except Exception:
                arr = np.ravel(ch.time)[cl]
            if np.isscalar(arr):
                continue
            spk = np.asarray(arr).ravel().astype(float)
            spk = np.sort(spk[np.isfinite(spk)])
        if spk.size < 2:
            continue

        yield elec, cl, spk


def split_spike_struct_by_side(spike_struct):
    left = []
    right = []

    for ch in np.ravel(spike_struct):
        elec = str(ch.electrode)
        if "_L_" in elec:
            left.append(ch)
        elif "_R_" in elec:
            right.append(ch)

    return np.array(left, dtype=object), np.array(right, dtype=object)


def build_spike_labels_df(
    spike_struct, STATS, allowed_clusters, ibi_merge_factor, *, spike_cache=None,
):
    rows = []

    for ch in np.ravel(spike_struct):
        elec = str(ch.electrode)

        for cl, arr in enumerate(np.ravel(ch.time)):
            if (elec, cl) not in allowed_clusters:
                continue

            key = (elec, int(cl))
            if spike_cache is not None and key in spike_cache:
                spk = spike_cache[key]
            else:
                spk = np.asarray(arr).flatten().astype(float)
                spk = np.sort(spk)
            if spk.size == 0:
                continue

            bursts = detect_bursts(spk, elec, cl, STATS, ibi_merge_factor=ibi_merge_factor)

            burst_id = np.full(spk.shape, -1, dtype=int)

            bursts_sorted = sorted(bursts, key=lambda b: float(b["Start_ms"]))
            j = 0
            for b_idx, b in enumerate(bursts_sorted):
                s = float(b["Start_ms"]); e = float(b["End_ms"])
                while j < spk.size and spk[j] < s:
                    j += 1
                k = j
                while k < spk.size and spk[k] <= e:
                    burst_id[k] = b_idx
                    k += 1
                j = k

            is_burst = burst_id >= 0

            for t, ib, bid in zip(spk, is_burst, burst_id):
                rows.append(
                    dict(
                        Electrode=elec,
                        Cluster=int(cl),
                        Spike_ms=float(t),
                        IsBurst=bool(ib),
                        BurstIndex=int(bid),
                    )
                )

    return pd.DataFrame(rows)


def pool_spike_struct_per_channel(spike_struct, record_len_s):
    class PooledCh:
        pass

    pooled_struct = []

    for ch in np.ravel(spike_struct):
        elec = str(ch.electrode)
        snr_arr = np.asarray(getattr(ch, "snr", [])).flatten()

        pooled = []
        for cl, arr in enumerate(np.ravel(ch.time)):
            spk = np.asarray(arr).flatten().astype(float)
            if spk.size < 2:
                continue
            fr_hz = float(spk.size / record_len_s) if record_len_s > 0 else float("nan")
            if fr_hz < FR_MIN_HZ:
                continue
            snr_val = float(snr_arr[cl]) if cl < snr_arr.size else float("nan")
            if snr_fails_filter(snr_val):
                continue
            pooled.append(spk)

        if not pooled:
            continue

        pooled_spk = np.sort(np.concatenate(pooled))
        if pooled_spk.size < 2:
            continue

        pc = PooledCh()
        pc.electrode = elec
        pc.time = np.empty((1,), dtype=object)
        pc.time[0] = pooled_spk
        pc.snr = np.array([np.nan], dtype=float)
        pc.dataSegmentLength = float(record_len_s)
        for name in ("analyzed_start_s", "analyzed_end_s", "duration_source", "duration_estimated"):
            if hasattr(ch, name):
                setattr(pc, name, getattr(ch, name))

        pooled_struct.append(pc)

    return np.array(pooled_struct, dtype=object)


def get_thresholding_method_name() -> tuple[str, int]:
    if maxisi_toggle:
        return "maxISI", 1
    if alpha_meanisi_toggle:
        return "alpha_meanISI", 2
    if rankSurprise_toggle:
        return "rankSurprise", 3
    return "unknown method", 0


def build_run_params(method_name: str, base_thr: float) -> dict:
    """Build a dict of all run parameters for the active method.

    Used to generate directory names, figure titles, and saved JSON.
    """
    common = {
        "method": method_name,
        "SNR_MIN": SNR_MIN,
        "SNR_MAX": SNR_MAX,
        "FR_MIN_HZ": FR_MIN_HZ,
        "MIN_SPIKES_IN_BURST": MIN_SPIKES_IN_BURST,
        "MIN_BURST_DURATION": MIN_BURST_DURATION,
        "MAX_BURST_DURATION": MAX_BURST_DURATION,
        "MIN_UNIT_BURST_DURATION_MS": globals().get(
            "MIN_UNIT_BURST_DURATION_MS", MIN_BURST_DURATION
        ),
        "MIN_REGION_BURST_SUPPORT_DURATION_MS": globals().get(
            "MIN_REGION_BURST_SUPPORT_DURATION_MS", MIN_BURST_DURATION
        ),
        "MIN_NETWORK_BURST_SUPPORT_DURATION_MS": globals().get(
            "MIN_NETWORK_BURST_SUPPORT_DURATION_MS", MIN_BURST_DURATION
        ),
        "MAX_NETWORK_BURST_SUPPORT_DURATION_MS": globals().get(
            "MAX_NETWORK_BURST_SUPPORT_DURATION_MS", MAX_BURST_DURATION
        ),
        "pooling": pooling_toggle,
        "MIN_UNIQUE_CHANNELS_NETWORK": MIN_UNIQUE_CHANNELS_NETWORK,
    }

    if method_name == "maxISI":
        common.update({
            "base_thr": base_thr,
            "ibi_merge_factor": ibi_merge_factor,
        })
    elif method_name == "alpha_meanISI":
        common.update({
            "alpha": ALPHA_MAXISI,
            "min_thr_ms": MIN_MAXISI_MS,
            "max_thr_ms": MAX_MAXISI_MS,
            "ibi_merge_factor": ibi_merge_factor,
        })
    elif method_name == "rankSurprise":
        from .detector_profile import DETECTOR_IMPLEMENTATION_VERSION
        shuffle_enabled = bool(RS_WIN_SHUFF_STAGE1_ENABLE)
        automatic = shuffle_enabled and bool(RS_WIN_SHUFF_AUTO_FROM_RECORDING)
        common.update({
            "detector_implementation_version": DETECTOR_IMPLEMENTATION_VERSION,
            "regional_calculation_identity": "numbered_region_and_hemisphere",
            "network_eligibility_region_policy": "anatomical_infer_region",
            "offset_reference_interval_policy": "declared_observation_bounds",
            "merged_unit_RS_policy": "max_constituent",
            "RS_alpha_cluster": RS_alpha_percentage_stage1,
            "RS_limit_cluster": RS_Percentile_Limit_stage1,
            "RS_alpha_region": RS_alpha_percentage_region,
            "RS_limit_region": RS_Percentile_Limit_region,
            "RS_alpha_network": RS_alpha_percentage_network,
            "RS_limit_network": RS_Percentile_Limit_network,
            "region_burst": RS_region_burst_toggle,
            "network_burst": RS_NETWORK_ONSETS_TOGGLE,
            "RS_offset_null": bool(RS_OFFSET_NULL_ENABLE),
            "RS_offset_null_seed": int(RS_OFFSET_NULL_SEED),
            "RS_offset_null_max_offset_ms": RS_OFFSET_NULL_MAX_OFFSET_MS,
            "RS_win_shuff_stage1": shuffle_enabled,
            "RS_win_shuff_auto_from_recording": automatic,
            "RS_win_shuff_mode": "automatic" if automatic else "custom" if shuffle_enabled else "disabled",
            "RS_win_shuff_window_ms": (
                float(RS_WIN_SHUFF_WINDOW_MS) if shuffle_enabled and not automatic else None
            ),
            "RS_win_shuff_bin_ms": (
                float(RS_WIN_SHUFF_BIN_MS) if shuffle_enabled and not automatic else None
            ),
            "RS_win_shuff_implementation_version": (
                WIN_SHUFF_IMPLEMENTATION_VERSION if shuffle_enabled else None
            ),
            "RS_win_shuff_auto_window_fraction": 0.10 if automatic else None,
            "RS_win_shuff_auto_bin_fraction": 0.10 if automatic else None,
            "RS_stage1_local_segment_enable": bool(RS_STAGE1_LOCAL_SEGMENT_ENABLE),
            "RS_stage1_segment_mode": str(RS_STAGE1_SEGMENT_MODE),
            "RS_stage1_segment_len_s": float(RS_STAGE1_SEGMENT_LEN_S),
            "RS_stage1_segment_overlap_fraction": float(RS_STAGE1_SEGMENT_OVERLAP_FRACTION),
            "MIN_UNIQUE_REGIONS_NETWORK": int(MIN_UNIQUE_REGIONS_NETWORK),
        })

    return common


def _win_shuff_tag(params: dict) -> str:
    if params.get("RS_win_shuff_auto_from_recording"):
        tag = "winAuto(10pct,10pct)"
    else:
        window = _format_number(params["RS_win_shuff_window_ms"])
        bin_size = _format_number(params["RS_win_shuff_bin_ms"])
        tag = f"win({window},{bin_size})"
    version = params.get("RS_win_shuff_implementation_version")
    return f"{tag}_{version}" if version else tag


def run_tag_from_params(params: dict) -> str:
    """Short, filesystem-safe directory name derived from run parameters."""
    m = params["method"]
    # Keep tags compact and stable over time. Use underscores for small joins and
    # double-underscores for major sections (easy to visually scan).
    #
    # Common suffix used by all methods (requested: keep SNR range).
    common_suffix = f"SNR={params['SNR_MIN']}-{params['SNR_MAX']}"
    parts: list[str] = []

    if m == "maxISI":
        parts += [
            f"thr={params['base_thr']}",
            f"ibi={params['ibi_merge_factor']}",
            f"minSpk={params['MIN_SPIKES_IN_BURST']}",
            f"FR={params['FR_MIN_HZ']}hz",
            f"minDur={params['MIN_BURST_DURATION']}ms",
            *([f"maxDur={params['MAX_BURST_DURATION']}ms"] if params.get("MAX_BURST_DURATION") is not None else []),
            common_suffix,
        ]
        if params["pooling"]:
            parts.append("pooled")
    elif m == "alpha_meanISI":
        parts += [
            f"alpha={params['alpha']}",
            f"range={params['min_thr_ms']}-{params['max_thr_ms']}ms",
            f"ibi={params['ibi_merge_factor']}",
            f"minSpk={params['MIN_SPIKES_IN_BURST']}",
            f"FR={params['FR_MIN_HZ']}hz",
            f"minDur={params['MIN_BURST_DURATION']}ms",
            *([f"maxDur={params['MAX_BURST_DURATION']}ms"] if params.get("MAX_BURST_DURATION") is not None else []),
            common_suffix,
        ]
        if params["pooling"]:
            parts.append("pooled")
    elif m == "rankSurprise":
        head = _rs_tag_head(params)
        mins = _rs_minimums_tag(params)
        # Preferred style: ..._SNR=..._FR=..._region__network
        toggles = ""
        # (Requested) omit region/network toggles from the run tag.
        if params.get("RS_offset_null"):
            toggles += "__offNull" if toggles else "offNull"
        if params.get("RS_win_shuff_stage1"):
            toggles += ("__" if toggles else "") + _win_shuff_tag(params)
        if params.get("RS_stage1_local_segment_enable"):
            seg_mode = str(params.get("RS_stage1_segment_mode", "nonoverlap"))
            seg_len = _format_number(float(params.get("RS_stage1_segment_len_s", RS_STAGE1_SEGMENT_LEN_S)))
            ov = _format_number(float(params.get("RS_stage1_segment_overlap_fraction", RS_STAGE1_SEGMENT_OVERLAP_FRACTION)))
            toggles += f"__seg({seg_mode},{seg_len}s,ov={ov})" if toggles else f"seg({seg_mode},{seg_len}s,ov={ov})"
        tag = f"{head}_{mins}_{common_suffix}{toggles}"
        parts += [tag]
        if params["pooling"]:
            parts.append("pooled")

    # For RS we intentionally keep one large head string, then append optional toggles
    # as __region and __network (matching existing conventions in scripts).
    return "__".join(str(p) for p in parts)


def run_tag_from_params_kilosort(params: dict, *, good_only: bool, sep_shank: bool) -> str:
    """
    Kilosort output folders use a shorter, KS-specific tag:
    - Keep RS alpha/limits + minSpk/minDur/minCh + region/network toggles
    - Drop SNR/FR (KS clusters use NaN SNR; FR filtering is implicit in stats)
    - Append optional flags: GoodOnly, SepShank
    """
    m = params["method"]
    if m != "rankSurprise":
        # For now, only RS is used for KS runs; fall back to default naming.
        return run_tag_from_params(params)

    head = _rs_tag_head(params)

    flags = []
    if good_only:
        flags.append("GoodOnly")
    if sep_shank:
        flags.append("SepShank")
    flags_str = ("_" + "_".join(flags)) if flags else ""

    mins = _rs_minimums_tag(params)

    toggles = ""
    if params.get("RS_win_shuff_stage1"):
        toggles += ("__" if toggles else "_") + _win_shuff_tag(params)
    if params.get("RS_stage1_local_segment_enable"):
        seg_mode = str(params.get("RS_stage1_segment_mode", "nonoverlap"))
        seg_len = _format_number(float(params.get("RS_stage1_segment_len_s", RS_STAGE1_SEGMENT_LEN_S)))
        ov = _format_number(float(params.get("RS_stage1_segment_overlap_fraction", RS_STAGE1_SEGMENT_OVERLAP_FRACTION)))
        toggles += f"__seg({seg_mode},{seg_len}s,ov={ov})" if toggles else f"_seg({seg_mode},{seg_len}s,ov={ov})"

    return f"{head}{flags_str}_{mins}{toggles}"


def figure_title_from_params(
    params: dict, patient: str, period: str, *, compact_kilosort: bool = False,
) -> str:
    """Human-readable figure title from run parameters."""
    m = params["method"]
    if compact_kilosort:
        header = f"{patient} \u2014 {m}"
    else:
        header = f"{patient} \u2022 {period} \u2014 {m}"

    if m == "maxISI":
        if compact_kilosort:
            detail = (
                f"thr={params['base_thr']}  IBI\u00d7{params['ibi_merge_factor']}  "
                f"dur\u2265{params['MIN_BURST_DURATION']}ms"
            )
        else:
            detail = (
                f"{params['SNR_MIN']}\u2264SNR\u2264{params['SNR_MAX']}  "
                f"thr={params['base_thr']}  IBI\u00d7{params['ibi_merge_factor']}  "
                f"dur\u2265{params['MIN_BURST_DURATION']}ms"
            )
    elif m == "alpha_meanISI":
        if compact_kilosort:
            detail = (
                f"\u03b1={params['alpha']}  thr=[{params['min_thr_ms']},{params['max_thr_ms']}]ms  "
                f"IBI\u00d7{params['ibi_merge_factor']}  "
                f"dur\u2265{params['MIN_BURST_DURATION']}ms"
            )
        else:
            detail = (
                f"{params['SNR_MIN']}\u2264SNR\u2264{params['SNR_MAX']}  "
                f"\u03b1={params['alpha']}  thr=[{params['min_thr_ms']},{params['max_thr_ms']}]ms  "
                f"IBI\u00d7{params['ibi_merge_factor']}  "
                f"dur\u2265{params['MIN_BURST_DURATION']}ms"
            )
    elif m == "rankSurprise":
        if compact_kilosort:
            detail = (
                f"\u03b1c={params['RS_alpha_cluster']:.0%}  "
                f"\u03b1r={params['RS_alpha_region']:.0%}  "
                f"\u03b1n={params['RS_alpha_network']:.0%}  "
            )
            if params.get("RS_win_shuff_stage1"):
                if params.get("RS_win_shuff_auto_from_recording"):
                    detail += "WinShuff(auto: 10% interval / 10% window)  "
                else:
                    win_w = _format_number(params["RS_win_shuff_window_ms"])
                    win_b = _format_number(params["RS_win_shuff_bin_ms"])
                    detail += f"WinShuff({win_w},{win_b})  "
        else:
            detail = (
                f"{params['SNR_MIN']}\u2264SNR\u2264{params['SNR_MAX']}  "
                f"\u03b1c={params['RS_alpha_cluster']:.0%}  "
                f"\u03b1r={params['RS_alpha_region']:.0%}  "
                f"\u03b1n={params['RS_alpha_network']:.0%}  "
            )
        if params.get("RS_offset_null"):
            mx = params.get("RS_offset_null_max_offset_ms")
            mx_txt = f"{float(mx):.0f}ms" if mx is not None else "auto"
            # Presentation rename: show channel shift range, not the random seed.
            detail += f"  ChanShift({mx_txt})"
    else:
        detail = ""

    if params.get("pooling"):
        detail += "  [POOLED]"

    return f"{header}:  {detail}" if detail else header


def save_run_params(params: dict, run_dir: Path) -> None:
    """Write run parameters as a JSON file inside the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2, default=str)


def burst_in_window(t_ms: float, window) -> bool:
    if isinstance(window, WindowIndex):
        return time_in_window_index(t_ms, window)
    for t0, t1 in window:
        if t0 <= t_ms <= t1:
            return True
    return False

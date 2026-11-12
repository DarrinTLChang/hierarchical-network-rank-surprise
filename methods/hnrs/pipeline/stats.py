# stats.py
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import scipy.io as spio
from pathlib import Path
from config import *
from .observation import stage1_observation_bounds_ms, validate_observation_bounds_s


# Keep the same table contract when preprocessing retains no units. Stage 1 can
# then report no_analyzable_data instead of failing while constructing the index.
CLUSTER_STATS_COLUMNS = [
    "Electrode", "Cluster", "FR_Hz", "ISI_mean_ms", "ISI_median_ms",
    "ISI_std_ms", "Start_ms", "End_ms", "Duration_ms", "thr", "SNR",
]


def snr_fails_filter(snr_val: float) -> bool:
    """True if cluster should be excluded: NaN, below SNR_MIN, or above SNR_MAX (if set)."""
    if np.isnan(snr_val) or snr_val < SNR_MIN:
        return True
    if SNR_MAX is not None and snr_val > SNR_MAX:
        return True
    return False


def build_cache(
    mat_file: str | None,
    cluster_stats_csv: Path,
    pooled_cluster_stats_csv: Path,
    base_thr: float,
    record_len_s: float,
    adaptie_thr_toggle: bool,
    pooling_toggle,
    *,
    spike_struct=None,
    skip_snr_filter: bool = False,
    observation_bounds_ms: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Scan spike data and compute per-cluster statistics.

    Provide either ``mat_file`` (loads variable spikeTime) or ``spike_struct`` in memory
    (e.g. from :func:`pipeline.kilosort_loader.load_kilosort_dir`).

    ``skip_snr_filter`` (e.g. for Kilosort loads with NaN SNR) skips SNR_MIN / SNR_MAX checks;
    FR_MIN_HZ and minimum spike count still apply.

    Writes:
      - isi_stats.csv
      - cluster_stats.csv

    Returns:
      DataFrame indexed by (Electrode, Cluster)
    """
    if spike_struct is not None:
        st = spike_struct
    elif mat_file is not None:
        mat = spio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        st = mat["spikeTime"]
    else:
        raise ValueError("build_cache requires mat_file or spike_struct")

    if observation_bounds_ms is not None:
        start_s, end_s = validate_observation_bounds_s(np.asarray(observation_bounds_ms, dtype=float) / 1000.0)
        bounds = (start_s * 1000.0, end_s * 1000.0)
    elif any(hasattr(ch, "dataSegmentLength") or hasattr(ch, "analyzed_start_s")
             or hasattr(ch, "analyzed_end_s") for ch in np.ravel(st)):
        bounds, _source = stage1_observation_bounds_ms(st)
    else:
        # Compatibility for in-memory callers whose record_len_s argument is
        # their explicit exposure declaration (including empty statistics).
        start_s, end_s = validate_observation_bounds_s((0.0, record_len_s))
        bounds = (start_s * 1000.0, end_s * 1000.0)
    exposure_s = (bounds[1] - bounds[0]) / 1000.0

    def analyzed_spikes(values) -> np.ndarray:
        spikes = np.asarray(values, dtype=float).ravel()
        return np.sort(spikes[np.isfinite(spikes) & (spikes >= bounds[0]) & (spikes <= bounds[1])])

    if not skip_snr_filter:
        print(f"[DEBUG build_cache] SNR_MIN={SNR_MIN}, SNR_MAX={SNR_MAX}")
    else:
        print("[DEBUG build_cache] SNR filter skipped (skip_snr_filter=True); FR_MIN_HZ still applies")
    rows: list[dict[str, Any]] = []
    _skipped_snr = 0
    _kept = 0

    for ch in np.ravel(st):
        elec = str(ch.electrode)
        snr_arr = np.asarray(ch.snr).flatten()
        
        for cl, arr in enumerate(np.ravel(ch.time)):
            snr_val = float(snr_arr[cl]) if cl < snr_arr.size else float("nan")
            spk = analyzed_spikes(arr)
            if spk.size < 2:
                continue

            fr_hz = spk.size / exposure_s
            if fr_hz < FR_MIN_HZ:
                continue
            if not skip_snr_filter and snr_fails_filter(snr_val):
                _skipped_snr += 1
                print(f"  [SKIP] {elec} cl={cl} SNR={snr_val:.3f} (fails filter)")
                continue
            _kept += 1

            isis = np.diff(spk.astype(float))

            isi_mean = float(np.mean(isis))
            isi_median = float(np.median(isis))

            isi_std = float(np.std(isis, ddof=0))
            start_ms = float(spk[0])
            end_ms   = float(spk[-1])
            duration_ms = end_ms - start_ms

            if adaptie_thr_toggle:
                if fr_hz > 0:
                    thr = ALPHA_MAXISI * isi_mean
                else:
                    thr = MAX_MAXISI_MS 

                thr = float(np.clip(thr, MIN_MAXISI_MS, MAX_MAXISI_MS))
            else:
                thr = float(base_thr)

            rows.append(
                dict(
                    Electrode=elec,
                    Cluster=cl,
                    FR_Hz=fr_hz,
                    ISI_mean_ms=isi_mean,
                    ISI_median_ms= isi_median,
                    ISI_std_ms=isi_std,
                    Start_ms=start_ms,
                    End_ms=end_ms,
                    Duration_ms=duration_ms,
                    thr=thr,
                    SNR=snr_val,
                )
            )

    df = pd.DataFrame(rows, columns=CLUSTER_STATS_COLUMNS)
    df.to_csv(cluster_stats_csv, index=False)

    if not skip_snr_filter:
        print(f"[DEBUG build_cache] Kept {_kept} clusters, skipped {_skipped_snr} by SNR filter")
    else:
        print(f"[DEBUG build_cache] Kept {_kept} clusters (SNR not used for filtering)")
    if _kept > 0:
        snr_vals = [r["SNR"] for r in rows if not np.isnan(r["SNR"])]
        if snr_vals:
            print(f"[DEBUG build_cache] SNR range of kept: {min(snr_vals):.2f} – {max(snr_vals):.2f}")
    print(f"• Wrote {len(df)} rows → {cluster_stats_csv}")

    if pooling_toggle:
        pooled_rows: list[dict[str, Any]] = []
        for ch in np.ravel(st):
            elec = str(ch.electrode)
            snr_arr = np.asarray(ch.snr).flatten()
            pooled = []
            for cl, arr in enumerate(np.ravel(ch.time)):
                snr_val = float(snr_arr[cl]) if cl < snr_arr.size else float("nan")
                spk = analyzed_spikes(arr)
                if spk.size < 2:
                    continue
                fr_hz_cl = spk.size / exposure_s
                if fr_hz_cl < FR_MIN_HZ:
                    continue
                if not skip_snr_filter and snr_fails_filter(snr_val):
                    continue
                pooled.append(spk)
            if not pooled:
                continue
            pooled_spk = np.sort(np.concatenate(pooled))
            fr_hz = pooled_spk.size / exposure_s
            isis = np.diff(pooled_spk.astype(float))

            pooled_isi_mean = float(np.mean(isis))
            pooled_isi_median = float(np.median(isis))
            pooled_isi_std = float(np.std(isis, ddof=0))
            start_ms = float(pooled_spk[0])
            end_ms   = float(pooled_spk[-1])
            duration_ms = end_ms - start_ms

            if adaptie_thr_toggle:
                if fr_hz > 0:
                    thr = ALPHA_MAXISI * pooled_isi_mean
                else:
                    thr = MAX_MAXISI_MS 
                thr = float(np.clip(thr, MIN_MAXISI_MS, MAX_MAXISI_MS))
            else:
                thr = float(base_thr)

            pooled_rows.append(
                dict(
                    Electrode=elec,
                    Cluster= 0,
                    FR_Hz=fr_hz,
                    ISI_mean_ms=pooled_isi_mean,
                    ISI_median_ms= pooled_isi_median,
                    ISI_std_ms= pooled_isi_std,
                    Start_ms=start_ms,
                    End_ms=end_ms,
                    Duration_ms=duration_ms,
                    thr=thr,
                    SNR=np.nan,
                )
            )
        df_pooled = pd.DataFrame(pooled_rows, columns=CLUSTER_STATS_COLUMNS)
        df_pooled.to_csv(pooled_cluster_stats_csv, index=False)
        print(f"• Wrote {len(df_pooled)} rows → {pooled_cluster_stats_csv}")

    df_idx = df.set_index(["Electrode", "Cluster"])

    if pooling_toggle:
        df_pooled_idx = df_pooled.set_index(["Electrode", "Cluster"])
    else:
        df_pooled_idx = None

    return df_idx, df_pooled_idx

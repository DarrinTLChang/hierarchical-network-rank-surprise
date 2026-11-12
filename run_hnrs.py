"""Run the manuscript's native HNRS detector on a spikeTime MAT file.

The MAT file must contain a ``spikeTime`` struct with electrode names,
per-cluster spike times in milliseconds, SNR, and recording duration in
seconds (``dataSegmentLength``). This command does not score the output
against ground truth or apply the paper's separate event consolidation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
HNRS_HOME = ROOT / "methods" / "hnrs"
PROFILE_FILES = ROOT / "methods" / "profiles"
MODES = {"network": "network", "network-null": "network_null", "burst-pause": "burst_pause"}


def _parse_seed(value: str) -> int:
    seed = int(value)
    if not 0 <= seed < 2**32:
        raise argparse.ArgumentTypeError("seed must be an unsigned 32-bit integer")
    return seed


def _load_profile(mode: str, shuffle_off: bool, seed: int | None):
    from pipeline.detector_profile import DetectorProfile

    name = MODES[mode]
    path = PROFILE_FILES / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    values = raw.get("detector_profile", raw).copy()
    values["custom_windows_s"] = tuple(tuple(window) for window in values["custom_windows_s"])
    profile = DetectorProfile(**values)
    if shuffle_off:
        profile_name = (
            "burstpause_stage1_no_window_shuffle_v2"
            if mode == "burst-pause"
            else f"{profile.profile_name}_shuffle_off"
        )
        profile = replace(profile, profile_name=profile_name, win_shuff_enabled=False)
    if seed is not None:
        profile = replace(profile, win_shuff_seed=seed, offset_null_seed=seed)
    return profile


def _native_network(spike_struct, output: Path, profile, bounds, source: str) -> str:
    import numpy as np

    from pipeline import detection
    from pipeline.detector_profile import detector_profile_context
    from pipeline.stats import build_cache
    from pipeline.utils import (
        build_sorted_spike_cache,
        burst_in_window,
        infer_region,
        iter_units_from_stats,
        split_spike_struct_by_side,
    )

    duration_s = (bounds[1] - bounds[0]) / 1000.0
    cache = build_sorted_spike_cache(spike_struct)
    left, right = split_spike_struct_by_side(spike_struct)

    def iterator(channels, stats):
        return iter_units_from_stats(channels, stats, spike_cache=cache)

    with detector_profile_context(profile):
        stats, _ = build_cache(
            None,
            output / "cluster_stats.csv",
            output / "pooled_cluster_stats.csv",
            0.0,
            duration_s,
            False,
            False,
            spike_struct=spike_struct,
            skip_snr_filter=not profile.apply_unit_quality_filters,
            observation_bounds_ms=bounds,
        )
        retained = stats.reset_index()[["Electrode", "Cluster", "FR_Hz", "SNR"]].copy()
        retained["spike_count"] = np.rint(retained.FR_Hz * duration_s).astype(int)
        retained.to_csv(output / "retained_units.csv", index=False)
        result = detection.rs_burst_detection(
            spike_struct,
            left,
            right,
            stats,
            output,
            iterator,
            infer_region,
            burst_in_window,
            flat_output=True,
            observation_bounds_ms=bounds,
            observation_source=source,
        )
    return str(result.analysis_status["status"])


def _native_burst_pause(spike_struct, output: Path, profile, bounds, source: str) -> str:
    import numpy as np
    import pandas as pd

    from pipeline.detection import _detect_stage1_unit_bursts
    from pipeline.detector_profile import detector_profile_context
    from pipeline.burst_csv import UNIT_BURST_COLUMNS
    from pipeline.utils import parse_electrode

    if len(spike_struct) != 1 or len(spike_struct[0].time) != 1:
        raise ValueError("burst-pause mode expects a MAT input with exactly one unit")
    channel = spike_struct[0]
    electrode = str(channel.electrode)
    spikes = np.sort(np.asarray(channel.time[0], dtype=float).ravel())
    diagnostics: list[dict] = []
    with detector_profile_context(profile):
        bursts = _detect_stage1_unit_bursts(
            spikes,
            electrode,
            0,
            observation_bounds_ms=bounds,
            observation_source=source,
            diagnostics=diagnostics,
            side_tag=parse_electrode(electrode)["side"],
        )
    rows = [
        dict(
            Electrode=str(burst["Electrode"]),
            Cluster=int(burst["Cluster"]) + 1,
            burst_start_ms=float(burst["Start_ms"]),
            burst_end_ms=float(burst["End_ms"]),
            burst_duration_ms=float(burst["Duration_ms"]),
            num_onsets=int(burst["Num_Spikes"]),
            RS=float(burst["RS"]),
            RS_aggregation=burst.get("RS_aggregation", "single_interval"),
            Method=str(burst["Method"]),
        )
        for burst in bursts
    ]
    arm = "on" if profile.win_shuff_enabled else "off"
    pd.DataFrame(rows, columns=UNIT_BURST_COLUMNS).sort_values(
        ["burst_start_ms", "burst_end_ms"]
    ).to_csv(output / f"hnrs_{arm}_unit_bursts.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / f"diagnostics_{arm}.csv", index=False)
    return "complete"


def main(argv: list[str] | None = None) -> int:
    # The upstream pipeline prints arrows/check marks in progress messages.
    # Keep redirected output working on narrow Windows console encodings.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="path to the spikeTime MAT file")
    parser.add_argument("output", type=Path, help="new or empty directory for CSV and JSON files")
    parser.add_argument(
        "--mode", choices=tuple(MODES), default="network",
        help=("network for all three stages (default), network-null for the null-control profile, "
              "or burst-pause for one unit and Stage 1 only"),
    )
    parser.add_argument("--shuffle-off", action="store_true", help="disable Stage-1 window shuffling")
    parser.add_argument(
        "--seed", type=_parse_seed,
        help="set both the window-shuffle and onset-offset seeds; defaults to the archived profile",
    )
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error(f"output directory must be new or empty: {args.output}")

    sys.path.insert(0, str(HNRS_HOME))
    from scipy.io import loadmat
    from pipeline.observation import stage1_observation_bounds_ms
    from pipeline.utils import apply_observation_interval

    values = loadmat(args.input, squeeze_me=True, struct_as_record=False, variable_names=["spikeTime"])
    if "spikeTime" not in values:
        parser.error("MAT input must contain a variable named spikeTime")
    raw = values["spikeTime"]
    bounds, source = stage1_observation_bounds_ms(raw)
    clipped = apply_observation_interval(raw, bounds, source)
    profile = _load_profile(args.mode, args.shuffle_off, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    detector = _native_burst_pause if args.mode == "burst-pause" else _native_network
    status = detector(clipped, args.output, profile, bounds, source)
    with args.input.open("rb") as handle:
        input_hash = sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            input_hash.update(block)
    metadata = {
        "mode": args.mode,
        "input_sha256": input_hash.hexdigest(),
        "observation_bounds_ms": list(bounds),
        "observation_source": source,
        "detector_profile": profile.to_dict(),
        "detector_profile_digest": profile.digest,
        "analysis_status": status,
        "post_detector_event_consolidation": False,
    }
    (args.output / "params.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"HNRS {args.mode}: {status}; output written to {args.output}")
    return 0 if status in ("complete", "no_analyzable_data") else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Check fresh native comparator runs against a fixed reference catalog."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import pandas as pd

METHODS = ("logIBEI", "Network-CMA", "meaRtools/Otsu")


def classify_support(start: float, end: float, supports: list[dict]) -> dict:
    """Apply the secondary two-region rule without changing event boundaries."""
    clipped = []
    for row in supports:
        lo = max(float(start), float(row["start_ms"]))
        hi = min(float(end), float(row["end_ms"]))
        if lo < hi:
            clipped.append((lo, hi, str(row["region"])))
    regions = {region for _, _, region in clipped}
    concurrent = set()
    for index, (lo, hi, region) in enumerate(clipped):
        for other_lo, other_hi, other_region in clipped[:index]:
            if region != other_region and max(lo, other_lo) < min(hi, other_hi):
                concurrent.update((region, other_region))
    kind = "multi" if concurrent else "single" if len(regions) == 1 else "unsupported"
    return {"predicted_class": kind, "regions": regions, "concurrent_regions": concurrent}


def verify_run(run_root: Path, reference_path: Path,
               reference_task: str = "general_events", tolerance_ms: float = 1e-7) -> dict:
    reference = pd.read_csv(reference_path, float_precision="round_trip", keep_default_na=False)
    if "task" in reference:
        reference = reference[reference.task.eq(reference_task)]
    reference = reference[reference.method.isin(METHODS)]
    jobs = sorted(run_root.glob("trial*/recording_settings.json"))
    if not jobs:
        raise ValueError("No completed comparator recording_settings.json files found")
    totals = defaultdict(int)
    checked_classes = 0
    versions = set()
    trials = []
    largest_difference = 0.0
    for metadata_path in jobs:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trial = int(metadata["trial"])
        trials.append(trial)
        assert metadata["quality"]["snr_lo"] == 1.2
        assert metadata["quality"]["snr_hi"] == 25
        assert metadata["quality"]["fr_min"] > 0.8
        assert metadata["quality"]["min_spikes"] == 2
        assert metadata["logIBEI_peak_search_ms"] == 100
        assert metadata["NetworkCMA_k"] == 0.75
        assert metadata["meaRtools_Sigma"] == 45
        assert metadata["output_merge_added"] is False
        assert metadata["output_duration_filter_added"] is False
        assert metadata["ground_truth_values_read"] is False
        versions.add(metadata["MATLAB_version"])
        folder = metadata_path.parent
        native = pd.read_csv(folder / "native_population_events.csv",
                             float_precision="round_trip", keep_default_na=False)
        support = pd.read_csv(folder / "support_events.csv",
                              float_precision="round_trip", keep_default_na=False)
        by_event = defaultdict(list)
        for row in support.to_dict("records"):
            by_event[str(row["method"]), int(row["event_id"])].append(row)
        for method in METHODS:
            actual = native[native.method.eq(method)].sort_values(["start_ms", "end_ms"])
            expected = reference[reference.method.eq(method) & reference.trial.eq(trial)]
            expected = expected.sort_values(["start_ms", "end_ms"])
            if len(actual) != len(expected):
                raise AssertionError(f"{method}, trial {trial}: count {len(actual)} != {len(expected)}")
            for row, target in zip(actual.to_dict("records"), expected.to_dict("records")):
                for field in ("start_ms", "end_ms"):
                    difference = abs(float(row[field]) - float(target[field]))
                    largest_difference = max(largest_difference, difference)
                    if not math.isclose(float(row[field]), float(target[field]),
                                        rel_tol=0, abs_tol=tolerance_ms):
                        raise AssertionError(f"{method}, trial {trial}: {field} differs")
                expected_class = target.get("predicted_class", target.get("kind"))
                if expected_class in {"single", "multi", "unsupported"}:
                    evidence = classify_support(row["start_ms"], row["end_ms"],
                                                by_event[method, int(row["event_id"])])
                    if evidence["predicted_class"] != expected_class:
                        raise AssertionError(
                            f"{method}, trial {trial}, event {row['event_id']}: anatomical class differs")
                    checked_classes += 1
            totals[method] += len(actual)
    assert len(trials) == len(set(trials))
    return {
        "verified": True, "recordings": len(trials), "recording_ids": trials,
        "native_population_counts": dict(totals),
        "anatomical_classes_checked": checked_classes,
        "maximum_boundary_difference_ms": largest_difference,
        "tolerance_ms": tolerance_ms, "MATLAB_versions": sorted(versions),
        "reference_file": reference_path.name,
        "comparison": "fresh detector population boundaries and support-derived secondary classes",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("--reference-task", default="general_events")
    parser.add_argument("--tolerance-ms", type=float, default=1e-7)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_root, args.reference_csv,
                                args.reference_task, args.tolerance_ms), indent=2))


if __name__ == "__main__":
    main()

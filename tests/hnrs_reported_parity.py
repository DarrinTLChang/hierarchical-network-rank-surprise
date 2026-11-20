"""Compare fresh HNRS catalogs/scores with references bundled in this release.

The required run root is produced by tests/hnrs_native_parity.py and contains
trialNN/seedMASTER/ folders. It is not the differently nested reproduce.py output
layout. No private archive or workstation path is required. Partial run roots
are supported, and the report states how many runs/scores were checked.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    from score_benchmarks import score_catalog
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="required fresh trialNN/seedMASTER root from hnrs_native_parity.py")
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error("run root must be an existing directory")
    reference_root = ROOT / "datasets/network_burst/reference_catalogs"
    references = {name: pd.read_csv(reference_root / f"{name}.csv", float_precision="round_trip").fillna("")
                  for name in ("general_events", "exclusive_events")}
    expected_scores = pd.read_csv(ROOT / "results/network_burst/raw_recording_scores.csv", float_precision="round_trip")
    expected_scores = expected_scores.loc[expected_scores.method.eq("HNRS")].set_index(["trial", "master_seed", "task"])
    runs = 0
    score_rows = 0
    interval_max_error = 0.0
    score_max_error = 0.0
    compared = []
    for folder in sorted(args.runs.glob("trial*/seed*")):
        if not folder.is_dir():
            continue
        trial, seed = int(folder.parent.name[5:]), int(folder.name[4:])
        truth = ROOT / f"datasets/network_burst/trials/trial{trial:02d}/event_manifest.csv"
        actual = {}
        for name, reference in references.items():
            got = pd.read_csv(folder / f"{name}.csv", float_precision="round_trip").fillna("")
            expected = reference.loc[reference.trial.eq(trial) & reference.master_seed.eq(seed)].copy()
            fields = ["prediction_id", "predicted_class", "regions", "source_stage", "native_event_ids", "unit_burst_ids"]
            pd.testing.assert_frame_equal(got[fields].reset_index(drop=True), expected[fields].reset_index(drop=True), check_exact=True)
            for name_column in ("start_ms", "end_ms"):
                errors = np.abs(got[name_column].to_numpy() - expected[name_column].to_numpy())
                interval_max_error = max(interval_max_error, float(errors.max(initial=0)))
                np.testing.assert_allclose(got[name_column], expected[name_column], rtol=0, atol=1e-9)
            actual[name] = got
        for task, calls in (("general_events", actual["general_events"]),
                            ("region_aware", actual["exclusive_events"]),
                            ("network_targets_native", actual["exclusive_events"].loc[actual["exclusive_events"].predicted_class.eq("multi")])):
            scores, _ = score_catalog(calls, truth, task)
            for score in scores:
                expected = expected_scores.loc[(trial, seed, score["task"])]
                for name_column in ("tp", "fp", "fn", "jaccard", "f1", "precision", "recall", "false_per_min",
                                    "off_event_fp", "wrong_class_fp", "wrong_class_fn", "unmatched_in_event_fp", "split_targets", "merge_predictions"):
                    value = score[name_column]
                    if value is None:
                        assert pd.isna(expected[name_column])
                        continue
                    delta = abs(value - expected[name_column])
                    score_max_error = max(score_max_error, float(delta))
                    np.testing.assert_allclose(value, expected[name_column], rtol=0, atol=1e-15)
                compared.append(dict(trial=trial, master_seed=seed, task=score["task"]))
                score_rows += 1
        runs += 1
    if not runs:
        parser.error("no trialNN/seedMASTER run folders found; nothing was verified")
    proof = dict(verified=True, runs=runs, compared_score_rows=score_rows,
                 catalog_id_class_region_and_membership_exact=True,
                 maximum_reported_interval_error_ms=interval_max_error,
                 maximum_score_error=score_max_error, compared=compared)
    (args.runs / "reported_catalog_verification.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in proof.items() if key != "compared"}, indent=2))


if __name__ == "__main__":
    main()

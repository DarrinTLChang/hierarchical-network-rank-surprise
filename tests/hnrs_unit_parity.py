"""Rerun both unit-only arms against bundled50-recording interval/score references.

All input/reference paths are relative to this release. Only a new output root
is required; no original workstation, external archive, or generator is used.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
import json
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_hnrs import run_recording
from score_benchmarks import burst_pause


def job(arguments):
    row, shuffle_off, output_root = arguments
    import numpy as np
    import pandas as pd
    arm = "rs" if shuffle_off else "on"
    trial = int(row["recording"])
    dest = Path(output_root) / f"recording{trial:02d}" / arm
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (dest.parent / f"{arm}.log").open("w", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
        metadata = run_recording(ROOT / row["input_path"], dest, mode="burst-pause", trial=trial, shuffle_off=shuffle_off)
    relative = Path(row["input_path"]).relative_to("datasets/burst_pause").parent
    job_id = "bp50/" + relative.as_posix()
    expected = pd.read_csv(ROOT / "results/burst_pause/interval_catalog.csv", float_precision="round_trip")
    expected = expected.loc[expected.job_id.eq(job_id) & expected.arm.eq(arm)]
    actual = pd.read_csv(dest / "reported_unit.csv", float_precision="round_trip")
    columns = ["Electrode", "Cluster", "burst_start_ms", "burst_end_ms"]
    # Original MATLAB RS CSVs were written to fewer significant digits than
    # Python's float round-trip. Match identities/counts exactly and endpoints
    # within the original cross-implementation audit tolerance (1e-9ms).
    pd.testing.assert_frame_equal(actual[columns[:2]].reset_index(drop=True), expected[columns[:2]].reset_index(drop=True), check_exact=True)
    errors = np.abs(actual[columns[2:]].to_numpy() - expected[columns[2:]].to_numpy())
    np.testing.assert_allclose(actual[columns[2:]], expected[columns[2:]], rtol=0, atol=1e-9)
    score = burst_pause(dest / "reported_unit.csv", ROOT / row["truth_path"])
    reference_scores = pd.read_csv(ROOT / "results/burst_pause/singleunit50_recording_metrics.csv", float_precision="round_trip")
    ref = reference_scores.loc[reference_scores.job_id.eq(job_id) & reference_scores.arm.eq(arm)].iloc[0]
    for name in ("tp", "fp", "fn", "precision", "recall", "f1", "jaccard"):
        expected_value = ref["event_jaccard" if name == "jaccard" else name]
        np.testing.assert_allclose(score[name], expected_value, rtol=0, atol=1e-15)
    diagnostics = pd.read_csv(dest / f"diagnostics_{'off' if shuffle_off else 'on'}.csv")
    if not shuffle_off:
        assert int(diagnostics.iloc[0]["seed"]) == 17010698844467267777
        assert int(diagnostics.iloc[0]["remainder_spike_count"]) == 0
    return dict(trial=trial, arm=arm, job_id=job_id, interval_reference_atol_ms=1e-9,
                max_interval_error_ms=float(errors.max(initial=0)),
                metrics_reference_exact=True, input_sha256=metadata["input_sha256"], **score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new audit output directory")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.output.exists():
        parser.error("audit output must be a new directory")
    with (ROOT / "datasets/recording_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = [row for row in csv.DictReader(stream) if row["cohort"] == "burst_pause"]
    args.output.mkdir(parents=True)
    reports = []
    jobs = [(row, off, str(args.output)) for row in records for off in (False, True)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(job, item) for item in jobs]):
            report = future.result()
            reports.append(report)
            print(f"{len(reports)}/{len(jobs)} recording{report['trial']} {report['arm']} exact", flush=True)
    import pandas as pd
    frame = pd.DataFrame(reports)
    frame.to_csv(args.output / "recording_scores.csv", index=False)
    summary = frame.groupby("arm")[["f1", "jaccard", "precision", "recall"]].agg(["mean", "std"])
    summary.to_csv(args.output / "summary.csv")
    (args.output / "verification.json").write_text(json.dumps(dict(verified=True, n_recordings=len(records),
        native_runs=len(reports), interval_reference_atol_ms=1e-9, original_rs_equivalent_unit_control=True,
        exact_current_reference_metrics=True, runs=sorted(reports, key=lambda row: (row["trial"], row["arm"]))),
        indent=2) + "\n", encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()

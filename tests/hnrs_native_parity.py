"""Fresh native pipeline parity audit (optional full synthetic regression).

Uses only the released recording manifest and fixed MAT inputs by default.
No saved detections are used as detector inputs. Optional exact CSV-value comparison
requires an explicitly supplied, independently available native-output root;
that root is not bundled and has no workstation-specific default. Its layout is
trialNN/seedMASTER/u01_r03_n15/{unit,region,network}_LR.csv. Without that optional
comparison, this still reruns inputs and scores generated, consolidated catalogs.
The output layout can be checked against bundled reported references using
tests/hnrs_reported_parity.py. This script never generates synthetic inputs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_hnrs import run_recording
from score_benchmarks import score_catalog, summarize_scores


def job(arguments):
    row, seed, output_root, reference_root = arguments
    import pandas as pd
    trial = int(row["recording"])
    dest = Path(output_root) / f"trial{trial:02d}" / f"seed{seed}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (dest.parent / f"seed{seed}.log").open("w", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
        metadata = run_recording(ROOT / row["input_path"], dest, trial=trial,
                                  recording_id=int(row["recording_id"]), master_seed=seed)
    if reference_root:
        reference = Path(reference_root) / f"trial{trial:02d}" / f"seed{seed}" / "u01_r03_n15"
        for name in ("unit_LR.csv", "region_LR.csv", "network_LR.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(dest / name, float_precision="round_trip"),
                                          pd.read_csv(reference / name, float_precision="round_trip"), check_exact=True)
    general, exclusive = pd.read_csv(dest / "general_events.csv"), pd.read_csv(dest / "exclusive_events.csv")
    scores = []
    for task, calls in (("general_events", general), ("region_aware", exclusive),
                        ("network_targets_native", exclusive.loc[exclusive.predicted_class.eq("multi")])):
        metrics, _ = score_catalog(calls, ROOT / row["truth_path"], task)
        scores.extend(dict(trial=trial, method="HNRS", master_seed=seed, **metric) for metric in metrics)
    return dict(trial=trial, master_seed=seed, scores=scores,
                exact_native_reference_parity=bool(reference_root), elapsed_s=time.perf_counter() - started,
                input_sha256=metadata["input_sha256"],
                output_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in dest.iterdir() if path.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/recording_manifest.csv")
    parser.add_argument("--output", type=Path, required=True, help="new audit output directory")
    parser.add_argument("--native-reference-root", type=Path,
                        help="optional user-supplied native golden outputs (not bundled; no default)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.output.exists():
        parser.error("audit output must be a new directory")
    if args.native_reference_root is not None and not args.native_reference_root.is_dir():
        parser.error("native reference root must be an existing directory")
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        records = [row for row in csv.DictReader(stream) if row["cohort"] == "network_burst"]
    records.sort(key=lambda row: int(row["recording"]))
    if args.limit:
        records = records[:args.limit]
    if not records:
        parser.error("manifest contains no network_burst recordings")
    args.output.mkdir(parents=True)
    jobs = [(row, seed, str(args.output), str(args.native_reference_root) if args.native_reference_root else None)
            for row in records for seed in (123, 456, 789)]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(job, item) for item in jobs]):
            report = future.result()
            reports.append(report)
            print(f"{len(reports)}/{len(jobs)} recording{report['trial']} seed{report['master_seed']} verified", flush=True)
    raw = [score for report in reports for score in report["scores"]]
    recording, summary = summarize_scores(raw)
    import pandas as pd
    pd.DataFrame(raw).to_csv(args.output / "raw_scores.csv", index=False)
    pd.DataFrame(recording).to_csv(args.output / "recording_scores.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output / "summary.csv", index=False)
    proof = dict(verified=True, n_recordings=len(records), native_runs=len(reports),
                 masters=[123, 456, 789], profile_cutoffs=[.01, .03, .15],
                 native_reference_parity=bool(args.native_reference_root), summary=summary,
                 runs=sorted(reports, key=lambda row: (row["trial"], row["master_seed"])))
    (args.output / "verification.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()

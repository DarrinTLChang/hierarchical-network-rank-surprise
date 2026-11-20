"""Run and score the released fixed synthetic inputs; never generate data."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "datasets" / "recording_manifest.csv"
METHODS = ("HNRS", "logIBEI", "Network-CMA", "meaRtools/Otsu")


def read_manifest():
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def input_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Invalid released input path: {name}")
    return path


def numbers(value):
    result = set()
    for item in value.split(","):
        if "-" in item:
            start, end = map(int, item.split("-", 1))
            if end < start:
                raise argparse.ArgumentTypeError("ranges must be increasing")
            result.update(range(start, end + 1))
        else:
            result.add(int(item))
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError("use positive recording numbers")
    return result


def new_directory(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Choose a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_rows(path, rows):
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")


def run_job(job):
    from run_hnrs import run_recording
    row, seed, out, shuffle_off = job
    path = input_path(row["input_path"])
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != row["input_sha256"]:
        raise ValueError(f"Released input checksum mismatch: {row['input_path']}")
    mode = ("burst-pause" if row["cohort"] in {"burst_pause", "synthetic_example"}
            else "network-null" if row["cohort"] == "network_null" else "network")
    scope = Path(row["cohort"]) / row["condition"] / f"trial{int(row['recording']):03d}"
    if row["shift_realization"]:
        scope /= "realization" + row["shift_realization"]
    scope /= f"seed{seed}"
    run_recording(path, Path(out) / scope, mode=mode, master_seed=seed,
                  recording_id=int(row["recording_id"]), dataset_tag=row["dataset_tag"],
                  seed=seed if mode != "network" else None,
                  trial=int(row["recording"]), shuffle_off=shuffle_off)
    return {**row, "master_seed": seed, "mode": mode, "shuffle_off": int(shuffle_off),
            "output_subdir": scope.as_posix()}


def run_hnrs(args):
    records = [row for row in read_manifest() if row["cohort"] == args.cohort]
    if args.trials:
        records = [row for row in records if int(row["recording"]) in args.trials]
    if args.condition:
        records = [row for row in records if row["condition"] == args.condition]
    if not records:
        raise ValueError("No manifest recordings match this selection")
    if args.shuffle_off and args.cohort not in {"burst_pause", "synthetic_example"}:
        raise ValueError("The shuffle-off comparison is unit-only")
    defaults = ([123] if args.cohort in {"burst_pause", "synthetic_example"} else
                [123, 456, 789, *range(1001, 1008)] if args.cohort == "network_null"
                else [123, 456, 789])
    seeds = sorted(args.seeds) if args.seeds else defaults
    out = new_directory(args.out)
    jobs = [(row, seed, str(out), args.shuffle_off) for row in records for seed in seeds]
    completed = []
    if args.jobs == 1:
        iterator = map(run_job, jobs)
        for result in iterator:
            completed.append(result)
            print(f"Completed {len(completed)}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                completed.append(future.result())
                print(f"Completed {len(completed)}/{len(jobs)}", flush=True)
    completed.sort(key=lambda row: (int(row["recording"]), row["condition"],
                                    row["shift_realization"], int(row["master_seed"])))
    write_rows(out / "run_manifest.csv", completed)
    print("Saved run_manifest.csv. Use score-runs to evaluate these fresh outputs.")


def finish_scores(out, scores, audit):
    from score_benchmarks import summarize_scores
    write_rows(out / "scores.csv", scores)
    if audit:
        write_rows(out / "prediction_audit.csv", audit)
    if scores:
        recording, summary = summarize_scores(scores)
        write_rows(out / "recording_seedmean_scores.csv", recording)
        write_rows(out / "summary.csv", summary)
        for row in summary:
            value = row["mean_jaccard"]
            print(f"{row['method']}: {row['task']}: mean Jaccard={value:.6f} "
                  f"({row['n_recordings']} recordings)")


def score_runs(args):
    import pandas as pd
    from score_benchmarks import burst_pause, score_catalog
    root = args.runs.resolve()
    with (root / "run_manifest.csv").open(encoding="utf-8", newline="") as handle:
        runs = list(csv.DictReader(handle))
    out = new_directory(args.out)
    scores, audits, counts = [], [], []
    for row in runs:
        directory = (root / row["output_subdir"]).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Invalid output path in run manifest")
        key = {"method": "RS" if int(row["shuffle_off"]) else "HNRS",
               "trial": int(row["recording"]), "master_seed": int(row["master_seed"]),
               "recording_id": int(row["recording_id"]), "cohort": row["cohort"]}
        if row["mode"] == "burst-pause":
            result = burst_pause(directory / "reported_unit.csv", input_path(row["truth_path"]))
            scores.append({**key, "task": "burst_pause", **result,
                           "false_per_min": result["fp"] * 60 / float(row["duration_s"])})
        elif row["cohort"] in {"network_null", "network_shift"}:
            calls = pd.read_csv(directory / "reported_network.csv", float_precision="round_trip")
            start, end = ((5000., 245000.) if row["cohort"] == "network_shift"
                          else (0., float(row["duration_s"]) * 1000))
            count = int(((calls.start_ms < end) & (calls.end_ms > start)).sum()) if len(calls) else 0
            counts.append({**key, "condition": row["condition"],
                           "shift_realization": row["shift_realization"],
                           "network_calls": count, "exposure_s": (end - start) / 1000})
        else:
            truth = pd.read_csv(input_path(row["truth_path"]), float_precision="round_trip")
            for filename, task in (("general_events.csv", "general_events"),
                                    ("reported_network.csv", "network_targets_native"),
                                    ("exclusive_events.csv", "region_aware")):
                calls = pd.read_csv(directory / filename, float_precision="round_trip")
                rows, audit = score_catalog(calls, truth, task, float(row["duration_s"]))
                scores.extend({**key, **entry} for entry in rows)
                audits.extend({**key, "analysis": task, **entry} for entry in audit)
    finish_scores(out, scores, audits)
    if counts:
        write_rows(out / "network_counts.csv", counts)
        print("Saved network_counts.csv; these controls are not ground-truth accuracy scores.")


def score_reference(args):
    """Recompute scores from released catalogs, not from regenerated detections."""
    import pandas as pd
    from score_benchmarks import score_catalog
    base = ROOT / "results" / "network_burst"
    read = lambda path: pd.read_csv(path, float_precision="round_trip")
    general = read(base / "general_events.csv")
    hnrs_network = read(base / "hnrs_network_events.csv")
    population = read(base / "native_population_events.csv")
    regional = read(base / "region_aware_events.csv")
    scores, audits = [], []
    records = [row for row in read_manifest() if row["cohort"] == "network_burst"]
    for record in records:
        trial = int(record["recording"])
        truth = read(input_path(record["truth_path"]))
        for method in METHODS:
            seeds = [123, 456, 789] if method == "HNRS" else [0]
            for seed in seeds:
                key = {"method": method, "trial": trial, "master_seed": seed}
                def select(frame):
                    selected = frame[(frame.method == method) & (frame.trial == trial)]
                    if method == "HNRS":
                        selected = selected[selected.master_seed == seed]
                    return selected
                catalogs = ((select(general if method == "HNRS" else population), "general_events"),
                            (select(hnrs_network if method == "HNRS" else population), "network_targets_native"),
                            (select(regional), "region_aware"))
                for calls, task in catalogs:
                    rows, audit = score_catalog(calls, truth, task, 230. if method == "meaRtools/Otsu" else 250.)
                    scores.extend({**key, **entry} for entry in rows)
                    audits.extend({**key, "analysis": task, **entry} for entry in audit)
    out = new_directory(args.out)
    finish_scores(out, scores, audits)
    print("This command rescored archived catalogs; it did not rerun any detector.")


def score_comparators(args):
    import pandas as pd
    from score_benchmarks import assign_region_labels, score_catalog
    records = [row for row in read_manifest() if row["cohort"] == args.cohort]
    if args.trials:
        records = [row for row in records if int(row["recording"]) in args.trials]
    if not records:
        raise ValueError("No matching comparator recordings")
    scores, audits = [], []
    for record in records:
        trial = int(record["recording"])
        directory = args.runs / f"trial{trial:02d}"
        native = pd.read_csv(directory / "native_population_events.csv", float_precision="round_trip")
        support = pd.read_csv(directory / "support_events.csv", float_precision="round_trip")
        labeled = pd.DataFrame(assign_region_labels(native, support))
        truth = pd.read_csv(input_path(record["truth_path"]), float_precision="round_trip")
        for method in METHODS[1:]:
            calls = native[native.method == method]
            secondary = labeled[labeled.method == method] if len(labeled) else []
            key = {"method": method, "trial": trial, "master_seed": 0}
            for predictions, task in ((calls, "general_events"),
                                      (calls, "network_targets_native"),
                                      (secondary, "region_aware")):
                rows, audit = score_catalog(predictions, truth, task,
                                             230. if method == "meaRtools/Otsu" else 250.)
                scores.extend({**key, **entry} for entry in rows)
                audits.extend({**key, "analysis": task, **entry} for entry in audit)
    finish_scores(new_directory(args.out), scores, audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-hnrs", help="run HNRS on released fixed inputs")
    run.add_argument("--cohort", required=True, choices=("burst_pause", "network_burst",
                     "network_development", "network_null", "network_shift", "synthetic_example"))
    run.add_argument("--trials", type=numbers, help="e.g. 1 or 1-5,10; otherwise all")
    run.add_argument("--condition", help="optional manifest condition filter")
    run.add_argument("--seeds", type=numbers, help="comma-separated master/literal seeds")
    run.add_argument("--shuffle-off", action="store_true", help="unit-only original-RS control")
    run.add_argument("--jobs", type=int, choices=range(1, 33), default=1)
    run.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    run.set_defaults(action=run_hnrs)
    score = commands.add_parser("score-runs", help="score fresh outputs with their run manifest")
    score.add_argument("--runs", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.set_defaults(action=score_runs)
    reference = commands.add_parser("score-reference", help="rescore the archived current catalogs")
    reference.add_argument("--out", type=Path, required=True)
    reference.set_defaults(action=score_reference)
    comparators = commands.add_parser("score-comparators", help="score fresh MATLAB adapter outputs")
    comparators.add_argument("--runs", type=Path, required=True)
    comparators.add_argument("--cohort", choices=("network_burst", "network_development"),
                             default="network_burst")
    comparators.add_argument("--trials", type=numbers, help="match the trials actually run")
    comparators.add_argument("--out", type=Path, required=True)
    comparators.set_defaults(action=score_comparators)
    args = parser.parse_args()
    try:
        args.action(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()

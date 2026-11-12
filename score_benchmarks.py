"""Score HNRS burst detections against the released synthetic truth.

Burst-Pause accepts one recording's unit detections. Network Burst scores
the released HNRS event catalogs; it does not generate those catalogs, run
any detector, or reproduce the other comparator implementations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from math import isfinite
from pathlib import Path
import re
from statistics import mean


def csv_rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        yield from reader


def span(row, start, end, *, point=False):
    a, b = float(row[start]), float(row[end])
    if not (isfinite(a) and isfinite(b) and 0 <= a <= b):
        raise ValueError(f"invalid interval: {a:g} to {b:g} ms")
    if a == b:
        if not point:
            raise ValueError(f"zero-duration interval: {a:g} ms")
        # Preserve the original Burst-Pause matcher's point-interval convention.
        b += 1e-6
    return a, b


def metrics(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
    }


def ordered_matches(truth, detections):
    """Maximize ordered match count, then total overlap (not IoU)."""
    truth = sorted(truth, key=lambda interval: interval[0])
    detections = sorted(detections)
    # Each cell stores count, summed overlap in ms, and the final action
    # (1 = skip truth, 2 = skip detection, 3 = match, in priority order).
    previous = [(0, 0.0, 0)] + [(0, 0.0, 2)] * len(detections)
    for start, end in truth:
        current = [(0, 0.0, 1)]
        for j, (ds, de) in enumerate(detections, 1):
            skip_truth = (*previous[j][:2], 1)
            skip_detection = (*current[j - 1][:2], 2)
            best = max(skip_truth, skip_detection)
            overlap = min(end, de) - max(start, ds)
            if overlap > 0:
                count, total, _ = previous[j - 1]
                best = max(best, (count + 1, total + overlap, 3))
            current.append(best)
        previous = current
    return previous[-1][:2]


def unit_intervals(path, start, end):
    by_unit = defaultdict(list)
    for row in csv_rows(path, ("Electrode", "Cluster", start, end)):
        cluster = float(row["Cluster"])
        if not row["Electrode"] or not cluster.is_integer():
            raise ValueError(f"{path}: invalid electrode/cluster identity")
        by_unit[row["Electrode"], int(cluster)].append(span(row, start, end, point=True))
    return by_unit


def burst_pause(detections_path, truth_path):
    truth = unit_intervals(truth_path, "start_ms", "end_ms")
    detections = unit_intervals(detections_path, "burst_start_ms", "burst_end_ms")
    if not truth:
        raise ValueError(f"{truth_path}: no planted bursts")
    n_truth, n_detected = (sum(map(len, obj.values())) for obj in (truth, detections))
    tp = sum(ordered_matches(truth.get(key, ()), detections.get(key, ()))[0]
             for key in truth.keys() | detections.keys())
    return metrics(tp, n_detected - tp, n_truth - tp)


def iou_matches(detections, truth):
    """Maximize positive-overlap matches, then total temporal IoU."""
    if not detections or not truth:
        return []
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    p, t = np.asarray(detections, dtype=float), np.asarray(truth, dtype=float)
    overlap = np.maximum(0, np.minimum(p[:, 1, None], t[None, :, 1])
                         - np.maximum(p[:, 0, None], t[None, :, 0]))
    union = ((p[:, 1] - p[:, 0])[:, None] + (t[:, 1] - t[:, 0])[None, :] - overlap)
    valid = overlap > 0
    n, m = overlap.shape
    base = min(n, m) + 1.0  # Cardinality takes precedence over total IoU.
    reward = np.zeros((n, m + n))  # Dummy columns allow unmatched detections.
    reward[:, :m] = np.where(valid, base + overlap / union, -base)
    rows, cols = linear_sum_assignment(reward, maximize=True)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if j < m and valid[i, j]]


def read_catalog(path, *, labeled):
    required = ["trial", "start_ms", "end_ms"]
    if labeled:
        required.append("predicted_class")
    by_trial = defaultdict(list)
    for row in csv_rows(path, required):
        event = span(row, "start_ms", "end_ms")
        label = row["predicted_class"] if labeled else "unlabeled"
        if labeled and label not in {"single", "multi"}:
            raise ValueError(f"{path}: unexpected HNRS classification {label!r}")
        by_trial[int(row["trial"])].append((event, label))
    return by_trial


def network_rows(trials_dir, general_path, exclusive_path):
    general = read_catalog(general_path, labeled=False)
    exclusive = read_catalog(exclusive_path, labeled=True)
    truth_paths = {}
    for path in trials_dir.glob("trial*/event_manifest.csv"):
        suffix = re.fullmatch(r"trial(\d+)", path.parent.name)
        if suffix:
            trial = int(suffix[1])
            if trial in truth_paths:
                raise ValueError(f"{trials_dir}: duplicate recording number {trial}")
            truth_paths[trial] = path
    if not truth_paths:
        raise ValueError(f"{trials_dir}: no trialNN/event_manifest.csv files")
    if unknown := (general.keys() | exclusive.keys()) - truth_paths.keys():
        raise ValueError(f"catalog trials without truth: {', '.join(map(str, sorted(unknown)))}")

    for trial, path in sorted(truth_paths.items()):
        truth = [
            (span(row, "actual_start_ms", "actual_end_ms"), row["event_type"])
            for row in csv_rows(path, ("actual_start_ms", "actual_end_ms", "event_type"))
        ]
        if sorted(label for _, label in truth) != ["multi"] * 6 + ["single"] * 12:
            raise ValueError(f"{path}: expected 6 network and 12 regional-only planted events")
        intervals = [interval for interval, _ in truth]
        unclassified = [event for event, _ in general.get(trial, ())]
        tp = len(iou_matches(unclassified, intervals))
        yield {"trial": trial, "task": "general_events",
               **metrics(tp, len(unclassified) - tp, len(truth) - tp)}

        # Match network predictions first, then regional predictions to the
        # remaining truth. Timing eligibility is independent of true class.
        available = list(range(len(truth)))
        for label, task in (("multi", "network_classification_on"),
                            ("single", "regional_classification_on")):
            calls = [event for event, kind in exclusive.get(trial, ()) if kind == label]
            pairs = iou_matches(calls, [intervals[j] for j in available])
            claimed = {available[j] for _, j in pairs}
            tp = sum(truth[j][1] == label for j in claimed)
            targets = sum(kind == label for _, kind in truth)
            yield {"trial": trial, "task": task,
                   **metrics(tp, len(calls) - tp, targets - tp)}
            available = [j for j in available if j not in claimed]


def network_summary(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    summary = {}
    for task, group in by_task.items():
        totals = {key: sum(row[key] for row in group) for key in ("tp", "fp", "fn")}
        summary[task] = {
            "n_recordings": len(group),
            **totals,
            "pooled_precision": metrics(**totals)["precision"],
            "mean_recall": mean(row["recall"] for row in group),
            "mean_f1": mean(row["f1"] for row in group),
            "mean_jaccard": mean(row["jaccard"] for row in group),
            "mean_false_per_min": mean(row["fp"] for row in group) * 60 / 250,
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="benchmark", required=True)
    bp = commands.add_parser("burst-pause", help="score one native HNRS unit export")
    bp.add_argument("detections", type=Path, help="unit_LR.csv from a single recording")
    bp.add_argument("truth", type=Path, help="the same recording's unit_ground_truth.csv")
    net = commands.add_parser("network-burst", help="score archived HNRS network benchmark catalogs")
    net.add_argument("dataset", type=Path, help="Network Burst dataset directory")
    net.add_argument("--trials-dir", type=Path, help="override the dataset's trials directory")
    net.add_argument("--general-events", type=Path, help="override reference_catalogs/general_events.csv")
    net.add_argument("--exclusive-events", type=Path, help="override reference_catalogs/exclusive_events.csv")
    net.add_argument("--csv", type=Path, help="write a new per-recording scores CSV")
    args = parser.parse_args(argv)
    try:
        if args.benchmark == "burst-pause":
            summary = burst_pause(args.detections, args.truth)
        else:
            refs = args.dataset / "reference_catalogs"
            rows = list(network_rows(
                args.trials_dir or args.dataset / "trials",
                args.general_events or refs / "general_events.csv",
                args.exclusive_events or refs / "exclusive_events.csv",
            ))
            summary = network_summary(rows)
            if args.csv:
                with args.csv.open("x", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

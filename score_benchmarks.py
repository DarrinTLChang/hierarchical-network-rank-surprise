"""Portable scoring for current synthetic burst recovery and classification.

Matching maximizes positive-overlap cardinality, then summed temporal IoU.
Unit-only Burst-Pause instead uses the original ordered, total-overlap DP.
Detectors and truth-blind output operations are separate from this module.
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
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
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


def _records(value):
    """Accept paths, data frames, or iterables of row mappings."""
    if isinstance(value, (str, Path)):
        return list(csv_rows(Path(value), ()))
    if hasattr(value, "to_dict"):
        return value.to_dict("records")
    return [dict(row) for row in value]


def _prediction_rows(value):
    rows = _records(value)
    normalized = []
    aliases = {"regional": "single", "network": "multi", "population": "unlabeled"}
    for i, row in enumerate(rows):
        a, b = span(row, "start_ms", "end_ms")
        label = str(row.get("predicted_class", row.get("kind", "unlabeled")))
        label = aliases.get(label, label)
        normalized.append({**row, "start_ms": a, "end_ms": b,
                           "prediction_id": str(row.get("prediction_id", row.get("event_id", i))),
                           "predicted_class": label, "regions": str(row.get("regions", row.get("region", "")))})
    return normalized


def _truth_rows(value):
    rows = _records(value)
    result = []
    for i, row in enumerate(rows):
        a, b = span(row, "actual_start_ms", "actual_end_ms")
        label = row["event_type"]
        if label not in {"single", "multi"}:
            raise ValueError(f"unexpected planted class: {label!r}")
        result.append({**row, "actual_start_ms": a, "actual_end_ms": b,
                       "event_type": label, "event_id": int(row.get("event_id", i)),
                       "regions": str(row.get("regions", ""))})
    return result


def assign_region_labels(population, support):
    """Secondary region-aware labels from exact native support membership.

    Each support row has event_id (or prediction_id), region, start_ms,end_ms.
    Boundaries are clipped to the population event for concurrency only. No
    native population boundary, onset-count rule, duration gate, or merge is
    changed. Support must already reflect the method-specific native members
    (especially logIBEI); overlapping arbitrary unit bursts are not substitutes.
    Pass one recording/method at a time, or include method and trial keys.
    """
    calls = _prediction_rows(population)
    by_event = defaultdict(list)
    def key(row, identifier):
        def stable(value):
            try:
                number = float(value)
                return str(int(number)) if number.is_integer() else str(value)
            except (ValueError, TypeError, OverflowError):
                return str(value)
        return (str(row.get("method", "")), stable(row.get("trial", "")), stable(identifier))
    for row in _records(support):
        identifier = row.get("event_id", row.get("prediction_id"))
        by_event[key(row, identifier)].append(row)
    labeled = []
    for call in calls:
        pieces = []
        event_support = by_event.get(key(call, call.get("event_id", call["prediction_id"])))
        if event_support is None:
            event_support = by_event.get(key(call, call["prediction_id"]), ())
        for row in event_support:
            a, b = max(call["start_ms"], float(row["start_ms"])), min(call["end_ms"], float(row["end_ms"]))
            if b > a:
                pieces.append((a, b, str(row.get("region", row.get("regions", "")))))
        regions = sorted({piece[2] for piece in pieces if piece[2]})
        concurrent_regions = {region for i, a in enumerate(pieces) for b in pieces[i + 1:]
                              if a[2] != b[2] and min(a[1], b[1]) > max(a[0], b[0])
                              for region in (a[2], b[2])}
        concurrent = bool(concurrent_regions)
        label = "multi" if concurrent else "single" if len(regions) == 1 else "unsupported"
        # meaRtools has no native unit membership: its network region list
        # records the concurrently supported subset from within-region runs.
        if concurrent and str(call.get("method", "")).startswith("meaRtools"):
            regions = sorted(concurrent_regions)
        labeled.append({**call, "predicted_class": label, "regions": ";".join(regions)})
    return labeled


def score_catalog(predictions, truth, task="general_events", exposure_s=250.0):
    """Score one method/recording/seed; return (metric rows, prediction audit).

    ``general_events`` ignores region/class labels. ``network_targets_native``
    treats every supplied native population output as a network-target call;
    matches to regional targets are false positives relative to that target.
    ``region_aware`` uses network-first matching, then regional calls against
    still-unclaimed targets; planted class does not restrict timing eligibility.
    Unsupported secondary labels abstain. No output post-processing is applied.
    """
    if exposure_s <= 0:
        raise ValueError("exposure_s must be positive")
    if task not in {"general_events", "network_targets_native", "region_aware",
                    "network_classification_on", "regional_classification_on"}:
        raise ValueError(f"unknown task: {task}")
    calls, targets = _prediction_rows(predictions), _truth_rows(truth)
    p = [(row["start_ms"], row["end_ms"]) for row in calls]
    t = [(row["actual_start_ms"], row["actual_end_ms"]) for row in targets]
    overlap = [[max(0.0, min(b, d) - max(a, c)) for c, d in t] for a, b in p]
    if task == "network_targets_native":
        calls = [{**row, "predicted_class": "multi"} for row in calls]
    labels = [row["predicted_class"] for row in calls]
    kinds = [row["event_type"] for row in targets]
    matches = {}
    if task == "general_events":
        matches = dict(iou_matches(p, t))
        agree = matches.copy()
        class_tasks = [(None, "general_events")]
    else:
        available = list(range(len(targets)))
        for label in ("multi", "single"):
            indices = [i for i, kind in enumerate(labels) if kind == label]
            pairs = iou_matches([p[i] for i in indices], [t[j] for j in available])
            matches.update({indices[i]: available[j] for i, j in pairs})
            claimed = {available[j] for _, j in pairs}
            available = [j for j in available if j not in claimed]
        agree = {i: j for i, j in matches.items() if labels[i] == kinds[j]}
        class_tasks = [("single", "regional_classification_on"), ("multi", "network_classification_on")]
        if task == "network_targets_native":
            class_tasks = [("multi", task)]
        elif task == "network_classification_on":
            class_tasks = [("multi", task)]
        elif task == "regional_classification_on":
            class_tasks = [("single", task)]
    audit = []
    for i, row in enumerate(calls):
        j = matches.get(i)
        touches = sum(value > 0 for value in overlap[i])
        unsupported = task != "general_events" and labels[i] not in {"single", "multi"}
        status = ("unsupported" if unsupported else "matched_correct_class" if i in agree else
                  "matched_wrong_class" if j is not None else "unmatched_in_event" if touches else "off_event")
        detail = {**row, "status": status, "overlapped_targets": touches,
                  "matched_event_id": targets[j]["event_id"] if j is not None else None,
                  "matched_target_class": kinds[j] if j is not None else ""}
        if i in agree:
            start_error, end_error = p[i][0] - t[j][0], p[i][1] - t[j][1]
            detail.update(onset_error_ms=start_error, offset_error_ms=end_error,
                          absolute_onset_error_ms=abs(start_error), absolute_offset_error_ms=abs(end_error),
                          duration_error_ms=end_error - start_error,
                          iou=overlap[i][j] / (p[i][1] - p[i][0] + t[j][1] - t[j][0] - overlap[i][j]))
        audit.append(detail)
    scores = []
    for label, task_name in class_tasks:
        indices = list(range(len(calls))) if label is None else [i for i, kind in enumerate(labels) if kind == label]
        target_indices = list(range(len(targets))) if label is None else [j for j, kind in enumerate(kinds) if kind == label]
        tp = sum(i in agree for i in indices)
        fp, fn = len(indices) - tp, len(target_indices) - tp
        wrong = sum(i in matches and i not in agree for i in indices)
        incoming = sum(kinds[j] == label and labels[i] != label for i, j in matches.items()) if label else 0
        off = sum(not any(overlap[i]) for i in indices)
        scores.append({"task": task_name, "target_class": label or "all", **metrics(tp, fp, fn),
                       "predictions": len(indices), "targets": len(target_indices),
                       "false_per_min": 60 * fp / exposure_s, "exposure_s": exposure_s,
                       "off_event_fp": off, "wrong_class_fp": wrong, "wrong_class_fn": incoming,
                       "unmatched_in_event_fp": fp - off - wrong,
                       "split_targets": sum(sum(overlap[i][j] > 0 for i in indices) >= 2 for j in target_indices),
                       "merge_predictions": sum(sum(value > 0 for value in overlap[i]) >= 2 for i in indices)})
    return scores, audit


def summarize_scores(rows, *, bootstrap_samples=10000, bootstrap_seed=20261116):
    """Average repeated seeds within recording before recording-level summaries.

    Returns (averaged recording rows, task/method summary rows). Counts in the
    summary are sums of recording seed means and can be fractional. Precision
    is pooled over these counts; seeds never increase the recording count.
    """
    import numpy as np
    import pandas as pd
    frame = pd.DataFrame(_records(rows))
    if frame.empty:
        return [], []
    if "method" not in frame:
        frame["method"] = "HNRS"
    if "trial" not in frame:
        raise ValueError("summary rows require trial recording identifiers")
    numeric = [name for name in frame.select_dtypes("number").columns if name not in {"trial", "master_seed", "seed"}]
    recording = frame.groupby(["method", "task", "trial"], sort=False, as_index=False)[numeric].mean()
    summaries = []
    for (method, task_name), group in recording.groupby(["method", "task"], sort=False):
        group = group.sort_values("trial")
        n = len(group)
        draws = np.random.default_rng(bootstrap_seed).integers(n, size=(bootstrap_samples, n))
        tp, fp, fn = [float(group[name].sum()) for name in ("tp", "fp", "fn")]
        summary = dict(method=method, task=task_name, n_recordings=n, tp=tp, fp=fp, fn=fn,
                       pooled_precision=tp / (tp + fp) if tp + fp else None,
                       pooled_recall=tp / (tp + fn) if tp + fn else None,
                       pooled_f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
                       pooled_jaccard=tp / (tp + fp + fn) if tp + fp + fn else 0.0)
        for name in ("jaccard", "f1", "precision", "recall", "false_per_min"):
            values = group[name].to_numpy(float)
            if np.isfinite(values).any():
                summary["mean_" + name] = float(np.nanmean(values))
                low, high = np.nanquantile(np.nanmean(values[draws], axis=1), [.025, .975])
                summary[name + "_ci95_low"], summary[name + "_ci95_high"] = float(low), float(high)
            else:
                summary["mean_" + name] = None
                summary[name + "_ci95_low"] = summary[name + "_ci95_high"] = None
        summaries.append(summary)
    return recording.to_dict("records"), summaries


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
    # Never pool independently rerun seed catalogs into one prediction stream.
    general, exclusive = defaultdict(list), defaultdict(list)
    for path, grouped in ((general_path, general), (exclusive_path, exclusive)):
        for row in csv_rows(path, ("trial", "start_ms", "end_ms")):
            seed = int(row.get("master_seed", row.get("seed", 0)) or 0)
            grouped[int(row["trial"]), seed].append(row)
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
    catalog_trials = {trial for trial, _ in general.keys() | exclusive.keys()}
    if unknown := catalog_trials - truth_paths.keys():
        raise ValueError(f"catalog trials without truth: {', '.join(map(str, sorted(unknown)))}")

    for trial, path in sorted(truth_paths.items()):
        truth = _truth_rows(path)
        if sorted(row["event_type"] for row in truth) != ["multi"] * 6 + ["single"] * 12:
            raise ValueError(f"{path}: expected 6 network and 12 regional-only planted events")
        seeds = sorted({seed for recording, seed in general.keys() | exclusive.keys() if recording == trial})
        if not seeds:
            raise ValueError(f"recording {trial} has no prediction catalog; empty streams need an explicit run manifest")
        for seed in seeds:
            local_general = general.get((trial, seed), [])
            local_exclusive = exclusive.get((trial, seed), [])
            native_network = [row for row in local_exclusive if row["predicted_class"] == "multi"]
            for task, calls in (("general_events", local_general), ("region_aware", local_exclusive),
                                ("network_targets_native", native_network)):
                scores, _ = score_catalog(calls, truth, task)
                for score in scores:
                    yield {"trial": trial, "master_seed": seed, "method": "HNRS", **score}


def network_summary(rows):
    _, summary = summarize_scores(rows)
    return {row["task"]: row for row in summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="benchmark", required=True)
    bp = commands.add_parser("burst-pause", help="score one native HNRS unit export")
    bp.add_argument("detections", type=Path, help="unit_LR.csv from a single recording")
    bp.add_argument("truth", type=Path, help="the same recording's unit_ground_truth.csv")
    net = commands.add_parser("network-burst", help="score seed-specific HNRS catalogs with within-recording seed averages")
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

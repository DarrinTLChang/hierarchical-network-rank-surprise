"""Truth-blind, post-detection catalogs for the manuscript HNRS outputs.

Native unit onsets and native regional candidates are used for membership.
Reported interval consolidation never changes the inputs to a later stage.
The general-detection catalog pools all regional calls across region labels;
it is not a network-stage catalog. Exclusive regional calls share no native
unit-burst contributor with an accepted native network candidate.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
import csv
import json
from math import isclose
from pathlib import Path
from statistics import median
import sys

HNRS_HOME = Path(__file__).resolve().parent / "methods" / "hnrs"
if str(HNRS_HOME) not in sys.path:
    sys.path.insert(0, str(HNRS_HOME))
from pipeline.hnrs_output_consolidation import NativeOutputEvent, merge_output_events
from pipeline.utils import infer_region, parse_electrode

CATALOG_COLUMNS = ("trial", "method", "prediction_id", "start_ms", "end_ms",
                   "predicted_class", "regions", "source_stage", "native_event_ids",
                   "unit_burst_ids", "full_end_ms")


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows, columns=CATALOG_COLUMNS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _index(indices, units):
    order = sorted(indices, key=lambda i: (units[i]["start"], i))
    return [units[i]["start"] for i in order], order


def _members(row, index):
    times, indices = index
    return indices[bisect_left(times, float(row["rs_core_start_ms"])):
                   bisect_right(times, float(row["rs_core_end_ms"]))]


def _units(rows, trial):
    units = []
    for i, row in enumerate(rows):
        electrode = row["Electrode"]
        side = parse_electrode(electrode)["side"]
        region = infer_region(electrode)
        region_key = f"{region}_{side}"
        start, end = float(row["burst_start_ms"]), float(row["burst_end_ms"])
        if end <= start:
            raise ValueError("native unit bursts must have positive duration")
        units.append(dict(index=i, start=start, end=end, region=region,
                          region_key=region_key, side=side,
                          unit=f"{electrode}|{int(row['Cluster'])}",
                          id=f"trial{trial:02d}:{side}:unit:{i:06d}"))
    return units


def _native(trial, i, stage, row, indices, units):
    if len(set(indices)) < int(row["num_onsets"]) or int(row["num_onsets"]) < 3:
        raise ValueError("candidate membership does not reproduce native onset count")
    picked = [units[j] for j in indices]
    sides = {unit["side"] for unit in picked}
    if len(sides) != 1:
        raise ValueError("a reported stream must identify one analyzed hemisphere")
    side, = sides
    start, end = float(row["burst_start_ms"]), float(row["burst_end_ms"])
    core_start, core_end = float(row["rs_core_start_ms"]), float(row["rs_core_end_ms"])
    expected_end = core_start + median(unit["end"] - unit["start"] for unit in picked)
    if (picked[0]["start"] != core_start or picked[-1]["start"] != core_end
            or start != core_start or not isclose(end, expected_end, rel_tol=0, abs_tol=1e-7)):
        raise ValueError("native core membership or median reported end does not agree")
    labels = tuple(sorted({unit["region"] for unit in picked}))
    if stage == "regional" and len(labels) != 1:
        raise ValueError("regional candidate includes more than one recording region")
    if stage == "network" and len(labels) < 2:
        raise ValueError("network candidate lacks the required two recording regions")
    return NativeOutputEvent(
        native_event_id=f"trial{trial:02d}:{side}:{stage}:{i:06d}", source_index=i,
        start_ms=start, end_ms=end, stream_id=side, region_labels=labels,
        unit_ids=tuple(unit["unit"] for unit in picked),
        unit_burst_ids=tuple(unit["id"] for unit in picked),
        contributing_indices=tuple(sorted(indices)), rs_score=float(row["RS"]),
        core_start_ms=core_start, core_end_ms=core_end, num_onsets=int(row["num_onsets"]))


def _consolidate(trial, events, grouping, kind, units, gap_ms):
    rows = []
    for group in merge_output_events(events, gap_ms=gap_ms, grouping=grouping):
        rows.append(dict(trial=trial, method="HNRS",
                         prediction_id=f"scale1p00:trial{trial:02d}:{group.stream_id}:{group.merged_event_id}",
                         start_ms=group.start_ms, end_ms=group.end_ms,
                         predicted_class=kind, regions=";".join(group.region_labels),
                         source_stage="network" if grouping == "network" else "regional",
                         native_event_ids=";".join(group.native_event_ids),
                         unit_burst_ids=";".join(group.unit_burst_ids),
                         full_end_ms=max(units[i]["end"] for i in group.contributing_indices)))
    return rows


def build_network_catalogs(native_directory, *, trial=1, gap_ms=25.0):
    """Return separate general, exclusive, all-regional, and network catalogs.

    Only native accepted CSVs are read. No truth labels, scoring windows,
    maximum merged-duration filter, or new participation rule enters here.
    """
    native_directory = Path(native_directory)
    units = _units(read_rows(native_directory / "unit_LR.csv"), trial)
    by_region = defaultdict(list)
    for unit in units:
        by_region[unit["region_key"]].append(unit["index"])
    regional_indices = {key: _index(indices, units) for key, indices in by_region.items()}
    regional = []
    for i, row in enumerate(read_rows(native_directory / "region_LR.csv")):
        indices = _members(row, regional_indices[row["Region"]])
        regional.append(_native(trial, i, "regional", row, indices, units))
    qualified = {i for event in regional for i in event.contributing_indices}
    # Native unit onset membership remains unchanged after regional reporting.
    network_index = _index(qualified, units)
    network = [_native(trial, i, "network", row, _members(row, network_index), units)
               for i, row in enumerate(read_rows(native_directory / "network_LR.csv"))]
    used = {uid for event in network for uid in event.unit_burst_ids}
    local = [event for event in regional if used.isdisjoint(event.unit_burst_ids)]
    net = _consolidate(trial, network, "network", "multi", units, gap_ms)
    reg = _consolidate(trial, local, "regional", "single", units, gap_ms)
    all_regional = _consolidate(trial, regional, "regional", "single", units, gap_ms)
    general = _consolidate(trial, regional, "pooled_regional", "unlabeled", units, gap_ms)
    diagnostics = dict(trial=trial, unit_bursts=len(units), raw_regional=len(regional),
                       raw_network=len(network), merged_network=len(net),
                       exclusive_regional=len(reg), general=len(general),
                       regional_single_distinct_unit=sum(len(event.unit_ids) == 1 for event in regional),
                       network_single_distinct_unit=sum(len(event.unit_ids) == 1 for event in network))
    return {"general_events": general, "exclusive_events": net + reg,
            "reported_regional": all_regional, "reported_network": net,
            "diagnostics": diagnostics}


def consolidate_unit_rows(rows, *, gap_ms=25.0):
    """Consolidate within unit without overwriting native accepted bursts."""
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        copied = dict(row)
        copied["burst_start_ms"] = float(row["burst_start_ms"])
        copied["burst_end_ms"] = float(row["burst_end_ms"])
        grouped[row["Electrode"], int(row["Cluster"])].append((i, copied))
    output = []
    for key in sorted(grouped):
        groups = []
        for i, row in sorted(grouped[key], key=lambda pair: (pair[1]["burst_start_ms"], pair[1]["burst_end_ms"], pair[0])):
            if groups and row["burst_start_ms"] <= groups[-1]["burst_end_ms"] + gap_ms:
                group = groups[-1]
                group["burst_end_ms"] = max(group["burst_end_ms"], row["burst_end_ms"])
                group["native_indices"].append(i)
            else:
                groups.append({**row, "native_indices": [i]})
        for group in groups:
            group["burst_duration_ms"] = group["burst_end_ms"] - group["burst_start_ms"]
            group["native_indices"] = ";".join(map(str, group["native_indices"]))
            group["native_count"] = len(group["native_indices"].split(";"))
            # No new RS score is inferred for a merged reporting interval.
            if group["native_count"] > 1:
                group["RS"] = ""
                group["RS_aggregation"] = "not_recomputed_after_output_consolidation"
                group["num_onsets"] = ""
            output.append(group)
    return output


def export_reported_catalogs(native_directory, *, trial=1, gap_ms=25.0, mode="network", shuffle_off=False):
    native_directory = Path(native_directory)
    if mode == "burst-pause":
        arm = "off" if shuffle_off else "on"
        path = native_directory / f"hnrs_{arm}_unit_bursts.csv"
        rows = read_rows(path)
        # Original RS retains native outputs; the matched shuffle-off port is
        # validated against that original implementation in the reference data.
        reported = rows if shuffle_off else consolidate_unit_rows(rows, gap_ms=gap_ms)
        fields = tuple(rows[0]) if rows else ("Electrode", "Cluster", "burst_start_ms", "burst_end_ms", "burst_duration_ms", "num_onsets", "RS", "RS_aggregation", "Method")
        fields = tuple(dict.fromkeys((*fields, "native_indices", "native_count")))
        write_rows(native_directory / "reported_unit.csv", reported, fields)
        return {"unit_bursts": len(rows), "reported_unit_bursts": len(reported)}
    catalogs = build_network_catalogs(native_directory, trial=trial, gap_ms=gap_ms)
    for name in ("general_events", "exclusive_events", "reported_regional", "reported_network"):
        write_rows(native_directory / f"{name}.csv", catalogs[name])
    unit_rows = read_rows(native_directory / "unit_LR.csv")
    fields = tuple(dict.fromkeys((*unit_rows[0], "native_indices", "native_count"))) if unit_rows else (
        "Electrode", "Cluster", "burst_start_ms", "burst_end_ms", "burst_duration_ms", "native_indices", "native_count")
    write_rows(native_directory / "reported_unit.csv", consolidate_unit_rows(unit_rows, gap_ms=gap_ms), fields)
    (native_directory / "catalog_diagnostics.json").write_text(
        json.dumps(catalogs["diagnostics"], indent=2) + "\n", encoding="utf-8")
    return catalogs["diagnostics"]

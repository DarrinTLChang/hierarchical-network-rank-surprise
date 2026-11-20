"""Validate shared scoring against fresh native comparator/support exports.

This runs no detector and applies no consolidation or duration filter to native
comparator calls. Supply a directory containing trialNN adapter output folders
produced by comparators/run_comparators.m. Score/class references come entirely
from this release's results directory; no private archive/default path is used.
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
    from score_benchmarks import assign_region_labels, score_catalog
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="required fresh comparator trialNN output root")
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error("run root must be an existing directory")
    refs = pd.read_csv(ROOT / "results/network_burst/region_aware_events.csv", float_precision="round_trip")
    refs = refs.loc[~refs.method.eq("HNRS")].fillna("")
    expected_scores = pd.read_csv(ROOT / "results/network_burst/raw_recording_scores.csv", float_precision="round_trip")
    expected_scores = expected_scores.loc[~expected_scores.method.eq("HNRS")].set_index(["trial", "method", "task"])
    calls_checked = scores_checked = 0
    max_error = 0.0
    rows = []
    for folder in sorted(args.runs.glob("trial*")):
        if not folder.is_dir():
            continue
        trial = int(folder.name[5:])
        population = pd.read_csv(folder / "native_population_events.csv", float_precision="round_trip")
        support = pd.read_csv(folder / "support_events.csv", float_precision="round_trip")
        labeled = pd.DataFrame(assign_region_labels(population, support))
        for method, calls in population.groupby("method", sort=False):
            local = labeled.loc[labeled.method.eq(method)].sort_values(["start_ms", "end_ms"]).reset_index(drop=True)
            expected = refs.loc[refs.trial.eq(trial) & refs.method.eq(method)].sort_values(["start_ms", "end_ms"]).reset_index(drop=True)
            # The region-aware reference catalog omits unsupported abstentions;
            # keep them in the fresh native audit, but they claim no target.
            classified = local.loc[local.predicted_class.isin(["single", "multi"])].reset_index(drop=True)
            assert len(classified) == len(expected), (trial, method, len(classified), len(expected))
            np.testing.assert_allclose(classified[["start_ms", "end_ms"]], expected[["start_ms", "end_ms"]], rtol=0, atol=1e-9)
            assert classified.predicted_class.tolist() == expected.predicted_class.tolist(), (trial, method, "classes")
            assert classified.regions.tolist() == expected.regions.tolist(), (trial, method, "regions")
            calls_checked += len(local)
            truth = ROOT / f"datasets/network_burst/trials/trial{trial:02d}/event_manifest.csv"
            exposure = 230 if method == "meaRtools/Otsu" else 250
            for task, inputs in (("general_events", calls), ("network_targets_native", calls), ("region_aware", local)):
                scores, _ = score_catalog(inputs, truth, task, exposure)
                for score in scores:
                    expected = expected_scores.loc[(trial, method, score["task"])]
                    for name in ("tp", "fp", "fn", "jaccard", "f1", "precision", "recall", "false_per_min"):
                        if score[name] is None:
                            assert pd.isna(expected[name])
                            continue
                        delta = abs(score[name] - expected[name])
                        max_error = max(max_error, float(delta))
                        np.testing.assert_allclose(score[name], expected[name], rtol=0, atol=1e-15)
                    scores_checked += 1
                    rows.append(dict(trial=trial, method=method, master_seed=0, **score))
    if not scores_checked:
        parser.error("no comparator trialNN outputs found; nothing was verified")
    proof = dict(verified=True, native_calls_checked=calls_checked, score_rows_checked=scores_checked,
                 support_derived_classes_and_regions_exact=True, maximum_score_error=max_error)
    pd.DataFrame(rows).to_csv(args.runs / "portable_scores.csv", index=False)
    (args.runs / "portable_scoring_verification.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()

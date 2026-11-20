"""Focused native/reference-contract and truth-blind reporting regression tests."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "methods" / "hnrs"))
from run_hnrs import _load_profile, derive_stream_seed
from hnrs_catalogs import consolidate_unit_rows
from pipeline.hnrs_output_consolidation import NativeOutputEvent, merge_output_events
from score_benchmarks import assign_region_labels, iou_matches, ordered_matches, score_catalog, summarize_scores


class Contracts(unittest.TestCase):
    def test_recording_specific_seed_contract(self):
        self.assertEqual(derive_stream_seed(123, 73001, "unit-shuffle"), 3891747325)
        self.assertEqual(derive_stream_seed(123, 73001, "source-offset"), 2397624824)
        self.assertNotEqual(derive_stream_seed(123, 73001, "unit-shuffle"),
                            derive_stream_seed(123, 73002, "unit-shuffle"))

    def test_current_profiles(self):
        p = _load_profile("network", False, None, master_seed=123,
                          recording_id=73001, duration_ms=250000)
        self.assertEqual((p.stage1_alpha, p.region_alpha, p.network_alpha), (.01, .03, .15))
        self.assertEqual((p.win_shuff_window_ms, p.win_shuff_bin_ms), (25000, 2500))
        self.assertEqual(p.min_unique_regions_network, 2)
        self.assertFalse(p.posthoc_merge_enabled)
        unit = _load_profile("burst-pause", False, None)
        self.assertEqual(unit.stage1_alpha, .01)
        self.assertEqual(unit.min_unit_burst_duration_ms, 0)
        null = _load_profile("network-null", False, 456, duration_ms=250000)
        self.assertEqual((null.win_shuff_seed, null.offset_null_seed), (456, 456))

    def test_transitive_merge_inclusive_gap_and_region_separation(self):
        events = [NativeOutputEvent(str(i), i, a, b, "L", (region,))
                  for i, (a, b, region) in enumerate([(0., 10., "A"), (35., 40., "A"),
                                                     (65., 80., "A"), (11., 20., "B")])]
        regional = merge_output_events(events, grouping="regional")
        self.assertEqual([(g.start_ms, g.end_ms) for g in regional], [(0., 80.), (11., 20.)])
        self.assertEqual(len(merge_output_events(events, grouping="pooled_regional")), 1)
        self.assertEqual([(e.start_ms, e.end_ms) for e in events], [(0, 10), (35, 40), (65, 80), (11, 20)])

    def test_unit_merge_does_not_invent_rs(self):
        rows = [dict(Electrode="e", Cluster=1, burst_start_ms=a, burst_end_ms=b,
                     RS=5, num_onsets=3) for a, b in [(0, 10), (35, 45)]]
        output = consolidate_unit_rows(rows)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["burst_end_ms"], 45)
        self.assertEqual(output[0]["RS"], "")
        self.assertEqual(rows[0]["burst_end_ms"], 10)

    def test_positive_overlap_and_cardinality(self):
        self.assertEqual(iou_matches([(0, 10)], [(10, 20)]), [])
        self.assertEqual(len(iou_matches([(0, 15), (0, 9)], [(0, 10), (10, 20)])), 2)
        self.assertEqual(ordered_matches([(0, 10), (20, 30)], [(0, 8), (21, 30)])[0], 2)

    def test_classification_vs_general_and_native_targets(self):
        truth = [dict(event_id=1, actual_start_ms=0, actual_end_ms=10, event_type="single", regions="A"),
                 dict(event_id=2, actual_start_ms=20, actual_end_ms=30, event_type="multi", regions="A;B")]
        calls = [dict(prediction_id="x", start_ms=0, end_ms=10, predicted_class="multi"),
                 dict(prediction_id="y", start_ms=20, end_ms=30, predicted_class="single")]
        self.assertEqual(score_catalog(calls, truth)[0][0]["tp"], 2)
        native = score_catalog(calls, truth, "network_targets_native")[0][0]
        self.assertEqual((native["tp"], native["fp"], native["fn"]), (1, 1, 0))
        secondary, _ = score_catalog(calls, truth, "region_aware")
        self.assertEqual([(r["tp"], r["fp"], r["fn"]) for r in secondary], [(0, 1, 1), (0, 1, 1)])

    def test_region_concurrency_inside_native_boundary(self):
        calls = [dict(event_id=1, start_ms=0, end_ms=10)]
        support = [dict(event_id=1, region="A", start_ms=0, end_ms=5),
                   dict(event_id=1, region="B", start_ms=5, end_ms=10)]
        self.assertEqual(assign_region_labels(calls, support)[0]["predicted_class"], "unsupported")
        support[1]["start_ms"] = 4
        self.assertEqual(assign_region_labels(calls, support)[0]["predicted_class"], "multi")

    def test_seed_average_not_independent_recordings(self):
        rows = [dict(method="HNRS", task="general_events", trial=t, master_seed=s,
                     tp=1, fp=0, fn=0, jaccard=1, f1=1, precision=1, recall=1, false_per_min=0)
                for t in (1, 2) for s in (123, 456, 789)]
        recording, summary = summarize_scores(rows, bootstrap_samples=10)
        self.assertEqual(len(recording), 2)
        self.assertEqual(summary[0]["n_recordings"], 2)
        self.assertEqual(summary[0]["tp"], 2)


if __name__ == "__main__":
    unittest.main()

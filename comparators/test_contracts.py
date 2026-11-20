"""Small static interval fixtures; these are not benchmark generators."""
import unittest

from verify_run import classify_support


class AnatomicalSupportTests(unittest.TestCase):
    def test_concurrent_distinct_regions(self):
        result = classify_support(0, 3, [
            {"start_ms": 0, "end_ms": 2, "region": "R1"},
            {"start_ms": 1, "end_ms": 3, "region": "R2"},
        ])
        self.assertEqual(result["predicted_class"], "multi")

    def test_touching_is_not_positive_overlap(self):
        result = classify_support(0, 3, [
            {"start_ms": 0, "end_ms": 1, "region": "R1"},
            {"start_ms": 1, "end_ms": 2, "region": "R2"},
        ])
        self.assertEqual(result["predicted_class"], "unsupported")

    def test_one_region_remains_single(self):
        result = classify_support(0, 3, [
            {"start_ms": 0, "end_ms": 2, "region": "R1"},
            {"start_ms": 1, "end_ms": 3, "region": "R1"},
        ])
        self.assertEqual(result["predicted_class"], "single")

    def test_clipping_to_parent_matters(self):
        result = classify_support(0, 1, [
            {"start_ms": 0, "end_ms": 2, "region": "R1"},
            {"start_ms": 1, "end_ms": 3, "region": "R2"},
        ])
        self.assertEqual(result["predicted_class"], "single")

    def test_no_support_remains_unsupported(self):
        self.assertEqual(classify_support(0, 1, [])["predicted_class"], "unsupported")


if __name__ == "__main__":
    unittest.main()

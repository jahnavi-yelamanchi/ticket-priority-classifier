import unittest

from app.benchmarking import percentile


class BenchmarkUtilityTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank_on_unsorted_values(self) -> None:
        self.assertEqual(percentile([5.0, 1.0, 4.0, 2.0, 3.0], 0.95), 5.0)

    def test_percentile_rejects_empty_samples(self) -> None:
        with self.assertRaises(ValueError):
            percentile([], 0.5)

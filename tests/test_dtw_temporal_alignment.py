import unittest

import numpy as np

from ai_trainer.dtw_compare import _dtw_dp
from ai_trainer.online_dtw import _subsequence_dtw_last_row_min


class DtwTemporalAlignmentTests(unittest.TestCase):
    def test_full_dtw_uses_actual_path_length_for_normalization(self):
        cost, path_len = _dtw_dp(np.ones((3, 6), dtype=float))

        self.assertEqual(path_len, 6)
        self.assertEqual(cost / path_len, 1.0)

        _, equal_length_path = _dtw_dp(np.ones((3, 3), dtype=float))
        self.assertEqual(equal_length_path, 3)

    def test_subsequence_dtw_uses_actual_path_length_for_normalization(self):
        distance = _subsequence_dtw_last_row_min(np.ones((3, 6), dtype=float))

        self.assertEqual(distance, 1.0)

    def test_optional_warping_window_keeps_different_length_endpoints_reachable(self):
        cost, path_len = _dtw_dp(np.ones((3, 6), dtype=float), window_ratio=0.2)

        self.assertTrue(np.isfinite(cost))
        self.assertGreater(path_len, 0)


if __name__ == "__main__":
    unittest.main()

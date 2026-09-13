import unittest

import numpy as np

from ai_trainer.scoring import (
    binary_normal_probability,
    binary_roc_auc,
    fit_binary_logistic_calibration,
)


class HierarchicalScoringTests(unittest.TestCase):
    def test_logistic_calibration_separates_simple_distance_features(self):
        features = np.asarray(
            [[0.1, 0.2], [0.2, 0.1], [1.0, 0.9], [0.9, 1.0]],
            dtype=np.float64,
        )
        labels = np.asarray([True, True, False, False])
        calibration = fit_binary_logistic_calibration(
            features,
            labels,
            ("distance_a", "distance_b"),
        )
        normal_probability = binary_normal_probability(features[0], calibration)
        error_probability = binary_normal_probability(features[-1], calibration)
        self.assertGreater(normal_probability, error_probability)
        self.assertGreaterEqual(calibration["calibration_balanced_accuracy"], 0.99)

    def test_logistic_calibration_imputes_infinite_distances(self):
        features = np.asarray([[0.1], [0.2], [1.0], [np.inf]])
        calibration = fit_binary_logistic_calibration(
            features,
            np.asarray([True, True, False, False]),
            ("distance",),
        )
        probability = binary_normal_probability(np.asarray([np.inf]), calibration)
        self.assertTrue(np.isfinite(probability))

    def test_roc_auc_is_tie_aware(self):
        auc = binary_roc_auc(
            np.asarray([False, False, True, True]),
            np.asarray([0.0, 0.5, 0.5, 1.0]),
        )
        self.assertAlmostEqual(auc, 0.875)


if __name__ == "__main__":
    unittest.main()

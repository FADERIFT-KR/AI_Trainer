import unittest

import numpy as np

from ai_trainer.dtw_compare import PHASES, phase_aware_weighted_dtw


def _features(length: int) -> dict[str, np.ndarray]:
    return {"joint_coords_3d": np.linspace(0.0, 1.0, length)[:, None]}


def _bounds(length: int) -> dict[str, list[int]]:
    return {phase: [0, length] for phase in PHASES}


def _config(max_ratio: float = 3.0) -> dict:
    return {
        "features": {"joint_coords_3d": {"metric": "euclidean"}},
        "phase_weights": {phase: 1.0 for phase in PHASES},
        "dtw_alignment": {
            "phase_resample_frames": {phase: 12 for phase in PHASES},
            "warping_window_ratio": 0.25,
            "max_phase_length_ratio": max_ratio,
        },
    }


class DtwAlignmentTests(unittest.TestCase):
    def test_equal_motion_at_different_duration_is_resampled(self) -> None:
        result = phase_aware_weighted_dtw(
            _features(10),
            _bounds(10),
            _features(20),
            _bounds(20),
            {"joint_coords_3d": 1.0},
            _config(),
        )
        self.assertTrue(result["alignment_valid"])
        self.assertAlmostEqual(result["total"], 0.0)
        self.assertTrue(all(value == 2.0 for value in result["length_ratio_by_phase"].values()))

    def test_excessive_phase_duration_ratio_is_rejected(self) -> None:
        result = phase_aware_weighted_dtw(
            _features(2),
            _bounds(2),
            _features(10),
            _bounds(10),
            {"joint_coords_3d": 1.0},
            _config(max_ratio=3.0),
        )
        self.assertFalse(result["alignment_valid"])
        self.assertTrue(np.isinf(result["total"]))
        self.assertEqual(set(result["invalid_phases"]), set(PHASES))

    def test_error_class_can_override_phase_weights(self) -> None:
        config = _config()
        config["class_phase_weights"] = {
            "오류": {phase: 2.0 for phase in PHASES}
        }
        query = {"joint_coords_3d": np.zeros((4, 1))}
        reference = {"joint_coords_3d": np.ones((4, 1))}
        baseline = phase_aware_weighted_dtw(
            query,
            _bounds(4),
            reference,
            _bounds(4),
            {"joint_coords_3d": 1.0},
            config,
        )
        weighted = phase_aware_weighted_dtw(
            query,
            _bounds(4),
            reference,
            _bounds(4),
            {"joint_coords_3d": 1.0},
            config,
            class_label="오류",
        )
        self.assertAlmostEqual(weighted["total"], baseline["total"] * 2.0)


if __name__ == "__main__":
    unittest.main()

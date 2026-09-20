from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.skeleton_filter import refine_skeleton_sequence  # noqa: E402


JOINTS = ["Hip", "LHip", "LKnee", "LAnkle", "Neck", "LShoulder", "LElbow", "LWrist"]


def synthetic_motion() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    t = np.arange(120) / 30.0
    root_y = 0.3 * np.cos(2 * np.pi * 0.45 * t)
    clean = np.zeros((len(t), len(JOINTS), 3), dtype=np.float64)
    offsets = {
        "Hip": (0.0, 0.0, 0.0), "LHip": (-0.2, 0.0, 0.0),
        "LKnee": (-0.2, -0.5, 0.03), "LAnkle": (-0.2, -1.0, 0.0),
        "Neck": (0.0, 0.65, 0.0), "LShoulder": (-0.3, 0.62, 0.0),
        "LElbow": (-0.55, 0.40, 0.02), "LWrist": (-0.72, 0.18, 0.03),
    }
    for index, name in enumerate(JOINTS):
        clean[:, index] = np.asarray(offsets[name])
        clean[:, index, 1] += root_y
    noisy = clean + rng.normal(scale=0.006, size=clean.shape)
    noisy[37, JOINTS.index("LKnee")] += np.array([0.35, -0.28, 0.22])
    noisy[81, JOINTS.index("LWrist")] += np.array([-0.30, 0.25, -0.20])
    return clean, noisy


class SkeletonFilterTests(unittest.TestCase):
    def test_refinement_removes_spikes_and_improves_clean_motion_error(self) -> None:
        clean, noisy = synthetic_motion()
        refined, report = refine_skeleton_sequence(noisy, JOINTS)
        raw_error = float(np.sqrt(np.mean((noisy - clean) ** 2)))
        refined_error = float(np.sqrt(np.mean((refined - clean) ** 2)))
        self.assertGreaterEqual(report.spike_points, 2)
        self.assertLess(refined_error, raw_error * 0.55)
        self.assertLess(report.refined_jerk_rms_normalized, report.raw_jerk_rms_normalized)
        self.assertLess(report.refined_bone_length_cv, report.raw_bone_length_cv)

    def test_short_sequence_is_supported_without_phase_shift_failure(self) -> None:
        coords = np.zeros((4, len(JOINTS), 3), dtype=np.float64)
        for index in range(len(JOINTS)):
            coords[:, index, 1] = index * 0.1
        refined, report = refine_skeleton_sequence(coords, JOINTS)
        self.assertEqual(refined.shape, coords.shape)
        self.assertTrue(np.isfinite(refined).all())
        self.assertEqual(report.frames, 4)


if __name__ == "__main__":
    unittest.main()

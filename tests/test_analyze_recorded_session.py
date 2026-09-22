"""Recorded-session diagnostics must expose occluded display jumps."""
from __future__ import annotations

import unittest

import numpy as np

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES
from scripts.analyze_recorded_session import occluded_display_jumps


class RecordedSessionAnalysisTests(unittest.TestCase):
    def test_occluded_far_wrist_jump_is_reported(self):
        joint = COMMON_JOINT_NAMES.index("RWrist")
        first = np.zeros((len(COMMON_JOINT_NAMES), 3))
        second = first.copy()
        second[joint, 0] = 0.5
        world = np.zeros((33, 4))
        world[:, 3] = 1.0
        world[16, 3] = 0.1
        rows = [
            {"video_frame": 0, "elapsed_ms": 0.0, "display_aligned_3d": first.tolist(),
             "world_landmarks": world.tolist()},
            {"video_frame": 1, "elapsed_ms": 100.0, "display_aligned_3d": second.tolist(),
             "world_landmarks": world.tolist()},
        ]
        hits = occluded_display_jumps(rows, "RWrist", 16)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["video_frame"], 1)
        self.assertEqual(hits[0]["jump_normalized"], 0.5)
        rows[1]["world_landmarks"][16][3] = 0.9
        self.assertEqual(occluded_display_jumps(rows, "RWrist", 16), [])

    def test_older_recording_without_display_coordinates_is_supported(self):
        rows = [{"video_frame": 0, "elapsed_ms": 0.0, "world_landmarks": None},
                {"video_frame": 1, "elapsed_ms": 100.0, "world_landmarks": None}]
        self.assertEqual(occluded_display_jumps(rows, "RWrist", 16), [])


if __name__ == "__main__":
    unittest.main()

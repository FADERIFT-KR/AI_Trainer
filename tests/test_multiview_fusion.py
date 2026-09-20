from __future__ import annotations

import unittest

import numpy as np

from ai_trainer.camera_views import VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.multiview_fusion import fuse_recorded_views


_IDX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}


def _base_pose(progress: float) -> np.ndarray:
    pose = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=float)
    for index in range(len(COMMON_JOINT_NAMES)):
        side = -1.0 if COMMON_JOINT_NAMES[index].startswith("L") else 1.0
        pose[index] = (0.04 * side * (index + 1), 0.03 * index, 0.01 * index)
    pose[_IDX["Hip"]] = 0.0
    pose[_IDX["Neck"]] = (0.0, 0.75 - 0.20 * np.sin(np.pi * progress), 0.08)
    pose[_IDX["LKnee"]] = (-0.18, -0.42 + 0.10 * np.sin(np.pi * progress), 0.32)
    pose[_IDX["RKnee"]] = (0.18, -0.42 + 0.10 * np.sin(np.pi * progress), 0.32)
    pose[_IDX["LAnkle"]] = (-0.20, -0.92, 0.05)
    pose[_IDX["RAnkle"]] = (0.20, -0.92, 0.05)
    return pose


def _rows(view: str, length: int) -> list[dict]:
    rows = []
    for frame in range(length):
        progress = frame / (length - 1)
        truth = _base_pose(progress)
        observed = truth.copy()
        if view == VIEW_FRONT:
            # Front depth is intentionally poor.
            observed[_IDX["LKnee"], 2] += 2.0
        else:
            # Side lateral coordinates are intentionally poor.
            observed[_IDX["LKnee"], 0] += 2.0
        landmarks = np.zeros((33, 4), dtype=float)
        landmarks[:, 3] = 1.0
        rows.append({
            "video_frame": frame,
            "analysis_frame": frame,
            "display_aligned_3d": observed.tolist(),
            "world_landmarks": landmarks.tolist(),
            "frozen_3d": [False] * len(COMMON_JOINT_NAMES),
            "completed_rep": ({"frame_range": [0, length - 1]} if frame == length - 1 else None),
        })
    return rows


class MultiViewFusionTests(unittest.TestCase):
    def test_front_lateral_and_side_depth_are_preferred_after_phase_alignment(self):
        rows = {
            VIEW_FRONT: _rows(VIEW_FRONT, 17),
            VIEW_LEFT: _rows(VIEW_LEFT, 11),
            VIEW_RIGHT: _rows(VIEW_RIGHT, 23),
        }
        result = fuse_recorded_views(rows, {view: 10.0 for view in rows})
        self.assertEqual(len(result.repetitions), 1)
        repetition = result.repetitions[0]
        self.assertEqual(repetition.source_views, (VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT))
        self.assertEqual(repetition.coordinates.shape, (23, len(COMMON_JOINT_NAMES), 3))

        middle = repetition.coordinates[len(repetition.coordinates) // 2, _IDX["LKnee"]]
        truth = _base_pose(0.5)[_IDX["LKnee"]]
        bad_side_x = _rows(VIEW_LEFT, 3)[1]["display_aligned_3d"][_IDX["LKnee"]][0]
        bad_front_z = _rows(VIEW_FRONT, 3)[1]["display_aligned_3d"][_IDX["LKnee"]][2]
        self.assertLess(abs(middle[0] - truth[0]), abs(bad_side_x - truth[0]))
        self.assertLess(abs(middle[2] - truth[2]), abs(bad_front_z - truth[2]))

        # Every repetition frame maps back to its own view timeline despite
        # the three recordings having different frame counts.
        for view, view_rows in rows.items():
            self.assertEqual(len(result.frame_poses[view]), len(view_rows))
            self.assertTrue(all(source == "phase_fused" for source in result.frame_sources[view]))

    def test_requires_front_and_at_least_one_side_for_comprehensive_output(self):
        rows = {VIEW_FRONT: _rows(VIEW_FRONT, 8)}
        result = fuse_recorded_views(rows)
        self.assertEqual(result.repetitions, ())
        self.assertTrue(all(source == "single_view" for source in result.frame_sources[VIEW_FRONT]))


if __name__ == "__main__":
    unittest.main()

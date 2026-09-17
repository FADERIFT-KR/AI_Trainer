import unittest
from types import SimpleNamespace

import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.heel_semantic import heel_motion_evidence, validate_heel_candidate
from ai_trainer.online_dtw import OnlineSquatSession


I = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}


class HeelSemanticTests(unittest.TestCase):
    def _sequence(self, heel_motion: float) -> np.ndarray:
        points = np.zeros((10, len(COMMON_JOINT_NAMES), 2), dtype=float)
        points[:, I["Neck"], 1] = -1.0
        for side in ("L", "R"):
            points[:, I[f"{side}Ankle"], 1] = 1.0
            points[:, I[f"{side}Heel"], 1] = 1.0 - np.linspace(0.0, heel_motion, 10)
            points[:, I[f"{side}BigToe"], 1] = 1.0
        return points

    def test_clear_no_evidence_rejects_only_heel_candidate(self):
        evidence = heel_motion_evidence(self._sequence(0.03), no_max=0.08, yes_min=0.10)
        self.assertEqual(evidence["verdict"], "NO")
        final, reason = validate_heel_candidate(
            "발뒤꿈치오류", "발뒤꿈치오류", evidence, 0.20
        )
        self.assertEqual(final, "자세추정불확실")
        self.assertEqual(reason, "UNKNOWN_HEEL_SEMANTIC_CONFLICT")

    def test_positive_evidence_retains_heel_error(self):
        evidence = heel_motion_evidence(self._sequence(0.12), no_max=0.08, yes_min=0.10)
        final, reason = validate_heel_candidate(
            "발뒤꿈치오류", "발뒤꿈치오류", evidence, 0.20
        )
        self.assertEqual((final, reason), ("발뒤꿈치오류", None))

    def test_existing_unknown_and_other_classes_are_unchanged(self):
        evidence = {"verdict": "NO"}
        self.assertEqual(
            validate_heel_candidate("발뒤꿈치오류", "자세추정불확실", evidence, 0.0),
            ("자세추정불확실", None),
        )
        self.assertEqual(
            validate_heel_candidate("정상", "정상", evidence, 0.0),
            ("정상", None),
        )

    def test_partial_dtw_is_recomputed_only_at_interval_or_event(self):
        calls = []
        session = SimpleNamespace(
            state="descend",
            weights_cfg={"temporal_alignment": {"partial_dtw_interval_frames": 6}},
            partial_dtw_last_frame=None,
            partial_dtw_last_state=None,
            partial_dtw_cache=None,
            _partial_online_distance=lambda frame: calls.append(frame) or {"frame": frame},
        )
        run = OnlineSquatSession._throttled_partial_online_distance
        self.assertEqual(run(session, 10, None), {"frame": 10})
        self.assertEqual(run(session, 14, None), {"frame": 10})
        self.assertEqual(run(session, 16, None), {"frame": 16})
        self.assertEqual(run(session, 17, "bottom"), {"frame": 17})
        self.assertEqual(calls, [10, 16, 17])


if __name__ == "__main__":
    unittest.main()

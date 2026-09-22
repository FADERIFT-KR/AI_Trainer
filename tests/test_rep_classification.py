import unittest
from unittest.mock import patch

import numpy as np
import torch

from ai_trainer.dl_classifier import build_fixed_vector
from ai_trainer.online_dtw import OnlineSquatSession


class CheckingClassifier:
    def predict_proba(self, feat, bounds):
        assert build_fixed_vector(feat, bounds) is not None
        return np.array([0.9, 0.05, 0.03, 0.02])


class RepClassificationTests(unittest.TestCase):
    def test_preparation_reaches_classifier_with_nonzero_emit_offset(self):
        session = OnlineSquatSession(
            model=torch.nn.Identity(), device=torch.device("cpu"),
            db_operational={"정상": [], "발뒤꿈치오류": []}, weights_cfg={},
            dl_classifier=CheckingClassifier(),
        )
        session.emit_offset = 7
        session.aligned_seq = list(np.random.default_rng(0).normal(size=(10, 18, 3)))
        session.rep_start_idx = 9
        session.phase_boundaries_running = {
            "준비": [7, 9], "하강": [9, 11], "최저점": [11, 13],
            "상승": [13, 15], "종료": [15, 17],
        }
        detail = {"min_distance": 1.0, "best_detail": {"per_feature_contrib": {}}}
        with patch("ai_trainer.online_dtw.resolve_weights", return_value={}), patch(
            "ai_trainer.online_dtw.multi_reference_distance", return_value=detail
        ):
            session._finalize_rep(16)
        result = session.completed_reps[-1]
        self.assertEqual(result.classifier_source, "dl")
        self.assertEqual(result.phase_bounds["준비"], [0, 2])
        self.assertEqual(result.predicted_class, "정상")

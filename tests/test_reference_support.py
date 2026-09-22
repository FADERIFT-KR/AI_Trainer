import json
from pathlib import Path
import unittest

import numpy as np
import torch

from ai_trainer.dtw_compare import PHASES, resolve_weights
from ai_trainer.features import extract_all_features
from ai_trainer.online_dtw import OnlineSquatSession
from ai_trainer.reference_support import ReferenceSupport
from tests.test_pose_timing import squat_pose


class ReferenceSupportTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "configs/dtw_feature_weights.json").read_text(encoding="utf-8"))
        self.coords = np.stack([squat_pose(t) for t in np.linspace(0, 4, 20)])
        self.feat = extract_all_features(self.coords)
        self.bounds = {p: [i*4, (i+1)*4] for i, p in enumerate(PHASES)}
        self.weights = resolve_weights(self.config, self.config["default_profile"], None)

    def reference(self, offset):
        feat = {k: v.copy() for k, v in self.feat.items()}
        for k in ("knee_flexion_angle", "hip_flexion_angle", "ankle_angle"):
            feat[k] += offset
        return {"feat": feat, "bounds": self.bounds, "meta": {}}

    def test_identical_candidate_references_get_equal_distances_across_labels(self):
        labels = ("정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류")
        refs = {label: [self.reference(0.1)] for label in labels}
        session = OnlineSquatSession(torch.nn.Identity(), torch.device("cpu"), refs, self.config)
        session.aligned_seq = list(self.coords)
        session.emit_offset = 0
        session.rep_start_idx = 4
        session.phase_boundaries_running = self.bounds
        session._finalize_rep(19)
        values = list(session.completed_reps[0].raw_distance_by_class.values())
        self.assertGreater(values[0], 0)
        np.testing.assert_allclose(values, values[0])

    def test_reference_coverage_rejects_distant_query_without_renaming_it_normal(self):
        support = ReferenceSupport({"error": [self.reference(0), self.reference(0.05), self.reference(0.1)]}, self.weights, self.config)
        close = support.check("error", 0.0, self.bounds)
        self.assertTrue(close["supported"])
        distant = support.check("error", close["radius"] * 2, self.bounds)
        self.assertFalse(distant["supported"])
        self.assertGreater(distant["distance"], distant["radius"])

    def test_single_reference_cannot_estimate_coverage(self):
        support = ReferenceSupport({"error": [self.reference(0)]}, self.weights, self.config)
        result = support.check("error", 0, self.bounds)
        self.assertFalse(result["supported"])
        self.assertIsNone(result["radius"])

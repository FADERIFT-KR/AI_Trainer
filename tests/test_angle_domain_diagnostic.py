import json
import tempfile
import unittest

import numpy as np

from ai_trainer.angle_domain_diagnostic import AngleDomainDiagnostic, angle_definition, frame_angles
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.features import extract_all_features
from ai_trainer.two_d_diagnostic import extract_2d_features
from tests.test_posture_score import sequence


class AngleDomainDiagnosticTests(unittest.TestCase):
    def test_definition_uses_common_skeleton_triplets(self):
        definition = angle_definition()
        self.assertEqual(definition["hip"]["left"]["joints"], ["Neck", "LHip", "LKnee"])
        self.assertEqual(definition["knee"]["right"]["joints"], ["RHip", "RKnee", "RAnkle"])
        self.assertEqual(definition["hip"]["left"]["indices"], [17, 6, 8])
        self.assertEqual(len(definition["skeleton_order"]), len(COMMON_JOINT_NAMES))

    def test_standing_angles_use_interior_convention(self):
        row = frame_angles(sequence(0.0, frames=1)[0])
        self.assertGreater(row["knee_avg"], 150.0)
        self.assertGreater(row["hip_avg"], 150.0)

    def test_writer_records_recomputed_live_values_without_mutating_inputs(self):
        refs2d, refs3d = [], []
        for index, amplitude in enumerate((0.9, 1.0)):
            coords = sequence(amplitude)
            raw = coords[:, :, :2] * 100.0
            feat, summary = extract_2d_features(raw)
            refs2d.append({"id": f"r{index}", "raw": raw, "feat": feat, "summary": summary})
            refs3d.append({"meta": {"medoid_id": f"r{index}"}, "feat": extract_all_features(coords)})
        cfg = {
            "hip_3d_minus_2d_excursion_min_deg": -20.0,
            "hip_3d_minus_2d_excursion_max_deg": 20.0,
            "knee_3d_minus_2d_excursion_min_deg": -20.0,
            "knee_3d_minus_2d_excursion_max_deg": 20.0,
        }
        raw_before = refs2d[0]["raw"].copy()
        with tempfile.TemporaryDirectory() as directory:
            writer = AngleDomainDiagnostic(refs2d, refs3d, directory, cfg)
            writer.record(
                rep=1, raw2d=refs2d[0]["raw"],
                coords3d=refs3d[0]["feat"]["joint_coords_3d"].reshape(-1, 18, 3),
                standing2d_frames=refs2d[0]["raw"][:5],
                standing3d_frames=refs3d[0]["feat"]["joint_coords_3d"].reshape(-1, 18, 3)[:5],
                detector_bottom=12,
                scorer_payload={"overall_match": 90.0, "diagnostic": {
                    "hip_excursion_2d": refs2d[0]["summary"]["hip_excursion"],
                    "knee_excursion_2d": refs2d[0]["summary"]["knee_excursion"],
                }},
                production={"raw": "정상", "final": "정상"},
            )
            path = writer.path
            writer.close()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["record_type"], "completed_rep")
        self.assertIn("normalized_landmarks", rows[-1])
        self.assertGreater(rows[1]["statistics"]["3d"]["hip"]["standing"]["median"], 80.0)
        self.assertTrue(np.array_equal(raw_before, refs2d[0]["raw"]))


if __name__ == "__main__":
    unittest.main()

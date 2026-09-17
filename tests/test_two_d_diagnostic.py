import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_trainer.two_d_diagnostic import TwoDDiagnostic, _distance, extract_2d_features


def motion(frames=12):
    p = np.zeros((frames, 18, 2), dtype=np.float32)
    for t in range(frames):
        p[t, :, 0] = np.linspace(100, 300, 18)
        p[t, :, 1] = np.linspace(80, 420, 18)
        p[t, 16] = (200, 240 + 30 * np.sin(np.pi * t / (frames - 1)))
        p[t, 17] = (200, 140)
    return p


class TwoDDiagnosticTests(unittest.TestCase):
    def test_feature_extraction_shape_and_finite_values(self):
        features, summary = extract_2d_features(motion())
        self.assertEqual(features.shape, (12, 12))
        self.assertTrue(np.isfinite(features).all())
        self.assertIn("bottom_knee_angle", summary)

    def test_class_distance_calculation_uses_only_2d_features(self):
        features, _ = extract_2d_features(motion())
        self.assertEqual(_distance(features, features), 0.0)
        self.assertGreater(_distance(features, features + 1), 0.0)

    def test_unknown_margin_handling(self):
        features, _ = extract_2d_features(motion())
        obj = TwoDDiagnostic.__new__(TwoDDiagnostic)
        obj.enabled = False
        obj._stream = None
        obj.refs = {
            name: [{"id": name, "feat": features}]
            for name in ("정상", "고관절오류", "발뒤꿈치오류", "엉덩이하방오류")
        }
        result = obj.diagnose(motion(), {"final": "자세추정불확실"})
        self.assertEqual(result["best"], "2D_ONLY_UNKNOWN")

    def test_diagnostic_exception_is_isolated(self):
        obj = TwoDDiagnostic.__new__(TwoDDiagnostic)
        obj.enabled = False
        obj._stream = None
        obj.refs = {}
        result = obj.diagnose(np.zeros((1, 18, 2)), {"final": "정상"})
        self.assertEqual(result["best"], "2D_DIAGNOSTIC_FAILED")

    def test_operational_reference_2d_loading(self):
        from scripts.build_reference_db import TL_ZIP, VL_ZIP
        project = Path(__file__).resolve().parent.parent
        manifest = project / "output" / "reference_db" / "manifest.json"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "output" / "reference_db"
            target.mkdir(parents=True)
            (target / "manifest.json").write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
            diag = TwoDDiagnostic(root, TL_ZIP, VL_ZIP)
            try:
                self.assertTrue(diag.enabled, diag.error)
                self.assertEqual({k: len(v) for k, v in diag.refs.items()}, {
                    "정상": 4, "발뒤꿈치오류": 4, "엉덩이하방오류": 4, "고관절오류": 4,
                })
            finally:
                diag.close()


if __name__ == "__main__":
    unittest.main()

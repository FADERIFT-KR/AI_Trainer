import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ai_trainer.game_ui.pose_diagnostics import PoseDiagnostics
from ai_trainer.online_dtw import RepResult


class PoseDiagnosticsTests(unittest.TestCase):
    def test_trace_serializes_coordinates_and_classifier_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            trace = PoseDiagnostics(path)
            try:
                trace.write(
                    common3d=np.zeros((18, 3)), frozen3d=np.zeros(18, dtype=bool),
                    completed_rep=RepResult(0, (7, 16), "정상", {}, 90.0, [], classifier_source="dl"),
                )
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(record["completed_rep"]["classifier_source"], "dl")
                self.assertEqual(len(record["common3d"]), 18)
                with self.assertRaises(FileExistsError):
                    PoseDiagnostics(path)
            finally:
                trace.close()

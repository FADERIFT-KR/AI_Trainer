import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from ai_trainer.game_ui.pipeline_worker import PipelineStatus
from ai_trainer.game_ui.screens import CompareScreen, TARGET_REPS
from ai_trainer.online_dtw import RepResult


class ClassificationResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_reference_coverage_does_not_hide_classification_or_normal_score(self):
        screen = CompareScreen()
        try:
            for i in range(TARGET_REPS):
                rep = RepResult(i, (0, 10), "고관절오류", {}, 99.0 if i == 0 else None, [],
                                assessment_supported=False, assessment_note="reference mismatch")
                status = PipelineStatus(np.zeros((100, 100, 3), dtype=np.uint8), 10, True,
                                        1, 0, True, "", "prep", 1, i+1, None, rep, None, None, None)
                screen._on_status(status)
            self.assertEqual(screen._session_reps[0]["display_class"], "정상")
            self.assertTrue(all(r["display_class"] == "고관절오류" for r in screen._session_reps[1:]))
            self.assertIn(f"정상 1 / {TARGET_REPS}", screen.game_result_body.text())
            self.assertNotIn("판정 보류", screen.game_result_body.text())
        finally:
            screen.stop()
            screen.close()

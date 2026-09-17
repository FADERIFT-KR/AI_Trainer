import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from ai_trainer.rep_detector_2d import Adaptive2DRepDetector
from ai_trainer.rep_diagnostics import RepDiagnosticRecorder


def row(frame, state="prep", event=None, progress=0.1, reason=None):
    return {
        "frame": frame,
        "previous_state": state,
        "current_state": state,
        "combined_progress": progress,
        "event": event,
        "event_reason": reason,
    }


class RepDiagnosticTests(unittest.TestCase):
    def test_adaptive_detector_exposes_actual_frame_conditions(self):
        detector = Adaptive2DRepDetector(40.0, 20.0)
        detector.update(1, 8.0, 4.0)
        debug = detector.last_debug
        self.assertAlmostEqual(debug["knee_progress"], 0.2)
        self.assertAlmostEqual(debug["hip_progress"], 0.2)
        self.assertAlmostEqual(debug["combined_progress"], 0.2)
        self.assertTrue(debug["start_condition"])
        self.assertEqual(debug["counter"], 1)
        self.assertEqual(debug["required_counter"], 2)

    def test_transition_and_abort_are_written(self):
        with tempfile.TemporaryDirectory() as td:
            rec = RepDiagnosticRecorder(td, now=datetime(2026, 9, 3, 1, 2, 3))
            rec.record_detector(row(1, event="rep_start", progress=0.25, reason="PRIMARY_2D_PROGRESS"))
            rec.record_detector(row(2, state="descend", event="abort", progress=0.1,
                                    reason="LOW_PEAK_PROGRESS_EARLY_RETURN"))
            rec.close()
            records = [json.loads(line) for line in rec.trace_path.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(rec.summary_path.read_text(encoding="utf-8"))
            self.assertEqual([x["event"] for x in records], ["rep_start", "abort"])
            self.assertEqual(summary["attempts"][0]["result"], "ABORTED")
            self.assertEqual(summary["attempts"][0]["reason"], "LOW_PEAK_PROGRESS_EARLY_RETURN")

    def test_framing_skip_is_written_and_marks_unfinished_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            rec = RepDiagnosticRecorder(td)
            rec.record_detector(row(1, progress=0.1))
            rec.record_framing_skip(camera_frame=9, reason="FRAMING_NOT_OK", position_state="TOO_FAR")
            rec.close()
            trace = rec.trace_path.read_text(encoding="utf-8")
            summary = json.loads(rec.summary_path.read_text(encoding="utf-8"))
            self.assertIn('"event": "FRAME_SKIPPED"', trace)
            self.assertEqual(summary["attempts"][0]["result"], "FRAMING_INTERRUPTED")

    def test_completed_rep_summary(self):
        with tempfile.TemporaryDirectory() as td:
            rec = RepDiagnosticRecorder(td)
            rec.record_detector(row(1, event="rep_start", progress=0.25))
            rec.record_detector(row(2, state="descend", progress=0.8))
            rec.record_detector(row(3, state="ascend", event="rep_end", progress=0.1,
                                    reason="2D_PROGRESS_RECOVERED"))
            rec.close()
            attempt = json.loads(rec.summary_path.read_text(encoding="utf-8"))["attempts"][0]
            self.assertEqual(attempt["result"], "COMPLETED")
            self.assertEqual(attempt["path"], ["prep", "descend", "ascend"])

    def test_unfinished_attempt_is_stalled(self):
        with tempfile.TemporaryDirectory() as td:
            rec = RepDiagnosticRecorder(td)
            rec.record_detector(row(1, progress=0.06))
            rec.close()
            attempt = json.loads(rec.summary_path.read_text(encoding="utf-8"))["attempts"][0]
            self.assertEqual(attempt["result"], "STALLED")
            self.assertEqual(attempt["reason"], "SESSION_ENDED_BEFORE_REP_END")

    def test_write_initialization_failure_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as td:
            blocked = Path(td) / "not_a_directory"
            blocked.write_text("occupied", encoding="utf-8")
            rec = RepDiagnosticRecorder(blocked)
            rec.record_detector(row(1, event="rep_start"))
            rec.record_framing_skip(camera_frame=2, reason="FRAMING_NOT_OK", position_state="test")
            rec.close()
            self.assertFalse(rec.enabled)
            self.assertIsNotNone(rec.error)


if __name__ == "__main__":
    unittest.main()

"""Video/timeline alignment and explicit export regression tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ai_trainer.squat.game_ui.session_recording import SessionRecording, ViewRecorder, json_safe
from scripts.analyze_recorded_session import analyze


class SessionRecordingTests(unittest.TestCase):
    def test_video_timeline_and_export(self):
        with tempfile.TemporaryDirectory() as selected_parent:
            recording = SessionRecording()
            original_dir = recording.directory
            try:
                recorder = ViewRecorder(recording.directory, "front", cv2, camera_index=2)
                for index in range(5):
                    frame = np.full((64, 96, 3), index * 30, dtype=np.uint8)
                    landmarks = np.zeros((33, 4), dtype=float)
                    landmarks[:, 3] = 1.0
                    landmarks[25, 0] = 0.4 if index == 2 else 0.0
                    recorder.record(frame, 20.0, {
                        "world_landmarks": landmarks,
                        "completed_rep": {"rep_index": 0, "predicted_class": "정상"}
                        if index == 4 else None,
                    })
                recorder.close()

                self.assertEqual(list(Path(selected_parent).iterdir()), [])
                saved = recording.export(selected_parent, {"final_result": "정상"})
                saved_again = recording.export(selected_parent, {"final_result": "정상"})
                self.assertNotEqual(saved, saved_again)

                metadata = json.loads((saved / "front.json").read_text(encoding="utf-8"))
                rows = [json.loads(line) for line in (saved / "front.jsonl")
                        .read_text(encoding="utf-8").splitlines()]
                self.assertEqual(metadata["frame_count"], 5)
                self.assertEqual([row["video_frame"] for row in rows], list(range(5)))
                self.assertEqual(rows[4]["completed_rep"]["predicted_class"], "정상")

                capture = cv2.VideoCapture(str(saved / metadata["video_file"]))
                try:
                    count = 0
                    while capture.read()[0]:
                        count += 1
                finally:
                    capture.release()
                self.assertEqual(count, 5)

                report = analyze(saved)
                self.assertEqual(report["views"]["front"]["repetitions"][0]["video_frame"], 4)
                self.assertEqual(report["views"]["front"]["left_knee_spike_candidates"][0]["video_frame"], 2)
                self.assertEqual(report["missing_views"], ["left", "right"])
            finally:
                recording.close()
            self.assertFalse(original_dir.exists())
            self.assertTrue(saved.exists())

    def test_json_safe_uses_null_for_unusable_measurements(self):
        result = json_safe({"x": np.array([np.nan, np.inf, 0.123456789])})
        self.assertEqual(result, {"x": [None, None, 0.123457]})
        self.assertEqual(json.dumps(result, allow_nan=False),
                         '{"x": [null, null, 0.123457]}')

    def test_empty_recording_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as selected_parent:
            recording = SessionRecording()
            try:
                with self.assertRaisesRegex(RuntimeError, "저장할 녹화 영상"):
                    recording.export(selected_parent, {})
            finally:
                recording.close()

    def test_temporary_source_cannot_be_selected_as_destination(self):
        recording = SessionRecording()
        try:
            with self.assertRaisesRegex(ValueError, "임시 녹화 폴더"):
                recording.export(recording.directory, {})
        finally:
            recording.close()


if __name__ == "__main__":
    unittest.main()

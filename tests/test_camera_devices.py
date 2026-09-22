from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.core.s1_capture.camera_devices import CameraDevice, default_camera_index  # noqa: E402


class CameraDeviceTests(unittest.TestCase):
    def test_default_index_uses_nonnegative_environment_value(self) -> None:
        with patch.dict(os.environ, {"AI_TRAINER_CAMERA_INDEX": "3"}):
            self.assertEqual(default_camera_index(), 3)
        with patch.dict(os.environ, {"AI_TRAINER_CAMERA_INDEX": "invalid"}):
            self.assertEqual(default_camera_index(), 0)
        with patch.dict(os.environ, {"AI_TRAINER_CAMERA_INDEX": "-2"}):
            self.assertEqual(default_camera_index(), 0)

    def test_button_label_exposes_index_and_verification_state(self) -> None:
        self.assertEqual(CameraDevice(2, "USB Camera").button_label, "카메라 2 · USB Camera")
        self.assertIn("기본값", CameraDevice(0, "OpenCV 영상 장치", verified=False).button_label)


if __name__ == "__main__":
    unittest.main()

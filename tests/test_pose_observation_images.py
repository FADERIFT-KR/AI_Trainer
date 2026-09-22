"""Synthetic observation/image linkage; no webcam, game, or existing data writes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt5.QtWidgets import QApplication
from ai_trainer.game_ui.pose_diagnostics import PoseDiagnostics
from ai_trainer.live_pose.render import draw_2d_pose
from scripts.view_pose_trace import load_trace, load_observation_image, create_window
from tests.test_pose_trace_viewer import record


class ObservationImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_exact_pixels_ids_overlay_and_navigation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '한글 진단.jsonl'
            original = np.zeros((60, 100, 3), np.uint8)
            original[:, :30] = [10, 30, 210]  # Asymmetric: detect accidental mirroring.
            original[:, 70:] = [220, 70, 5]
            landmarks = np.tile([.5, .5, 0, 1], (33, 1))
            landmarks[23, :2] = [.2, .3]
            landmarks[25, :2] = [.7, .8]
            saved = original.copy()
            overlay = draw_2d_pose(original, landmarks)
            np.testing.assert_array_equal(original, saved)
            self.assertFalse(np.array_equal(original, overlay))
            trace = PoseDiagnostics(path)
            try:
                for ident in (7, 11):
                    data = record()
                    data.update(schema_version=3, image_size=[100, 60], image_landmarks=landmarks,
                                timestamp=ident / 10, sample_index=ident * 3)
                    trace.write_observation(original, observation_id=ident, mirrored=True, **data)
            finally:
                trace.close()
            rows = load_trace(path)
            for ident, row in zip((7, 11), rows):
                self.assertEqual(row.record['observation_id'], ident)
                self.assertEqual(row.record['observation_image']['observation_id'], ident)
                self.assertIn(f'{ident:08d}', row.record['observation_image']['path'])
                loaded, message = load_observation_image(row.record, path)
                np.testing.assert_array_equal(loaded, original)
                self.assertIn('True', message)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            window = create_window(rows, path)
            window.index.setValue(1)
            self.assertIn('observation_id=11', window.image_info.text())
            for label in window.camera_images:
                self.assertFalse(label.pixmap().isNull())
            window.close()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                PoseDiagnostics(path)

    def test_collision_and_failed_image_do_not_append_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            trace = PoseDiagnostics(path)
            image = np.zeros((2, 3, 3), np.uint8)
            try:
                with patch('cv2.imencode', return_value=(False, None)):
                    with self.assertRaisesRegex(RuntimeError, '저장 실패'):
                        trace.write_observation(image, observation_id=0, mirrored=False, image_size=[3, 2])
                self.assertEqual(path.read_text(), '')
                trace.image_dir.mkdir()
                collision = trace.image_dir / 'observation_00000000.png'
                collision.write_bytes(b'preserve')
                with self.assertRaisesRegex(RuntimeError, '저장 실패'):
                    trace.write_observation(image, observation_id=0, mirrored=False, image_size=[3, 2])
                self.assertEqual(collision.read_bytes(), b'preserve')
                self.assertEqual(path.read_text(), '')
            finally:
                trace.close()

    def test_json_failure_is_explicit_and_may_leave_orphan_image(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = PoseDiagnostics(Path(directory) / 'trace.jsonl')
            try:
                with patch.object(trace, 'write', side_effect=OSError('disk full')):
                    with self.assertRaisesRegex(RuntimeError, '세션을 중단'):
                        trace.write_observation(np.zeros((2, 3, 3), np.uint8), observation_id=1,
                                                mirrored=False, image_size=[3, 2])
                self.assertEqual(len(list(trace.image_dir.glob('*.png'))), 1)
            finally:
                trace.close()

    def test_legacy_and_mismatched_image_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            self.assertIsNone(load_observation_image(record(), path)[0])
            data = {'observation_id': 1, 'observation_image': {'observation_id': 2}}
            self.assertIn('불일치', load_observation_image(data, path)[1])
            data['observation_image'].update(observation_id=1, path='../outside.png')
            self.assertIn('폴더 밖', load_observation_image(data, path)[1])

    def test_coordinate_only_writer_creates_no_images(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            trace = PoseDiagnostics(path)
            try:
                trace.write(**record())
                self.assertFalse(trace.image_dir.exists())
            finally:
                trace.close()
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_modified_or_missing_png_is_not_displayed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            trace = PoseDiagnostics(path)
            try:
                trace.write_observation(np.zeros((2, 3, 3), np.uint8), observation_id=0,
                                        mirrored=False, image_size=[3, 2])
            finally:
                trace.close()
            data = json.loads(path.read_text(encoding='utf-8'))
            image_path = path.parent / data['observation_image']['path']
            image_path.write_bytes(b'corrupt synthetic fixture')
            self.assertIn('SHA-256', load_observation_image(data, path)[1])
            data['observation_image']['path'] = 'missing.png'
            self.assertIsNone(load_observation_image(data, path)[0])


if __name__ == '__main__':
    unittest.main()

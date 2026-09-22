"""Offline viewer checks; only synthetic fixtures in temporary directories are written."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication
from scripts.view_pose_trace import load_trace, metrics, observation, create_window


def record():
    world = np.zeros((33, 4))
    world[:, 3] = 1
    world[11, :3], world[12, :3] = [-.2, 1, 0], [.2, 1, 0]
    for hip, knee, ankle, x in ((23, 25, 27, -.2), (24, 26, 28, .2)):
        world[hip, :3] = [x, .5, 0]
        world[knee, :3] = [x, 0, 0]
        world[ankle, :3] = [x, -.5, 0]
    raw = observation({"world_landmarks": world.tolist()}).raw
    return dict(world_landmarks=world.tolist(), common3d=raw.tolist(), aligned_frame=None,
                timestamp=12.34, sample_index=7, phase="prep", frozen3d=[False]*18)


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_mapping_and_metrics(self):
        row = observation(record())
        np.testing.assert_allclose(row.raw[16], [0, .5, 0])
        np.testing.assert_allclose(row.raw[17], [0, 1, 0])
        # Neck is on the midline, not directly above either lateral hip.
        hip_angle = 180 - np.degrees(np.arctan(.2/.5))
        np.testing.assert_allclose(metrics(row.raw), [180, hip_angle, .5, 180, hip_angle, .5])
        bent = row.common.copy()
        bent[11] += [0, .5, .5]
        self.assertAlmostEqual(metrics(bent)[3], 90)

    def test_invalid_stage_is_not_replaced(self):
        data = record()
        data['common3d'][0][0] = None
        row = observation(data)
        self.assertIsNone(row.common)
        self.assertIsNotNone(row.raw)
        self.assertIsNone(row.aligned)
        self.assertTrue(np.isnan(metrics(row.common)).all())

    def test_degenerate_angles_are_unavailable(self):
        self.assertTrue(np.isnan(metrics(np.zeros((18, 3)))[0]))

    def test_empty_corrupt_and_wrong_record_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '손상 기록.jsonl'
            for contents, message in (('', '빈 JSONL'), ('{}\n{bad\n', '2행 JSON'),
                                      ('{}\n\n', '2행이 비어'), ('[]\n', '객체')):
                path.write_text(contents, encoding='utf-8')
                with self.assertRaisesRegex(ValueError, message):
                    load_trace(path)

    def test_unicode_path_navigation_render_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '한글 공백 기록.jsonl'
            first, second = record(), record()
            second['aligned_frame'] = second['common3d']
            second['sample_index'] = 13
            second['frozen3d'][9] = True
            path.write_text('\n'.join(json.dumps(r) for r in (first, second)), encoding='utf-8')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = load_trace(path)
            window = create_window(rows, path)
            self.assertFalse(window.images[0].pixmap().isNull())
            self.assertIn('표시 불가', window.images[2].text())
            window.index.setValue(1)
            self.assertIn('sample_index=13', window.info.text())
            self.assertIn('RKnee', window.info.text())
            self.assertFalse(window.images[2].pixmap().isNull())
            window.index.setValue(0)
            self.assertIn('표시 불가', window.images[2].text())
            window.close()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()

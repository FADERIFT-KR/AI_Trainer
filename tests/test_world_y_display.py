"""Read-only regression checks on existing traces; no screenshots/files written."""
import hashlib
import json
import os
from pathlib import Path
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt5.QtWidgets import QApplication
from scripts.view_pose_trace import load_trace, create_window

ROOT = Path(__file__).resolve().parents[1]


def visual(label):
    pix = label.pixmap()
    return pix.toImage() if pix is not None else label.text()


class WorldYDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def check_trace(self, path):
        if not path.exists():
            self.skipTest(f'Local fixture unavailable: {path}')
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = load_trace(path)
        window = create_window(rows, path)
        try:
            for index in (0, 24):
                window.flip_world_y.setChecked(False)
                window.index.setValue(index)
                record_before = json.dumps(rows[index].record, sort_keys=True)
                image_paths = [path.parent / rows[index].record['observation_image']['path']] if 'observation_image' in rows[index].record else []
                hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in image_paths]
                pictures = [visual(x) for x in window.images + window.camera_images]
                stats = [window.stats.item(i,j).text() for i in range(6) for j in range(4)]
                length = window.length_info.text()
                for coords in (rows[index].raw, rows[index].common):
                    for axes in ([0,1], [2,1]):
                        a = window.raw_view.transform(coords[:, axes])
                        b = window.flipped_raw_view.transform(coords[:, axes])
                        np.testing.assert_array_equal(a[:,0], b[:,0])
                        np.testing.assert_allclose(a[:,1]+b[:,1], 300)
                window.flip_world_y.setChecked(True)
                changed = [visual(x) for x in window.images + window.camera_images]
                for j in (0,1,5):
                    self.assertNotEqual(pictures[j], changed[j])
                for j in (2,3,4):
                    self.assertEqual(pictures[j], changed[j])
                self.assertEqual(changed[0], changed[5])
                self.assertEqual(stats, [window.stats.item(i,j).text() for i in range(6) for j in range(4)])
                self.assertEqual(length, window.length_info.text())
                self.assertEqual(record_before, json.dumps(rows[index].record, sort_keys=True))
                window.flip_world_y.setChecked(False)
                self.assertEqual(pictures, [visual(x) for x in window.images + window.camera_images])
                self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in image_paths])
        finally:
            window.close()
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_image_trace_rows_zero_and_24(self):
        self.check_trace(ROOT / 'output/diagnostics/capture_5403f09a618a4d778b50d9f667885a44/trace.jsonl')

    def test_legacy_trace_rows_zero_and_24(self):
        self.check_trace(ROOT / 'output/diagnostics/squat_pose_20260920_230655_706.jsonl')

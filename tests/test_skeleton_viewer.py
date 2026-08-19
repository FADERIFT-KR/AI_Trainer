import unittest

import numpy as np

from src.skeleton_viewer import fit_panel, normalized_to_pixel


class SkeletonViewerTests(unittest.TestCase):
    def test_normalized_coordinate_conversion(self) -> None:
        self.assertEqual(normalized_to_pixel(0.5, 0.25, 640, 480), (320, 120))

    def test_fit_panel_preserves_requested_size(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        panel = fit_panel(frame, width=300, height=200)
        self.assertEqual(panel.shape, (200, 300, 3))


if __name__ == "__main__":
    unittest.main()

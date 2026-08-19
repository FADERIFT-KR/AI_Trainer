import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.reference_player import AnnotationRange, load_annotation_range, load_keypoints


class ReferencePlayerTests(unittest.TestCase):
    def test_load_keypoints(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "points.csv"
            path.write_text("image_filename,LHip_x,LHip_y\n0.jpg,10.5,20.25\n", encoding="utf-8")
            frames = load_keypoints(path)
        self.assertEqual(frames, [{"LHip": (10.5, 20.25)}])

    def test_load_valid_annotation(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.json"
            path.write_text(json.dumps({"annotations": [{"start_frame": 53, "end_frame": 201}]}),
                            encoding="utf-8")
            value = load_annotation_range(path)
        self.assertEqual(value, AnnotationRange(53, 201))
        self.assertTrue(value.contains(53))
        self.assertTrue(value.contains(201))
        self.assertFalse(value.contains(52))

    def test_load_range_from_damaged_json(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.json"
            path.write_text('{"broken": "text, "start_frame": 4, "end_frame": 9}', encoding="utf-8")
            value = load_annotation_range(path)
        self.assertEqual(value, AnnotationRange(4, 9))


if __name__ == "__main__":
    unittest.main()

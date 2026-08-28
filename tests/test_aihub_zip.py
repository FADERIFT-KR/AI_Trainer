"""Tests for AI Hub label access from both zip and extracted directories."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.aihub_zip import AiHubZip, JOINT_NAMES, SequenceKey


SEQUENCE = SequenceKey("정상", "초급", "CB01", "1")
VIRTUAL_BASE = SEQUENCE.base_dir


def _csv_bytes(dimensions: int) -> bytes:
    output = io.StringIO(newline="")
    fieldnames = ["image_filename"] + [
        f"{joint}_{axis}"
        for joint in JOINT_NAMES
        for axis in ("x", "y", "z")[:dimensions]
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    row: dict[str, str | float] = {"image_filename": "frame_0001.jpg"}
    for joint_index, joint in enumerate(JOINT_NAMES):
        row[f"{joint}_x"] = joint_index + 0.1
        row[f"{joint}_y"] = joint_index + 0.2
        if dimensions == 3:
            row[f"{joint}_z"] = joint_index + 0.3
    writer.writerow(row)
    return output.getvalue().encode("utf-8-sig")


def _fixture_files() -> dict[str, bytes]:
    return {
        f"{VIRTUAL_BASE}/3d_points.csv": _csv_bytes(3),
        f"{VIRTUAL_BASE}/camera1/local_keypoints/Motion2-2.csv": _csv_bytes(2),
        f"{VIRTUAL_BASE}/camera1/video/annotation.json": json.dumps(
            {"annotations": [{"start_frame": 0, "end_frame": 0}]},
            ensure_ascii=False,
        ).encode("utf-8-sig"),
    }


class AiHubDataSourceContract:
    source: Path

    def assert_source_contract(self) -> None:
        with AiHubZip(self.source) as data:
            self.assertEqual(data.find_sequences(), [SEQUENCE])
            self.assertEqual(data.list_cameras(SEQUENCE), [1])

            frames_3d, coords_3d = data.read_3d(SEQUENCE)
            self.assertEqual(frames_3d, ["frame_0001.jpg"])
            self.assertEqual(coords_3d.shape, (1, len(JOINT_NAMES), 3))
            np.testing.assert_allclose(coords_3d[0, 0], [0.1, 0.2, 0.3])

            frames_2d, coords_2d = data.read_2d(SEQUENCE, 1)
            self.assertEqual(frames_2d, ["frame_0001.jpg"])
            self.assertEqual(coords_2d.shape, (1, len(JOINT_NAMES), 2))
            np.testing.assert_allclose(coords_2d[0, -1], [25.1, 25.2])

            self.assertEqual(
                data.read_annotation(SEQUENCE, 1),
                {"annotations": [{"start_frame": 0, "end_frame": 0}]},
            )


class ExtractedDirectoryTests(AiHubDataSourceContract, unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.source = Path(self._temp_dir.name)
        for virtual_path, content in _fixture_files().items():
            relative_path = Path(*virtual_path.split("/")[2:])
            destination = self.source / "에어스쿼트" / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def test_dataset_root_uses_the_same_reading_contract_as_zip(self) -> None:
        self.assert_source_contract()

    def test_air_squat_directory_itself_is_accepted(self) -> None:
        self.source = self.source / "에어스쿼트"
        self.assert_source_contract()

    def test_directory_without_air_squat_data_is_rejected(self) -> None:
        empty_dir = self.source / "empty"
        empty_dir.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "에어스쿼트 디렉터리"):
            AiHubZip(empty_dir)


class ZipCompatibilityTests(AiHubDataSourceContract, unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.source = Path(self._temp_dir.name) / "TL.zip"
        with zipfile.ZipFile(self.source, "w") as archive:
            for virtual_path, content in _fixture_files().items():
                archive.writestr(virtual_path, content)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def test_existing_zip_contract_is_preserved(self) -> None:
        self.assert_source_contract()


if __name__ == "__main__":
    unittest.main()

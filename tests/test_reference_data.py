"""Tests for the AI Hub air-squat reference-data pipeline.

All source data used here is synthetic.  The fixtures mirror the relevant
AI Hub JSON and 3-D CSV fields without redistributing any dataset records.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO
from io import StringIO
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_trainer.reference.builder import BuildConfig, BuildError, build_reference
from ai_trainer.reference.demo import create_demo_preview
from ai_trainer.reference.io import (
    DataFormatError,
    SkeletonSequence,
    discover_source_pairs,
    is_3d_keypoint_csv,
    iter_normal_air_squat_annotations,
    load_metadata,
    read_3d_keypoints,
    read_pair_manifest,
    slice_annotation,
)
from ai_trainer.reference.processing import (
    ProcessingError,
    aggregate_repetitions,
    normalize_skeleton,
    process_sequence,
    segment_repetitions,
)
from ai_trainer.reference.runner import main as run_reference
from ai_trainer.reference.schema import (
    AIHUB_JOINTS,
    COMMON_JOINTS,
    FEATURE_NAMES,
    is_normal_air_squat,
    normalize_label,
)
from ai_trainer.reference.visualization import VisualizationError, render_reference_preview


def squat_depths(repetitions: int = 1, *, cycle_frames: int = 61) -> np.ndarray:
    """Return complete, separated standing-squat-standing depth cycles."""

    standing = np.zeros(15, dtype=np.float64)
    cycle = np.sin(np.linspace(0.0, math.pi, cycle_frames, dtype=np.float64)) ** 2
    pieces = [standing]
    for index in range(repetitions):
        pieces.append(cycle)
        if index + 1 < repetitions:
            pieces.append(standing)
    pieces.append(standing)
    return np.concatenate(pieces)


def synthetic_skeleton(depths: np.ndarray) -> np.ndarray:
    """Create a non-degenerate 26-joint skeleton for each squat depth."""

    frames: list[np.ndarray] = []
    for raw_depth in np.asarray(depths, dtype=np.float64):
        depth = float(raw_depth)
        hip_y = 1.00 - 0.42 * depth
        knee_y = 0.52 - 0.12 * depth
        knee_z = 0.06 + 0.32 * depth
        trunk_z = 0.18 * depth

        joints: dict[str, tuple[float, float, float]] = {
            "Hip": (0.00, hip_y, 0.00),
            "LHip": (0.20, hip_y, 0.00),
            "RHip": (-0.20, hip_y, 0.00),
            "LKnee": (0.20, knee_y, knee_z),
            "RKnee": (-0.20, knee_y, knee_z),
            "LAnkle": (0.20, 0.00, 0.00),
            "RAnkle": (-0.20, 0.00, 0.00),
            "LBigToe": (0.20, -0.03, 0.25),
            "RBigToe": (-0.20, -0.03, 0.25),
            "LSmallToe": (0.15, -0.03, 0.23),
            "RSmallToe": (-0.15, -0.03, 0.23),
            "LHeel": (0.20, 0.00, -0.10),
            "RHeel": (-0.20, 0.00, -0.10),
            "Neck": (0.00, hip_y + 0.75, trunk_z),
            "LShoulder": (0.28, hip_y + 0.72, trunk_z),
            "RShoulder": (-0.28, hip_y + 0.72, trunk_z),
            "LElbow": (0.38, hip_y + 0.48, trunk_z + 0.03),
            "RElbow": (-0.38, hip_y + 0.48, trunk_z + 0.03),
            "LWrist": (0.40, hip_y + 0.25, trunk_z + 0.10),
            "RWrist": (-0.40, hip_y + 0.25, trunk_z + 0.10),
            "Head": (0.00, hip_y + 0.91, trunk_z + 0.01),
            "Nose": (0.00, hip_y + 0.98, trunk_z + 0.04),
            "LEye": (0.035, hip_y + 1.00, trunk_z + 0.035),
            "REye": (-0.035, hip_y + 1.00, trunk_z + 0.035),
            "LEar": (0.075, hip_y + 0.97, trunk_z),
            "REar": (-0.075, hip_y + 0.97, trunk_z),
        }
        frames.append(np.asarray([joints[name] for name in AIHUB_JOINTS], dtype=np.float64))
    return np.stack(frames)


def common_skeleton(depths: np.ndarray) -> np.ndarray:
    native = synthetic_skeleton(depths)
    lookup = {name: index for index, name in enumerate(AIHUB_JOINTS)}
    return native[:, [lookup[name] for name in COMMON_JOINTS], :]


def write_keypoints_csv(
    path: Path,
    depths: np.ndarray,
    *,
    dimensions: int = 3,
    omit_virtual_joints: bool = False,
    first_frame: int = 0,
) -> np.ndarray:
    """Write a synthetic AI Hub-style keypoint CSV and return native points."""

    points = synthetic_skeleton(depths)
    excluded = {"Hip", "Neck"} if omit_virtual_joints else set()
    joints = [joint for joint in AIHUB_JOINTS if joint not in excluded]
    axes = "xyz"[:dimensions]
    fieldnames = ["image_filename"] + [f"{joint}_{axis}" for joint in joints for axis in axes]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        joint_lookup = {name: index for index, name in enumerate(AIHUB_JOINTS)}
        for frame_index, frame in enumerate(points, start=first_frame):
            row: dict[str, object] = {"image_filename": f"frame_{frame_index:06d}.jpg"}
            source_row = frame_index - first_frame
            for joint in joints:
                for axis_index, axis in enumerate(axes):
                    row[f"{joint}_{axis}"] = f"{points[source_row, joint_lookup[joint], axis_index]:.9f}"
            writer.writerow(row)
    return points


def normal_annotation(start: int, end: int, **updates: object) -> dict[str, object]:
    annotation: dict[str, object] = {
        "annotation_no": 17,
        "motion_category1": "하체",
        "motion_category2": "에어 스쿼트",
        "motion_category3": "",
        "motion_category4": "정상",
        "start_frame": start,
        "end_frame": end,
    }
    annotation.update(updates)
    return annotation


def write_metadata(
    path: Path,
    annotations: list[object],
    *,
    video_name: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"annotations": annotations}
    if video_name is not None:
        payload["video_name"] = video_name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_pair_manifest(
    path: Path,
    rows: list[tuple[str, str, str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("annotation_json", "keypoints_3d_csv", "annotation_index", "sample_id")
        )
        writer.writerows(rows)


class LabelSelectionTests(unittest.TestCase):
    def test_normalized_korean_and_english_labels_are_selected(self) -> None:
        korean = normal_annotation(0, 10)
        english = normal_annotation(
            0,
            10,
            motion_category2=" AIR-SQUAT ",
            motion_category4="Correct",
        )

        self.assertEqual(normalize_label(" AIR_SQUAT "), "airsquat")
        self.assertTrue(is_normal_air_squat(korean))
        self.assertTrue(is_normal_air_squat(english))

    def test_wrong_exercise_and_abnormal_squat_are_rejected(self) -> None:
        wrong_exercise = normal_annotation(0, 10, motion_category2="데드리프트")
        abnormal = normal_annotation(0, 10, motion_category4="무릎 모임")
        metadata = {
            "annotations": [
                wrong_exercise,
                "not an annotation object",
                normal_annotation(0, 10),
                abnormal,
            ]
        }

        self.assertFalse(is_normal_air_squat(wrong_exercise))
        self.assertFalse(is_normal_air_squat(abnormal))
        self.assertEqual(
            list(iter_normal_air_squat_annotations(metadata)),
            [(2, metadata["annotations"][2])],
        )
        self.assertEqual(list(iter_normal_air_squat_annotations(metadata, 0)), [])
        self.assertEqual(
            list(iter_normal_air_squat_annotations(metadata, 2)),
            [(2, metadata["annotations"][2])],
        )


class InputAndPairingTests(unittest.TestCase):
    def test_2d_csv_is_rejected_and_3d_csv_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            depths = squat_depths()
            two_d = root / "clip_2d.csv"
            three_d = root / "clip_3d.csv"
            write_keypoints_csv(two_d, depths, dimensions=2)
            expected_native = write_keypoints_csv(three_d, depths)

            self.assertFalse(is_3d_keypoint_csv(two_d))
            self.assertTrue(is_3d_keypoint_csv(three_d))
            sequence = read_3d_keypoints(three_d)

            self.assertEqual(sequence.points.shape, (len(depths), len(COMMON_JOINTS), 3))
            np.testing.assert_array_equal(sequence.frame_numbers, np.arange(len(depths)))
            self.assertEqual(sequence.image_filenames[3], "frame_000003.jpg")
            native_lookup = {name: index for index, name in enumerate(AIHUB_JOINTS)}
            common_lookup = {name: index for index, name in enumerate(COMMON_JOINTS)}
            np.testing.assert_allclose(
                sequence.points[:, common_lookup["LKnee"]],
                expected_native[:, native_lookup["LKnee"]],
            )

    def test_reader_reconstructs_virtual_hip_and_neck_midpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "without_virtual_3d.csv"
            write_keypoints_csv(path, squat_depths(), omit_virtual_joints=True)

            sequence = read_3d_keypoints(path)
            index = {name: COMMON_JOINTS.index(name) for name in ("Hip", "LHip", "RHip", "Neck", "LShoulder", "RShoulder")}
            np.testing.assert_allclose(
                sequence.points[:, index["Hip"]],
                (sequence.points[:, index["LHip"]] + sequence.points[:, index["RHip"]]) / 2,
            )
            np.testing.assert_allclose(
                sequence.points[:, index["Neck"]],
                (
                    sequence.points[:, index["LShoulder"]]
                    + sequence.points[:, index["RShoulder"]]
                )
                / 2,
            )

    def test_explicit_manifest_preserves_index_and_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "labels" / "clip.json"
            csv_path = root / "keypoints" / "clip.csv"
            manifest = root / "pairs.csv"
            depths = squat_depths()
            write_metadata(json_path, [normal_annotation(0, len(depths) - 1)])
            write_keypoints_csv(csv_path, depths)
            write_pair_manifest(
                manifest,
                [("labels/clip.json", "keypoints/clip.csv", "0", "athlete-001")],
            )

            pairs = read_pair_manifest(manifest, root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].annotation_json, json_path.resolve())
            self.assertEqual(pairs[0].keypoints_csv, csv_path.resolve())
            self.assertEqual(pairs[0].annotation_index, 0)
            self.assertEqual(pairs[0].sample_id, "athlete-001")
            self.assertEqual(pairs[0].pairing_method, "manifest")

    def test_auto_pairing_filters_labels_and_ignores_2d_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            depths = squat_depths()
            valid_json = root / "session42_annotations.json"
            valid_csv = root / "session42_keypoints_3d.csv"
            write_metadata(
                valid_json,
                [normal_annotation(0, len(depths) - 1)],
                video_name="session42.mp4",
            )
            write_keypoints_csv(valid_csv, depths)
            write_keypoints_csv(root / "session42_keypoints_2d.csv", depths, dimensions=2)
            write_metadata(
                root / "deadlift_annotations.json",
                [normal_annotation(0, len(depths) - 1, motion_category2="데드리프트")],
                video_name="deadlift.mp4",
            )

            result = discover_source_pairs(root)

            self.assertEqual(result.normal_air_squat_json_count, 1)
            self.assertEqual(result.three_dimensional_csv_count, 1)
            self.assertEqual(len(result.pairs), 1)
            self.assertEqual(result.pairs[0].annotation_json, valid_json.resolve())
            self.assertEqual(result.pairs[0].keypoints_csv, valid_csv.resolve())
            self.assertEqual(result.pairs[0].pairing_method, "automatic")
            self.assertEqual(result.unmatched_json, [])
            self.assertEqual(result.ambiguous_json, [])

    def test_auto_pairing_reports_equal_best_matches_as_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            depths = squat_depths()
            annotation_path = root / "clip_annotations.json"
            write_metadata(
                annotation_path,
                [normal_annotation(0, len(depths) - 1)],
                video_name="clip.mp4",
            )
            write_keypoints_csv(root / "clip_keypoints_3d_cam1.csv", depths)
            write_keypoints_csv(root / "clip_keypoints_3d_cam2.csv", depths)

            result = discover_source_pairs(root)

            self.assertEqual(result.pairs, [])
            self.assertEqual(result.ambiguous_json, [annotation_path.name])
            self.assertEqual(result.three_dimensional_csv_count, 2)

    def test_slice_uses_inclusive_source_frame_ids(self) -> None:
        points = common_skeleton(np.zeros(8))
        sequence = SkeletonSequence(
            points=points,
            frame_numbers=np.arange(100, 108),
            image_filenames=tuple(f"frame_{value}.jpg" for value in range(100, 108)),
        )

        sliced = slice_annotation(
            sequence,
            {"start_frame": 102, "end_frame": 105},
            fps=30.0,
        )

        self.assertEqual(sliced.points.shape[0], 4)
        self.assertEqual((sliced.row_start, sliced.row_end), (2, 5))
        self.assertEqual((sliced.source_frame_start, sliced.source_frame_end), (102, 105))


class GeometryAndSegmentationTests(unittest.TestCase):
    def test_normalization_is_translation_scale_and_rotation_invariant(self) -> None:
        source = common_skeleton(squat_depths())
        normalized, valid_ratio = normalize_skeleton(source)
        angle = math.radians(37.0)
        rotation_z = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        angle_x = math.radians(-23.0)
        rotation_x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(angle_x), -math.sin(angle_x)],
                [0.0, math.sin(angle_x), math.cos(angle_x)],
            ]
        )
        rotation = rotation_z @ rotation_x
        transformed = 3.7 * np.einsum("tjc,kc->tjk", source, rotation)
        transformed += np.asarray([12.5, -7.0, 3.25])

        transformed_normalized, transformed_ratio = normalize_skeleton(transformed)

        self.assertEqual(valid_ratio, 1.0)
        self.assertEqual(transformed_ratio, 1.0)
        np.testing.assert_allclose(transformed_normalized, normalized, atol=2e-6, rtol=2e-6)
        hip_index = COMMON_JOINTS.index("Hip")
        np.testing.assert_array_equal(normalized[:, hip_index], 0.0)

    def test_one_repetition_is_phase_aligned_to_exact_bottom_index_50(self) -> None:
        repetitions = process_sequence(
            common_skeleton(squat_depths(1)),
            target_frames=101,
            min_flexion_deg=25.0,
            min_frames=15,
            min_valid_ratio=1.0,
            max_missing_gap_frames=0,
        )

        self.assertEqual(len(repetitions), 1)
        repetition = repetitions[0]
        self.assertEqual(repetition.positions.shape, (101, len(COMMON_JOINTS), 3))
        self.assertEqual(repetition.features.shape, (101, len(FEATURE_NAMES)))
        self.assertEqual(int(np.argmin(repetition.features[:, :2].mean(axis=1))), 50)
        self.assertLess(repetition.source_start, repetition.source_bottom)
        self.assertLess(repetition.source_bottom, repetition.source_end)
        self.assertGreater(repetition.flexion_range_deg, 25.0)

        arrays, keep, distances = aggregate_repetitions(repetitions, mad_threshold=4.0)
        self.assertEqual(int(arrays["bottom_index"]), 50)
        self.assertEqual(int(arrays["phase"][50]), 1)
        np.testing.assert_array_equal(keep, [True])
        self.assertEqual(distances.shape, (1,))

    def test_multiple_complete_repetitions_are_segmented_separately(self) -> None:
        repetitions = process_sequence(
            common_skeleton(squat_depths(3)),
            target_frames=101,
            min_flexion_deg=25.0,
            min_frames=15,
            min_valid_ratio=1.0,
            max_missing_gap_frames=0,
        )

        self.assertEqual(len(repetitions), 3)
        bottoms = [item.source_bottom for item in repetitions]
        self.assertEqual(bottoms, sorted(bottoms))
        for repetition in repetitions:
            self.assertEqual(int(np.argmin(repetition.features[:, :2].mean(axis=1))), 50)
            self.assertEqual(repetition.positions.shape[0], 101)


class BuilderIntegrationTests(unittest.TestCase):
    def _make_manifest_build(self, root: Path, repetitions: int = 3) -> BuildConfig:
        input_root = root / "input"
        output = root / "output"
        input_root.mkdir()
        depths = squat_depths(repetitions)
        json_path = input_root / "labels" / "workout_annotations.json"
        csv_path = input_root / "keypoints" / "workout_keypoints_3d.csv"
        manifest = input_root / "pairs.csv"
        write_metadata(
            json_path,
            [normal_annotation(0, len(depths) - 1)],
            video_name="workout.mp4",
        )
        write_keypoints_csv(csv_path, depths)
        write_pair_manifest(
            manifest,
            [("labels/workout_annotations.json", "keypoints/workout_keypoints_3d.csv", "0", "synthetic")],
        )
        return BuildConfig(
            input_root=input_root,
            output_dir=output,
            pairs_manifest=manifest,
            target_frames=101,
            min_frames=15,
            min_flexion_deg=25.0,
            min_repetitions=repetitions,
        )

    def test_end_to_end_build_writes_expected_artifacts_and_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._make_manifest_build(root)

            result = build_reference(config)

            self.assertEqual(result.discovered_pairs, 1)
            self.assertEqual(result.processed_annotations, 1)
            self.assertEqual(result.total_repetitions, 3)
            self.assertEqual(result.accepted_repetitions, 3)
            self.assertEqual(result.rejected_repetitions, 0)
            self.assertEqual(result.skipped_sources, 0)
            expected_files = {
                "reference.npz",
                "repetitions.npz",
                "manifest.csv",
                "metadata.json",
                "build_report.json",
                "reference_preview.png",
            }
            self.assertEqual({path.name for path in config.output_dir.iterdir()}, expected_files)

            preview = config.output_dir / "reference_preview.png"
            self.assertTrue(preview.is_file())
            self.assertGreater(preview.stat().st_size, 8)
            self.assertEqual(preview.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

            with np.load(config.output_dir / "reference.npz") as reference:
                self.assertEqual(reference["positions_median"].shape, (101, len(COMMON_JOINTS), 3))
                self.assertEqual(reference["positions_q10"].shape, (101, len(COMMON_JOINTS), 3))
                self.assertEqual(reference["positions_q90"].shape, (101, len(COMMON_JOINTS), 3))
                self.assertEqual(reference["features_median"].shape, (101, len(FEATURE_NAMES)))
                self.assertEqual(reference["features_q10"].shape, (101, len(FEATURE_NAMES)))
                self.assertEqual(reference["features_q90"].shape, (101, len(FEATURE_NAMES)))
                self.assertEqual(reference["phase"].shape, (101,))
                self.assertEqual(int(reference["bottom_index"]), 50)
                self.assertEqual(int(reference["accepted_repetition_count"]), 3)
                self.assertEqual(int(reference["total_repetition_count"]), 3)
                self.assertEqual(reference["joint_names"].tolist(), list(COMMON_JOINTS))
                self.assertEqual(reference["feature_names"].tolist(), list(FEATURE_NAMES))

            with np.load(config.output_dir / "repetitions.npz") as repetitions:
                self.assertEqual(
                    repetitions["positions"].shape,
                    (3, 101, len(COMMON_JOINTS), 3),
                )
                self.assertEqual(
                    repetitions["features"].shape,
                    (3, 101, len(FEATURE_NAMES)),
                )
                self.assertEqual(repetitions["repetition_ids"].shape, (3,))
                np.testing.assert_array_equal(repetitions["accepted"], np.ones(3, dtype=bool))

            with (config.output_dir / "manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["pairing_method"] for row in rows}, {"manifest"})
            self.assertEqual({row["target_frames"] for row in rows}, {"101"})
            self.assertEqual({row["accepted"] for row in rows}, {"true"})
            self.assertEqual({len(row["annotation_sha256"]) for row in rows}, {64})
            self.assertEqual({len(row["keypoints_sha256"]) for row in rows}, {64})
            self.assertEqual(
                [row["repetition_id"] for row in rows],
                ["synthetic-a000-r01", "synthetic-a000-r02", "synthetic-a000-r03"],
            )

            metadata = json.loads((config.output_dir / "metadata.json").read_text(encoding="utf-8"))
            report = json.loads((config.output_dir / "build_report.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source"]["dataset_id"], 71422)
            self.assertEqual(metadata["selection"]["motion"], "에어 스쿼트")
            self.assertEqual(metadata["selection"]["total_repetitions"], 3)
            self.assertEqual(report["discovery"], {"mode": "manifest"})
            self.assertEqual(report["processed_annotations"], 1)

    def test_duplicate_camera_views_of_same_csv_interval_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            input_root.mkdir()
            depths = squat_depths(1)
            shared_csv = input_root / "shared_keypoints_3d.csv"
            first_json = input_root / "camera1_annotations.json"
            second_json = input_root / "camera2_annotations.json"
            manifest = input_root / "pairs.csv"
            annotation = normal_annotation(0, len(depths) - 1)
            write_keypoints_csv(shared_csv, depths)
            write_metadata(first_json, [annotation], video_name="camera1.mp4")
            write_metadata(second_json, [annotation], video_name="camera2.mp4")
            write_pair_manifest(
                manifest,
                [
                    (first_json.name, shared_csv.name, "0", "view-1"),
                    (second_json.name, shared_csv.name, "0", "view-2"),
                ],
            )

            result = build_reference(
                BuildConfig(
                    input_root=input_root,
                    output_dir=root / "output",
                    pairs_manifest=manifest,
                    min_repetitions=1,
                )
            )

            self.assertEqual(result.discovered_pairs, 2)
            self.assertEqual(result.processed_annotations, 1)
            self.assertEqual(result.total_repetitions, 1)
            with (root / "output" / "manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["repetition_id"], "view-1-a000-r01")

    def test_existing_artifacts_require_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._make_manifest_build(root, repetitions=1)
            build_reference(config)

            with self.assertRaisesRegex(BuildError, "--overwrite"):
                build_reference(config)

            overwritten = build_reference(
                BuildConfig(**{**config.__dict__, "overwrite": True, "include_repetitions": False})
            )
            self.assertEqual(overwritten.total_repetitions, 1)
            self.assertFalse((config.output_dir / "repetitions.npz").exists())

    def test_preview_plot_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._make_manifest_build(root, repetitions=1)
            preview = config.output_dir / "reference_preview.png"

            build_reference(config)
            self.assertTrue(preview.is_file())

            build_reference(
                BuildConfig(
                    **{**config.__dict__, "include_plot": False, "overwrite": True}
                )
            )

            self.assertFalse(preview.exists())


class PythonRunnerTests(unittest.TestCase):
    def test_create_demo_preview_writes_png_to_requested_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "nested" / "demo.png"

            result = create_demo_preview(destination)

            self.assertEqual(result, destination.resolve())
            self.assertTrue(destination.is_file())
            self.assertGreater(destination.stat().st_size, 8)
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_zero_argument_run_falls_back_to_demo_and_can_be_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            preview = (
                project_root
                / "data"
                / "reference"
                / "demo"
                / "reference_preview.png"
            )

            for _ in range(2):
                output = StringIO()
                with redirect_stdout(output):
                    exit_code = run_reference([], project_root=project_root)

                self.assertEqual(exit_code, 0)
                self.assertIn("synthetic", output.getvalue().casefold())
                self.assertTrue(preview.is_file())
                self.assertEqual(preview.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

            self.assertFalse(
                (project_root / "data" / "reference" / "air_squat").exists()
            )

    def test_explicit_invalid_build_args_do_not_fall_back_to_demo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            missing_input = project_root / "missing-input"
            build_output = project_root / "build-output"
            demo_preview = (
                project_root
                / "data"
                / "reference"
                / "demo"
                / "reference_preview.png"
            )
            stdout = StringIO()
            stderr = StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = run_reference(
                    [
                        "--input",
                        str(missing_input),
                        "--output",
                        str(build_output),
                    ],
                    project_root=project_root,
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("Input directory not found", stderr.getvalue())
            self.assertFalse(demo_preview.exists())
            self.assertFalse(build_output.exists())


class FailureTests(unittest.TestCase):
    def test_preview_rejects_nonfinite_positions(self) -> None:
        positions = common_skeleton(np.zeros(3, dtype=np.float64))
        positions[1, 0, 0] = np.nan

        with self.assertRaisesRegex(VisualizationError, "NaN or infinite"):
            render_reference_preview(positions, COMMON_JOINTS, 1, BytesIO())

    def test_metadata_and_csv_schema_failures_are_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid_json = root / "invalid.json"
            invalid_json.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(DataFormatError, "root must be an object"):
                load_metadata(invalid_json)

            missing_annotations = root / "missing.json"
            missing_annotations.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(DataFormatError, "annotations list"):
                load_metadata(missing_annotations)

            incomplete_csv = root / "incomplete.csv"
            incomplete_csv.write_text("Nose_x,Nose_y,Nose_z\n0,1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(DataFormatError, "Missing required 3-D columns"):
                read_3d_keypoints(incomplete_csv)

    def test_invalid_slice_and_incomplete_motion_fail(self) -> None:
        sequence = SkeletonSequence(
            points=common_skeleton(np.zeros(20)),
            frame_numbers=np.full(20, -1, dtype=np.int64),
            image_filenames=tuple("" for _ in range(20)),
        )
        with self.assertRaisesRegex(DataFormatError, "Invalid frame interval"):
            slice_annotation(
                sequence,
                {"start_frame": 10, "end_frame": 2},
                fps=30.0,
            )

        standing_features = np.zeros((30, len(FEATURE_NAMES)), dtype=np.float32)
        standing_features[:, :2] = 175.0
        with self.assertRaisesRegex(ProcessingError, "Knee-flexion range"):
            segment_repetitions(standing_features, min_flexion_deg=25.0, min_frames=15)

    def test_missing_joint_quality_limits_are_enforced(self) -> None:
        source = common_skeleton(squat_depths(1))
        short_gap = source.copy()
        short_gap[20, COMMON_JOINTS.index("LWrist"), 0] = np.nan
        with self.assertRaisesRegex(ProcessingError, "Valid joint ratio"):
            process_sequence(
                short_gap,
                target_frames=101,
                min_flexion_deg=25.0,
                min_frames=15,
                min_valid_ratio=1.0,
                max_missing_gap_frames=1,
            )

        long_gap = source.copy()
        long_gap[20:23, COMMON_JOINTS.index("LWrist"), 0] = np.nan
        with self.assertRaisesRegex(ProcessingError, "3-frame gap"):
            normalize_skeleton(long_gap, max_missing_gap_frames=2)

    def test_degenerate_skeleton_and_bad_manifest_fail(self) -> None:
        with self.assertRaisesRegex(ProcessingError, "Body scale"):
            normalize_skeleton(np.zeros((20, len(COMMON_JOINTS), 3), dtype=np.float64))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "pairs.csv"
            manifest.write_text("annotation_json,keypoints_3d_csv\nmissing.json,missing.csv\n", encoding="utf-8")
            with self.assertRaisesRegex(DataFormatError, "JSON not found"):
                read_pair_manifest(manifest, root)

    def test_builder_rejects_no_pair_insufficient_repetitions_and_bad_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            input_root.mkdir()
            depths = squat_depths(1)
            write_metadata(
                input_root / "only_annotations.json",
                [normal_annotation(0, len(depths) - 1)],
            )
            write_keypoints_csv(input_root / "only_2d.csv", depths, dimensions=2)

            with self.assertRaisesRegex(BuildError, "No unambiguous"):
                build_reference(BuildConfig(input_root=input_root, output_dir=root / "none"))

            csv_path = input_root / "single_keypoints_3d.csv"
            manifest = input_root / "pairs.csv"
            write_keypoints_csv(csv_path, depths)
            write_pair_manifest(
                manifest,
                [("only_annotations.json", "single_keypoints_3d.csv", "0", "single")],
            )
            with self.assertRaisesRegex(BuildError, "Only 1 complete repetitions"):
                build_reference(
                    BuildConfig(
                        input_root=input_root,
                        output_dir=root / "insufficient",
                        pairs_manifest=manifest,
                        min_repetitions=2,
                    )
                )

            invalid_configs = (
                (BuildConfig(input_root=input_root, output_dir=root / "bad", target_frames=100), "target_frames"),
                (BuildConfig(input_root=input_root, output_dir=root / "bad", fps=0), "fps"),
                (BuildConfig(input_root=input_root, output_dir=root / "bad", min_frames=4), "min_frames"),
                (BuildConfig(input_root=input_root, output_dir=root / "bad", min_repetitions=0), "min_repetitions"),
                (BuildConfig(input_root=input_root, output_dir=root / "bad", min_valid_ratio=0), "min_valid_ratio"),
                (BuildConfig(input_root=input_root, output_dir=root / "bad", max_missing_gap_frames=-1), "max_missing_gap_frames"),
            )
            for config, message in invalid_configs:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(BuildError, message):
                        build_reference(config)


if __name__ == "__main__":
    unittest.main()

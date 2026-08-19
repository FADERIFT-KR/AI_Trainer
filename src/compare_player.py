"""Side-by-side reference video and live webcam pose comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.geometry import joint_angle
from src.pose_tracker import PoseMeasurement, PoseTracker
from src.reference_player import draw_skeleton, load_annotation_range, load_keypoints


FEATURE_NAMES = ("left knee", "right knee", "left hip", "right hip")


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Comparison:
    similarity: float
    mean_error: float
    largest_error_name: str
    largest_error: float


def reference_measurement(points: dict[str, tuple[float, float]]) -> PoseMeasurement:
    def point(name: str) -> Point:
        x_value, y_value = points[name]
        return Point(x_value, y_value)

    return PoseMeasurement(
        left_knee_angle=joint_angle(point("LHip"), point("LKnee"), point("LAnkle")),
        right_knee_angle=joint_angle(point("RHip"), point("RKnee"), point("RAnkle")),
        left_hip_angle=joint_angle(point("LShoulder"), point("LHip"), point("LKnee")),
        right_hip_angle=joint_angle(point("RShoulder"), point("RHip"), point("RKnee")),
    )


def compare_measurements(reference: PoseMeasurement, live: PoseMeasurement) -> Comparison:
    reference_values = (
        reference.left_knee_angle, reference.right_knee_angle,
        reference.left_hip_angle, reference.right_hip_angle,
    )
    live_values = (
        live.left_knee_angle, live.right_knee_angle,
        live.left_hip_angle, live.right_hip_angle,
    )
    errors = [abs(expected - actual) for expected, actual in zip(reference_values, live_values)]
    mean_error = sum(errors) / len(errors)
    largest_index = max(range(len(errors)), key=errors.__getitem__)
    # A 90-degree mean difference represents no useful match for these squat features.
    similarity = max(0.0, min(100.0, 100.0 * (1.0 - mean_error / 90.0)))
    return Comparison(similarity, mean_error, FEATURE_NAMES[largest_index], errors[largest_index])


def frame_for_elapsed(elapsed: float, fps: float, frame_count: int, speed: float = 1.0) -> int:
    """Return the looping reference frame for wall-clock playback time."""
    if fps <= 0 or frame_count <= 0 or speed <= 0:
        raise ValueError("fps, frame_count, and speed must be positive")
    return int(elapsed * fps * speed) % frame_count


def fit_panel(frame: np.ndarray, width: int = 640, height: int = 600) -> np.ndarray:
    """Letterbox a frame without changing its aspect ratio."""
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, round(frame.shape[1] * scale))
    resized_height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (resized_width, resized_height))
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - resized_width) // 2
    y_offset = (height - resized_height) // 2
    panel[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return panel


def put_text(frame: np.ndarray, text: str, y: int, color: tuple[int, int, int], scale: float = 0.7) -> None:
    cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def play_and_compare(
    video_path: Path,
    csv_path: Path,
    annotation_path: Path,
    camera_index: int,
    label: str,
) -> int:
    keypoint_frames = load_keypoints(csv_path)
    reference_values = [reference_measurement(points) for points in keypoint_frames]
    annotation = load_annotation_range(annotation_path)
    video = cv2.VideoCapture(str(video_path))
    camera = cv2.VideoCapture(camera_index)
    if not video.isOpened():
        print(f"Could not open reference video: {video_path}")
        return 1
    if not camera.isOpened():
        video.release()
        print(f"Could not open camera {camera_index}.")
        return 1

    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = min(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), len(keypoint_frames))
    if frame_count <= 0:
        video.release()
        camera.release()
        print("The reference video or keypoint CSV has no frames.")
        return 1
    frame_index = -1
    reference_frame = None
    paused = False
    playback_started = perf_counter()
    paused_elapsed = 0.0
    try:
        with PoseTracker() as tracker:
            while True:
                ok_camera, camera_frame = camera.read()
                if not ok_camera:
                    print("Could not read a frame from the camera.")
                    return 1
                camera_frame = cv2.flip(camera_frame, 1)
                results = tracker.process(cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB))
                tracker.draw(camera_frame, results)
                live_value = tracker.measure(results)

                elapsed = paused_elapsed if paused else perf_counter() - playback_started
                desired_index = frame_for_elapsed(elapsed, fps, frame_count)
                if reference_frame is None or desired_index != frame_index:
                    # Seek when analysis falls behind or the short sample loops. This keeps
                    # the reference motion at its true speed instead of slowing with inference.
                    video.set(cv2.CAP_PROP_POS_FRAMES, desired_index)
                    ok_video, next_reference_frame = video.read()
                    if not ok_video:
                        print("Could not read the reference video.")
                        return 1
                    reference_frame = next_reference_frame
                    frame_index = desired_index
                current_index = frame_index

                display_reference = reference_frame.copy()
                draw_skeleton(display_reference, keypoint_frames[current_index])
                ref_value = reference_values[current_index]
                active = annotation.contains(current_index)

                left_panel = fit_panel(display_reference)
                right_panel = fit_panel(camera_frame)
                put_text(left_panel, "REFERENCE DATASET", 32, (0, 220, 255))
                put_text(left_panel, f"Frame {current_index} | {label if active else 'outside label range'}",
                         66, (0, 0, 255) if active else (220, 220, 220), 0.65)
                put_text(left_panel,
                         f"Knee L/R {ref_value.left_knee_angle:.0f}/{ref_value.right_knee_angle:.0f}",
                         100, (255, 255, 255), 0.62)
                put_text(left_panel,
                         f"Hip  L/R {ref_value.left_hip_angle:.0f}/{ref_value.right_hip_angle:.0f}",
                         130, (255, 255, 255), 0.62)

                put_text(right_panel, "LIVE WEBCAM", 32, (0, 255, 0))
                if live_value is None:
                    put_text(right_panel, "Move your full body into view", 68, (0, 0, 255), 0.65)
                else:
                    comparison = compare_measurements(ref_value, live_value)
                    if comparison.similarity >= 85:
                        status, color = "CLOSE MATCH", (0, 255, 0)
                    elif comparison.similarity >= 65:
                        status, color = "ADJUST POSE", (0, 220, 255)
                    else:
                        status, color = "NOT MATCHING", (0, 0, 255)
                    put_text(right_panel, f"Motion similarity: {comparison.similarity:5.1f}%  {status}",
                             68, color, 0.65)
                    put_text(right_panel,
                             f"Largest difference: {comparison.largest_error_name} {comparison.largest_error:.1f} deg",
                             102, color, 0.58)
                    put_text(right_panel,
                             f"Knee L/R {live_value.left_knee_angle:.0f}/{live_value.right_knee_angle:.0f}",
                             136, (255, 255, 255), 0.58)
                    put_text(right_panel,
                             f"Hip  L/R {live_value.left_hip_angle:.0f}/{live_value.right_hip_angle:.0f}",
                             168, (255, 255, 255), 0.58)

                combined = np.hstack((left_panel, right_panel))
                put_text(combined, "SPACE: pause  R: restart  Q: quit", combined.shape[0] - 18,
                         (255, 255, 255), 0.65)
                cv2.imshow("AI Trainer - Reference vs Live", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    if paused:
                        playback_started = perf_counter() - paused_elapsed
                        paused = False
                    else:
                        paused_elapsed = perf_counter() - playback_started
                        paused = True
                if key == ord("r"):
                    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_index = -1
                    reference_frame = None
                    playback_started = perf_counter()
                    paused_elapsed = 0.0
                    paused = False
    finally:
        video.release()
        camera.release()
        cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a live pose with a labelled reference video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--label", default="HIP ERROR SAMPLE")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(play_and_compare(
        args.video, args.keypoints, args.annotation, args.camera, args.label
    ))

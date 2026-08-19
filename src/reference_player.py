"""Play a labelled pose video with CSV keypoints drawn over each frame."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2


Point = tuple[float, float]

# Connections used by the 26-joint AI Hub-style CSV files.
SKELETON_CONNECTIONS = (
    ("LEar", "LEye"), ("LEye", "Nose"), ("Nose", "REye"), ("REye", "REar"),
    ("Head", "Neck"), ("Neck", "LShoulder"), ("Neck", "RShoulder"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LShoulder", "LHip"), ("RShoulder", "RHip"),
    ("LHip", "RHip"), ("Neck", "Hip"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"),
    ("LAnkle", "LHeel"), ("LHeel", "LBigToe"), ("LBigToe", "LSmallToe"),
    ("RAnkle", "RHeel"), ("RHeel", "RBigToe"), ("RBigToe", "RSmallToe"),
)


@dataclass(frozen=True)
class AnnotationRange:
    start_frame: int
    end_frame: int

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


def load_keypoints(path: Path) -> list[dict[str, Point]]:
    """Load x/y joint pairs from a frame-per-row CSV file."""
    frames: list[dict[str, Point]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("The keypoint CSV has no header")
        joints = [name[:-2] for name in reader.fieldnames if name.endswith("_x")]
        if not joints:
            raise ValueError("The keypoint CSV has no *_x columns")
        for row_number, row in enumerate(reader, start=2):
            points: dict[str, Point] = {}
            for joint in joints:
                x_value = row.get(f"{joint}_x")
                y_value = row.get(f"{joint}_y")
                if x_value in (None, "") or y_value in (None, ""):
                    raise ValueError(f"Missing {joint} coordinate at CSV row {row_number}")
                points[joint] = (float(x_value), float(y_value))
            frames.append(points)
    if not frames:
        raise ValueError("The keypoint CSV has no data rows")
    return frames


def load_annotation_range(path: Path) -> AnnotationRange:
    """Read the labelled frame range, tolerating legacy mojibake JSON files."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        data = json.loads(text)
        annotation = data["annotations"][0]
        return AnnotationRange(int(annotation["start_frame"]), int(annotation["end_frame"]))
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        start = re.search(r'"start_frame"\s*:\s*(\d+)', text)
        end = re.search(r'"end_frame"\s*:\s*(\d+)', text)
        if not start or not end:
            raise ValueError("Could not find start_frame/end_frame in annotation JSON")
        return AnnotationRange(int(start.group(1)), int(end.group(1)))


def draw_skeleton(frame: object, points: dict[str, Point]) -> None:
    for first, second in SKELETON_CONNECTIONS:
        if first in points and second in points:
            start = tuple(round(value) for value in points[first])
            end = tuple(round(value) for value in points[second])
            cv2.line(frame, start, end, (0, 220, 255), 3, cv2.LINE_AA)
    for x_value, y_value in points.values():
        cv2.circle(frame, (round(x_value), round(y_value)), 5, (0, 60, 255), -1, cv2.LINE_AA)


def play(video_path: Path, csv_path: Path, annotation_path: Path, label: str) -> int:
    keypoint_frames = load_keypoints(csv_path)
    annotation = load_annotation_range(annotation_path)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print(f"Could not open video: {video_path}")
        return 1

    video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames != len(keypoint_frames):
        print(f"Warning: video has {video_frames} frames, CSV has {len(keypoint_frames)} rows.")

    delay = max(1, round(1000 / (video.get(cv2.CAP_PROP_FPS) or 30)))
    frame_index = 0
    paused = False
    try:
        while frame_index < min(video_frames, len(keypoint_frames)):
            if not paused:
                ok, frame = video.read()
                if not ok:
                    break
                draw_skeleton(frame, keypoint_frames[frame_index])
                active = annotation.contains(frame_index)
                color = (0, 0, 255) if active else (255, 255, 255)
                status = label if active else "OUTSIDE LABELLED RANGE"
                cv2.putText(frame, f"Frame {frame_index}/{video_frames - 1}", (25, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, status, (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                cv2.putText(frame, "SPACE: pause  Q: quit", (25, frame.shape[0] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("AI Trainer - Reference Dataset", frame)
                frame_index += 1
            key = cv2.waitKey(0 if paused else delay) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        video.release()
        cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a reference pose dataset sample")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--label", default="HIP ERROR")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(play(args.video, args.keypoints, args.annotation, args.label))

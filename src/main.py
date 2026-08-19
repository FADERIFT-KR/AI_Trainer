"""Run the first AI Trainer webcam milestone."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2

from src.pose_tracker import PoseMeasurement, PoseTracker


CSV_FIELDS = ("timestamp", "left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time squat pose tracker")
    parser.add_argument("--camera", type=int, default=0, help="webcam device index")
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"), help="CSV directory")
    return parser.parse_args()


def open_recording(output_dir: Path) -> tuple[object, csv.DictWriter]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / datetime.now().strftime("squat_%Y%m%d_%H%M%S.csv")
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    return handle, writer


def write_measurement(writer: csv.DictWriter, elapsed: float, value: PoseMeasurement) -> None:
    writer.writerow({
        "timestamp": f"{elapsed:.3f}",
        "left_knee_angle": f"{value.left_knee_angle:.2f}",
        "right_knee_angle": f"{value.right_knee_angle:.2f}",
        "left_hip_angle": f"{value.left_hip_angle:.2f}",
        "right_hip_angle": f"{value.right_hip_angle:.2f}",
    })


def draw_status(frame: object, measurement: PoseMeasurement | None, recording: bool) -> None:
    lines = (
        (f"Knee  L {measurement.left_knee_angle:5.1f}  R {measurement.right_knee_angle:5.1f}",
         f"Hip   L {measurement.left_hip_angle:5.1f}  R {measurement.right_hip_angle:5.1f}")
        if measurement else ("Move your full body into the camera view",)
    )
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (20, 35 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "REC" if recording else "R: record  Q: quit", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (255, 255, 255), 2)


def run(camera_index: int, output_dir: Path) -> int:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Camera {camera_index} could not be opened.")
        return 1
    recording_handle = None
    writer = None
    started_at = perf_counter()
    try:
        with PoseTracker() as tracker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a frame from the camera.")
                    return 1
                frame = cv2.flip(frame, 1)
                results = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tracker.draw(frame, results)
                measurement = tracker.measure(results)
                if writer and measurement:
                    write_measurement(writer, perf_counter() - started_at, measurement)
                draw_status(frame, measurement, writer is not None)
                cv2.imshow("AI Trainer - Squat Pose", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    if recording_handle:
                        recording_handle.close()
                        recording_handle, writer = None, None
                    else:
                        recording_handle, writer = open_recording(output_dir)
                        started_at = perf_counter()
    finally:
        if recording_handle:
            recording_handle.close()
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(run(arguments.camera, arguments.output_dir))

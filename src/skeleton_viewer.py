"""Show a raw webcam feed beside an extracted pose-coordinate view."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.pose_tracker import PoseTracker


PANEL_WIDTH = 700
PANEL_HEIGHT = 600
SKELETON_AREA_WIDTH = 430
SKELETON_AREA_HEIGHT = 520
KEY_JOINTS = (
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
)


def normalized_to_pixel(x_value: float, y_value: float, width: int, height: int) -> tuple[int, int]:
    """Convert MediaPipe normalized image coordinates to panel pixels."""
    return round(x_value * width), round(y_value * height)


def fit_panel(frame: np.ndarray, width: int = PANEL_WIDTH, height: int = PANEL_HEIGHT) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, round(frame.shape[1] * scale))
    resized_height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (resized_width, resized_height))
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - resized_width) // 2
    y_offset = (height - resized_height) // 2
    panel[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return panel


def put_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
) -> None:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_coordinate_view(results: object, pose_module: object) -> np.ndarray:
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    put_text(panel, "EXTRACTED SKELETON COORDINATES", (18, 28), (0, 255, 255), 0.66)
    put_text(panel, "x/y: normalized image position, z: relative depth", (18, 52), (180, 180, 180), 0.45)
    if not results.pose_landmarks:
        put_text(panel, "Move your full body into the camera view", (85, 300), (0, 0, 255), 0.65)
        return panel

    landmarks = results.pose_landmarks.landmark
    x_origin, y_origin = 15, 62
    for first_index, second_index in pose_module.POSE_CONNECTIONS:
        first = landmarks[first_index]
        second = landmarks[second_index]
        if first.visibility < 0.5 or second.visibility < 0.5:
            continue
        first_pixel = normalized_to_pixel(first.x, first.y, SKELETON_AREA_WIDTH, SKELETON_AREA_HEIGHT)
        second_pixel = normalized_to_pixel(second.x, second.y, SKELETON_AREA_WIDTH, SKELETON_AREA_HEIGHT)
        start = (first_pixel[0] + x_origin, first_pixel[1] + y_origin)
        end = (second_pixel[0] + x_origin, second_pixel[1] + y_origin)
        cv2.line(panel, start, end, (0, 210, 255), 3, cv2.LINE_AA)

    for index, landmark in enumerate(landmarks):
        if landmark.visibility < 0.5:
            continue
        x_pixel, y_pixel = normalized_to_pixel(
            landmark.x, landmark.y, SKELETON_AREA_WIDTH, SKELETON_AREA_HEIGHT
        )
        center = (x_pixel + x_origin, y_pixel + y_origin)
        cv2.circle(panel, center, 4, (0, 80, 255), -1, cv2.LINE_AA)
        put_text(panel, str(index), (center[0] + 4, center[1] - 4), (150, 255, 150), 0.32)

    table_x = 458
    put_text(panel, "JOINT       x      y      z", (table_x, 86), (255, 255, 255), 0.43)
    for row_index, name in enumerate(KEY_JOINTS):
        enum_value = pose_module.PoseLandmark[name]
        value = landmarks[enum_value.value]
        short_name = name.replace("LEFT_", "L_").replace("RIGHT_", "R_")
        put_text(
            panel,
            f"{short_name:<10} {value.x: .2f} {value.y: .2f} {value.z: .2f}",
            (table_x, 112 + row_index * 28),
            (180, 255, 180) if value.visibility >= 0.5 else (100, 100, 100),
            0.39,
        )
    put_text(panel, "MediaPipe landmark IDs", (table_x, 370), (180, 180, 180), 0.42)
    put_text(panel, "11/12 shoulder", (table_x, 396), (180, 180, 180), 0.42)
    put_text(panel, "23/24 hip", (table_x, 420), (180, 180, 180), 0.42)
    put_text(panel, "25/26 knee", (table_x, 444), (180, 180, 180), 0.42)
    put_text(panel, "27/28 ankle", (table_x, 468), (180, 180, 180), 0.42)
    return panel


def run(camera_index: int) -> int:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Could not open camera {camera_index}.")
        return 1
    try:
        with PoseTracker() as tracker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a frame from the camera.")
                    return 1
                frame = cv2.flip(frame, 1)
                results = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                webcam_panel = fit_panel(frame)
                put_text(webcam_panel, "LIVE WEBCAM", (18, 30), (0, 255, 0), 0.72)
                coordinate_panel = draw_coordinate_view(results, tracker.pose_module)
                combined = np.hstack((webcam_panel, coordinate_panel))
                put_text(combined, "Q: quit", (18, combined.shape[0] - 18), (255, 255, 255), 0.6)
                cv2.imshow("AI Trainer - Webcam and Skeleton Coordinates", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show webcam and extracted skeleton coordinates")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(run(arguments.camera))

#!/usr/bin/env python3
"""Replay saved squat videos with 2-D pose overlay and a 3-D skeleton view."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import cv2  # noqa: E402

from ai_trainer.live_pose.core import FrameProcessor  # noqa: E402
from ai_trainer.live_pose.mediapipe_pose import MediaPipePoseDetector  # noqa: E402


VIEW_ORDER = ("front", "left", "right")
DISPLAY_SIZE = (640, 360)


def _videos(session: Path, requested_view: str | None) -> list[tuple[str, Path]]:
    views = (requested_view,) if requested_view else VIEW_ORDER
    videos = [(view, session / f"{view}.mp4") for view in views]
    missing = [str(path) for _, path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError("Saved video not found: " + ", ".join(missing))
    return videos


def replay(session: Path, model: Path, requested_view: str | None) -> int:
    videos = _videos(session, requested_view)
    window_name = "AI Trainer - saved video 2D / estimated 3D skeleton"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_SIZE[0] * 2, DISPLAY_SIZE[1])
    paused = False
    try:
        with MediaPipePoseDetector(model, min_detection_confidence=0.4, min_presence_confidence=0.4, min_tracking_confidence=0.4) as detector:
            processor = FrameProcessor(
                detector,
                mirror=False,
                skeleton_width=DISPLAY_SIZE[0],
                skeleton_height=DISPLAY_SIZE[1],
            )
            for view, path in videos:
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise OSError(f"Could not open saved video: {path}")
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                delay_ms = max(1, round(1000.0 / fps))
                frame_index = 0
                last_panel = None
                try:
                    while True:
                        if not paused:
                            ok, frame = capture.read()
                            if not ok:
                                break
                            processed = processor.process(frame)
                            camera = cv2.resize(processed.video_bgr, DISPLAY_SIZE)
                            skeleton = cv2.resize(processed.skeleton_bgr, DISPLAY_SIZE)
                            cv2.putText(camera, f"SAVED {view.upper()} / 2D POSE", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
                            cv2.putText(skeleton, "MEDIAPIPE WORLD 3D / 33 LANDMARKS", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(skeleton, "SPACE: pause   ESC: close", (14, DISPLAY_SIZE[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1, cv2.LINE_AA)
                            last_panel = cv2.hconcat((camera, skeleton))
                            frame_index += 1
                        if last_panel is None:
                            break
                        cv2.imshow(window_name, last_panel)
                        key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
                        if key in (27, ord("q")):
                            return 0
                        if key == ord(" "):
                            paused = not paused
                finally:
                    capture.release()
    finally:
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True, help="Directory containing front/left/right MP4 files")
    parser.add_argument("--view", choices=VIEW_ORDER, help="Replay only one saved view")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "pose_landmarker_full.task")
    args = parser.parse_args()
    return replay(args.session.resolve(), args.model.resolve(), args.view)


if __name__ == "__main__":
    raise SystemExit(main())

"""Temporary per-view video and frame diagnostics, exported after assessment.

The webcam thread writes its own clip and JSONL timeline.  The UI owns the
temporary directory across front/left/right workers and copies it only after
the user chooses a destination on the result screen.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ai_trainer.camera_views import VIEWS


def json_safe(value: Any) -> Any:
    """Convert numpy diagnostics to strict JSON; missing numbers become null."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


class ViewRecorder:
    """Write the exact undistorted camera input and matching JSONL frame rows."""

    def __init__(self, directory: str | Path, view: str, cv2_module: Any,
                 camera_index: int, calibration_applied: bool = False):
        if view not in VIEWS:
            raise ValueError(f"Unsupported view: {view}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.view = view
        self.cv2 = cv2_module
        self.camera_index = camera_index
        self.calibration_applied = calibration_applied
        self.frame_count = 0
        self.video_path: Path | None = None
        self.timeline_path = self.directory / f"{view}.jsonl"
        self._writer = None
        self._timeline = None
        self._start_clock: float | None = None
        self._start_utc: str | None = None
        self._fps = 0.0
        self._size: tuple[int, int] | None = None

    def _open(self, frame: np.ndarray, measured_fps: float) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Recording frames must be uint8 BGR images")
        height, width = frame.shape[:2]
        if width < 2 or height < 2:
            raise ValueError("Recording frame size is invalid")
        self._size = (width, height)
        self._fps = float(np.clip(measured_fps if np.isfinite(measured_fps) and measured_fps > 0 else 30.0,
                                  5.0, 60.0))
        for extension, codec in ((".mp4", "mp4v"), (".avi", "MJPG")):
            path = self.directory / f"{self.view}{extension}"
            writer = self.cv2.VideoWriter(str(path), self.cv2.VideoWriter_fourcc(*codec),
                                          self._fps, self._size)
            if writer.isOpened():
                self._writer = writer
                self.video_path = path
                break
            writer.release()
            if path.exists():
                path.unlink()
        if self._writer is None:
            raise RuntimeError("녹화용 MP4/AVI 코덱을 열 수 없습니다.")
        self._timeline = self.timeline_path.open("w", encoding="utf-8", newline="\n")
        self._start_clock = time.perf_counter()
        self._start_utc = datetime.now(timezone.utc).isoformat()

    def record(self, frame_bgr: np.ndarray, measured_fps: float, diagnostics: dict) -> None:
        frame = np.ascontiguousarray(frame_bgr)
        if self._writer is None:
            self._open(frame, measured_fps)
        if frame.shape[:2] != (self._size[1], self._size[0]) or frame.dtype != np.uint8:
            raise ValueError("Camera frame size or format changed during recording")
        self._writer.write(frame)
        row = {"video_frame": self.frame_count,
               "elapsed_ms": round((time.perf_counter() - self._start_clock) * 1000, 3),
               "view": self.view, **diagnostics}
        self._timeline.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._timeline is not None:
            self._timeline.close()
            self._timeline = None
        if self.frame_count:
            (self.directory / f"{self.view}.json").write_text(
                json.dumps({"schema_version": 1, "view": self.view,
                            "camera_index": self.camera_index,
                            "calibration_applied": self.calibration_applied,
                            "video_file": self.video_path.name,
                            "timeline_file": self.timeline_path.name,
                            "video_fps": self._fps,
                            "frame_size": list(self._size),
                            "frame_count": self.frame_count,
                            "start_utc": self._start_utc,
                            "video_content": "undistorted camera input before pose overlay",
                            "timing_note": "Video uses a fixed FPS; elapsed_ms in JSONL is the measured per-frame clock."},
                           ensure_ascii=False, indent=2), encoding="utf-8")


class SessionRecording:
    """Own temporary clips until the user elects to save a finished session."""

    def __init__(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="ai_trainer_squat_")
        self.directory = Path(self._temporary.name)

    def export(self, parent_directory: str | Path, summary: dict) -> Path:
        parent = Path(parent_directory).expanduser().resolve()
        if not parent.is_dir():
            raise ValueError(f"저장 폴더가 없습니다: {parent}")
        temporary_root = self.directory.resolve()
        if parent == temporary_root or temporary_root in parent.parents:
            raise ValueError("임시 녹화 폴더 안에는 저장할 수 없습니다. 다른 폴더를 선택하세요.")
        source_files = []
        for view in VIEWS:
            info = self.directory / f"{view}.json"
            if not info.is_file():
                continue
            payload = json.loads(info.read_text(encoding="utf-8"))
            video = self.directory / payload["video_file"]
            timeline = self.directory / payload["timeline_file"]
            if not video.is_file() or not timeline.is_file():
                raise RuntimeError(f"{view} 녹화 자료가 완성되지 않았습니다.")
            source_files.extend((video, timeline, info))
        if not any(file.suffix in (".mp4", ".avi") for file in source_files):
            raise RuntimeError("저장할 녹화 영상이 없습니다.")

        base = datetime.now().strftime("squat_session_%Y%m%d_%H%M%S")
        for suffix in range(1000):
            target = parent / (base if suffix == 0 else f"{base}_{suffix:03d}")
            try:
                target.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError("중복되지 않는 저장 폴더 이름을 만들 수 없습니다.")
        try:
            for source in source_files:
                shutil.copy2(source, target / source.name)
            (target / "session.json").write_text(
                json.dumps(json_safe({
                    "schema_version": 1,
                    "saved_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": summary,
                    "views": [json.loads((target / f"{view}.json").read_text(encoding="utf-8"))
                              for view in VIEWS if (target / f"{view}.json").exists()],
                    "missing_views": [view for view in VIEWS if not (target / f"{view}.json").exists()],
                }), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(f"저장이 완료되지 않았습니다. 부분 파일 위치: {target} ({error})") from error
        return target

    def close(self) -> None:
        self._temporary.cleanup()

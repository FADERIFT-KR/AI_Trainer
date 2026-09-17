#!/usr/bin/env python3
"""Diagnostic-only Windows camera backend comparison; does not alter production settings."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def fourcc_text(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))


def measure(name: str, backend: int, frames: int, width: int, height: int, fps: int) -> dict:
    capture = cv2.VideoCapture(0, backend)
    if not capture.isOpened():
        capture.release()
        return {"backend": name, "opened": False}
    properties = [
        ("width", cv2.CAP_PROP_FRAME_WIDTH, width),
        ("height", cv2.CAP_PROP_FRAME_HEIGHT, height),
        ("fps", cv2.CAP_PROP_FPS, fps),
        ("buffer_size", cv2.CAP_PROP_BUFFERSIZE, 1),
    ]
    set_results = {label: bool(capture.set(prop, value)) for label, prop, value in properties}
    actual = {label: float(capture.get(prop)) for label, prop, _ in properties}
    actual["fourcc_value"] = float(capture.get(cv2.CAP_PROP_FOURCC))
    actual["fourcc"] = fourcc_text(actual["fourcc_value"])
    samples = []
    failures = 0
    for _ in range(frames):
        started = time.perf_counter()
        ok, frame = capture.read()
        elapsed = (time.perf_counter() - started) * 1000.0
        if ok and frame is not None:
            samples.append(elapsed)
        else:
            failures += 1
    capture.release()
    values = np.asarray(samples, dtype=float)
    return {
        "backend": name,
        "opened": True,
        "requested": {"width": width, "height": height, "fps": fps, "buffer_size": 1},
        "set_results": set_results,
        "actual": actual,
        "frames": len(samples),
        "failures": failures,
        "capture_ms": {
            "avg": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        },
        "capture_fps": 1000.0 / statistics.mean(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()
    results = []
    for name in ("CAP_DSHOW", "CAP_MSMF"):
        backend = getattr(cv2, name, None)
        if backend is not None:
            results.append(measure(name, backend, args.frames, 1280, 720, 30))
    out_dir = ROOT / "output" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / time.strftime("camera_backends_%Y%m%d_%H%M%S.json")
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

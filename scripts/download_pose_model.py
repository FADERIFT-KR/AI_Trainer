"""Explicitly download the pinned official MediaPipe lite pose model."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
MINIMUM_MODEL_BYTES = 1_000_000
MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"


def _valid_model(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < MINIMUM_MODEL_BYTES:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256 or not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        return {
            "pose_detector.tflite",
            "pose_landmarks_detector.tflite",
        }.issubset(archive.namelist())


def download_model(output: str | Path, *, overwrite: bool = False) -> Path:
    destination = Path(output).expanduser().resolve()
    if destination.exists() and not overwrite:
        if _valid_model(destination):
            return destination
        raise RuntimeError(f"Existing model file is invalid: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "AI-Trainer/1.0 pose-model-downloader"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_handle:
            temporary_path = Path(output_handle.name)
            while chunk := response.read(1024 * 1024):
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if not _valid_model(temporary_path):
            size = temporary_path.stat().st_size if temporary_path.exists() else 0
            with temporary_path.open("rb") as handle:
                header_hex = handle.read(16).hex(" ")
            raise RuntimeError(
                "Downloaded pose model is incomplete or has an unexpected format "
                f"(size={size:,}, header={header_hex})"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = download_model(args.output, overwrite=args.overwrite)
    except (RuntimeError, OSError, urllib.error.URLError) as error:
        print(f"Pose model download failed: {error}", file=sys.stderr)
        return 2
    print(f"Pose model ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

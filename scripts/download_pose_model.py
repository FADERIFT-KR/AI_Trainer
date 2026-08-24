"""Explicitly download the pinned official MediaPipe pose model.

기본값을 "full" 모델로 사용한다. "lite"는 가장 가볍고 빠르지만 정확도가 떨어져
실사용 환경(조명/거리/각도가 AI Hub 스튜디오만큼 이상적이지 않음)에서 관절
visibility가 낮게 잡히는 문제가 있었다("발이 안 보여요" 같은 위치 안내가 실제로는
위치가 아니라 인식 정확도 부족 때문에 자주 뜨는 원인 중 하나). MPS/CPU에서도
충분히 실시간으로 돌아가는 속도라 정확도를 우선한다.
"""

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
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "pose_landmarker_full.task"
MINIMUM_MODEL_BYTES = 1_000_000
MODEL_SHA256 = "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad"


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

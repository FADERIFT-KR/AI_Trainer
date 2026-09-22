"""Command-line entry point for the real-time pose window."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ai_trainer.runtime_paths import configured_dataset_path, default_model_path, save_dataset_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show laptop camera video and estimated 3-D pose in one PyQt window."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--model", type=Path, help="Local MediaPipe Pose Landmarker .task bundle")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Extracted AI Hub dataset directory; persists as the default for reference building",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--no-mirror", action="store_true", help="Disable selfie-view mirroring")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    project_root: str | Path | None = None,
) -> int:
    args = make_parser().parse_args(argv)
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path.cwd().resolve()
    )
    requested_model = args.model or default_model_path(root)
    model_path = requested_model if requested_model.is_absolute() else root / requested_model
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        print(
            f"Pose model not found: {model_path}\n"
            "Download it first: python scripts/download_pose_model.py",
            file=sys.stderr,
        )
        return 2

    if args.dataset is not None:
        dataset_path = args.dataset.expanduser().resolve()
        if not dataset_path.is_dir():
            print(f"Dataset directory not found: {dataset_path}", file=sys.stderr)
            return 2
        try:
            save_dataset_path(dataset_path)
        except OSError as error:
            print(f"Could not save dataset setting: {error}", file=sys.stderr)
            return 2
    else:
        dataset_path = configured_dataset_path()
        if dataset_path is not None and not dataset_path.is_dir():
            print(
                f"Saved dataset directory is unavailable and will not be used: {dataset_path}",
                file=sys.stderr,
            )
            dataset_path = None

    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication

        from .window import LivePoseWindow
        from .worker import CameraConfig
    except ImportError as error:
        print(
            f"Live pose dependency is missing: {error}. "
            "Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    try:
        config = CameraConfig(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            requested_fps=args.fps,
            mirror=not args.no_mirror,
            confidence=args.confidence,
        )
    except ValueError as error:
        print(f"Invalid live pose configuration: {error}", file=sys.stderr)
        return 2

    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    application = QApplication(sys.argv[:1])
    application.setApplicationName("AI Trainer Live Pose")
    window = LivePoseWindow(model_path, config, dataset_path=dataset_path)
    window.show()
    window.start()
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""ChArUco 기반 단일 카메라 렌즈 보정.

먼저 인쇄할 표적을 만든다.

    python scripts/calibrate_camera.py --board-output output/charuco_5x7.png --board-only

PNG를 실제 크기(기본 square=30 mm, marker=22 mm)로 인쇄한 뒤, 같은 카메라와
해상도로 다음을 실행한다.

    python scripts/calibrate_camera.py --camera 0

미리보기에서 보드를 화면 전체에 걸쳐 가까이/멀리/기울여 움직인 뒤 Space를 눌러
20개 이상의 서로 다른 장면을 수집한다. 보정값은 기본적으로
``configs/local_camera_calibration.json``에 저장되며 게임과 live-pose 창이 자동으로
읽어 raw 프레임을 MediaPipe 추론 전에 undistort한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ai_trainer.camera_calibration import (  # noqa: E402
    CameraCalibrationError,
    calibrate_charuco_views,
    create_charuco_board,
)
from ai_trainer.live_pose.worker import CameraConfig, _open_camera  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "configs" / "local_camera_calibration.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (기본: 0)")
    parser.add_argument("--width", type=int, default=1280, help="캡처 너비 (기본: 1280)")
    parser.add_argument("--height", type=int, default=720, help="캡처 높이 (기본: 720)")
    parser.add_argument("--fps", type=int, default=30, help="요청 FPS (기본: 30)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="보정 JSON 출력 경로")
    parser.add_argument("--views", type=int, default=20, help="수집할 유효 보드 장면 수 (기본: 20)")
    parser.add_argument("--min-corners", type=int, default=12, help="한 장면에서 필요한 최소 ChArUco corner 수")
    parser.add_argument("--max-rms", type=float, default=1.0, help="허용 RMS reprojection error(px)")
    parser.add_argument("--squares-x", type=int, default=5, help="ChArUco board 가로 square 수")
    parser.add_argument("--squares-y", type=int, default=7, help="ChArUco board 세로 square 수")
    parser.add_argument("--square-mm", type=float, default=30.0, help="인쇄한 square 한 변 길이(mm)")
    parser.add_argument("--marker-mm", type=float, default=22.0, help="인쇄한 ArUco marker 한 변 길이(mm)")
    parser.add_argument("--board-output", type=Path, default=None, help="인쇄용 ChArUco PNG 출력 경로")
    parser.add_argument("--board-only", action="store_true", help="보드 PNG만 만들고 종료")
    return parser


def write_board(board, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = board.generateImage((1500, 2100), marginSize=60, borderBits=1)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"보드 이미지를 저장하지 못했습니다: {path}")
    print(f"인쇄용 ChArUco 보드 저장: {path}")
    print("  인쇄할 때 크기 조정 없이 100%로 출력하고, square 길이가 설정값과 같은지 자로 확인하세요.")


def draw_capture_status(frame: np.ndarray, *, corners, n_samples: int, target_views: int, min_corners: int) -> np.ndarray:
    canvas = frame.copy()
    n_corners = 0 if corners is None else len(corners)
    if corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(canvas, corners)
    ready = n_corners >= min_corners
    color = (70, 220, 70) if ready else (40, 80, 240)
    lines = [
        f"ChArUco corners: {n_corners} / {min_corners}",
        f"Captured views: {n_samples} / {target_views}",
        "SPACE: capture   R: reset   Q/ESC: quit",
        "Move the board across the full frame and tilt it between captures.",
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (16, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return canvas


def capture_views(args, board) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, int]]:
    detector = cv2.aruco.CharucoDetector(board)
    config = CameraConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        requested_fps=args.fps,
        mirror=False,
    )
    capture = _open_camera(cv2, config)
    views: list[tuple[np.ndarray, np.ndarray]] = []
    image_size: tuple[int, int] | None = None
    try:
        while len(views) < args.views:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError("카메라 프레임을 읽지 못했습니다.")
            height, width = frame_bgr.shape[:2]
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            corners, ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
            preview = draw_capture_status(
                frame_bgr,
                corners=corners,
                n_samples=len(views),
                target_views=args.views,
                min_corners=args.min_corners,
            )
            cv2.imshow("AI Trainer - ChArUco camera calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                raise KeyboardInterrupt
            if key in (ord("r"), ord("R")):
                views.clear()
                image_size = None
                print("수집한 장면을 초기화했습니다.")
            elif key == ord(" "):
                if corners is None or ids is None or len(ids) < args.min_corners:
                    print(f"보드 corner가 부족합니다 ({0 if ids is None else len(ids)}/{args.min_corners}).")
                    continue
                current_size = (width, height)
                if image_size is not None and current_size != image_size:
                    raise RuntimeError(f"캡처 해상도가 바뀌었습니다: {image_size} -> {current_size}")
                image_size = current_size
                views.append((corners.copy(), ids.copy()))
                print(f"장면 수집: {len(views)}/{args.views} ({len(ids)} corners)")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    if image_size is None:
        raise RuntimeError("보정용 장면이 수집되지 않았습니다.")
    return views, image_size


def main() -> int:
    args = build_parser().parse_args()
    if args.views < 12:
        raise SystemExit("--views는 최소 12여야 합니다.")
    if args.min_corners < 4:
        raise SystemExit("--min-corners는 최소 4여야 합니다.")
    if args.square_mm <= 0 or args.marker_mm <= 0 or args.marker_mm >= args.square_mm:
        raise SystemExit("square/marker 길이를 확인하세요 (0 < marker < square).")

    board = create_charuco_board(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_mm / 1000.0,
        marker_length_m=args.marker_mm / 1000.0,
    )
    if args.board_output is not None:
        write_board(board, args.board_output)
    if args.board_only:
        return 0

    print("ChArUco 보드 캡처를 시작합니다. Space로 서로 다른 장면을 저장하세요.")
    try:
        views, image_size = capture_views(args, board)
    except KeyboardInterrupt:
        print("사용자가 보정을 취소했습니다.")
        return 130

    calibration = calibrate_charuco_views(
        board,
        views,
        image_size,
        camera_index=args.camera,
        min_views=12,
    )
    if calibration.rms_reprojection_error > args.max_rms:
        raise SystemExit(
            f"RMS reprojection error가 너무 큽니다: {calibration.rms_reprojection_error:.3f}px "
            f"(허용 {args.max_rms:.3f}px). 보드를 더 크게, 더 다양한 각도로 다시 촬영하세요."
        )
    calibration.save(args.output)
    print(f"보정 저장: {args.output}")
    print(
        f"  camera={args.camera}, image={image_size[0]}x{image_size[1]}, "
        f"views={calibration.n_views}, RMS={calibration.rms_reprojection_error:.3f}px"
    )
    print("이 파일은 게임과 live-pose 창에서 다음 실행부터 자동 적용됩니다.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CameraCalibrationError, RuntimeError, ValueError) as error:
        raise SystemExit(f"카메라 보정 실패: {error}") from error

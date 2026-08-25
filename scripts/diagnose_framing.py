#!/usr/bin/env python3
"""framing_check가 왜 실패하는지 실시간으로 숫자를 찍어서 진단하는 스크립트.

GUI(scripts/run_game.py) 없이 터미널에 height_ratio 등 핵심 수치만 출력한다.
"카메라에 조금 더/덜 가까이 서주세요" 메시지가 왜 뜨는지 원인을 바로
확인하려는 목적.

정해진 시간(기본 12초)만 돌고 자동 종료 + 마지막에 요약을 출력한다
(Ctrl+C로 끊어야 하는 무한루프가 아님 — 터미널에서 긴 로그를 계속 복사하다
스크롤 내용이 섞여 들어가는 걸 피하기 위함). 요약 몇 줄만 보내주면 된다.

⚠️ 이 코딩 에이전트(샌드박스) 프로세스에서 실행하면 macOS 카메라 권한을
받을 수 없다. 카메라 권한이 있는 실제 터미널에서 사용자가 직접 실행해야 한다.

사용:
    python scripts/diagnose_framing.py            # 12초 진단 후 자동 종료
    python scripts/diagnose_framing.py --seconds 20
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from ai_trainer.game_ui.framing_check import (  # noqa: E402
    MAX_BODY_HEIGHT_RATIO,
    MIN_BODY_HEIGHT_RATIO,
    check_framing,
)
from ai_trainer.live_pose.mediapipe_pose import MediaPipePoseDetector  # noqa: E402
from ai_trainer.live_pose.worker import CameraConfig, _open_camera  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=12.0, help="진단을 돌릴 시간(초). 기본 12초")
    parser.add_argument(
        "--warmup", type=float, default=10.0,
        help="측정 시작 전 대기 시간(초). 명령어 실행 후 카메라 앞에 설 시간을 준다. 기본 10초",
    )
    args = parser.parse_args()

    config = CameraConfig()
    print(f"카메라 index: {config.camera_index}  (바꾸려면 AI_TRAINER_CAMERA_INDEX 환경변수)")
    detector = MediaPipePoseDetector(
        DEFAULT_MODEL_PATH,
        min_detection_confidence=config.confidence,
        min_presence_confidence=config.confidence,
        min_tracking_confidence=config.confidence,
    )
    capture = _open_camera(cv2, config)

    print(f"기준: MIN_BODY_HEIGHT_RATIO={MIN_BODY_HEIGHT_RATIO}  MAX_BODY_HEIGHT_RATIO={MAX_BODY_HEIGHT_RATIO}")

    # 명령어를 실행한 직후에는 아직 카메라 앞에 서 있지 않은 경우가 많아, 측정을
    # 바로 시작하면 "사람 감지 안됨"만 잔뜩 찍히고 정작 서 있는 동안은 측정이
    # 끝나버리는 문제가 있었다. 카운트다운으로 자리 잡을 시간을 먼저 준다.
    if args.warmup > 0:
        print(f"{args.warmup:.0f}초 뒤 측정을 시작합니다 — 그 전에 카메라 앞자리를 잡아주세요.")
        warmup_start = time.monotonic()
        last_shown = None
        while True:
            remaining = args.warmup - (time.monotonic() - warmup_start)
            if remaining <= 0:
                break
            shown = int(remaining) + 1
            if shown != last_shown:
                print(f"  {shown}...")
                last_shown = shown
            capture.read()  # 워밍업 동안에도 프레임을 계속 읽어 버퍼가 쌓이지 않게 함

    print(f"측정 시작! {args.seconds:.0f}초간 자동으로 진행 후 요약을 출력하고 종료합니다.\n")

    ratios: list[float] = []
    messages: Counter[str] = Counter()
    no_person_frames = 0
    no_body_box_frames = 0
    ok_frames = 0

    start = time.monotonic()
    last_print = 0.0
    try:
        while time.monotonic() - start < args.seconds:
            success, frame_bgr = capture.read()
            if not success or frame_bgr is None:
                continue
            frame_bgr = frame_bgr[:, ::-1] if config.mirror else frame_bgr
            h, w = frame_bgr.shape[:2]
            observation = detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            if observation is None:
                no_person_frames += 1
                continue

            result = check_framing(observation.image_landmarks, w, h)
            messages[result.message] += 1
            if result.ok:
                ok_frames += 1
            if result.body_box is not None:
                x0, y0, x1, y1 = result.body_box
                ratios.append((y1 - y0) / h)
            else:
                no_body_box_frames += 1

            now = time.monotonic()
            if now - last_print >= 1.0:  # 1초 간격으로만 진행상황 표시 (터미널 도배 방지)
                last_print = now
                elapsed = now - start
                print(f"  ...측정 중 ({elapsed:.0f}/{args.seconds:.0f}초)")
    except KeyboardInterrupt:
        print("\n(중간에 중단됨 — 지금까지 모은 데이터로 요약합니다)")
    finally:
        capture.release()
        detector.close()

    total = no_person_frames + no_body_box_frames + len(ratios)
    print("\n" + "=" * 50)
    print("요약")
    print("=" * 50)
    print(f"총 분석 프레임: {total}")
    print(f"  사람 감지 안됨: {no_person_frames}")
    print(f"  body_box 없음(관절 누락): {no_body_box_frames}")
    print(f"  body_box 있음: {len(ratios)}")
    if ratios:
        print(
            f"height_ratio: min={min(ratios):.3f}  max={max(ratios):.3f}  "
            f"median={statistics.median(ratios):.3f}  (기준 {MIN_BODY_HEIGHT_RATIO}~{MAX_BODY_HEIGHT_RATIO})"
        )
    print(f"ok=True 프레임: {ok_frames}/{total}")
    print("메시지별 빈도 (많은 순):")
    for msg, count in messages.most_common(5):
        print(f"  [{count:4d}] {msg}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""framing_check ROI를 실제 게임 화면과 동일하게 창으로 띄워서 보여주는 진단 스크립트.

diagnose_framing.py(터미널 텍스트만)와 달리, 카메라 화면 자체를 창으로 보여주고
그 위에 목표 가이드 박스(초록/빨강)와 실제 감지된 몸 박스, 상태 메시지를
게임(game_ui/pipeline_worker.py)과 동일한 방식으로 그린다 — 화면을 보면서
직접 자리를 맞출 수 있게 하려는 목적.

⚠️ 이 코딩 에이전트(샌드박스) 프로세스에서 실행하면 macOS 카메라 권한을
받을 수 없다. 카메라 권한이 있는 실제 터미널에서 사용자가 직접 실행해야 한다.

사용:
    python scripts/diagnose_framing_visual.py
    (창에서 q 또는 ESC로 종료)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ai_trainer.game_ui.framing_check import check_framing, guide_box as compute_guide_box  # noqa: E402
from ai_trainer.live_pose.mediapipe_pose import MediaPipePoseDetector  # noqa: E402
from ai_trainer.live_pose.render import draw_2d_pose  # noqa: E402
from ai_trainer.live_pose.worker import CameraConfig, _open_camera  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"
WINDOW_NAME = "framing 진단 (q 또는 ESC로 종료)"


def main() -> int:
    config = CameraConfig()
    print(f"카메라 index: {config.camera_index}  (바꾸려면 AI_TRAINER_CAMERA_INDEX 환경변수)")
    detector = MediaPipePoseDetector(
        DEFAULT_MODEL_PATH,
        min_detection_confidence=config.confidence,
        min_presence_confidence=config.confidence,
        min_tracking_confidence=config.confidence,
    )
    capture = _open_camera(cv2, config)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_shown = False
    consecutive_failures = 0

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success or frame_bgr is None:
                consecutive_failures += 1
                if consecutive_failures == 1 or consecutive_failures % 30 == 0:
                    # 조용히 재시도만 하면 "멈춘 건지 안 뜨는 건지" 알 수 없으므로,
                    # 실패가 계속되면 눈에 보이게 알려준다.
                    print(f"카메라에서 프레임을 못 읽는 중... ({consecutive_failures}회 연속 실패) "
                          f"— index/권한을 확인해주세요.")
                if consecutive_failures >= 300:  # 대략 수 초~수십 초 이상 계속 실패
                    print("프레임을 계속 못 읽어 종료합니다. AI_TRAINER_CAMERA_INDEX가 올바른지, "
                          "다른 앱이 이 카메라를 점유하고 있지 않은지 확인해주세요.")
                    return 1
                continue
            consecutive_failures = 0
            display_bgr = np.ascontiguousarray(frame_bgr[:, ::-1] if config.mirror else frame_bgr)
            h, w = display_bgr.shape[:2]
            gbox = compute_guide_box(w, h)

            observation = detector.process(np.ascontiguousarray(display_bgr[:, :, ::-1]))

            if observation is not None:
                display_bgr = draw_2d_pose(display_bgr, observation.image_landmarks)
                framing = check_framing(observation.image_landmarks, w, h)

                box_color = (90, 220, 90) if framing.ok else (60, 60, 240)
                cv2.rectangle(display_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), box_color, 2)
                if framing.body_box is not None:
                    cv2.rectangle(
                        display_bgr, (framing.body_box[0], framing.body_box[1]),
                        (framing.body_box[2], framing.body_box[3]), (0, 200, 255), 2,
                    )
                    ratio = (framing.body_box[3] - framing.body_box[1]) / h
                    cv2.putText(
                        display_bgr, f"height_ratio={ratio:.3f}", (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
                    )
                cv2.putText(
                    display_bgr, framing.message, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA,
                )
                if framing.low_confidence_joints:
                    joints_str = ", ".join(f"{name}({v:.2f})" for name, v in framing.low_confidence_joints)
                    cv2.putText(
                        display_bgr, f"인식 약함: {joints_str}", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1, cv2.LINE_AA,
                    )
            else:
                cv2.rectangle(display_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (60, 60, 240), 2)
                cv2.putText(
                    display_bgr, "사람 감지 안됨", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 240), 2, cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, display_bgr)
            if not window_shown:
                # macOS에서 Cocoa 창이 터미널 뒤에 숨어서 뜨는 경우가 있어,
                # 처음 한 번은 강제로 앞으로 가져온다.
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
                print("카메라 창이 떴습니다 (다른 창 뒤에 숨었다면 Mission Control이나 Cmd+Tab으로 찾아보세요).")
                window_shown = True
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

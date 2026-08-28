#!/usr/bin/env python3
"""녹화된 스쿼트 영상 파일로 실제 검출/판정 파이프라인을 검증한다.

라이브 웹캠(cv2.VideoCapture(0))은 이 개발 환경(샌드박스)에서 장치 접근이 막혀 있지만,
영상 "파일"은 cv2.VideoCapture(path)로 문제없이 열 수 있다. 이 스크립트는
game_ui.pipeline_worker.SquatPipelineWorker.run()과 동일한 파이프라인(MediaPipe 검출 ->
프레이밍 체크 -> Common Skeleton -> 2D->3D Lifting -> OnlineSquatSession)을 Qt 스레드 없이
그대로 돌려서, 사람이 실제로 촬영한 영상에 대해:
  1) 프레임별 진단 로그(콘솔) 출력 — 프레이밍/신뢰도/phase/실시간 판정
  2) 판정 내용을 그대로 그려 넣은 주석 영상 저장 (앱 화면과 동일한 오버레이)
  3) REP 종료 시점 요약 + 세션 전체 요약(오류 라벨이 뜬 비율 등)
을 만들어준다.

사용법:
    python scripts/analyze_squat_video.py <영상경로> [--out 주석영상.mp4] [--mirror]

주의: 세션 활성화(캘리브레이션 시작)는 실제 앱과 동일하게 "화각이 FRAMING_STABLE_SECONDS
이상 안정적으로 좋을 때"부터 시작한다 — 즉 영상 앞부분에 사용자가 가이드박스 안에 정면으로
서 있는 구간이 있어야 그 이후부터 phase/DTW 판정이 시작된다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_trainer.game_ui.error_explain import annotate_error  # noqa: E402
from ai_trainer.game_ui.framing_check import check_framing  # noqa: E402
from ai_trainer.game_ui.framing_check import guide_box as compute_guide_box  # noqa: E402
from ai_trainer.game_ui.joint_overlay import draw_joint_feedback  # noqa: E402
from ai_trainer.game_ui.pipeline_worker import (  # noqa: E402
    DB_DIR,
    DEFAULT_MODEL_PATH,
    LIFTING_CKPT,
    OFFLINE_REPORT_PATH,
    WEIGHTS_CFG_PATH,
)
from ai_trainer.game_ui.pose_bridge import CommonSkeleton3DBridge, CommonSkeletonBridge  # noqa: E402
from ai_trainer.joint_feedback import compute_joint_scores  # noqa: E402
from ai_trainer.lifting_model import TemporalLiftingNet  # noqa: E402
from ai_trainer.live_pose.mediapipe_pose import MediaPipePoseDetector  # noqa: E402
from ai_trainer.live_pose.render import draw_2d_pose  # noqa: E402
from ai_trainer.online_dtw import OnlineSquatSession  # noqa: E402
from ai_trainer.reference_db_io import load_reference_db  # noqa: E402
from ai_trainer.scoring import PASS_SCORE_THRESHOLD, distance_to_score  # noqa: E402

FRAMING_STABLE_SECONDS = 1.0
FRAMING_DEBOUNCE_FRAMES = 5
BAD_LABEL_MIN_STREAK = 4
JUDGE_MARGIN_THRESHOLD = 0.05
PHASE_LABEL_KR = {"prep": "준비", "descend": "하강", "bottom": "최저점", "ascend": "상승", None: "-"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=str, help="분석할 스쿼트 영상 파일 경로")
    ap.add_argument("--out", type=str, default=None, help="주석 오버레이 영상 저장 경로 (기본: <입력명>_analyzed.mp4)")
    ap.add_argument("--mirror", action="store_true", help="좌우 반전 (셀피 화면처럼 촬영된 경우)")
    ap.add_argument("--confidence", type=float, default=0.4, help="MediaPipe/프레이밍 최소 신뢰도")
    ap.add_argument("--no-video-out", action="store_true", help="주석 영상을 저장하지 않고 로그만 출력")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"영상을 찾을 수 없습니다: {video_path}")
    out_path = Path(args.out) if args.out else video_path.with_name(video_path.stem + "_analyzed.mp4")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("모델 로딩 중...")
    lifting_model = TemporalLiftingNet(n_joints=18, hidden=128)
    lifting_model.load_state_dict(torch.load(LIFTING_CKPT, map_location=device))
    lifting_model.to(device).eval()

    weights_cfg = json.loads(WEIGHTS_CFG_PATH.read_text(encoding="utf-8"))
    db = load_reference_db(DB_DIR)
    score_calib = None
    if OFFLINE_REPORT_PATH.exists():
        score_calib = json.loads(OFFLINE_REPORT_PATH.read_text(encoding="utf-8"))["score_calibration_ground_truth"]

    # pipeline_worker.py와 동일: 실시간 3D 소스는 자체 lifting 모델이 아니라 MediaPipe
    # 자체 world_landmarks(CommonSkeleton3DBridge) — 그래서 비교 대상도 ground_truth tier.
    session = OnlineSquatSession(
        model=lifting_model, device=device, db_operational=db["ground_truth"],
        weights_cfg=weights_cfg, score_calib=score_calib,
    )
    bridge = CommonSkeletonBridge(min_visibility=args.confidence)  # 화면 그리기(2D)용
    bridge3d = CommonSkeleton3DBridge(min_visibility=args.confidence)  # 판정(3D)용
    detector = MediaPipePoseDetector(
        DEFAULT_MODEL_PATH,
        min_detection_confidence=args.confidence,
        min_presence_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"영상: {video_path.name}  fps={src_fps:.1f}  frames={total_frames}")

    writer = None
    session_active = False
    framing_effective_ok = False
    framing_streak = 0
    framing_ok_since_frame: int | None = None
    bad_streak_key = None
    bad_streak_count = 0
    judge_counts: Counter = Counter()
    frame_idx = -1
    t_start = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        display_bgr = np.ascontiguousarray(frame_bgr[:, ::-1] if args.mirror else frame_bgr)
        observation = detector.process(np.ascontiguousarray(display_bgr[:, :, ::-1]))

        h, w = display_bgr.shape[:2]
        gbox = compute_guide_box(w, h)
        pose_found = observation is not None
        framing_ok = False
        framing_message = "카메라 앞에 서주세요"
        phase = None
        judge_text, judge_kind = "대기 중", "neutral"

        if pose_found:
            video_bgr = draw_2d_pose(display_bgr, observation.image_landmarks)
            framing = check_framing(observation.image_landmarks, w, h, relax_distance=session_active)
            framing_message = framing.message
            if framing.ok == framing_effective_ok:
                framing_streak = 0
            else:
                framing_streak += 1
                if framing_streak >= FRAMING_DEBOUNCE_FRAMES:
                    framing_effective_ok = framing.ok
                    framing_streak = 0
            framing_ok = framing_effective_ok

            box_color = (90, 220, 90) if framing_ok else (60, 60, 240)
            cv2.rectangle(video_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), box_color, 2)
            if not framing_ok:
                cv2.putText(video_bgr, framing_message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 240), 2, cv2.LINE_AA)

            # 실제 앱의 3-2-1 카운트다운 대신: 화각이 FRAMING_STABLE_SECONDS 이상
            # 안정적으로 좋으면 그 즉시 세션 시작(카운트다운 대기 없이 바로 진행).
            if not session_active:
                if framing_ok:
                    if framing_ok_since_frame is None:
                        framing_ok_since_frame = frame_idx
                    elif (frame_idx - framing_ok_since_frame) / src_fps >= FRAMING_STABLE_SECONDS:
                        session_active = True
                        print(f"[frame {frame_idx}] 화각 안정 확인 -> 세션 시작(캘리브레이션 개시)")
                else:
                    framing_ok_since_frame = None

            if framing_ok and session_active:
                common2d, frozen_mask, mean_conf = bridge.update(observation.image_landmarks, w, h)
                common3d, _frozen3d, _mean_conf3d = bridge3d.update(observation.world_landmarks)
                status = session.push_frame_3d(common3d)
                if status is not None and status.get("status") == "ok":
                    phase = status["phase"]
                    partial = status["partial_distance"]
                    live_score = None

                    if partial is not None:
                        dvals = partial["distance_by_class"]
                        if score_calib is not None and "정상" in dvals:
                            live_score = distance_to_score(dvals["정상"], score_calib)
                        best_class = min(dvals, key=dvals.get)
                        sorted_d = sorted(dvals.values())
                        margin = sorted_d[1] - sorted_d[0] if len(sorted_d) >= 2 else 0.0
                        passes = live_score is not None and live_score >= PASS_SCORE_THRESHOLD

                        if passes:
                            judge_text, judge_kind = "자세 양호", "good"
                            bad_streak_key, bad_streak_count = None, 0
                            annotate = False
                        elif margin <= JUDGE_MARGIN_THRESHOLD:
                            judge_text, judge_kind = "확인 필요", "neutral"
                            bad_streak_key, bad_streak_count = None, 0
                            annotate = False
                        elif best_class == "정상":
                            judge_text, judge_kind = "자세 양호", "good"
                            bad_streak_key, bad_streak_count = None, 0
                            annotate = False
                        else:
                            key = (phase, best_class)
                            if key == bad_streak_key:
                                bad_streak_count += 1
                            else:
                                bad_streak_key, bad_streak_count = key, 1
                            if bad_streak_count >= BAD_LABEL_MIN_STREAK:
                                judge_text, judge_kind = f"자세 확인 필요 ({best_class})", "bad"
                                annotate = True
                            else:
                                judge_text, judge_kind = "확인 필요", "neutral"
                                annotate = False

                        if annotate:
                            annotate_error(video_bgr, common2d, best_class)
                        judge_counts[judge_kind] += 1
                        score_txt = f" · {live_score:.0f}%" if live_score is not None else ""
                        print(
                            f"[frame {frame_idx:4d}] phase={PHASE_LABEL_KR.get(phase,'-'):4s} "
                            f"judge={judge_text}{score_txt}  dist={ {k: round(v,3) for k,v in dvals.items()} }"
                        )

                    jf = status["joint_feedback"]
                    if jf is not None:
                        joint_scores = compute_joint_scores(jf["user_frame"], jf["ref_frame"])
                        draw_joint_feedback(video_bgr, common2d, joint_scores)

                    if status["event"] == "rep_end":
                        r = status["completed_rep"]
                        score_text = f"{r.score_vs_normal:.0f}%" if r.score_vs_normal is not None else "-"
                        print(
                            f"\n=== REP {r.rep_index} 종료 (frame {r.frame_range[0]}~{r.frame_range[1]}) ===\n"
                            f"  판정: {r.predicted_class}  |  유사도: {score_text}  |  "
                            f"거리: { {k: round(v,3) for k,v in r.raw_distance_by_class.items()} }\n"
                            f"  주요 특징: {', '.join(n for n,_ in r.top_contributing_features)}\n"
                        )

            cv2.putText(video_bgr, f"phase: {PHASE_LABEL_KR.get(phase,'-')}", (12, h - 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(video_bgr, judge_text, (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 220, 90) if judge_kind == "good" else
                        (60, 60, 240) if judge_kind == "bad" else (0, 210, 255), 2, cv2.LINE_AA)
        else:
            video_bgr = display_bgr.copy()
            cv2.rectangle(video_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (60, 60, 240), 2)
            cv2.putText(video_bgr, "사람을 찾는 중...", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 240), 2, cv2.LINE_AA)

        if not args.no_video_out:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (video_bgr.shape[1], video_bgr.shape[0]))
            writer.write(video_bgr)

    cap.release()
    if writer is not None:
        writer.release()
    detector.close()

    elapsed = time.time() - t_start
    print(f"\n분석 완료 ({elapsed:.1f}초, {frame_idx+1}프레임). REP {len(session.completed_reps)}개 검출.")
    print(f"실시간 판정 프레임 분포: {dict(judge_counts)}")
    if not args.no_video_out:
        print(f"주석 영상 저장: {out_path}")
    for r in session.completed_reps:
        score_text = f"{r.score_vs_normal:.0f}%" if r.score_vs_normal is not None else "-"
        print(f"  REP {r.rep_index}: {r.predicted_class}  (유사도 {score_text}, frame {r.frame_range})")


if __name__ == "__main__":
    main()

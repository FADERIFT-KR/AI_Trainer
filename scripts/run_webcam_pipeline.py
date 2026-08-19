#!/usr/bin/env python3
"""Headless end-to-end 실시간 파이프라인.

Webcam RGB -> 2D Pose Estimation(MediaPipe) -> Common Skeleton Mapping ->
Temporal Buffer -> 2D->3D Lifting -> Calibration -> 3D Normalization ->
Online Phase Detection -> Online/Subsequence DTW -> (rep 종료 시) Offline
Weighted DTW -> Score/Error Class/Feature Contribution.

GUI(PyQt 등 앱) 없음 — 터미널 상태 출력 + (기본값) OpenCV 미리보기 창.
미리보기 창: 왼쪽 webcam RGB + 2D skeleton overlay, 오른쪽 현재 3D skeleton(lifting 결과).
창이 떠 있는 동안 'q' 키를 누르면 종료한다. `--headless`로 창을 끌 수 있다.
미리보기/저장 로직은 순수 표시용이며 파이프라인 계산(추론/DTW) 결과에는 관여하지 않는다.

⚠️ 이 스크립트를 이 코딩 에이전트(샌드박스 프로세스)에서 실행하면 macOS 카메라 권한을
얻지 못해 webcam 소스는 열리지 않는다(OpenCV `not authorized to capture video`).
카메라 권한이 있는 실제 터미널(VS Code 통합 터미널 등)에서 사용자가 직접 실행해야
webcam 소스가 동작한다. `video:<path>`로 소스를 지정하면 이 문제와 무관하게 어떤
환경에서도 동작한다.

사용 예:
    # 실제 웹캠, 미리보기 창 표시 (기본값)
    python scripts/run_webcam_pipeline.py --source webcam:0 --viz-out output/webcam_check.mp4

    # 오른쪽 패널에 정상 동작 참고 영상을 반복 재생해 따라하기 (본인 3D skeleton 대신)
    python scripts/run_webcam_pipeline.py --source webcam:0 --reference-video output/reference_demo.mp4

    # 미리보기 창 없이 headless로 실행 (기존 동작)
    python scripts/run_webcam_pipeline.py --source webcam:0 --headless

    # 로컬 영상 파일로 테스트
    python scripts/run_webcam_pipeline.py --source video:/path/to/clip.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_trainer.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS  # noqa: E402
from ai_trainer.lifting_model import TemporalLiftingNet  # noqa: E402
from ai_trainer.mediapipe_mapping import CommonSkeletonTracker  # noqa: E402
from ai_trainer.online_dtw import OnlineSquatSession  # noqa: E402
from ai_trainer.pose_estimator import MediaPipePoseEstimator  # noqa: E402
from ai_trainer.reference_db_io import load_reference_db  # noqa: E402
from ai_trainer.render import draw_skeleton_panel, fit_transform  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_CFG_PATH = ROOT / "configs" / "dtw_feature_weights.json"
DB_DIR = ROOT / "output" / "reference_db"
OFFLINE_REPORT_PATH = ROOT / "output" / "dtw_eval" / "offline_eval_report.json"

VIZ_PANEL_W, VIZ_PANEL_H = 480, 480
WINDOW_NAME = "AI Trainer - Squat Pipeline Preview (q: quit)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="webcam:<index> 또는 video:<path>")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--min-visibility", type=float, default=0.5)
    p.add_argument("--viz-out", default=None, help="검증용 시각화 mp4 저장 경로 (webcam 프레임 + 2D skeleton + 3D 정면 투영)")
    p.add_argument("--headless", action="store_true", help="OpenCV 미리보기 창을 띄우지 않음 (기본값: 창을 띄움)")
    p.add_argument(
        "--reference-video", default=None,
        help="오른쪽 패널에 반복 재생할 정상 동작 참고 영상 경로. 지정하지 않으면 기존처럼 사용자 본인의 lifting 3D skeleton을 표시.",
    )
    return p.parse_args()


def build_canvas(
    frame_bgr: np.ndarray,
    common2d: np.ndarray | None,
    session: OnlineSquatSession,
    ref_frame: np.ndarray | None = None,
) -> np.ndarray:
    """왼쪽: webcam + 2D skeleton overlay. 오른쪽: --reference-video가 있으면 그 영상(따라하기용),
    없으면 사용자 본인의 현재 3D skeleton(lifting 결과). 표시/저장 전용, 계산에 영향 없음."""
    canvas = np.zeros((VIZ_PANEL_H, VIZ_PANEL_W * 2, 3), dtype=np.uint8)
    h, w = frame_bgr.shape[:2]
    frame_resized = cv2.resize(frame_bgr, (VIZ_PANEL_W, VIZ_PANEL_H))

    if common2d is not None:
        # frame_resized와 동일한 스케일(가로/세로 각각 실제 리사이즈 비율)로 직접 매핑해야
        # 원본 이미지 속 관절 위치와 화면에 겹쳐 그려지는 스켈레톤이 어긋나거나(예: 사람보다 커져서
        # 화면 밖으로 잘림) 하지 않는다. (기존 버그: 스켈레톤 자체의 bbox로 다시 맞추던 fit_transform을
        # 여기 재사용해 실제 프레임 스케일과 불일치했음)
        scale = np.array([VIZ_PANEL_W / w, VIZ_PANEL_H / h])
        pts = common2d * scale
        valid = np.isfinite(pts).all(axis=-1)
        for (i, j), color in zip(COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR):
            if valid[i] and valid[j]:
                cv2.line(frame_resized, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)), color, 2, cv2.LINE_AA)
        for i in range(pts.shape[0]):
            if valid[i]:
                cv2.circle(frame_resized, tuple(pts[i].astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.putText(frame_resized, "no person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    canvas[:, :VIZ_PANEL_W] = frame_resized

    if ref_frame is not None:
        canvas[:, VIZ_PANEL_W:] = cv2.resize(ref_frame, (VIZ_PANEL_W, VIZ_PANEL_H))
        cv2.putText(canvas, "Reference (따라하기)", (VIZ_PANEL_W + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    elif session.aligned_seq:
        latest = session.aligned_seq[-1][:, [0, 1]]
        tf3d = fit_transform(latest[None], VIZ_PANEL_W, VIZ_PANEL_H, flip_y=True)
        draw_skeleton_panel(
            canvas, (VIZ_PANEL_W, 0), VIZ_PANEL_W, VIZ_PANEL_H, tf3d(latest),
            "3D lateral-vertical (body-centered)", None, COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
        )
    else:
        right = canvas[:, VIZ_PANEL_W:]
        cv2.putText(right, "calibrating...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return canvas


def read_looping(cap: cv2.VideoCapture) -> np.ndarray | None:
    """참고 영상을 프레임이 끝나면 처음으로 되감아 계속 반복 재생한다."""
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return frame if ret else None


def open_source(source: str) -> cv2.VideoCapture:
    if source.startswith("webcam:"):
        idx = int(source.split(":", 1)[1])
        return cv2.VideoCapture(idx)
    if source.startswith("video:"):
        path = source.split(":", 1)[1]
        return cv2.VideoCapture(path)
    raise ValueError(f"알 수 없는 --source 형식: {source} (webcam:<n> 또는 video:<path>)")


def main() -> None:
    args = parse_args()

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"[오류] 프레임 소스를 열 수 없습니다: {args.source}", file=sys.stderr)
        print(
            "  - webcam:N 인 경우: 이 프로세스에 macOS 카메라 권한이 없을 수 있습니다. "
            "시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 터미널 앱 권한을 확인하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TemporalLiftingNet(n_joints=18, hidden=128)
    model.load_state_dict(torch.load(ROOT / "output" / "lifting_baseline" / "model_best.pt", map_location=device))
    model.to(device).eval()

    weights_cfg = json.loads(WEIGHTS_CFG_PATH.read_text(encoding="utf-8"))
    db = load_reference_db(DB_DIR)
    score_calib = None
    if OFFLINE_REPORT_PATH.exists():
        score_calib = json.loads(OFFLINE_REPORT_PATH.read_text(encoding="utf-8"))["score_calibration"]

    pose_estimator = MediaPipePoseEstimator()
    tracker = CommonSkeletonTracker(min_visibility=args.min_visibility)
    session = OnlineSquatSession(
        model=model, device=device, db_operational=db["operational"], weights_cfg=weights_cfg, score_calib=score_calib,
    )

    viz_writer = None
    if args.viz_out:
        Path(args.viz_out).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        viz_writer = cv2.VideoWriter(args.viz_out, fourcc, 20.0, (VIZ_PANEL_W * 2, VIZ_PANEL_H))

    ref_cap = None
    if args.reference_video:
        ref_cap = cv2.VideoCapture(args.reference_video)
        if not ref_cap.isOpened():
            print(f"[경고] --reference-video를 열 수 없습니다: {args.reference_video} (본인 3D skeleton으로 대체)", file=sys.stderr)
            ref_cap = None

    print(f"파이프라인 시작: source={args.source}  device={device}  min_visibility={args.min_visibility}")
    print("=" * 78)

    rep_count = 0
    frame_idx = 0
    n_detected = 0
    t_start = time.perf_counter()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        ts_ms = int((time.perf_counter() - t_start) * 1000) if args.source.startswith("webcam:") else frame_idx * 33
        pose = pose_estimator.estimate(frame_bgr, timestamp_ms=ts_ms)

        quit_requested = False

        ref_frame = read_looping(ref_cap) if ref_cap is not None else None

        if not pose.detected:
            print(f"[frame {frame_idx:5d}] 사람 미검출")
            if viz_writer is not None or not args.headless:
                canvas = build_canvas(frame_bgr, None, session, ref_frame)
                if viz_writer is not None:
                    viz_writer.write(canvas)
                if not args.headless:
                    cv2.imshow(WINDOW_NAME, canvas)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        quit_requested = True
            frame_idx += 1
            if quit_requested:
                break
            continue
        n_detected += 1

        common2d, frozen_mask, mean_conf = tracker.update(pose.landmarks_px, pose.visibility)
        n_frozen = int(frozen_mask.sum())

        status = session.push_frame(common2d)

        if viz_writer is not None or not args.headless:
            canvas = build_canvas(frame_bgr, common2d, session, ref_frame)
            if viz_writer is not None:
                viz_writer.write(canvas)
            if not args.headless:
                cv2.imshow(WINDOW_NAME, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    quit_requested = True

        if quit_requested:
            frame_idx += 1
            break

        if status is None or status.get("status") == "calibrating":
            frame_idx += 1
            continue

        margin_str = ""
        if status["partial_distance"]:
            dvals = sorted(status["partial_distance"]["distance_by_class"].values())
            if len(dvals) >= 2:
                margin_str = f" margin={dvals[1]-dvals[0]:+.2f}"

        conf_flag = f" [frozen={n_frozen}/18]" if n_frozen else ""
        print(
            f"[frame {status['emit_frame']:5d}] phase={status['phase']:8s} rep#{rep_count:2d} "
            f"conf={mean_conf:.2f}{conf_flag}  pelvis_h={status['pelvis_height']:.2f} vel={status['velocity']:+.3f}"
            + (f"  partial({status['partial_distance']['phase']})={ {k: round(v,2) for k,v in status['partial_distance']['distance_by_class'].items()} }{margin_str}" if status["partial_distance"] else "")
        )

        if status["event"] == "rep_end":
            rep_count += 1
            r = status["completed_rep"]
            dists = r.raw_distance_by_class
            sorted_d = sorted(dists.values())
            margin = sorted_d[1] - sorted_d[0] if len(sorted_d) >= 2 else 0.0
            print("=" * 78)
            print(f"REP #{rep_count} 완료 (frame {r.frame_range[0]}~{r.frame_range[1]})")
            print(f"  predicted_class = {r.predicted_class}")
            print(f"  raw_distance    = { {k: round(v,3) for k,v in dists.items()} }")
            print(f"  distance_margin = {margin:.3f}")
            print(f"  score(vs 정상)  = {r.score_vs_normal}")
            print(f"  top_features    = {r.top_contributing_features}")
            print("=" * 78)

        frame_idx += 1

    elapsed = time.perf_counter() - t_start
    cap.release()
    if ref_cap is not None:
        ref_cap.release()
    pose_estimator.close()
    if viz_writer is not None:
        viz_writer.release()
        print(f"\n시각화 저장: {args.viz_out}")
    if not args.headless:
        cv2.destroyAllWindows()

    fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print("\n" + "=" * 78)
    print(f"총 프레임 {frame_idx}개 (사람 검출 {n_detected}개), 총 rep {rep_count}회")
    print(f"경과 {elapsed:.1f}초, 평균 FPS {fps:.1f} (30FPS 실시간 기준 {'충족' if fps >= 30 else '미충족'})")
    print(f"고정 latency: T=9 lifting window로 인한 4프레임 지연 (~{4/max(fps,1)*1000:.0f}ms @ {fps:.0f}FPS)")


if __name__ == "__main__":
    main()

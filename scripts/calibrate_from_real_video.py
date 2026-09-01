#!/usr/bin/env python3
"""score_calibration_ground_truth를 "진짜" 실사용(MediaPipe) 영상으로 재계산한다.

기존 score_calibration_ground_truth(run_offline_dtw_eval.py가 만듦)는 AI Hub CSV끼리만
비교한 값(query도 build_ground_truth_reference로 만든 "순수 CSV" 데이터)이라, 실제
MediaPipe 3D(world_landmarks)로 들어오는 라이브 쿼리와는 거리 스케일이 전혀 안 맞았다
(2026-09-01 확인: 실측 REP들의 "정상" distance가 2.0~2.8인데 calib hi=1.475라서 무조건
0%로 클리핑되고 있었음 — 그 결과 PASS_SCORE_THRESHOLD 안전장치도 항상 무력화됨).

이 스크립트는 실제 녹화 영상(analyze_squat_video.py와 동일한 파이프라인: MediaPipe ->
CommonSkeleton3DBridge -> OnlineSquatSession.push_frame_3d)을 돌려 완료된 REP들의 "정상"
클래스 거리를 모아, 그 실측 분포로 lo/hi를 다시 잡는다.

주의: 지금은 영상 1개(제한된 샘플)로만 계산한 상태다 — 통계적으로 탄탄한 값이 아니라
"무조건 0%"라는 명백한 버그를 우선 없애기 위한 잠정치다. 관대한 여유(padding)를 둬서
이 소수 샘플에 과적합되지 않게 했다. 실사용 영상이 더 쌓이면(--video를 여러 개 넘기면)
자동으로 더 많은 샘플로 재계산된다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_trainer.game_ui.framing_check import check_framing  # noqa: E402
from ai_trainer.game_ui.pipeline_worker import DB_DIR, DEFAULT_MODEL_PATH, LIFTING_CKPT, OFFLINE_REPORT_PATH, WEIGHTS_CFG_PATH  # noqa: E402
from ai_trainer.game_ui.pose_bridge import CommonSkeleton3DBridge  # noqa: E402
from ai_trainer.lifting_model import TemporalLiftingNet  # noqa: E402
from ai_trainer.live_pose.mediapipe_pose import MediaPipePoseDetector  # noqa: E402
from ai_trainer.online_dtw import OnlineSquatSession  # noqa: E402
from ai_trainer.reference_db_io import load_reference_db  # noqa: E402

# 소수 샘플(영상 1~2개) 과적합 방지용 여유폭 — 관찰된 최솟값보다 더 낮게(=더 잘한 것으로),
# 관찰된 최댓값보다 더 높게(=더 못한 것도 0%로 바로 안 깎이게) 벌려서 lo/hi를 잡는다.
PAD_LO = 0.3
PAD_HI = 0.3


def collect_normal_distances(video_path: Path, confidence: float = 0.4) -> list[float]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    lifting_model = TemporalLiftingNet(n_joints=18, hidden=128)
    lifting_model.load_state_dict(torch.load(LIFTING_CKPT, map_location=device))
    lifting_model.to(device).eval()

    weights_cfg = json.loads(WEIGHTS_CFG_PATH.read_text(encoding="utf-8"))
    db = load_reference_db(DB_DIR)
    session = OnlineSquatSession(
        model=lifting_model, device=device, db_operational=db["ground_truth"], weights_cfg=weights_cfg,
    )
    bridge3d = CommonSkeleton3DBridge(min_visibility=confidence)
    detector = MediaPipePoseDetector(
        DEFAULT_MODEL_PATH, min_detection_confidence=confidence,
        min_presence_confidence=confidence, min_tracking_confidence=confidence,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    active = False
    ok_since_frame: int | None = None
    frame_idx = -1
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        h, w = frame_bgr.shape[:2]
        obs = detector.process(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        if obs is None:
            continue
        framing = check_framing(obs.image_landmarks, w, h, relax_distance=active)
        if not active:
            if framing.ok:
                if ok_since_frame is None:
                    ok_since_frame = frame_idx
                elif (frame_idx - ok_since_frame) / src_fps >= 1.0:
                    active = True
            else:
                ok_since_frame = None
            continue
        common3d, _frozen, _conf = bridge3d.update(obs.world_landmarks)
        session.push_frame_3d(common3d)

    detector.close()
    cap.release()
    return [r.raw_distance_by_class["정상"] for r in session.completed_reps]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", type=str, help="재계산에 쓸 실사용 영상 파일 경로(들)")
    args = ap.parse_args()

    all_normal_d: list[float] = []
    for v in args.videos:
        path = Path(v)
        print(f"처리 중: {path.name} ...")
        d = collect_normal_distances(path)
        print(f"  REP {len(d)}개, '정상' 거리: {[round(x, 3) for x in d]}")
        all_normal_d.extend(d)

    if len(all_normal_d) < 2:
        raise SystemExit("REP이 2개 미만 검출됐습니다 — calibration을 계산할 수 없습니다.")

    arr = np.array(all_normal_d)
    lo = max(0.0, float(arr.min()) - PAD_LO)
    hi = float(arr.max()) + PAD_HI
    print(f"\n전체 REP {len(all_normal_d)}개(영상 {len(args.videos)}개) 기준: lo={lo:.3f} hi={hi:.3f}")

    report = json.loads(OFFLINE_REPORT_PATH.read_text(encoding="utf-8"))
    old = report.get("score_calibration_ground_truth")
    report["score_calibration_ground_truth"] = {
        "lo": lo, "hi": hi, "n_normal": len(all_normal_d), "n_error": 0,
        "weight_profile": weight_profile_note(),
        "source": "calibrate_from_real_video.py (실사용 MediaPipe 영상 기반, CSV-only 계산 아님)",
        "videos": [str(Path(v).name) for v in args.videos],
    }
    OFFLINE_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"이전 calib: {old}")
    print(f"새 calib  : {report['score_calibration_ground_truth']}")
    print(f"저장: {OFFLINE_REPORT_PATH}")


def weight_profile_note() -> str:
    weights_cfg = json.loads(WEIGHTS_CFG_PATH.read_text(encoding="utf-8"))
    return weights_cfg["default_profile"]


if __name__ == "__main__":
    main()

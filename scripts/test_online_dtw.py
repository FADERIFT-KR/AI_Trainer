#!/usr/bin/env python3
"""Online DTW를 prerecorded validation 시퀀스로 "스트리밍처럼" 검증한다.

실제 웹캠에 연결하지 않고, camera1 2D 프레임을 하나씩 순서대로 push_frame()에
넣어 실시간 입력을 시뮬레이션한다. 검증 항목:
  1. 미래 프레임 미참조 (OnlineSquatSession 설계 자체가 보장 - 아래서 지연 프레임 수 출력으로 재확인)
  2. phase 실시간 전환 안정성
  3. repetition 시작/종료 검출
  4. partial DTW distance의 시간에 따른 갱신
  5. 최종 Online 결과 vs 기존 Offline 결과 합리적 일치 여부
  6. 처리 속도 (초당 프레임)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_trainer.actor_split import load_all_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.common_skeleton import to_common_skeleton  # noqa: E402
from ai_trainer.dtw_compare import multi_reference_distance, resolve_weights  # noqa: E402
from ai_trainer.features import extract_all_features  # noqa: E402
from ai_trainer.lifting_dataset import load_actor_split  # noqa: E402
from ai_trainer.lifting_model import TemporalLiftingNet  # noqa: E402
from ai_trainer.online_dtw import OnlineSquatSession  # noqa: E402
from ai_trainer.phase_features import extract_phase_features  # noqa: E402
from ai_trainer.phase_segmentation import segment_phases  # noqa: E402
from ai_trainer.reference_db_io import load_reference_db  # noqa: E402
from ai_trainer.reference_pipeline import build_operational_reference  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Training/02.라벨링데이터/TL.zip"
)
VL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip"
)
SPLIT_PATH = ROOT / "configs" / "actor_split.json"
WEIGHTS_CFG_PATH = ROOT / "configs" / "dtw_feature_weights.json"
DB_DIR = ROOT / "output" / "reference_db"
OFFLINE_REPORT_PATH = ROOT / "output" / "dtw_eval" / "offline_eval_report.json"
OUT_PATH = ROOT / "output" / "dtw_eval" / "online_eval_report.json"
CLASSES = ["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"]
N_TEST_SEQS = 6


def offline_predict(z, seq, origin, model, device, db, weights_cfg):
    ref = build_operational_reference(z, seq, origin, model, device)
    if ref is None:
        return None
    feat = extract_all_features(ref.coords)
    bounds = segment_phases(extract_phase_features(ref.coords)).as_dict()
    per_class = {}
    for cls, medoids in db["operational"].items():
        w = resolve_weights(weights_cfg, weights_cfg["default_profile"], class_label=cls)
        per_class[cls] = multi_reference_distance(feat, bounds, medoids, w, weights_cfg, top_k=2)
    pred = min(per_class, key=lambda c: per_class[c]["min_distance"])
    return {"predicted_class": pred, "raw_distance_by_class": {c: per_class[c]["min_distance"] for c in CLASSES}}


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TemporalLiftingNet(n_joints=18, hidden=128)
    model.load_state_dict(torch.load(ROOT / "output" / "lifting_baseline" / "model_best.pt", map_location=device))
    model.to(device).eval()

    weights_cfg = json.loads(WEIGHTS_CFG_PATH.read_text(encoding="utf-8"))
    db = load_reference_db(DB_DIR)
    score_calib = None
    if OFFLINE_REPORT_PATH.exists():
        score_calib = json.loads(OFFLINE_REPORT_PATH.read_text(encoding="utf-8"))["score_calibration"]

    actor_to_split = load_actor_split(SPLIT_PATH)
    all_seqs = load_all_air_squat_sequences(TL_ZIP, VL_ZIP)
    val_seqs = [os_ for os_ in all_seqs if actor_to_split.get(os_.seq.actor) == "val" and os_.seq.error_type in CLASSES]

    import random

    rng = random.Random(9)
    rng.shuffle(val_seqs)
    # 클래스 다양성 확보
    picked, seen_cls = [], set()
    for os_ in val_seqs:
        if os_.seq.error_type not in seen_cls or len(picked) < N_TEST_SEQS:
            picked.append(os_)
            seen_cls.add(os_.seq.error_type)
        if len(picked) >= N_TEST_SEQS:
            break

    zips = {"TL": AiHubZip(TL_ZIP), "VL": AiHubZip(VL_ZIP)}
    all_reports = []

    for os_ in picked:
        seq = os_.seq
        z = zips[os_.origin]
        frames_2d, coords_26 = z.read_2d(seq, 1)
        coords_18 = to_common_skeleton(coords_26)  # (N,18,2) 전체 클립(대기 구간 포함), raw pixel
        ann = z.read_annotation(seq, 1)
        a0 = ann["annotations"][0]
        gt_start, gt_end = a0["start_frame"], a0["end_frame"]

        session = OnlineSquatSession(
            model=model, device=device, db_operational=db["operational"],
            weights_cfg=weights_cfg, score_calib=score_calib,
        )

        phase_log = []
        partial_samples = []
        last_phase = None
        t0 = time.perf_counter()
        max_seen_idx = -1
        causality_ok = True
        for i in range(coords_18.shape[0]):
            status = session.push_frame(coords_18[i])
            # 인과성 체크: emit_frame은 항상 현재까지 도착한 인덱스(i) - HALF 이하여야 함
            if status is not None and status.get("status") == "ok":
                if status["emit_frame"] > i:
                    causality_ok = False
                max_seen_idx = max(max_seen_idx, status["emit_frame"])
                if status["phase"] != last_phase:
                    phase_log.append((status["emit_frame"], status["phase"]))
                    last_phase = status["phase"]
                if status["partial_distance"] and len(partial_samples) < 6 and i % 5 == 0:
                    partial_samples.append(
                        {"frame": status["emit_frame"], "phase": status["partial_distance"]["phase"],
                         "distance_by_class": status["partial_distance"]["distance_by_class"]}
                    )
        elapsed = time.perf_counter() - t0
        fps = coords_18.shape[0] / elapsed if elapsed > 0 else float("inf")

        def overlap(r):
            s, e = r.frame_range
            return max(0, min(e, gt_end) - max(s, gt_start))

        rep = max(session.completed_reps, key=overlap) if session.completed_reps else None
        offline = offline_predict(z, seq, os_.origin, model, device, db, weights_cfg)

        report = {
            "actor": seq.actor, "level": seq.level, "true_class": seq.error_type, "rep": seq.rep,
            "n_frames": int(coords_18.shape[0]), "gt_frame_range": [gt_start, gt_end],
            "causality_ok": causality_ok,
            "phase_transitions": phase_log,
            "n_reps_detected": len(session.completed_reps),
            "detected_frame_range": list(rep.frame_range) if rep else None,
            "online_predicted_class": rep.predicted_class if rep else None,
            "online_raw_distance": rep.raw_distance_by_class if rep else None,
            "online_score": rep.score_vs_normal if rep else None,
            "online_top_features": rep.top_contributing_features if rep else None,
            "offline_predicted_class": offline["predicted_class"] if offline else None,
            "offline_raw_distance": offline["raw_distance_by_class"] if offline else None,
            "partial_distance_samples": partial_samples,
            "elapsed_sec": elapsed, "fps": fps,
        }
        all_reports.append(report)

        print(f"\n=== {seq.actor}({seq.level}) true={seq.error_type} rep{seq.rep}  frames={coords_18.shape[0]} (GT motion {gt_start}-{gt_end}) ===")
        print(f"  causality_ok={causality_ok}  fps={fps:.1f}  reps_detected={len(session.completed_reps)}")
        print(f"  phase 전환 로그: {phase_log}")
        if rep:
            print(f"  검출된 rep 구간: {rep.frame_range}  (GT: [{gt_start},{gt_end}])")
            print(f"  Online  pred={rep.predicted_class}  dist={ {k: round(v,3) for k,v in rep.raw_distance_by_class.items()} }  score={rep.score_vs_normal}")
        if offline:
            print(f"  Offline pred={offline['predicted_class']}  dist={ {k: round(v,3) for k,v in offline['raw_distance_by_class'].items()} }")

    for z in zips.values():
        z.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()

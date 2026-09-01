#!/usr/bin/env python3
"""joint_feedback.py의 관절별 "합격 범위" 폭을 AI Hub CSV 실측 데이터로 다시 계산한다.

기존 ANGLE_TOLERANCE_DEG=15.0(모든 관절/phase 공통)은 실측 통계가 아니라 실사용
영상 몇 프레임을 보고 손으로 잡은 값이었다(2026-08-28). 이 스크립트는 "정상" 레이블
train 시퀀스 전부에서 관절별 각도가 실제 배우들 사이에 얼마나 자연스럽게 벌어지는지
계산해 그 폭으로 대체한다.

방법
----
1. "정상"(train) 시퀀스마다 phase(준비/하강/최저점/상승/종료)를 분할하고, 그 안에서
   관절 각도 시계열을 진행률 0~100%(K=20 지점)로 리샘플한다 — 그냥 프레임을 다 모아
   분산을 재면 사람마다 다른 하강/상승 "속도"가 자세 차이처럼 부풀어 보이기 때문에,
   "같은 순간(진행률)"끼리만 비교한다.
2. 각 (phase, 진행률 지점, 관절)에서 배우들 사이 각도의 10~90퍼센타일 폭을 구하고,
   그 폭의 절반을 "합격 허용오차"로 쓴다 — online_dtw.OnlineSquatSession이 실시간
   비교에 쓰는 "정상" 레퍼런스 프레임 하나와 사용자 프레임을 비교하는 것과 동일한
   상황(특정 순간의 각도 하나 대 하나 비교)을 재현하기 위함.
3. 진행률 지점(K=20)들의 값을 median으로 묶어 (phase, joint) 하나당 숫자 하나로
   요약해 configs/joint_angle_tolerance.json에 저장한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ai_trainer.actor_split import load_all_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES  # noqa: E402
from ai_trainer.joint_feedback import TRACKED_JOINTS, _angle_deg  # noqa: E402
from ai_trainer.lifting_dataset import load_actor_split  # noqa: E402
from ai_trainer.phase_features import extract_phase_features  # noqa: E402
from ai_trainer.phase_segmentation import segment_phases  # noqa: E402
from ai_trainer.reference_pipeline import build_ground_truth_reference  # noqa: E402

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
OUT_PATH = ROOT / "configs" / "joint_angle_tolerance.json"
PHASES = ["준비", "하강", "최저점", "상승", "종료"]
K = 20  # phase 진행률 리샘플링 포인트 수


def main() -> None:
    actor_to_split = load_actor_split(SPLIT_PATH)
    all_seqs = load_all_air_squat_sequences(TL_ZIP, VL_ZIP)
    normal_seqs = [os_ for os_ in all_seqs if os_.seq.error_type == "정상" and actor_to_split.get(os_.seq.actor) == "train"]
    print(f"정상(train) 시퀀스: {len(normal_seqs)}개, actor {len({o.seq.actor for o in normal_seqs})}명")

    zips = {"TL": AiHubZip(TL_ZIP), "VL": AiHubZip(VL_ZIP)}
    idx = {n: i for i, n in enumerate(COMMON_JOINT_NAMES)}

    by_phase_joint: dict[str, dict[str, list[np.ndarray]]] = {}
    n_ok = 0
    for os_ in normal_seqs:
        ref = build_ground_truth_reference(zips[os_.origin], os_.seq, os_.origin)
        if ref is None or ref.coords.shape[0] < 10:
            continue
        pf = extract_phase_features(ref.coords)
        bounds = segment_phases(pf).as_dict()
        n_ok += 1
        for phase in PHASES:
            s, e = bounds[phase]
            if e - s < 2:
                continue
            for joint_name, (a_name, vertex_name, b_name) in TRACKED_JOINTS.items():
                ai_, vi, bi = idx[a_name], idx[vertex_name], idx[b_name]
                angles = np.array(
                    [_angle_deg(ref.coords[t, ai_], ref.coords[t, vi], ref.coords[t, bi]) for t in range(s, e)]
                )
                orig_t = np.linspace(0, 1, len(angles))
                new_t = np.linspace(0, 1, K)
                resampled = np.interp(new_t, orig_t, angles)
                by_phase_joint.setdefault(phase, {}).setdefault(joint_name, []).append(resampled)
    for z in zips.values():
        z.close()
    print(f"유효 시퀀스: {n_ok}개, 진행률 리샘플 포인트 K={K}")

    tolerance: dict[str, dict[str, float]] = {}
    detail: dict[str, dict[str, dict]] = {}
    for phase in PHASES:
        tolerance[phase] = {}
        detail[phase] = {}
        for joint_name in TRACKED_JOINTS:
            arr = np.array(by_phase_joint[phase][joint_name])  # (n_seq, K)
            p10 = np.percentile(arr, 10, axis=0)
            p90 = np.percentile(arr, 90, axis=0)
            median_range90 = float(np.median(p90 - p10))
            half = round(median_range90 / 2, 1)
            tolerance[phase][joint_name] = half
            detail[phase][joint_name] = {"n_seq": int(arr.shape[0]), "median_p10_p90_range_deg": round(median_range90, 1)}
            print(f"  {phase:4s} {joint_name:12s} n_seq={arr.shape[0]:4d}  합격 허용오차=±{half:.1f}도")

    payload = {
        "description": (
            "관절별 '합격 범위' 허용오차(도). AI Hub '정상' train 시퀀스의 관절 각도가 "
            "phase 진행률상 같은 순간에 배우들 사이에서 실제로 얼마나 벌어지는지(10~90퍼센타일 "
            "폭의 절반)로 계산 — 2026-08-28 이전에 쓰던 고정값(모든 phase/관절 공통 15도)은 "
            "실측 통계가 아니라 어림값이었음."
        ),
        "method": f"정상(train) 시퀀스 -> phase 분할 -> 진행률 0~100%(K={K})로 리샘플 -> 진행률 지점별 배우간 각도 10~90퍼센타일 폭의 median의 절반",
        "n_sequences": n_ok,
        "tolerance_deg": tolerance,
        "detail": detail,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()

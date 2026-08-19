#!/usr/bin/env python3
"""정면 카메라 후보를 데이터 전수(여러 actor/repetition/오류유형)에 걸쳐 정량 검증한다.

각 camera의 2D Skeleton과, 동일 프레임 3D Skeleton의 3가지 orthographic
projection(X-Y, X-Z, Y-Z)을 각각 translation/scale 정규화 후 Procrustes
정합하여 잔차(disparity, 낮을수록 유사)를 계산한다.

카메라 하나를 "확정"하지 않고, 여러 actor/난이도/오류유형에 걸쳐
- 어떤 camera가 어떤 projection과 가장 지속적으로 유사한지
- actor/동작에 따라 결과가 바뀌는지
를 표로 보고한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd  # noqa: E402

from ai_trainer.actor_split import load_all_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import JOINT_NAMES, AiHubZip  # noqa: E402
from ai_trainer.camera_id import PROJECTIONS, evaluate_frame, sample_frame_indices  # noqa: E402

_NOSE, _LEYE, _REYE = JOINT_NAMES.index("Nose"), JOINT_NAMES.index("LEye"), JOINT_NAMES.index("REye")


def face_plausibility(coords_2d_frame) -> bool:
    """Nose_x가 LEye_x~REye_x 사이에 있으면 얼굴이 정상적으로 보이는(=정면성 있는) 프레임으로 본다.

    camera0/camera1처럼 Procrustes 상으로는 구분되지 않는(반사 허용) 앞/뒤 카메라 쌍을
    구분하기 위한 보조 지표. 뒤통수만 보이는 카메라는 얼굴 랜드마크가 기하학적으로
    앞뒤가 안 맞아 이 조건을 잘 만족하지 못한다.
    """
    nose_x = coords_2d_frame[_NOSE, 0]
    ley_x, rey_x = coords_2d_frame[_LEYE, 0], coords_2d_frame[_REYE, 0]
    return min(ley_x, rey_x) <= nose_x <= max(ley_x, rey_x)

TL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Training/02.라벨링데이터/TL.zip"
)
VL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip"
)
OUT_CSV = Path(__file__).resolve().parent.parent / "output" / "camera_projection_disparity.csv"

FRAMES_PER_SEQ_CAMERA = 10  # 시퀀스x카메라당 표본 프레임 수
ACTORS_PER_PREFIX = {"CA": 6, "CB": 8, "CI": 6}  # 난이도별 표본 actor 수
REPS_PER_CONDITION = 2  # (actor, 오류유형)당 최대 반복 수


def pick_sample_sequences(seed: int = 7):
    """난이도(prefix)별로 actor를 고르고, 각 actor의 모든 오류유형에서
    최대 REPS_PER_CONDITION개의 시퀀스를 뽑는다."""
    import random

    rng = random.Random(seed)
    all_seqs = load_all_air_squat_sequences(TL_ZIP, VL_ZIP)

    by_actor_cond: dict[tuple[str, str], list] = {}
    for os_ in all_seqs:
        key = (os_.seq.actor, os_.seq.error_type)
        by_actor_cond.setdefault(key, []).append(os_)

    actors_by_prefix: dict[str, list[str]] = {}
    for os_ in all_seqs:
        actors_by_prefix.setdefault(os_.seq.actor[:2], []).append(os_.seq.actor)
    for p in actors_by_prefix:
        actors_by_prefix[p] = sorted(set(actors_by_prefix[p]))

    chosen_actors: list[str] = []
    for prefix, n in ACTORS_PER_PREFIX.items():
        pool = actors_by_prefix.get(prefix, [])
        rng.shuffle(pool)
        chosen_actors.extend(pool[:n])

    chosen = []
    for actor in chosen_actors:
        for err in ["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"]:
            candidates = by_actor_cond.get((actor, err), [])
            rng.shuffle(candidates)
            chosen.extend(candidates[:REPS_PER_CONDITION])
    return chosen


def main() -> None:
    sample = pick_sample_sequences()
    print(f"표본 시퀀스: {len(sample)}개 (TL+VL 통합 풀에서 stratified sampling)")

    zips = {"TL": AiHubZip(TL_ZIP), "VL": AiHubZip(VL_ZIP)}
    rows = []
    n_done = 0
    for os_ in sample:
        z = zips[os_.origin]
        seq = os_.seq
        try:
            frames_3d, coords_3d = z.read_3d(seq)
        except Exception as e:  # noqa: BLE001
            print(f"[스킵] {seq}: 3D 읽기 실패 ({e})")
            continue

        cams = z.list_cameras(seq)
        for cam in cams:
            try:
                frames_2d, coords_2d = z.read_2d(seq, cam)
                ann = z.read_annotation(seq, cam)
            except Exception as e:  # noqa: BLE001
                continue

            n = min(len(frames_3d), len(frames_2d))
            if n < 5:
                continue

            start_f, end_f = 0, n - 1
            if ann.get("annotations"):
                a0 = ann["annotations"][0]
                start_f = max(0, min(a0["start_frame"], n - 1))
                end_f = max(start_f, min(a0["end_frame"], n - 1))

            for t in sample_frame_indices(start_f, end_f, FRAMES_PER_SEQ_CAMERA):
                if t >= n:
                    continue
                disp = evaluate_frame(coords_2d[t], coords_3d[t])
                rows.append(
                    {
                        "origin": os_.origin,
                        "actor": seq.actor,
                        "level": seq.level,
                        "error_type": seq.error_type,
                        "rep": seq.rep,
                        "camera": cam,
                        "frame": t,
                        "face_plausible": face_plausibility(coords_2d[t]),
                        **disp,
                    }
                )
        n_done += 1
        if n_done % 20 == 0:
            print(f"  ...{n_done}/{len(sample)} 시퀀스 처리")

    for z in zips.values():
        z.close()

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n원본 결과 저장: {OUT_CSV}  (행 {len(df)}개)")

    proj_cols = list(PROJECTIONS.keys())

    print("\n=== [A] camera x projection 평균 Procrustes disparity (낮을수록 유사, 전체 풀링) ===")
    pivot = df.groupby("camera")[proj_cols].agg(["mean", "std", "count"])
    print(pivot.round(4).to_string())

    print("\n=== [B] camera별 best-matching projection ===")
    means = df.groupby("camera")[proj_cols].mean()
    for cam, row in means.iterrows():
        best = row.idxmin()
        gap = sorted(row.values)[1] - row.min()
        print(f"  camera{cam}: best={best} (mean={row[best]:.4f}), 2위와 차이={gap:.4f}  전체={row.round(4).to_dict()}")

    print("\n=== [C] Y-Z projection 기준 camera 순위 (전체 풀링) ===")
    yz_rank = means["YZ"].sort_values()
    print(yz_rank.round(4).to_string())

    print("\n=== [D] 시퀀스별 'Y-Z와 가장 유사한 camera' 다수결 집계 (단일 시퀀스에 의존하지 않는지 확인) ===")
    per_seq = (
        df.groupby(["origin", "actor", "error_type", "rep", "camera"])["YZ"]
        .mean()
        .reset_index()
    )
    winners = per_seq.loc[per_seq.groupby(["origin", "actor", "error_type", "rep"])["YZ"].idxmin()]
    winner_counts = winners["camera"].value_counts().sort_index()
    print(f"  표본 시퀀스 수: {winners.shape[0]}")
    print(winner_counts.to_string())
    top_camera = winner_counts.idxmax()
    top_ratio = winner_counts.max() / winners.shape[0]
    print(f"  => 가장 자주 1위인 camera: camera{top_camera} ({winner_counts.max()}/{winners.shape[0]} = {top_ratio:.1%})")

    print("\n=== [E] 난이도(level)별 Y-Z 기준 camera 순위 (actor 그룹에 따라 바뀌는지) ===")
    for level in ["초급", "중급", "고급"]:
        sub = df[df["level"] == level]
        if sub.empty:
            continue
        rank = sub.groupby("camera")["YZ"].mean().sort_values()
        print(f"  [{level}] " + ", ".join(f"cam{c}={v:.3f}" for c, v in rank.items()))

    print("\n=== [F] 오류유형(error_type)별 Y-Z 기준 camera 순위 (동작 종류에 따라 바뀌는지) ===")
    for err in ["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"]:
        sub = df[df["error_type"] == err]
        if sub.empty:
            continue
        rank = sub.groupby("camera")["YZ"].mean().sort_values()
        print(f"  [{err}] " + ", ".join(f"cam{c}={v:.3f}" for c, v in rank.items()))

    print(
        "\n=== [G] camera0 vs camera1 얼굴 신뢰도(face_plausible) 비교 ==="
        "\n    (Procrustes는 반사를 허용해 '앞-뒤 축'은 구분해도 '정면 vs 후면'은 구분 못 함 -> 보조 지표로 구분)"
    )
    face01 = df[df["camera"].isin([0, 1])].groupby("camera")["face_plausible"].mean()
    print(face01.round(3).to_string())
    if 0 in face01 and 1 in face01:
        front_guess = int(face01.idxmax())
        print(f"  => face_plausible 비율이 더 높은 camera{front_guess}가 정면(얼굴이 보이는 쪽)일 가능성이 높음")


if __name__ == "__main__":
    main()

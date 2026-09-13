#!/usr/bin/env python3
"""난이도·오류유형별 Multi-Reference DB 구축.

1. actor split이 train인 시퀀스만 난이도·클래스별로 표본 추출한다.
2. 각 시퀀스에 Ground Truth Reference 전처리(Common Skeleton -> Hip-center ->
   Scale Norm -> Orientation Align)를 적용하고 phase feature/phase segmentation을 계산한다.
3. (pelvis_height, knee_flexion, hip_flexion) 3차원 시계열로 난이도·클래스 그룹 내부
   pairwise DTW 거리행렬을 만들고 k-medoids(k=4)로 대표 시퀀스를 선정한다.
4. 선정된 medoid에 대해서만 Operational Reference(학습된 lifting 모델 통과)도 생성한다.
5. manifest.json(metadata) + sequences.npz(좌표 배열)로 저장한다.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_trainer.actor_split import load_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.clustering import kmedoids, pairwise_dtw_distance_matrix, sequence_feature_matrix  # noqa: E402
from ai_trainer.lifting_dataset import load_actor_split  # noqa: E402
from ai_trainer.lifting_model import TemporalLiftingNet  # noqa: E402
from ai_trainer.phase_features import extract_phase_features  # noqa: E402
from ai_trainer.phase_segmentation import segment_phases  # noqa: E402
from ai_trainer.reference_pipeline import build_ground_truth_reference, build_operational_reference  # noqa: E402
from ai_trainer.dataset_config import DATASET_PATH  # noqa: E402
from ai_trainer.reference_levels import DIFFICULTY_LEVELS, REFERENCE_CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = ROOT / "configs" / "actor_split.json"
OUT_DIR = ROOT / "output" / "reference_db"

CLASSES = list(REFERENCE_CLASSES)
LEVELS = list(DIFFICULTY_LEVELS)
N_CANDIDATES_PER_GROUP = 40  # DTW 계산량을 위해 난이도·클래스 그룹당 후보 상한
K_MEDOIDS = 4
DTW_RADIUS = 5
SEED = 17


def sample_candidates(
    all_seqs,
    cls: str,
    level: str,
    actor_to_split: dict[str, str],
    split: str,
    n: int,
    seed: int,
):
    """지정 난이도·클래스·split에서 actor당 1개를 우선 표본 추출한다."""
    rng = random.Random(seed)
    pool = [
        os_
        for os_ in all_seqs
        if os_.seq.error_type == cls
        and os_.seq.level == level
        and actor_to_split.get(os_.seq.actor) == split
    ]
    by_actor = {}
    for os_ in pool:
        by_actor.setdefault(os_.seq.actor, []).append(os_)
    actors = list(by_actor.keys())
    rng.shuffle(actors)

    chosen = []
    for a in actors:
        cands = by_actor[a]
        chosen.append(rng.choice(cands))
        if len(chosen) >= n:
            return chosen
    # actor 수가 n보다 적으면 남은 시퀀스에서 추가로 채움
    remaining = [os_ for os_ in pool if os_ not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, n - len(chosen))])
    return chosen[:n]


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TemporalLiftingNet(n_joints=18, hidden=128)
    model.load_state_dict(torch.load(ROOT / "output" / "lifting_baseline" / "model_best.pt", map_location=device))
    model.to(device).eval()

    actor_to_split = load_actor_split(SPLIT_PATH)
    print(f"데이터셋: {DATASET_PATH}")
    all_seqs = load_air_squat_sequences(DATASET_PATH)
    zips = {"DATASET": AiHubZip(DATASET_PATH)}

    manifest = []
    seq_arrays: dict[str, np.ndarray] = {}

    for cls in CLASSES:
        for level in LEVELS:
            group_name = f"{cls}/{level}"
            print(f"\n=== [{group_name}] ===")
            candidates = sample_candidates(
                all_seqs,
                cls,
                level,
                actor_to_split,
                split="train",
                n=N_CANDIDATES_PER_GROUP,
                seed=SEED,
            )
            if len(candidates) < K_MEDOIDS:
                raise RuntimeError(
                    f"[{group_name}] train 후보 시퀀스 부족: "
                    f"{len(candidates)}개 (k={K_MEDOIDS} 필요)"
                )
            print(f"후보 {len(candidates)}개 시퀀스에 GT 전처리 적용 중...")

            built = []  # (os_, ref, feats, bounds, dtw_feat)
            for os_ in candidates:
                ref = build_ground_truth_reference(zips[os_.origin], os_.seq, os_.origin)
                if ref is None:
                    continue
                feats = extract_phase_features(ref.coords)
                bounds = segment_phases(feats)
                dtw_feat = sequence_feature_matrix(
                    feats.pelvis_height,
                    feats.knee_flexion_deg,
                    feats.hip_flexion_deg,
                )
                built.append((os_, ref, feats, bounds, dtw_feat))
            if len(built) < K_MEDOIDS:
                raise RuntimeError(
                    f"[{group_name}] GT 전처리 후 유효 시퀀스 부족: "
                    f"{len(built)}개 (k={K_MEDOIDS} 필요)"
                )
            print(
                f"유효 시퀀스 {len(built)}개, pairwise DTW 계산 중 "
                f"({len(built)*(len(built)-1)//2}쌍)..."
            )

            t0 = time.time()
            dist = pairwise_dtw_distance_matrix([b[4] for b in built], radius=DTW_RADIUS)
            print(f"DTW 완료 ({time.time()-t0:.1f}초)")

            medoid_idx, assignment = kmedoids(dist, k=K_MEDOIDS, seed=SEED)
            print(
                f"medoid {K_MEDOIDS}개 선정: "
                + ", ".join(
                    f"{built[i][0].seq.actor}/rep{built[i][0].seq.rep}"
                    for i in medoid_idx
                )
            )

            for rank, i in enumerate(medoid_idx):
                os_, gt_ref, feats, bounds, _ = built[i]
                seq = os_.seq
                cluster_size = int(
                    (assignment == np.where(medoid_idx == i)[0][0]).sum()
                )

                op_ref = build_operational_reference(
                    zips[os_.origin], seq, os_.origin, model, device
                )

                base_id = f"{cls}_{level}_{rank}_{seq.actor}_rep{seq.rep}"
                norm_meta = {
                    "scale_leg_length_gt": gt_ref.scale,
                    "scale_leg_length_operational": op_ref.scale if op_ref else None,
                    "orientation_reference_frames": 5,
                }
                common_meta = {
                    "medoid_id": base_id,
                    "medoid_rank": rank,
                    "class_label": cls,
                    "actor_id": seq.actor,
                    "difficulty_level": level,
                    "repetition_id": seq.rep,
                    "source_split": actor_to_split.get(seq.actor, "unknown"),
                    "origin_zip": os_.origin,
                    "frame_range": list(gt_ref.frame_range),
                    "sequence_length": int(gt_ref.coords.shape[0]),
                    "is_medoid": True,
                    "cluster_size": cluster_size,
                    "phase_boundaries": bounds.as_dict(),
                    "normalization_metadata": norm_meta,
                }

                manifest.append(
                    {
                        **common_meta,
                        "tier": "ground_truth",
                        "array_key": f"{base_id}__gt",
                    }
                )
                seq_arrays[f"{base_id}__gt"] = gt_ref.coords

                if op_ref is not None:
                    manifest.append(
                        {
                            **common_meta,
                            "tier": "operational",
                            "array_key": f"{base_id}__operational",
                        }
                    )
                    seq_arrays[f"{base_id}__operational"] = op_ref.coords
                else:
                    print(
                        f"  [경고] {base_id}: Operational Reference 생성 실패 "
                        "(camera1 없음 등)"
                    )

    for z in zips.values():
        z.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "classes": CLASSES,
                "levels": LEVELS,
                "k_medoids_per_level_class": K_MEDOIDS,
                "entries": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(OUT_DIR / "sequences.npz", **seq_arrays)
    print(f"\n저장 완료: {OUT_DIR / 'manifest.json'} ({len(manifest)}개 항목), {OUT_DIR / 'sequences.npz'}")


if __name__ == "__main__":
    main()

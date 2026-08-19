"""클래스 내부 시퀀스 pairwise DTW distance + 간단한 k-medoids.

과도하게 복잡한 clustering 라이브러리를 새로 들이지 않고, fastdtw(distance)와
직접 구현한 PAM류 k-medoids만 사용한다 (요청: "처음부터 과도하게 복잡하게 만들지 말라").
"""
from __future__ import annotations

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def sequence_feature_matrix(pelvis_height: np.ndarray, knee_flexion_deg: np.ndarray, hip_flexion_deg: np.ndarray) -> np.ndarray:
    """DTW distance 계산에 쓸 (T,3) feature. 각도는 0~1 스케일로 맞춰 pelvis_height와 균형."""
    return np.stack([pelvis_height, knee_flexion_deg / 180.0, hip_flexion_deg / 180.0], axis=1)


def pairwise_dtw_distance_matrix(sequences: list[np.ndarray], radius: int = 5) -> np.ndarray:
    """sequences: 길이가 서로 다를 수 있는 (T_i, D) 배열 리스트. fastdtw(근사 DTW) 사용."""
    n = len(sequences)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d, _ = fastdtw(sequences[i], sequences[j], radius=radius, dist=euclidean)
            dist[i, j] = dist[j, i] = d
    return dist


def kmedoids(dist: np.ndarray, k: int, seed: int = 0, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """단순 PAM류 k-medoids. dist: (N,N) 대칭 거리행렬.

    반환: (medoid_indices(k,), assignment(N,) — 각 포인트가 속한 medoid의 인덱스 위치(0..k-1))
    """
    n = dist.shape[0]
    rng = np.random.default_rng(seed)

    # 초기화: 서로 최대한 멀리 떨어진 점들을 고르는 farthest-point 방식
    medoids = [int(rng.integers(n))]
    for _ in range(1, k):
        d_to_nearest = dist[:, medoids].min(axis=1)
        medoids.append(int(np.argmax(d_to_nearest)))
    medoids = np.array(medoids)

    assignment = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d_to_medoids = dist[:, medoids]  # (n,k)
        new_assignment = np.argmin(d_to_medoids, axis=1)

        new_medoids = medoids.copy()
        for c in range(k):
            members = np.where(new_assignment == c)[0]
            if len(members) == 0:
                continue
            sub = dist[np.ix_(members, members)]
            cost = sub.sum(axis=1)
            new_medoids[c] = members[int(np.argmin(cost))]

        if np.array_equal(new_medoids, medoids) and np.array_equal(new_assignment, assignment):
            medoids, assignment = new_medoids, new_assignment
            break
        medoids, assignment = new_medoids, new_assignment

    return medoids, assignment

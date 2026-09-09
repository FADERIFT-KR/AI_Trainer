"""소형 1D-CNN 기반 스쿼트 자세 4클래스(정상/발뒤꿈치오류/엉덩이하방오류/고관절오류)
분류기.

DTW(medoid 거리 기반) 판정과 딥러닝을 actor 단위 5-fold 교차검증으로 정직하게
비교한 결과(2026-09-08, 262개 유효 시퀀스/46명 actor):

    DTW만    : AP 88.8% ± 7.4%   (fold별 74.5~95.0%로 요동 — seed/구성에 민감)
    DL만     : AP 95.8% ± 1.5%   (fold별 93.2~98.1%로 훨씬 안정적)

DTW를 섞는 앙상블도 시도했지만 섞을수록 AP는 오히려 떨어지고(정확도만 소폭 개선),
사용자가 "보정정확도(AP)가 중요하다"고 명시했으므로 REP 완료 시 최종 클래스 판정은
이 모델이 전담한다 — DTW는 판정에 전혀 관여하지 않는다. 대신 "왜 이 클래스로
판단했는지" 근거 텍스트(feature 기여도)를 만드는 데는 DTW의 phase-aware weighted
거리 분해를 계속 재사용한다(딥러닝 모델 자체는 "어떤 관절이 문제였는지" 같은
해석 가능한 근거를 주지 못하므로).

2026-09-10 추가 검증: 좌우 미러 증강(스쿼트는 좌우 대칭이라는 물리적 전제로 L/R
각도를 맞바꿔 학습 데이터를 2배로 늘리는 것)은 실제로는 AP를 살짝 깎아서(95.7%->
94.7%) 기각. 반대로 "같은 구조를 다른 시드 5개로 학습해 확률을 평균"하는 앙상블은
AP를 올리면서(95.7%->96.2%) 표준편차를 절반으로 줄였다(±2.2%->±1.0%) — 모델이
워낙 작아서(~7.7K 파라미터) 5개를 합쳐도 여전히 가볍다. 그래서 DLSquatClassifier는
모델 여러 개를 들고 있다가 softmax 확률을 평균하는 방식으로 바뀌었다.

입력은 원본 좌표가 아니라 이미 검증된 각도/속도 feature 8종만 쓴다(사용자 결정:
"각도만 보는 게 보수적으로 맞다") — DDTW(features.py._keogh_derivative)에 쓴 것과
동일한 기준각/미분이다. phase(준비/하강/최저점/상승/종료)마다 진행률 0~100%를
K_RESAMPLE개 지점으로 리샘플해 사람마다 다른 스쿼트 "속도"를 정규화하고, 5개
phase를 이어붙여 고정 길이 시퀀스를 만든다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CLASSES = ["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"]
PHASES = ["준비", "하강", "최저점", "상승", "종료"]
FEATURE_KEYS = [
    "knee_flexion_angle", "hip_flexion_angle", "ankle_angle", "torso_inclination",
    "knee_flexion_velocity", "hip_flexion_velocity", "ankle_velocity", "torso_inclination_velocity",
]
IN_CHANNELS = 14  # 2+2+2+1+2+2+2+1
K_RESAMPLE = 15


def resample_phase(arr: np.ndarray, k: int) -> np.ndarray:
    """arr: (n, d) -> (k, d), 진행률 0~1 기준 선형보간(compute_joint_angle_tolerance.py와
    동일한 방식)."""
    n = arr.shape[0]
    if n == 1:
        return np.repeat(arr, k, axis=0)
    orig_t = np.linspace(0, 1, n)
    new_t = np.linspace(0, 1, k)
    return np.stack([np.interp(new_t, orig_t, arr[:, d]) for d in range(arr.shape[1])], axis=1)


def build_fixed_vector(feat: dict[str, np.ndarray], bounds: dict) -> np.ndarray | None:
    """feat: features.extract_all_features() 결과. bounds: {phase: [s,e]}.
    반환: (5*K_RESAMPLE, IN_CHANNELS) 고정 길이 벡터, phase 하나라도 프레임이
    2개 미만이면(신뢰 불가) None."""
    blocks = []
    for phase in PHASES:
        s, e = bounds[phase]
        if e - s < 2:
            return None
        cols = [feat[key][s:e] for key in FEATURE_KEYS]
        phase_block = np.concatenate(cols, axis=1)
        blocks.append(resample_phase(phase_block, K_RESAMPLE))
    return np.concatenate(blocks, axis=0).astype(np.float32)


class SmallSquatCNN(nn.Module):
    """~7.5K 파라미터의 작은 1D-CNN. 데이터가 46명 actor/262개 시퀀스뿐이라
    LSTM/Transformer 같은 무거운 모델은 과적합 위험이 크다고 판단해 최소 구성으로
    맞췄다(BatchNorm+Dropout 0.3, weight_decay와 함께 씀)."""

    def __init__(self, in_ch: int = IN_CHANNELS, n_classes: int = len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, in_ch, T)
        z = self.net(x).squeeze(-1)
        return self.fc(z)


DEFAULT_ENSEMBLE_SIZE = 5


class DLSquatClassifier:
    """학습된 checkpoint(들) + 정규화 통계(mu/sigma, train set에서만 계산)를 로드해
    완료된 REP 하나를 4클래스 확률로 분류한다. 모델을 여러 개(ensemble) 들고 있으면
    softmax 확률을 평균한다 — 검증된 대로(2026-09-10) 단일 모델보다 AP가 높고
    표준편차가 절반이라 훨씬 안정적이다."""

    def __init__(self, models: list[SmallSquatCNN], mu: np.ndarray, sigma: np.ndarray):
        self.models = models
        for m in self.models:
            m.eval()
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def load(cls, checkpoint_dir: str | Path, norm_path: str | Path | None = None) -> "DLSquatClassifier":
        """checkpoint_dir: model_0.pt, model_1.pt, ... 를 담은 디렉터리(구버전 호환:
        checkpoint_dir이 단일 .pt 파일 경로면 모델 1개짜리로 취급). norm_path 생략 시
        checkpoint_dir과 같은 디렉터리의 norm_stats.npz를 쓴다."""
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.is_file():
            # 구버전 단일 체크포인트 경로 호환
            model = SmallSquatCNN()
            model.load_state_dict(torch.load(checkpoint_dir, map_location="cpu"))
            models = [model]
            norm_path = norm_path or checkpoint_dir.with_name("norm_stats.npz")
        else:
            paths = sorted(checkpoint_dir.glob("model_*.pt"))
            if not paths:
                raise FileNotFoundError(f"{checkpoint_dir}에 model_*.pt 체크포인트가 없습니다.")
            models = []
            for p in paths:
                m = SmallSquatCNN()
                m.load_state_dict(torch.load(p, map_location="cpu"))
                models.append(m)
            norm_path = norm_path or checkpoint_dir / "norm_stats.npz"
        norm = np.load(norm_path)
        return cls(models, norm["mu"], norm["sigma"])

    def predict_proba(self, feat: dict[str, np.ndarray], bounds: dict) -> np.ndarray | None:
        """반환: CLASSES 순서의 (4,) softmax 확률(앙상블 평균). phase가 너무 짧아
        신뢰할 수 없는 REP면 None(호출부는 이 경우 DTW로 fallback해야 한다)."""
        vec = build_fixed_vector(feat, bounds)
        if vec is None:
            return None
        x = vec.T[None, :, :]  # (1, in_ch, T)
        x = (x - self.mu) / self.sigma
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            probs = np.mean([torch.softmax(m(x_t), dim=1).numpy()[0] for m in self.models], axis=0)
        return probs

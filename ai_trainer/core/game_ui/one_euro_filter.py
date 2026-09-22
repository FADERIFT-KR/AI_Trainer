"""One-Euro Filter — 관절 픽셀좌표 프레임간 지터(jitter) 저역통과 필터.

정지 상태(예: "준비" 자세로 서 있을 때)에서는 강하게 스무딩해 떨림을 줄이고, 빠르게
움직일 때(예: 스쿼트 하강)는 자동으로 스무딩을 풀어 반응 지연을 최소화한다
(Casiez et al. 2012, 1€ Filter — 원래 터치/커서 트래킹용으로 고안됨).

`CommonSkeletonBridge`가 confidence 임계값을 넘긴("good") 관절에만 적용한다 — freeze된
관절(직전 값을 그대로 유지)까지 필터에 흘려보내면 "값이 안 바뀜"이 "정지"로 오인되어
속도 추정이 깨지므로 대상에서 제외한다.

dt는 실제 타임스탬프 대신 online_dtw.py와 동일하게 30fps 고정 가정을 쓴다 — 이미
vel_eps/debounce_n 등 프로젝트 전체가 이 가정을 공유하고 있고(CameraConfig.requested_fps=30),
카메라 실제 처리 속도가 크게 떨어지면 그쪽 임계값들도 같이 재보정이 필요한 것과 동일한
전제다.

기본 파라미터(min_cutoff=0.8, beta=0.02)는 "정지 시 노이즈는 강하게 죽이고, 스쿼트
하강/상승 같은 실제 동작 속도에서는 거의 필터링 없이 통과시킨다"는 목표로 대략적인
픽셀 이동량 크기(정지 지터 ~수십 px/sec vs 실제 동작 ~수백 px/sec)를 기준으로 잡은
초기값이다 — 실제 웹캠 검증 후 조정 필요할 수 있음.
"""
from __future__ import annotations

import numpy as np

DEFAULT_FREQ = 30.0


def _smoothing_factor(dt: float, cutoff: np.ndarray) -> np.ndarray:
    r = 2 * np.pi * cutoff * dt
    return r / (r + 1)


def _exp_smooth(a: np.ndarray, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    """(N,D) 좌표 배열 전체에 관절별(N개) 독립 1€ 필터를 벡터화 적용 (D=2 또는 3)."""

    def __init__(
        self,
        n_points: int,
        n_dims: int = 2,
        freq: float = DEFAULT_FREQ,
        min_cutoff: float = 0.8,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev = np.zeros((n_points, n_dims))
        self.initialized = np.zeros(n_points, dtype=bool)

    def __call__(self, x: np.ndarray, active: np.ndarray) -> np.ndarray:
        """x: (N,D) 이번 프레임 관측치(D=2인 픽셀 좌표 또는 D=3인 world 좌표 등).
        active: (N,) 이번 프레임에 필터링할("good") 관절 마스크.

        active=False인 관절은 필터 내부 상태를 갱신하지 않고 입력값을 그대로 반환한다
        (freeze된 관절은 이미 상위에서 직전 좌표로 대체되므로 여기서 손댈 필요가 없다).
        """
        if self.x_prev is None:
            self.x_prev = x.copy()
        out = x.copy()
        dt = 1.0 / self.freq

        idx = np.where(active)[0]
        if idx.size == 0:
            return out

        dx = (x[idx] - self.x_prev[idx]) / dt
        a_d = _smoothing_factor(dt, np.full_like(dx, self.d_cutoff))
        dx_hat = _exp_smooth(a_d, dx, self.dx_prev[idx])
        first_time = ~self.initialized[idx]
        dx_hat[first_time] = 0.0  # 첫 관측은 속도 추정 불가 -> 0으로 시작(초기 프레임 과도 스무딩 방지)

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _smoothing_factor(dt, cutoff)
        x_hat = _exp_smooth(a, x[idx], self.x_prev[idx])

        out[idx] = x_hat
        self.x_prev[idx] = x_hat
        self.dx_prev[idx] = dx_hat
        self.initialized[idx] = True
        return out

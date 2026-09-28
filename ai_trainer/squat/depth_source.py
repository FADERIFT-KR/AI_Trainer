"""정면 영상에서 3D를 얻는 방법 선택 — MediaPipe 자체 3D / 2D→3D lifting / 혼합.

배경: 정면 촬영에서 상체가 앞으로 기우는 것은 카메라 광축(깊이) 방향 움직임이라
단안 RGB에서 가장 추정하기 어렵다. 실측(front.mp4, 737프레임) 결과:

    상체 전후기울기       서있을때   최저점    차이
    MediaPipe world      -6.2도    -6.8도    0.6도   <- 기울기를 거의 못 잡음
    Lifting 모델         -4.8도   -24.8도   20.0도   <- 잡아냄

반대로 lifting 모델은 스쿼트 깊이를 과소평가하는 것이 확인된 바 있다(2026-08-28).
둘 중 무엇이 이 프로젝트에 맞는지는 실제 영상으로 비교해야 하므로, 실행 중에
바꿔가며 볼 수 있도록 세 모드를 모두 남긴다.

축 규약: lifting 모델 출력은 (전후, 좌우, 상하) 순서이고 좌우·상하 부호가 반대다
(front.mp4 729프레임 상관분석: lifting[0]->z +0.89, lifting[1]->x -0.98,
lifting[2]->y -0.99). `lifting_to_common3d()`가 이를 MediaPipe 규약(x=좌우,
y=상하(아래가 +), z=전후)으로 맞춰준다.
"""
from __future__ import annotations

import numpy as np

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES

SOURCE_MEDIAPIPE = "mediapipe"
SOURCE_LIFTING = "lifting"
SOURCE_HYBRID = "hybrid"

SOURCE_LABELS = (
    (SOURCE_MEDIAPIPE, "MediaPipe 3D", "깊이에 강함 · 정면에서 상체 기울기를 거의 못 잡음"),
    (SOURCE_LIFTING, "Lifting 모델", "2D에서 z 추정 · 상체 기울기를 잡음 · 깊이 과소평가 보고됨"),
    (SOURCE_HYBRID, "혼합", "좌우·상하는 MediaPipe, 전후(깊이)만 Lifting에서 가져옴"),
)
SOURCES = tuple(name for name, _label, _hint in SOURCE_LABELS)

_HIP = COMMON_JOINT_NAMES.index("Hip")
_NECK = COMMON_JOINT_NAMES.index("Neck")


def lifting_to_common3d(lifted: np.ndarray, reference_common3d: np.ndarray) -> np.ndarray:
    """lifting 출력(18,3)을 MediaPipe common3d 규약·크기로 맞춘다.

    lifting은 2D 픽셀을 몸통 길이로 나눈 무차원 좌표라 미터 단위가 아니다. 같은
    프레임의 MediaPipe 몸통 길이로 배율을 맞춰야 두 소스를 섞거나 바꿔 끼울 수 있다.
    """
    remapped = np.stack([-lifted[:, 1], -lifted[:, 2], lifted[:, 0]], axis=-1)
    remapped = remapped - remapped[_HIP]
    own_torso = float(np.linalg.norm(remapped[_NECK] - remapped[_HIP]))
    if own_torso < 1e-6:
        return remapped
    target_torso = float(np.linalg.norm(reference_common3d[_NECK] - reference_common3d[_HIP]))
    return remapped * (target_torso / own_torso)


def combine(source: str, mediapipe_3d: np.ndarray, lifted_common3d: np.ndarray | None) -> np.ndarray:
    """선택한 모드에 맞는 (18,3) 좌표를 돌려준다."""
    if source == SOURCE_MEDIAPIPE or lifted_common3d is None:
        return mediapipe_3d
    if source == SOURCE_LIFTING:
        return lifted_common3d
    if source == SOURCE_HYBRID:
        merged = mediapipe_3d.copy()
        merged[:, 2] = lifted_common3d[:, 2]  # 전후(깊이)만 교체
        return merged
    raise ValueError(f"알 수 없는 3D 소스입니다: {source}")


__all__ = ["SOURCES", "SOURCE_LABELS", "SOURCE_MEDIAPIPE", "SOURCE_LIFTING",
           "SOURCE_HYBRID", "combine", "lifting_to_common3d"]


class LiftingDepthEstimator:
    """2D Common Skeleton 프레임을 모아 lifting 모델로 3D를 만든다.

    `online_dtw.push_frame`과 같은 규약을 쓴다 — WINDOW_T 프레임 창의 가운데 프레임을
    추정하고, 2D 스케일은 처음 `calib_frames`개의 몸통 길이 median으로 한 번만 고정한다
    (프레임마다 다시 잡으면 사람이 카메라에 가까워질 때 좌표가 통째로 흔들린다).
    """

    def __init__(self, model, device, window: int, calib_frames: int = 8):
        self.model = model
        self.device = device
        self.window = window
        self.half = window // 2
        self.calib_frames = calib_frames
        self._buffer: list[np.ndarray] = []
        self._scale2d: float | None = None

    def push(self, common2d: np.ndarray):
        """새 2D 프레임을 넣고, 확정된 과거 프레임의 (index, lifted 3D)를 돌려준다.

        아직 창이 안 찼거나 스케일 보정 전이면 None.
        """
        import torch

        self._buffer.append(np.asarray(common2d, dtype=float).copy())
        center = len(self._buffer) - 1 - self.half
        if center < 0:
            return None
        if self._scale2d is None:
            if center + 1 < self.calib_frames:
                return None
            torso = [
                float(np.linalg.norm(frame[_NECK] - frame[_HIP]))
                for frame in self._buffer[: self.calib_frames]
            ]
            self._scale2d = max(float(np.median(torso)), 1e-6)

        low = max(0, center - self.half)
        window = np.stack(self._buffer[low : center + self.half + 1])
        if window.shape[0] < self.window:  # 시퀀스 시작부는 첫 프레임으로 왼쪽을 채운다
            pad = self.window - window.shape[0]
            window = np.concatenate([np.repeat(window[:1], pad, axis=0), window], axis=0)

        normalized = (window - window[:, _HIP : _HIP + 1, :]) / self._scale2d
        tensor = torch.from_numpy(normalized[None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            lifted = self.model(tensor)[0].cpu().numpy()
        return center, lifted


__all__ = __all__ + ["LiftingDepthEstimator"]

"""프레임 단위 관절별 자세 오차(위치+각도) 계산 — 실시간 UI 막대바/스켈레톤 색상용.

`online_dtw.OnlineSquatSession`이 이미 계산하는 phase-aware weighted DTW(전체 스쿼트
시퀀스가 기준 동작과 얼마나 유사한지)와는 역할이 다르다:

    DTW          = "이번 렙(rep) 전체가 기준 동작과 얼마나 비슷한가" (시퀀스 레벨)
    joint_feedback = "지금 이 순간, 이 관절이 기준 자세에서 얼마나 벗어났는가" (프레임 레벨)

이 모듈은 두 정규화된 3D 프레임(사용자 현재 프레임, 그 프레임에 DTW로 정렬된 reference
프레임)을 입력받아 관절별 오차만 계산한다 — "지금 프레임에 대응하는 reference 프레임이
무엇인지" 찾는 정렬 자체는 `online_dtw.OnlineSquatSession._joint_feedback_frame()`의
책임이다(서브시퀀스 DTW의 종점 탐색을 재사용).

좌표는 이미 Hip-center + Scale + Orientation 정규화가 끝난 공통 좌표계(leg_length
단위, `features.py`/`reference_pipeline.py`와 동일)를 사용하므로, 카메라와의 거리나
사용자 키 차이로 오차가 부풀려지지 않는다. 절대 raw 픽셀 좌표를 쓰지 않는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES

_TOLERANCE_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "joint_angle_tolerance.json"

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}

# 관절별 각도 정의: (끝점 A, 각도의 꼭짓점, 끝점 B).
#   무릎 = Hip-Knee-Ankle (요청사항)
#   고관절 = Shoulder-Hip-Knee (요청사항 — DTW의 hip_flexion_angle(Neck 기준)과는 다름,
#            여기서는 좌우 각각 어깨를 기준으로 써서 좌우 비대칭도 잡아낼 수 있게 함)
#   발목 = Knee-Ankle-BigToe (DTW의 ankle_angle과 동일 정의, features.py 참고) — 2026-09-03
#         추가. DTW가 "발뒤꿈치오류"를 이 각도로 판별하도록 바뀐 뒤(위치기반 heel_height
#         제거), 관절 패널에 발목이 없어서 "발뒤꿈치오류"인데 근거로는 무관한 무릎/고관절
#         각도만 보여주는 불일치가 생겼다 — DTW 판정 근거와 화면 설명을 일치시키기 위해 추가.
#
# 어깨(Hip-Shoulder-Elbow)는 의도적으로 제외했다(2026-08-28, 실사용 확인) — 이 각도는
# 상완을 얼마나 앞으로 뻗었는지를 재는데, "정상" 레퍼런스 프레임을 고르는 정렬 자체가
# F_squat_form_focus 가중치(팔 좌표 제외)로 이뤄져서 팔 위치와 무관하게 골라진다. 그
# 결과 사용자 팔이 안정적으로 앞으로 뻗어 있어도(정상 에어스쿼트 균형 자세) 매번 다른
# 팔 자세의 레퍼런스 프레임과 비교돼 각도오차 50도 이상이 계속 나와 항상 BAD로 뜨는
# 문제가 있었다. DTW 쪽에서 이미 같은 이유로 팔 raw 좌표를 뺀 것과 동일한 결정.
TRACKED_JOINTS: dict[str, tuple[str, str, str]] = {
    "Left Knee": ("LHip", "LKnee", "LAnkle"),
    "Right Knee": ("RHip", "RKnee", "RAnkle"),
    "Left Hip": ("LShoulder", "LHip", "LKnee"),
    "Right Hip": ("RShoulder", "RHip", "RKnee"),
    "Left Ankle": ("LKnee", "LAnkle", "LBigToe"),
    "Right Ankle": ("RKnee", "RAnkle", "RBigToe"),
}

# ============================================================
# 튜닝 파라미터 — 판정 기준을 바꾸고 싶으면 이 블록만 수정하면 된다.
#
# 2026-09-02: 위치오차(position_error)를 판정에서 완전히 뺐다 — "정렬된 서로 다른
# 두 사람"의 절대 관절 위치를 비교하는 값이라, leg_length로 정규화해도 개인마다
# 팔/몸통 비율이 달라 자세가 맞아도 항상 오차가 낀다는 게 실측으로 확인됐고
# (2026-08-28), 사용자도 "위치오차는 빼고 각도만 보수적으로 보는 게 맞다"고
# 확인함. position_error 필드 자체는 진단용으로 계속 계산은 하지만 score/status에는
# 전혀 관여하지 않는다. score/status는 이제 configs/joint_angle_tolerance.json의
# phase별 CSV 실측 허용오차(tolerance_deg) 하나만 기준으로 정해진다 — 각도오차가
# 허용오차의 절반 이내면 good, 허용오차 이내면 warning, 넘으면 bad.
# ============================================================
SCORE_RATIO_CAP = 2.0  # 진행바 0~100% 표시용: angle_err가 tolerance_deg의 이 배수면 100%로 클램프

# "합격 범위"로 화면에 보여줄 각도 허용 오차(도), phase/관절별로 다름 —
# configs/joint_angle_tolerance.json(scripts/compute_joint_angle_tolerance.py로 생성)을
# 그대로 읽는다. AI Hub "정상" train 시퀀스 129개에서, phase 진행률(0~100%)이 같은
# 순간끼리 배우들 사이 각도가 실제로 얼마나 벌어지는지(10~90퍼센타일 폭의 절반)를
# 계산한 값이다 — 그냥 프레임을 다 모아 분산을 재면 사람마다 다른 하강/상승 속도가
# 자세 차이처럼 부풀어 보여서, "같은 진행률 지점"끼리만 비교했다. 2026-08-28 이전에
# 쓰던 고정값(모든 phase/관절 공통 15도)은 실측 영상 몇 프레임을 보고 잡은 어림값이었다.
_ANGLE_TOLERANCE_BY_PHASE_JOINT: dict[str, dict[str, float]] = json.loads(
    _TOLERANCE_CFG_PATH.read_text(encoding="utf-8")
)["tolerance_deg"]
# 알 수 없는 phase가 들어오면(예: "prep"/"bottom" 같은 영문 상태값이 실수로 들어옴)
# 쓸 기본값 — 전체 phase 평균 근처인 하강/최저점 수준으로 잡는다.
_DEFAULT_ANGLE_TOLERANCE_DEG = 12.0


def _angle_tolerance_deg(phase: str | None, joint_name: str) -> float:
    by_joint = _ANGLE_TOLERANCE_BY_PHASE_JOINT.get(phase or "")
    if by_joint is None:
        return _DEFAULT_ANGLE_TOLERANCE_DEG
    return by_joint.get(joint_name, _DEFAULT_ANGLE_TOLERANCE_DEG)


# ============================================================

STATUS_GOOD, STATUS_WARNING, STATUS_BAD = "good", "warning", "bad"


@dataclass(frozen=True)
class JointScore:
    name: str  # 표시용 이름 ("Left Knee" 등)
    common_joint: str  # COMMON_JOINT_NAMES 상의 꼭짓점 이름 (스켈레톤에 그릴 때 위치 참조용)
    position_error: float  # raw, leg_length 단위
    angle_error_deg: float  # raw, degree
    user_angle_deg: float  # 지금 사용자 각도(도)
    ref_angle_deg: float  # 정렬된 "정상" 레퍼런스 프레임의 각도(도) — 합격 범위의 중심
    tolerance_deg: float  # 이 phase/관절의 합격 허용오차(도) — AI Hub 실측 기반, phase마다 다름
    score: float  # 0~1 정규화된 최종 오차 (낮을수록 좋음)
    status: str  # STATUS_GOOD | STATUS_WARNING | STATUS_BAD

    @property
    def tolerance_range_deg(self) -> tuple[float, float]:
        """"합격"으로 볼 각도 범위(ref_angle_deg ± tolerance_deg)."""
        return self.ref_angle_deg - self.tolerance_deg, self.ref_angle_deg + self.tolerance_deg

    @property
    def within_angle_tolerance(self) -> bool:
        return self.angle_error_deg <= self.tolerance_deg


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """b를 꼭짓점으로 하는 a-b-c 각도(도)."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    cos = float(np.dot(v1, v2) / (n1 * n2 + 1e-8))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _status_for(angle_err: float, tolerance_deg: float) -> str:
    ratio = angle_err / tolerance_deg if tolerance_deg > 1e-6 else float("inf")
    if ratio <= 0.5:
        return STATUS_GOOD
    if ratio <= 1.0:
        return STATUS_WARNING
    return STATUS_BAD


def compute_joint_scores(user_frame: np.ndarray, ref_frame: np.ndarray, phase: str | None = None) -> list[JointScore]:
    """user_frame/ref_frame: (18,3) 정규화 완료 좌표. 두 프레임은 이미 "같은 순간"으로
    정렬되어 있다고 가정한다(정렬 자체는 이 함수의 책임이 아님).

    phase: online_dtw._joint_feedback_frame()이 알려주는 현재 phase("하강"/"최저점"/"상승"
    등, 한국어 표기). 관절별 합격 허용오차(tolerance_deg)를 phase에 맞게 고르는 데 쓴다 —
    안 주면(None) 기본값(_DEFAULT_ANGLE_TOLERANCE_DEG)을 쓴다.

    position_error(위치오차)는 진단용으로 계산만 하고 score/status에는 안 쓴다(위 설명 참고).
    joint_score(진행바용, 0~1) = min(1, angle_err / (tolerance_deg * SCORE_RATIO_CAP))
    """
    scores: list[JointScore] = []
    for name, (a_name, vertex_name, b_name) in TRACKED_JOINTS.items():
        vi = _IDX[vertex_name]
        pos_err = float(np.linalg.norm(user_frame[vi] - ref_frame[vi]))

        ai, bi = _IDX[a_name], _IDX[b_name]
        user_angle = _angle_deg(user_frame[ai], user_frame[vi], user_frame[bi])
        ref_angle = _angle_deg(ref_frame[ai], ref_frame[vi], ref_frame[bi])
        angle_err = abs(user_angle - ref_angle)
        tolerance = _angle_tolerance_deg(phase, name)

        score = float(np.clip(angle_err / (tolerance * SCORE_RATIO_CAP), 0.0, 1.0)) if tolerance > 1e-6 else 1.0

        scores.append(
            JointScore(
                name=name, common_joint=vertex_name, position_error=pos_err,
                angle_error_deg=angle_err, user_angle_deg=user_angle, ref_angle_deg=ref_angle,
                tolerance_deg=tolerance,
                score=score, status=_status_for(angle_err, tolerance),
            )
        )
    return scores

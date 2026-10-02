from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES  # noqa: E402
from ai_trainer.core.s3_mapping.source_adapters import (  # noqa: E402
    MEDIAPIPE_INDEX,
    aihub_to_common,
    mediapipe_to_common,
)
from ai_trainer.core.s4_normalize.gravity_frame import align_sequence, fit_gravity_frame  # noqa: E402
from ai_trainer.squat.aihub_zip import JOINT_NAMES  # noqa: E402
from ai_trainer.squat.error_features import FEATURE_NAMES, rep_features  # noqa: E402


IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
SHANK = THIGH = 0.42


def pose(tibia_deg: float, thigh_elev_deg: float, trunk_deg: float, heel_lift: float = 0.0) -> np.ndarray:
    """정답 좌표계(x 오른쪽, y 정면, z 위, m)의 18관절 자세 1프레임."""
    tib, th, tr = np.radians([tibia_deg, thigh_elev_deg, trunk_deg])
    out = np.zeros((len(COMMON_JOINT_NAMES), 3))
    for side, sx in (("L", -0.1), ("R", 0.1)):
        ankle = np.array([sx, 0.0, 0.08])
        knee = ankle + SHANK * np.array([0.0, np.sin(tib), np.cos(tib)])
        hip = knee + THIGH * np.array([0.0, -np.cos(th), np.sin(th)])
        out[IDX[f"{side}Ankle"]] = ankle
        out[IDX[f"{side}Knee"]] = knee
        out[IDX[f"{side}Hip"]] = hip
        out[IDX[f"{side}Heel"]] = [sx, -0.06, 0.03 + heel_lift]
        out[IDX[f"{side}BigToe"]] = [sx * 1.2, 0.15, 0.02]
    pelvis = (out[IDX["LHip"]] + out[IDX["RHip"]]) / 2.0
    trunk = 0.5 * np.array([0.0, np.sin(tr), np.cos(tr)])
    for side, sx in (("L", -0.18), ("R", 0.18)):
        shoulder = pelvis + trunk + [sx, 0.0, 0.0]
        out[IDX[f"{side}Shoulder"]] = shoulder
        out[IDX[f"{side}Elbow"]] = shoulder + [0.0, 0.0, -0.28]
        out[IDX[f"{side}Wrist"]] = shoulder + [0.0, 0.0, -0.53]
    out[IDX["Hip"]] = pelvis
    out[IDX["Neck"]] = (out[IDX["LShoulder"]] + out[IDX["RShoulder"]]) / 2.0
    return out


def squat(frames: int = 31, heel_lift: float = 0.0, hold: int = 5) -> np.ndarray:
    """선 자세 유지 → 최저점(정강이 30°, 대퇴 -5°, 몸통 30°) → 선 자세 유지."""
    p = (1 - np.cos(np.linspace(0, 2 * np.pi, frames))) / 2
    p = np.concatenate([np.zeros(hold), p, np.zeros(hold)])
    return np.stack([pose(30 * x, 90 - 95 * x, 30 * x, heel_lift * x) for x in p])


EXACT = 1e-6  # 완전히 편 프레임만 선 자세로 쓰는 허용 오차(정답값 검사용)


def random_rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def features_of(common_seq: np.ndarray, tolerance_deg: float = 5.0) -> dict[str, float]:
    aligned, frame = align_sequence(common_seq, tolerance_deg=tolerance_deg)
    return rep_features(aligned, frame.standing)


class GravityFrameTests(unittest.TestCase):
    def test_canonical_input_recovers_axes_and_known_angles(self) -> None:
        seq = squat()
        aligned, frame = align_sequence(seq, tolerance_deg=EXACT)
        np.testing.assert_allclose(frame.rotation, np.eye(3), atol=1e-9)
        f = rep_features(aligned, frame.standing)
        self.assertAlmostEqual(f["trunk_incl_bottom"], 30.0, places=6)
        self.assertAlmostEqual(f["tibia_incl_bottom"], 30.0, places=6)
        self.assertAlmostEqual(f["thigh_elev_bottom"], -5.0, places=6)
        self.assertAlmostEqual(f["trunk_minus_tibia"], 0.0, places=6)
        self.assertAlmostEqual(f["foot_angle_rise"], 0.0, places=6)

    def test_aihub_mm_with_arbitrary_rotation_and_offset_matches_canonical(self) -> None:
        seq = squat()
        expected = features_of(seq)
        rot = random_rotation(3)
        world = seq @ rot.T + np.array([0.4, -1.2, 0.9])
        aihub = np.zeros((seq.shape[0], len(JOINT_NAMES), 3))
        for name in COMMON_JOINT_NAMES:
            aihub[:, JOINT_NAMES.index(name)] = world[:, IDX[name]] * 1000.0
        got = features_of(aihub_to_common(aihub))
        for name in FEATURE_NAMES:
            self.assertAlmostEqual(got[name], expected[name], places=6, msg=name)

    def test_mediapipe_camera_axes_match_canonical(self) -> None:
        """y 아래·z 카메라 반대 방향인 카메라 좌표, 45° 사선 + 10° 내려다보는 카메라."""
        seq = squat()
        expected = features_of(seq)
        yaw, pitch = np.radians(45.0), np.radians(10.0)
        to_camera = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # 위(z) → 카메라 -y
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        cam = seq @ (rx @ to_camera @ rz).T
        world33 = np.zeros((seq.shape[0], 33, 3))
        for name, mp_index in MEDIAPIPE_INDEX.items():
            world33[:, mp_index] = cam[:, IDX[name]]
        world33 -= (world33[:, 23:24] + world33[:, 24:25]) / 2.0  # MediaPipe 원점 = 고관절 중점
        got = features_of(mediapipe_to_common(world33))
        for name in FEATURE_NAMES:
            self.assertAlmostEqual(got[name], expected[name], places=6, msg=name)

    def test_mirrored_input_is_rejected(self) -> None:
        mirrored = squat() * np.array([-1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            fit_gravity_frame(mirrored)

    def test_heel_lift_is_measured_relative_to_toe(self) -> None:
        f = features_of(squat(heel_lift=0.05), tolerance_deg=EXACT)
        leg = SHANK + THIGH
        self.assertAlmostEqual(f["heel_rise_norm"], 0.05 / leg, places=6)
        horizontal = np.hypot(0.02, 0.21)  # 발끝→뒤꿈치 수평 거리 (좌우 0.02, 앞뒤 0.21)
        expected_rise = np.degrees(np.arctan2(0.06, horizontal) - np.arctan2(0.01, horizontal))
        self.assertAlmostEqual(f["foot_angle_rise"], expected_rise, places=6)

    def test_standing_frames_are_straight_leg_frames(self) -> None:
        seq = squat(frames=31, hold=5)
        frame = fit_gravity_frame(seq)
        self.assertTrue(frame.standing[:5].all() and frame.standing[-5:].all())
        self.assertFalse(frame.standing[5 + 15])  # 최저점

    def test_aihub_centers_are_replaced_by_midpoints(self) -> None:
        aihub = np.random.default_rng(0).normal(size=(2, 26, 3)) * 100
        common = aihub_to_common(aihub)
        mid_hip = (aihub[:, JOINT_NAMES.index("LHip")] + aihub[:, JOINT_NAMES.index("RHip")]) / 2000.0
        np.testing.assert_allclose(common[:, IDX["Hip"]], mid_hip)
        raw = aihub_to_common(aihub, derive_centers=False)
        np.testing.assert_allclose(raw[:, IDX["Hip"]], aihub[:, JOINT_NAMES.index("Hip")] / 1000.0)


if __name__ == "__main__":
    unittest.main()

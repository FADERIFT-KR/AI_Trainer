"""Schemas and label rules for AI Hub dataset 71422."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping


DATASET_ID = 71422
DATASET_NAME = "크로스핏 동작 데이터"
DATASET_VERSION = "1.1"
DATASET_URL = "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71422"
USAGE_POLICY_URL = (
    "https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105"
)
SCHEMA_VERSION = "1.0.0"

# The 26 joints documented by AI Hub for the dataset's 3-D CSV files.
AIHUB_JOINTS = (
    "Nose",
    "LEye",
    "REye",
    "LEar",
    "REar",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
    "Head",
    "Neck",
    "Hip",
    "LBigToe",
    "RBigToe",
    "LSmallToe",
    "RSmallToe",
    "LHeel",
    "RHeel",
)

# Joints retained in the runtime reference. They can all be mapped to MediaPipe
# Pose; Neck and Hip are virtual midpoints on the MediaPipe side.
COMMON_JOINTS = (
    "Nose",
    "Neck",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "Hip",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
    "LHeel",
    "RHeel",
    "LBigToe",
    "RBigToe",
)

MEDIAPIPE_MAPPING: Mapping[str, int | tuple[int, int]] = {
    "Nose": 0,
    "Neck": (11, 12),
    "LShoulder": 11,
    "RShoulder": 12,
    "LElbow": 13,
    "RElbow": 14,
    "LWrist": 15,
    "RWrist": 16,
    "Hip": (23, 24),
    "LHip": 23,
    "RHip": 24,
    "LKnee": 25,
    "RKnee": 26,
    "LAnkle": 27,
    "RAnkle": 28,
    "LHeel": 29,
    "RHeel": 30,
    "LBigToe": 31,
    "RBigToe": 32,
}

FEATURE_NAMES = (
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "left_hip_angle_deg",
    "right_hip_angle_deg",
    "left_ankle_angle_deg",
    "right_ankle_angle_deg",
    "trunk_lean_signed_deg",
    "knee_angle_asymmetry_deg",
    "hip_angle_asymmetry_deg",
    "left_knee_tracking",
    "right_knee_tracking",
    "pelvis_to_ankle_height",
    "stance_width",
    "heel_height_asymmetry",
)

CATEGORY_FIELDS = (
    "motion_category1",
    "motion_category2",
    "motion_category3",
)


def normalize_label(value: Any) -> str:
    """Normalize Korean/English labels without changing their meaning."""

    if value is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(value))
    return re.sub(r"[\s_\-]+", "", normalized).casefold()


def is_normal_air_squat(annotation: Mapping[str, Any]) -> bool:
    """Return whether an AI Hub annotation denotes a correct air squat."""

    categories = [normalize_label(annotation.get(field)) for field in CATEGORY_FIELDS]
    air_squat_labels = {"에어스쿼트", "airsquat"}
    if not any(label in air_squat_labels for label in categories):
        return False

    correctness = normalize_label(annotation.get("motion_category4"))
    normal_labels = {"정상", "normal", "correct", "정확"}
    return correctness in normal_labels


def schema_metadata() -> dict[str, Any]:
    """Serializable joint and coordinate metadata written with each build."""

    return {
        "schema_version": SCHEMA_VERSION,
        "native_joint_names": list(AIHUB_JOINTS),
        "joint_names": list(COMMON_JOINTS),
        "feature_names": list(FEATURE_NAMES),
        "mediapipe_mapping": {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in MEDIAPIPE_MAPPING.items()
        },
        "normalization": {
            "origin": "per-frame midpoint of LHip and RHip",
            "scale": "sequence median distance from hip midpoint to Neck",
            "x_axis": "standing-pose RHip to LHip direction",
            "y_axis": "standing-pose hip midpoint to Neck, orthogonalized to x",
            "z_axis": "right-handed cross product x by y",
            "units": "dimensionless body scale",
        },
    }

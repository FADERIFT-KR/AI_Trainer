from __future__ import annotations

import numpy as np
import pytest

from ai_trainer.augmentation.metric_validation import compare_sequences
from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES


def _pose() -> np.ndarray:
    index = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
    pose = np.zeros((3, len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
    for frame in range(3):
        pose[frame, index["Hip"]] = (0.2, 0.9 - frame * 0.1, 0.4)
        pose[frame, index["Neck"]] = (0.2, 1.6 - frame * 0.1, 0.4)
        pose[frame, index["LKnee"]] = (0.0, 0.5, 0.7)
        pose[frame, index["RKnee"]] = (0.4, 0.5, 0.7)
    return pose


def test_depth_error_survives_root_alignment() -> None:
    truth = _pose()
    predicted = truth.copy()
    predicted[:, COMMON_JOINT_NAMES.index("LKnee"), 2] += 0.05
    report = compare_sequences(predicted, truth)
    assert report.valid_frames == 3
    assert report.z_mae_mm == pytest.approx(50.0 / len(COMMON_JOINT_NAMES))
    assert report.per_joint_mm["LKnee"] == pytest.approx(50.0)
    assert report.root_mpjpe_mm > 0


def test_frame_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="same nonempty"):
        compare_sequences(_pose()[:2], _pose())

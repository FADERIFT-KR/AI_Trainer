"""MediaPipe 2D heel evidence used to validate only HEEL_ERROR DTW results."""
from __future__ import annotations

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES

I = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def heel_motion_evidence(points: np.ndarray, *, no_max: float, yes_min: float) -> dict:
    """Return torso-normalized heel motion evidence for one completed REP."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] < 2:
        return {"verdict": "AMBIGUOUS", "value": 0.0, "left": {}, "right": {},
                "no_max": float(no_max), "yes_min": float(yes_min)}

    prep_count = min(5, len(p))
    torso = max(float(np.median(np.linalg.norm(
        p[:prep_count, I["Neck"]] - p[:prep_count, I["Hip"]], axis=1
    ))), 1e-6)
    side_motion: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    for side in ("L", "R"):
        ankle_gap = (p[:, I[f"{side}Ankle"], 1] - p[:, I[f"{side}Heel"], 1]) / torso
        toe_gap = (p[:, I[f"{side}Heel"], 1] - p[:, I[f"{side}BigToe"], 1]) / torso
        standing = float(np.median(ankle_gap[:prep_count]))
        extreme_index = int(np.argmax(np.abs(ankle_gap - standing)))
        side_motion[side] = float(np.ptp(ankle_gap))
        details[side] = {
            "heel_ankle_motion": side_motion[side],
            "heel_ankle_standing": standing,
            "heel_ankle_bottom_or_extreme": float(ankle_gap[extreme_index]),
            "heel_toe_motion": float(np.ptp(toe_gap)),
        }

    value = max(side_motion.values())
    verdict = "NO" if value <= no_max else "YES" if value >= yes_min else "AMBIGUOUS"
    return {"verdict": verdict, "value": value, "left": details["L"], "right": details["R"],
            "no_max": float(no_max), "yes_min": float(yes_min)}


def validate_heel_candidate(
    raw_class: str,
    current_final: str,
    evidence: dict,
    relative_margin: float,
) -> tuple[str, str | None]:
    """Validate only a still-valid heel-error candidate; preserve prior UNKNOWNs."""
    if raw_class != "발뒤꿈치오류" or current_final != "발뒤꿈치오류":
        return current_final, None
    if evidence["verdict"] == "NO":
        return "자세추정불확실", "UNKNOWN_HEEL_SEMANTIC_CONFLICT"
    return current_final, None


__all__ = ["heel_motion_evidence", "validate_heel_candidate"]

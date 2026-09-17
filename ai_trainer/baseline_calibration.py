"""Standing-only 2D baseline collection for the live squat session."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StandingBaselineCalibrator:
    required_samples: int
    hip_min_deg: float
    knee_min_deg: float
    hip_side_min_deg: float
    knee_side_min_deg: float
    hip_delta_max_deg: float
    knee_delta_max_deg: float
    samples: list[dict] = field(default_factory=list)
    accepted_total: int = 0
    rejected_total: int = 0
    previous: tuple[float, float] | None = None

    def observe(self, frame: int, angles: dict, pelvis_height: float, angles_3d: dict) -> dict:
        hip = float(angles["hip_angle_2d"])
        knee = float(angles["knee_angle_2d"])
        hip_delta = None if self.previous is None else abs(hip - self.previous[0])
        knee_delta = None if self.previous is None else abs(knee - self.previous[1])
        self.previous = (hip, knee)
        reasons = []
        if hip < self.hip_min_deg or min(angles["left_hip_angle_2d"], angles["right_hip_angle_2d"]) < self.hip_side_min_deg:
            reasons.append("HIP_NOT_STANDING")
        if knee < self.knee_min_deg or min(angles["left_knee_angle_2d"], angles["right_knee_angle_2d"]) < self.knee_side_min_deg:
            reasons.append("KNEE_NOT_STANDING")
        if hip_delta is not None and hip_delta > self.hip_delta_max_deg:
            reasons.append("HIP_MOVING")
        if knee_delta is not None and knee_delta > self.knee_delta_max_deg:
            reasons.append("KNEE_MOVING")
        accepted = not reasons
        if accepted:
            self.accepted_total += 1
            self.samples.append({
                "frame": int(frame), "pelvis_height": float(pelvis_height),
                "hip_2d": hip, "knee_2d": knee,
                "hip_3d": float(angles_3d["hip_angle_3d"]),
                "knee_3d": float(angles_3d["knee_angle_3d"]),
            })
        else:
            self.rejected_total += 1
            self.samples.clear()  # eight consecutive valid standing frames
            self.previous = None  # an invalid pose must not poison the next valid streak
        return {
            "frame": int(frame), "left_hip": float(angles["left_hip_angle_2d"]),
            "right_hip": float(angles["right_hip_angle_2d"]), "hip": hip,
            "left_knee": float(angles["left_knee_angle_2d"]),
            "right_knee": float(angles["right_knee_angle_2d"]), "knee": knee,
            "hip_delta": hip_delta, "knee_delta": knee_delta,
            "accepted": accepted, "reject_reason": reasons,
            "consecutive": len(self.samples), "required": self.required_samples,
        }

    @property
    def ready(self) -> bool:
        return len(self.samples) >= self.required_samples

    def result(self) -> dict:
        if not self.ready:
            raise RuntimeError("standing baseline is not ready")
        chosen = self.samples[-self.required_samples:]
        return {
            "height": float(np.median([s["pelvis_height"] for s in chosen])),
            "hip_2d": float(np.median([s["hip_2d"] for s in chosen])),
            "knee_2d": float(np.median([s["knee_2d"] for s in chosen])),
            "hip_3d": float(np.median([s["hip_3d"] for s in chosen])),
            "knee_3d": float(np.median([s["knee_3d"] for s in chosen])),
            "frames": [s["frame"] for s in chosen],
            "accepted_total": self.accepted_total, "rejected_total": self.rejected_total,
        }


__all__ = ["StandingBaselineCalibrator"]

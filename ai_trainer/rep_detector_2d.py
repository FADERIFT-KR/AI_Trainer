"""Adaptive 2D squat REP detector, independent from DTW classification."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Adaptive2DRepDetector:
    knee_normal_excursion: float
    hip_normal_excursion: float
    start_fraction: float = 0.20
    valid_rep_fraction: float = 0.35
    recovery_fraction: float = 0.20
    reversal_fraction: float = 0.10
    transition_frames: int = 2
    recovery_frames: int = 3

    state: str = "prep"
    counter: int = 0
    peak_progress: float = 0.0
    rep_start_frame: int | None = None
    last_debug: dict = field(default_factory=dict, init=False, repr=False)

    def progress(self, knee_excursion: float, hip_excursion: float) -> float:
        knee = max(0.0, knee_excursion) / max(self.knee_normal_excursion, 1e-6)
        hip = max(0.0, hip_excursion) / max(self.hip_normal_excursion, 1e-6)
        return (knee + hip) / 2.0

    def update(
        self,
        frame: int,
        knee_excursion: float,
        hip_excursion: float,
        *,
        pelvis_velocity: float = 0.0,
        pelvis_velocity_eps: float = 0.004,
        pelvis_near_baseline: bool = False,
    ) -> str | None:
        value = self.progress(knee_excursion, hip_excursion)
        state_before = self.state
        counter_before = self.counter
        peak_before = self.peak_progress
        knee_progress = max(0.0, knee_excursion) / max(self.knee_normal_excursion, 1e-6)
        hip_progress = max(0.0, hip_excursion) / max(self.hip_normal_excursion, 1e-6)
        primary = value >= self.start_fraction
        assisted = value >= self.start_fraction / 2.0 and pelvis_velocity < -pelvis_velocity_eps
        drop_from_peak = max(peak_before, value) - value
        bottom_condition = drop_from_peak >= self.reversal_fraction
        abort_condition = max(peak_before, value) < self.valid_rep_fraction and value <= self.start_fraction
        recovered_2d = value <= self.recovery_fraction
        supported_recovery = pelvis_near_baseline and value <= self.valid_rep_fraction
        event = None
        if self.state == "prep":
            self.counter = self.counter + 1 if primary or assisted else 0
            if self.counter >= self.transition_frames:
                self.state = "descend"
                self.rep_start_frame = max(0, frame - self.transition_frames)
                self.peak_progress = value
                self.counter = 0
                event = "rep_start"
        elif self.state == "descend":
            self.peak_progress = max(self.peak_progress, value)
            reversed_enough = self.peak_progress - value >= self.reversal_fraction
            self.counter = self.counter + 1 if reversed_enough else 0
            if self.peak_progress < self.valid_rep_fraction and value <= self.start_fraction:
                self.reset()
                event = "abort"
            elif self.counter >= self.transition_frames:
                self.state = "bottom"
                self.counter = 0
                event = "bottom"
        elif self.state == "bottom":
            if value < self.peak_progress:
                self.state = "ascend"
                event = "ascend"
        else:
            self.counter = self.counter + 1 if recovered_2d or supported_recovery else 0
            if self.counter >= self.recovery_frames:
                self.state = "prep"
                self.counter = 0
                self.peak_progress = 0.0
                self.rep_start_frame = None
                event = "rep_end"
        reason = None
        if event == "abort":
            reason = "LOW_PEAK_PROGRESS_EARLY_RETURN"
        elif event == "rep_start":
            reason = "PRIMARY_2D_PROGRESS" if primary else "PELVIS_ASSISTED_START"
        elif event == "bottom":
            reason = "PROGRESS_REVERSED_FROM_PEAK"
        elif event == "ascend":
            reason = "PROGRESS_BELOW_PEAK"
        elif event == "rep_end":
            reason = "2D_PROGRESS_RECOVERED" if recovered_2d else "PELVIS_ASSISTED_RECOVERY"
        required = self.transition_frames if state_before in {"prep", "descend"} else (
            self.recovery_frames if state_before == "ascend" else 1
        )
        condition_pass = {
            "prep": primary or assisted,
            "descend": bottom_condition,
            "bottom": value < peak_before,
            "ascend": recovered_2d or supported_recovery,
        }.get(state_before, False)
        self.last_debug = {
            "evaluated_state": state_before,
            "current_state": self.state,
            "knee_progress": float(knee_progress),
            "hip_progress": float(hip_progress),
            "combined_progress": float(value),
            "peak_progress_before": float(peak_before),
            "peak_progress": float(self.peak_progress),
            "drop_from_peak": float(drop_from_peak),
            "start_condition": bool(primary),
            "assist_condition": bool(assisted),
            "bottom_condition": bool(bottom_condition),
            "abort_condition": bool(abort_condition),
            "progress_return_condition": bool(recovered_2d),
            "near_baseline_condition": bool(pelvis_near_baseline),
            "pelvis_return_condition": bool(supported_recovery),
            "condition_passed": bool(condition_pass),
            "counter_before": int(counter_before),
            "counter": int(self.counter),
            "required_counter": int(required),
            "event": event,
            "event_reason": reason,
        }
        return event

    def reset(self) -> None:
        self.state = "prep"
        self.counter = 0
        self.peak_progress = 0.0
        self.rep_start_frame = None


def validate_dtw_candidate(raw_class: str, *, depth_2d_pass: bool, consistency_pass: bool) -> str:
    """Apply semantic evidence without ever converting a rejected error to NORMAL."""
    if not consistency_pass:
        return "자세추정불확실"
    if raw_class == "엉덩이하방오류" and depth_2d_pass:
        return "자세추정불확실"
    if raw_class == "정상" and not depth_2d_pass:
        return "자세추정불확실"
    return raw_class

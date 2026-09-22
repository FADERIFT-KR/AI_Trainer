"""Causal sampling of received poses on the reference dataset's 30 Hz clock.

Interpolation uses only the last two observations, once both have arrived.
It adds no information and never extrapolates beyond the latest observation.
"""
from __future__ import annotations

import numpy as np


class PoseSampler:
    def __init__(self, fps: float = 30.0, max_gap: float = 0.5):
        self.dt = 1.0 / fps
        self.max_gap = max_gap
        self.previous = None
        self.previous_time = None
        self.next_time = None
        self.discontinuity = False
        self.sample_timestamps: list[float] = []

    def push(self, pose: np.ndarray, timestamp: float) -> list[np.ndarray]:
        if not np.isfinite(timestamp) or (self.previous_time is not None and timestamp <= self.previous_time):
            raise ValueError("Pose timestamps must be finite and strictly increasing")
        self.discontinuity = self.previous_time is not None and timestamp - self.previous_time > self.max_gap
        self.sample_timestamps = []
        if self.previous_time is None or self.discontinuity:
            samples = [pose.copy()]
            self.sample_timestamps.append(timestamp)
            self.next_time = timestamp + self.dt
        else:
            gap = timestamp - self.previous_time
            samples = []
            while self.next_time <= timestamp + 1e-9:
                alpha = np.clip((self.next_time - self.previous_time) / gap, 0, 1)
                samples.append((1-alpha) * self.previous + alpha * pose)
                self.sample_timestamps.append(self.next_time)
                self.next_time += self.dt
        self.previous = pose.copy()
        self.previous_time = timestamp
        return samples

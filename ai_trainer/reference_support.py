"""Empirical reference coverage, not a calibrated probability of correctness.

A nearest class always exists, even for an out-of-distribution pose. Compare
the query distance with leave-one-reference-out distances within that class.
The widest observed nearest-neighbour distance is a conservative coverage
radius. Passing this check does not establish that a real-world action is safe
or correct; failing it means the reference set does not support the label.
"""
from __future__ import annotations

import numpy as np

from .dtw_compare import PHASES, phase_aware_weighted_dtw


class ReferenceSupport:
    def __init__(self, references, weights, config):
        self.references = references
        self.weights = weights
        self.config = config
        self.cache = {}

    def check(self, label: str, distance: float, bounds: dict) -> dict:
        phases = tuple(p for p in PHASES if bounds[p][1] > bounds[p][0])
        key = (label, phases)
        if key not in self.cache:
            refs = self.references[label]
            if len(refs) < 2:
                self.cache[key] = None
            else:
                matrix = np.full((len(refs), len(refs)), np.inf)
                for i, a in enumerate(refs):
                    for j in range(i):
                        b = refs[j]
                        ba = {p: a["bounds"][p] if p in phases else [0, 0] for p in PHASES}
                        bb = {p: b["bounds"][p] if p in phases else [0, 0] for p in PHASES}
                        d = phase_aware_weighted_dtw(a["feat"], ba, b["feat"], bb, self.weights, self.config)["total"]
                        matrix[i, j] = matrix[j, i] = d
                self.cache[key] = float(matrix.min(axis=1).max())
        radius = self.cache[key]
        supported = radius is not None and np.isfinite(distance) and distance <= radius + 1e-8
        return {"supported": bool(supported), "distance": float(distance), "radius": radius,
                "method": "maximum_leave_one_reference_out_nearest_distance", "phases": list(phases)}

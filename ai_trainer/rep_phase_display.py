"""Display-only labels for the production REP detector state."""
from __future__ import annotations

REP_PHASE_LABELS = {
    "prep": "준비",
    "descend": "하강",
    "bottom": "최저",
    "ascend": "상승",
}


def rep_phase_label(state: str | None, *, active: bool = True) -> str:
    if state in REP_PHASE_LABELS:
        return REP_PHASE_LABELS[state]
    return "자세 확인 중" if active else "준비 중"


__all__ = ["REP_PHASE_LABELS", "rep_phase_label"]

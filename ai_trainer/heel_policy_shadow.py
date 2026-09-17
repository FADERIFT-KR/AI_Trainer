"""Pure offline helpers for shadow-evaluating heel-error policies."""
from __future__ import annotations

from collections import Counter
from math import isfinite

NORMAL = "정상"
HEEL = "발뒤꿈치오류"
UNKNOWN = "자세추정불확실"


def relative_heel_margin(normal_distance: float, heel_distance: float, epsilon: float = 1e-12) -> float:
    """Positive values mean that DTW prefers the heel-error class."""
    n, h = float(normal_distance), float(heel_distance)
    if not (isfinite(n) and isfinite(h)):
        raise ValueError("distances must be finite")
    return (n - h) / max(abs(n), float(epsilon))


def apply_policy(policy: str, raw: str, evidence: str, heel_direct: str | None = None) -> tuple[str, str | None]:
    """Apply a diagnostic policy without consulting ground truth."""
    if policy == "A_RAW_DTW" or raw != HEEL:
        return raw, None
    if policy in {"B_NO_REJECT", "E_PRODUCTION"}:
        return (UNKNOWN, "UNKNOWN_HEEL_SEMANTIC_CONFLICT") if evidence == "NO" else (HEEL, None)
    if policy == "C_YES_REQUIRED":
        if evidence == "YES":
            return HEEL, None
        reason = "UNKNOWN_HEEL_AMBIGUOUS" if evidence == "AMBIGUOUS" else "UNKNOWN_HEEL_CONFLICT"
        return UNKNOWN, reason
    if policy == "D_HEEL_DIRECT_2CLASS":
        return (HEEL, None) if heel_direct == HEEL else (UNKNOWN, "UNKNOWN_HEEL_DIRECT_CONFLICT")
    raise ValueError(f"unknown policy: {policy}")


def classification_metrics(rows: list[dict]) -> dict:
    """Two-class metrics where UNKNOWN counts as an incorrect/non-positive result."""
    if not rows:
        raise ValueError("rows must not be empty")
    per_class = {}
    for label in (NORMAL, HEEL):
        tp = sum(r["truth"] == label and r["result"] == label for r in rows)
        fp = sum(r["truth"] != label and r["result"] == label for r in rows)
        fn = sum(r["truth"] == label and r["result"] != label for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return {
        "evaluated_n": len(rows),
        "accuracy": sum(r["truth"] == r["result"] for r in rows) / len(rows),
        "balanced_accuracy": sum(v["recall"] for v in per_class.values()) / 2,
        "macro_precision": sum(v["precision"] for v in per_class.values()) / 2,
        "macro_recall": sum(v["recall"] for v in per_class.values()) / 2,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / 2,
        "heel_precision": per_class[HEEL]["precision"],
        "heel_recall": per_class[HEEL]["recall"],
        "heel_f1": per_class[HEEL]["f1"],
        "normal_recall": per_class[NORMAL]["recall"],
        "normal_to_heel_fp": sum(r["truth"] == NORMAL and r["result"] == HEEL for r in rows),
        "heel_to_unknown": sum(r["truth"] == HEEL and r["result"] == UNKNOWN for r in rows),
        "normal_to_unknown": sum(r["truth"] == NORMAL and r["result"] == UNKNOWN for r in rows),
        "unknown_n": sum(r["result"] == UNKNOWN for r in rows),
        "unknown_rate": sum(r["result"] == UNKNOWN for r in rows) / len(rows),
        "true_heel_preserved": per_class[HEEL]["tp"],
    }


def evidence_table(rows: list[dict]) -> list[dict]:
    out = Counter((r["truth"], r["evidence"], r["raw_prediction"]) for r in rows)
    return [{"ground_truth": k[0], "heel_evidence": k[1], "raw_dtw_prediction": k[2], "count": v}
            for k, v in sorted(out.items())]


def empirical_evidence(rows: list[dict]) -> list[dict]:
    result = []
    for evidence in ("NO", "AMBIGUOUS", "YES"):
        selected = [r for r in rows if r["evidence"] == evidence]
        n = len(selected)
        result.append({"heel_evidence": evidence, "n": n,
                       "p_heel": sum(r["truth"] == HEEL for r in selected) / n if n else None,
                       "p_normal": sum(r["truth"] == NORMAL for r in selected) / n if n else None})
    return result


__all__ = ["NORMAL", "HEEL", "UNKNOWN", "relative_heel_margin", "apply_policy",
           "classification_metrics", "evidence_table", "empirical_evidence"]

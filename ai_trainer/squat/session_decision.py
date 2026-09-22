"""Conservative fusion of sequence-model and view-condition evidence."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ai_trainer.squat.camera_views import VIEW_LABEL_KO, VIEWS

NORMAL_CLASS = "정상"
UNCERTAIN_CLASS = "판정 불확실"
PAPER_POSTURE_CLASS = "논문 자세 기준 위반"


@dataclass(frozen=True)
class ViewDecision:
    view: str
    label: str
    confirmed_errors: tuple[str, ...]
    normal_repetitions: int
    reason: str


@dataclass(frozen=True)
class SessionDecision:
    label: str
    view_decisions: tuple[ViewDecision, ...]
    confirmed_errors: tuple[str, ...]


def decide_view(repetitions: list[dict], view: str) -> ViewDecision:
    view_reps = [rep for rep in repetitions if rep.get("view_mode") == view]
    normal_count = sum(rep.get("model_class") == NORMAL_CLASS for rep in view_reps)
    evidence_count: Counter[str] = Counter()
    condition_violation_count: Counter[str] = Counter()
    paper_violation_count = 0
    paper_condition_count: Counter[str] = Counter()
    for rep in view_reps:
        model_class = rep.get("model_class")
        assessment = rep.get("condition_assessment") or {}
        paper_violations = assessment.get("paper_posture_violations") or ()
        if paper_violations:
            # A single low-confidence bottom frame should not overturn the
            # whole session.  The per-repetition result still shows it, while
            # the final decision needs the same evidence in two repetitions.
            paper_violation_count += 1
            for violation in paper_violations:
                condition = violation.get("condition", "논문 자세 조건")
                paper_condition_count[str(condition)] += 1
        support = assessment.get("error_support") or {}
        if model_class and model_class != NORMAL_CLASS and float(support.get(model_class, 0.0)) >= 0.60:
            # 오류 확정 표는 시퀀스 모델과 같은 오류 조건이 일치할 때만 준다.
            evidence_count[model_class] += 1
        predicted_by_rule = assessment.get("predicted_error")
        if predicted_by_rule and float(support.get(predicted_by_rule, 0.0)) >= 0.75:
            # 조건만으로 오류를 확정하지는 않지만, 반복 위반이면 정상 확정을 막는다.
            condition_violation_count[predicted_by_rule] += 1

    confirmed = tuple(sorted(error for error, count in evidence_count.items() if count >= 2))
    if confirmed:
        return ViewDecision(
            view=view,
            label=confirmed[0] if len(confirmed) == 1 else "복합오류",
            confirmed_errors=confirmed,
            normal_repetitions=normal_count,
            reason=f"동일 오류 조건이 3회 중 2회 이상 반복됨: {', '.join(confirmed)}",
        )
    if paper_violation_count >= 2:
        repeated = [name for name, count in paper_condition_count.items() if count >= 2]
        detail = ", ".join(repeated) if repeated else "논문 최저점 자세 조건"
        return ViewDecision(
            view=view,
            label=PAPER_POSTURE_CLASS,
            confirmed_errors=(PAPER_POSTURE_CLASS,),
            normal_repetitions=normal_count,
            reason=f"3회 중 {paper_violation_count}회 논문 기준 위반: {detail}",
        )
    if (
        len(view_reps) >= 3
        and normal_count >= 2
        and not any(count >= 2 for count in condition_violation_count.values())
    ):
        return ViewDecision(
            view=view,
            label=NORMAL_CLASS,
            confirmed_errors=(),
            normal_repetitions=normal_count,
            reason=f"모델 정상 {normal_count}/3, 반복 확인된 조건 오류 없음",
        )
    return ViewDecision(
        view=view,
        label=UNCERTAIN_CLASS,
        confirmed_errors=(),
        normal_repetitions=normal_count,
        reason="모델과 조건 근거가 일치하지 않거나 반복 근거가 부족함",
    )


def decide_session(repetitions: list[dict]) -> SessionDecision:
    decisions = tuple(decide_view(repetitions, view) for view in VIEWS)
    errors = tuple(sorted({error for decision in decisions for error in decision.confirmed_errors}))
    if errors:
        label = errors[0] if len(errors) == 1 else "복합오류"
    elif all(decision.label == NORMAL_CLASS for decision in decisions):
        label = NORMAL_CLASS
    else:
        label = UNCERTAIN_CLASS
    return SessionDecision(label=label, view_decisions=decisions, confirmed_errors=errors)


def format_session_decision(decision: SessionDecision) -> str:
    lines = [f"최종 판정: {decision.label}"]
    for item in decision.view_decisions:
        lines.append(f"{VIEW_LABEL_KO[item.view]}: {item.label} — {item.reason}")
    return "\n".join(lines)


__all__ = [
    "NORMAL_CLASS",
    "PAPER_POSTURE_CLASS",
    "SessionDecision",
    "UNCERTAIN_CLASS",
    "ViewDecision",
    "decide_session",
    "decide_view",
    "format_session_decision",
]

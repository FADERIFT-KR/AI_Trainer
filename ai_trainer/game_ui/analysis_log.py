"""In-memory posture analysis history for one exercise session."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime

from .error_explain import ERROR_EXPLANATIONS

UNKNOWN_CLASS = "자세추정불확실"
UNKNOWN_MESSAGES = {
    "UNKNOWN_POSE_VISIBILITY": "전신이 화면에 들어오지 않아 자세를 안정적으로 인식하지 못했습니다.",
    "UNKNOWN_2D_3D_MISMATCH": "2D와 3D 자세 추정이 일치하지 않아 결과를 확정하지 않았습니다.",
    "UNKNOWN_2D_3D_KNEE_MISMATCH": "무릎의 2D와 3D 자세 추정이 일치하지 않아 결과를 확정하지 않았습니다.",
    "UNKNOWN_2D_3D_HIP_MISMATCH": "고관절의 2D와 3D 자세 추정이 일치하지 않아 결과를 확정하지 않았습니다.",
    "UNKNOWN_2D_3D_BOTH_MISMATCH": "무릎과 고관절의 2D와 3D 자세 추정이 일치하지 않아 결과를 확정하지 않았습니다.",
    "UNKNOWN_DEPTH_CONFLICT": "동작 깊이 신호가 서로 달라 결과를 확정하지 않았습니다.",
    "UNKNOWN_DEPTH_AMBIGUOUS": "동작 깊이를 안정적으로 판단하지 못했습니다.",
}


@dataclass(frozen=True)
class PoseAnalysisEntry:
    timestamp: datetime
    exercise: str
    status: str
    predicted_class: str
    score: float | None
    dtw_distance: float | None
    errors: tuple[str, ...]
    error_messages: tuple[str, ...]
    body_parts: tuple[str, ...]
    good_points: tuple[str, ...]
    top_features: tuple[str, ...]
    rep: int | None
    source: str
    posture_components: dict | None = None


@dataclass(frozen=True)
class AnalysisSummary:
    exercise: str
    total_records: int
    total_reps: int
    normal_count: int
    error_count: int
    average_score: float | None
    error_counts: Counter
    provisional_count: int


@dataclass
class SessionAnalysisLog:
    """Stores one entry per completed rep and ignores duplicate rep events."""

    entries: list[PoseAnalysisEntry] = field(default_factory=list)
    _recorded_reps: set[int] = field(default_factory=set)
    _partial_candidate: str | None = None
    _partial_candidate_count: int = 0
    _last_partial_class: str | None = None
    _last_partial_at: datetime | None = None

    PARTIAL_STABLE_FRAMES = 8
    PARTIAL_COOLDOWN_SECONDS = 10.0

    def record_completed_rep(self, exercise: str, result: object) -> PoseAnalysisEntry | None:
        rep = int(result.rep_index) + 1
        if rep in self._recorded_reps:
            posture = getattr(result, "posture_score", None) or {}
            if posture.get("score_valid"):
                for index, entry in enumerate(self.entries):
                    if entry.rep == rep and entry.source == "rep":
                        self.entries[index] = replace(
                            entry, score=float(posture["overall"]), posture_components=posture
                        )
                        break
            print("\n========== ANALYSIS LOG TRACE ==========", flush=True)
            print("analysis_log add called: True", flush=True)
            print("analysis_log add result: duplicate ignored", flush=True)
            print(f"analysis_log item count: {len(self.entries)}", flush=True)
            print("========================================\n", flush=True)
            return None

        predicted_class = str(result.predicted_class)
        is_normal = predicted_class == "정상"
        explanation = ERROR_EXPLANATIONS.get(predicted_class)
        debug = getattr(result, "debug_summary", None) or {}
        unknown_message = UNKNOWN_MESSAGES.get(debug.get("reason_code")) if predicted_class == UNKNOWN_CLASS else None
        raw_distances = getattr(result, "raw_distance_by_class", {})
        dtw_distance = raw_distances.get(predicted_class)

        posture = getattr(result, "posture_score", None) or {}
        entry = PoseAnalysisEntry(
            timestamp=datetime.now(),
            exercise=exercise,
            status="normal" if is_normal else "error",
            predicted_class=predicted_class,
            score=posture.get("overall") if posture.get("score_valid") else None,
            dtw_distance=float(dtw_distance) if dtw_distance is not None else None,
            errors=() if is_normal else (predicted_class,),
            error_messages=(unknown_message,) if unknown_message else (() if is_normal or explanation is None else (explanation.text,)),
            body_parts=() if is_normal or explanation is None else explanation.body_parts,
            good_points=("DTW 기준 정상 자세 흐름에 가장 가깝게 판정되었습니다.",) if is_normal else (),
            top_features=tuple(name for name, _ in result.top_contributing_features),
            rep=rep,
            source="rep",
            posture_components=posture if posture else None,
        )
        self.entries.append(entry)
        self._recorded_reps.add(rep)
        self._last_partial_class = predicted_class
        self._last_partial_at = entry.timestamp
        self._partial_candidate = None
        self._partial_candidate_count = 0
        print("\n========== ANALYSIS LOG TRACE ==========", flush=True)
        print("analysis_log add called: True", flush=True)
        print("analysis_log add result: completed REP stored", flush=True)
        print(f"analysis_log item count: {len(self.entries)}", flush=True)
        print("========================================\n", flush=True)
        return entry

    def observe_partial(
        self,
        exercise: str,
        partial_distance: dict | None,
        *,
        now: datetime | None = None,
    ) -> PoseAnalysisEntry | None:
        """Record only stable live DTW state changes, with a duplicate cooldown."""
        if not partial_distance or not partial_distance.get("distance_by_class"):
            self._partial_candidate = None
            self._partial_candidate_count = 0
            return None

        distances = partial_distance["distance_by_class"]
        predicted_class = partial_distance.get("predicted_class", min(distances, key=distances.get))
        if predicted_class is None:
            self._partial_candidate = None
            self._partial_candidate_count = 0
            return None
        if predicted_class == self._partial_candidate:
            self._partial_candidate_count += 1
        else:
            self._partial_candidate = predicted_class
            self._partial_candidate_count = 1

        if self._partial_candidate_count < self.PARTIAL_STABLE_FRAMES:
            return None

        timestamp = now or datetime.now()
        if (
            predicted_class == self._last_partial_class
            and self._last_partial_at is not None
            and (timestamp - self._last_partial_at).total_seconds() < self.PARTIAL_COOLDOWN_SECONDS
        ):
            return None

        is_normal = predicted_class == "정상"
        explanation = ERROR_EXPLANATIONS.get(predicted_class)
        entry = PoseAnalysisEntry(
            timestamp=timestamp,
            exercise=exercise,
            status="normal" if is_normal else "error",
            predicted_class=predicted_class,
            score=None,
            dtw_distance=float(distances[predicted_class]),
            errors=() if is_normal else (predicted_class,),
            error_messages=() if is_normal or explanation is None else (explanation.text,),
            body_parts=() if is_normal or explanation is None else explanation.body_parts,
            good_points=("실시간 DTW 기준 정상 자세 흐름에 가장 가깝게 판정되었습니다.",) if is_normal else (),
            top_features=(),
            rep=None,
            source="partial",
        )
        self.entries.append(entry)
        self._last_partial_class = predicted_class
        self._last_partial_at = timestamp
        return entry

    def summary(self, exercise: str = "") -> AnalysisSummary:
        active_exercise = exercise or (self.entries[-1].exercise if self.entries else "-")
        completed_entries = [entry for entry in self.entries if entry.source == "rep"]
        normal_count = sum(entry.status == "normal" for entry in completed_entries)
        scores = [entry.score for entry in completed_entries if entry.score is not None]
        return AnalysisSummary(
            exercise=active_exercise,
            total_records=len(self.entries),
            total_reps=len({entry.rep for entry in self.entries if entry.rep is not None}),
            normal_count=normal_count,
            error_count=len(completed_entries) - normal_count,
            average_score=sum(scores) / len(scores) if scores else None,
            error_counts=Counter(error for entry in completed_entries for error in entry.errors),
            provisional_count=sum(entry.source == "partial" for entry in self.entries),
        )

    def clear(self) -> None:
        self.entries.clear()
        self._recorded_reps.clear()
        self._partial_candidate = None
        self._partial_candidate_count = 0
        self._last_partial_class = None
        self._last_partial_at = None

"""Non-modal dialog that presents the current in-memory analysis history."""
from __future__ import annotations

from html import escape

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QMessageBox, QPushButton, QTextBrowser, QVBoxLayout

from .analysis_log import SessionAnalysisLog


def _display_class(label: str) -> str:
    return "자세 인식 불안정" if label == "자세추정불확실" else label


class AnalysisHistoryDialog(QDialog):
    def __init__(self, history: SessionAnalysisLog, exercise: str, parent=None) -> None:
        super().__init__(parent)
        self.history = history
        self.exercise = exercise
        self.setWindowTitle("자세 분석 기록")
        self.setModal(False)
        self.resize(560, 620)

        self.content = QTextBrowser()
        self.content.setOpenExternalLinks(False)

        self.clear_button = QPushButton("기록 초기화")
        self.clear_button.clicked.connect(self._confirm_clear)
        close_button = QPushButton("닫기")
        close_button.clicked.connect(self.close)

        buttons = QHBoxLayout()
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.content, 1)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self) -> None:
        summary = self.history.summary(self.exercise)
        average = f"{summary.average_score:.1f}" if summary.average_score is not None else "점수 데이터 없음"

        good_items = [point for entry in self.history.entries for point in entry.good_points]
        good_html = self._counted_list(good_items, "기록된 정상 판정이 없습니다.")
        error_html = self._improvement_list(summary.error_counts)

        recent_rows = []
        for entry in self.history.entries[-20:][::-1]:
            label = "정상" if entry.status == "normal" else _display_class(entry.predicted_class)
            event_label = f"REP {entry.rep} · 최종" if entry.rep is not None else "잠정 실시간"
            details = []
            if entry.score is not None:
                details.append(f"자세 일치율 {entry.score:.1f}%")
            components = entry.posture_components or {}
            if components.get("score_valid"):
                details.append(
                    "깊이 {depth:.0f}% · 고관절 {hip:.0f}% · 무릎 {knee:.0f}% · "
                    "발뒤꿈치 안정 {heel_stability:.0f}% · 균형 {balance:.0f}% · "
                    "궤적 {trajectory:.0f}%".format(**components)
                )
            if entry.dtw_distance is not None:
                details.append(f"DTW {entry.dtw_distance:.4f}")
            if entry.error_messages:
                details.extend(entry.error_messages)
            suffix = f" · {' · '.join(details)}" if details else ""
            recent_rows.append(
                f"<li>{entry.timestamp.strftime('%H:%M:%S')} · {event_label} · "
                f"<b>{escape(label)}</b>{escape(suffix)}</li>"
            )
        recent_html = "<ul>" + "".join(recent_rows) + "</ul>" if recent_rows else "<p>아직 완료된 REP 기록이 없습니다.</p>"

        self.content.setHtml(
            "<style>body{font-family:'Malgun Gothic';font-size:14px;color:#e7ecf4;background:#171c26;}"
            "h2{color:#72a7ff;}h3{color:#72df8d;margin-top:18px;}li{margin:5px 0;}</style>"
            f"<h2>운동 요약</h2><p><b>운동:</b> {escape(summary.exercise)}<br>"
            f"<b>기록된 판정:</b> {summary.total_records}건 · <b>잠정 실시간:</b> {summary.provisional_count}건 · "
            f"<b>완료 REP:</b> {summary.total_reps}<br>"
            f"<b>정상:</b> {summary.normal_count}회 · <b>오류:</b> {summary.error_count}회<br>"
            f"<b>평균 자세 일치율:</b> {average}{'%' if summary.average_score is not None else ''}</p>"
            f"<h3>잘한 부분</h3>{good_html}"
            f"<h3>개선할 부분</h3>{error_html}"
            f"<h3>최근 기록</h3>{recent_html}"
        )

    @staticmethod
    def _counted_list(items, empty_text: str) -> str:
        from collections import Counter

        counts = Counter(items)
        if not counts:
            return f"<p>{escape(empty_text)}</p>"
        return "<ul>" + "".join(
            f"<li>{escape(str(item))} - {count}회</li>" for item, count in counts.most_common()
        ) + "</ul>"

    def _improvement_list(self, error_counts) -> str:
        if not error_counts:
            return "<p>기록된 자세 오류가 없습니다.</p>"
        rows = []
        for error, count in error_counts.most_common():
            matching = next((entry for entry in self.history.entries if error in entry.errors), None)
            details = []
            if matching is not None and matching.body_parts:
                details.append("관련 부위: " + ", ".join(matching.body_parts))
            if matching is not None and matching.error_messages:
                details.extend(matching.error_messages)
            detail_text = f" · {' · '.join(details)}" if details else ""
            rows.append(f"<li><b>{escape(error)}</b> - {count}회{escape(detail_text)}</li>")
        return "<ul>" + "".join(rows) + "</ul>"

    def _confirm_clear(self) -> None:
        answer = QMessageBox.question(
            self,
            "기록 초기화",
            "현재 운동 세션의 분석 기록을 모두 삭제할까요?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.history.clear()
            self.refresh()


__all__ = ["AnalysisHistoryDialog"]

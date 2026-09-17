"""Best-effort persistent diagnostics for live REP detection.

Logging must never be allowed to interrupt the camera or classification pipeline.
Every public method therefore catches I/O/serialization errors and disables only
the diagnostic writer.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class RepDiagnosticRecorder:
    def __init__(self, output_dir: str | Path, *, now: datetime | None = None) -> None:
        stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
        self.output_dir = Path(output_dir)
        self.trace_path = self.output_dir / f"rep_trace_{stamp}.jsonl"
        self.summary_path = self.output_dir / f"rep_trace_{stamp}_summary.json"
        self.enabled = True
        self.error: str | None = None
        self._attempts: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._next_attempt = 1
        self._last_detector: dict[str, Any] | None = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._stream = self.trace_path.open("x", encoding="utf-8", buffering=1)
        except Exception as error:  # diagnostics are deliberately non-fatal
            self.enabled = False
            self.error = f"{type(error).__name__}: {error}"
            self._stream = None

    def _write(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self._stream is None:
            return
        try:
            self._stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.enabled = False
            try:
                self._stream.close()
            except Exception:
                pass

    def _new_attempt(self, frame: int, state: str) -> dict[str, Any]:
        attempt = {
            "attempt_id": self._next_attempt,
            "start_frame": frame,
            "end_frame": None,
            "path": [state],
            "result": None,
            "reason": None,
            "framing_skipped_frames": 0,
        }
        self._next_attempt += 1
        self._attempts.append(attempt)
        self._active = attempt
        return attempt

    def record_detector(self, payload: dict[str, Any]) -> None:
        """Record one accepted detector frame and maintain attempt summaries."""
        try:
            row = dict(payload)
            frame = int(row["frame"])
            state = str(row.get("current_state", row.get("state", "prep")))
            event = row.get("event")
            progress = float(row.get("combined_progress", 0.0) or 0.0)
            # Diagnostic-only early candidate: captures attempts that never satisfy
            # the production rep_start condition without changing that condition.
            if self._active is None and (event == "rep_start" or progress >= 0.05):
                self._new_attempt(frame, str(row.get("previous_state", "prep")))
            if self._active is not None:
                row["attempt_id"] = self._active["attempt_id"]
                if not self._active["path"] or self._active["path"][-1] != state:
                    self._active["path"].append(state)
                if event in {"abort", "rep_end"}:
                    self._active["end_frame"] = frame
                    self._active["result"] = "ABORTED" if event == "abort" else "COMPLETED"
                    self._active["reason"] = row.get("event_reason")
                    self._active = None
            else:
                row["attempt_id"] = None
            row.setdefault("timestamp", datetime.now().isoformat(timespec="milliseconds"))
            row["record_type"] = "detector_frame"
            self._last_detector = row
            self._write(row)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def record_framing_skip(
        self, *, camera_frame: int, reason: str, position_state: str, **details: Any
    ) -> None:
        try:
            if self._active is not None:
                self._active["framing_skipped_frames"] += 1
            row = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "record_type": "pipeline_event",
                "camera_frame": int(camera_frame),
                "attempt_id": None if self._active is None else self._active["attempt_id"],
                "state": None if self._last_detector is None else self._last_detector.get("current_state"),
                "event": "FRAME_SKIPPED",
                "reason": reason,
                "position_state": position_state,
                "framing_ok": False,
                "frame_delivered": False,
            }
            row.update(details)
            self._write(row)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def close(self) -> None:
        try:
            if self._active is not None:
                self._active["result"] = (
                    "FRAMING_INTERRUPTED"
                    if self._active["framing_skipped_frames"] else "STALLED"
                )
                self._active["reason"] = (
                    "FRAMING_NOT_OK" if self._active["framing_skipped_frames"]
                    else "SESSION_ENDED_BEFORE_REP_END"
                )
                if self._last_detector is not None:
                    self._active["end_frame"] = self._last_detector.get("frame")
                self._active = None
            if self._stream is not None:
                try:
                    self._stream.flush()
                    self._stream.close()
                except Exception:
                    pass
            if self.enabled:
                payload = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "trace_file": self.trace_path.name,
                    "attempt_count": len(self._attempts),
                    "attempts": self._attempts,
                    "logging_error": self.error,
                }
                self.summary_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8",
                )
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"


__all__ = ["RepDiagnosticRecorder"]

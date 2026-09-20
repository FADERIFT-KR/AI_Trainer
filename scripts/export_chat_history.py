#!/usr/bin/env python3
"""Export user-visible Codex chat messages from local JSONL sessions.

System/developer instructions, reasoning, tool calls/results, and sub-agent
sessions are intentionally excluded.  The resulting text contains only root
thread user/assistant messages, grouped by local calendar date.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


AUTO_USER_PREFIXES = (
    "<recommended_plugins>",
    "<environment_context>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-dir", type=Path, required=True)
    parser.add_argument(
        "--before",
        default=None,
        help="optional exclusive local date, YYYY-MM-DD",
    )
    parser.add_argument("--timezone", default="Asia/Seoul")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_session_titles(index_path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    if not index_path.exists():
        return titles
    with index_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            thread_id = item.get("id")
            title = item.get("thread_name")
            if thread_id and title:
                titles[str(thread_id)] = str(title)
    return titles


def iter_root_sessions(codex_dir: Path):
    locations = [codex_dir / "sessions", codex_dir / "archived_sessions"]
    for location in locations:
        if not location.exists():
            continue
        for path in sorted(location.rglob("*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as stream:
                    first = json.loads(stream.readline())
            except (OSError, json.JSONDecodeError):
                continue
            if first.get("type") != "session_meta":
                continue
            meta = first.get("payload", {})
            if meta.get("id") != meta.get("session_id"):
                continue
            source = meta.get("source") or {}
            if isinstance(source, dict) and "subagent" in source:
                continue
            yield path, meta


def content_text(content: object) -> str:
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") not in {"input_text", "output_text", "text"}:
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts).strip()


def is_automatic_user_context(text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(prefix) for prefix in AUTO_USER_PREFIXES)


def main() -> None:
    args = parse_args()
    timezone = ZoneInfo(args.timezone)
    cutoff_local = (
        datetime.fromisoformat(args.before).replace(tzinfo=timezone)
        if args.before
        else None
    )
    titles = read_session_titles(args.codex_dir / "session_index.jsonl")

    messages: list[dict] = []
    root_sessions: dict[str, dict] = {}
    for path, meta in iter_root_sessions(args.codex_dir):
        session_id = str(meta["session_id"])
        root_sessions[session_id] = {
            "title": titles.get(session_id, "제목 없음"),
            "path": str(path),
        }
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("type") != "response_item":
                    continue
                payload = item.get("payload", {})
                if payload.get("type") != "message":
                    continue
                role = payload.get("role")
                if role not in {"user", "assistant"}:
                    continue
                text = content_text(payload.get("content"))
                if not text or (role == "user" and is_automatic_user_context(text)):
                    continue
                timestamp_text = item.get("timestamp")
                if not isinstance(timestamp_text, str):
                    continue
                timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
                local_timestamp = timestamp.astimezone(timezone)
                if cutoff_local is not None and local_timestamp >= cutoff_local:
                    continue
                messages.append(
                    {
                        "timestamp": local_timestamp,
                        "role": role,
                        "text": text,
                        "session_id": session_id,
                        "title": titles.get(session_id, "제목 없음"),
                    }
                )

    messages.sort(key=lambda item: item["timestamp"])
    # A repeated rollout can contain an exact replay at the same timestamp.
    deduplicated = []
    seen = set()
    for message in messages:
        key = (
            message["timestamp"].isoformat(),
            message["role"],
            message["text"],
            message["session_id"],
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(message)
    messages = deduplicated

    counts_by_date = Counter(message["timestamp"].date().isoformat() for message in messages)
    counts_by_role = Counter(message["role"] for message in messages)
    lines = [
        "Codex 채팅 전체 원문 내역",
        f"생성 시각: {datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        (
            f"추출 범위: {args.before} 00:00 ({args.timezone}) 이전"
            if args.before
            else "추출 범위: 로컬에 보존된 전체 루트 대화"
        ),
        "포함: 루트 대화의 사용자/어시스턴트 화면 메시지",
        "제외: 시스템·개발자 지침, 내부 추론, 도구 호출/출력, 하위 에이전트 로그, 자동 환경 정보",
        f"루트 대화 수: {len({message['session_id'] for message in messages})}",
        f"메시지 수: {len(messages)} (사용자 {counts_by_role['user']}, 어시스턴트 {counts_by_role['assistant']})",
        "",
        "날짜별 메시지 수:",
    ]
    for date, count in sorted(counts_by_date.items()):
        lines.append(f"- {date}: {count}개")

    current_date = None
    current_session = None
    for message in messages:
        date = message["timestamp"].date().isoformat()
        if date != current_date:
            lines.extend(["", "=" * 80, date, "=" * 80])
            current_date = date
            current_session = None
        if message["session_id"] != current_session:
            lines.extend(
                [
                    "",
                    f"[대화: {message['title']}]",
                    f"[세션 ID: {message['session_id']}]",
                ]
            )
            current_session = message["session_id"]
        role_label = "사용자" if message["role"] == "user" else "어시스턴트"
        time_label = message["timestamp"].strftime("%H:%M:%S")
        lines.extend(["", f"[{time_label}] {role_label}", message["text"]])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "root_sessions": len({message["session_id"] for message in messages}),
                "messages": len(messages),
                "user_messages": counts_by_role["user"],
                "assistant_messages": counts_by_role["assistant"],
                "dates": dict(sorted(counts_by_date.items())),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

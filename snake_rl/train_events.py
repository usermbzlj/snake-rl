"""Structured training events for CLI logs and the web console."""

from __future__ import annotations

import json
import sys
from typing import Any

EVENT_PREFIX = "SNAKE_EVENT "


def emit_event(event_type: str, **payload: Any) -> None:
    body = {"type": event_type, **payload}
    line = EVENT_PREFIX + json.dumps(body, ensure_ascii=False)
    print(line, flush=True)


def parse_event_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text.startswith(EVENT_PREFIX):
        return None
    raw = text[len(EVENT_PREFIX) :]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def write_event_to(stream: Any, event_type: str, **payload: Any) -> None:
    body = {"type": event_type, **payload}
    stream.write(EVENT_PREFIX + json.dumps(body, ensure_ascii=False) + "\n")
    if stream is sys.stdout:
        stream.flush()

"""Append-only log of everything that happened in a run.

This is the source of truth. Nothing here is ever edited or deleted -
it exists so the full history can always be reconstructed. What gets
sent to the model each turn is a separate, smaller *view* over this log
(see agent/context.py), not the log itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Any
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_message(self) -> dict[str, Any]:
        """Render as a chat-completion message dict."""
        return {"role": self.role, "content": self.content, **self.extra}


class EventLog:
    """A plain, growable list of events. No hidden behavior."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def append(self, role: str, content: Any, **extra: Any) -> Event:
        event = Event(role=role, content=content, extra=extra)
        self.events.append(event)
        return event

    def tail(self, n: int) -> list[Event]:
        return self.events[-n:] if n else []

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

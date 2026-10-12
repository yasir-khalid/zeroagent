"""The session: an append-only log of everything that happened in a run.

This is the source of truth, and it lives on disk: one JSON object per line in
a .jsonl file, written the moment each event happens. Nothing is ever edited
or deleted, so the log is always at least as big as the context window and
usually much bigger - compaction adds a summary event, it doesn't remove the
events it summarized.

Because every event is on disk before the next step starts, the harness
process can die at any point and a new one can pick up the session by reading
the file back (see ConversationContext). What gets sent to the model each turn
is a separate, smaller *view* over this log (see agent/context.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Events with these roles are chat messages. Anything else (like "compaction")
# is a note the harness keeps for itself and never sends to the model as-is.
MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class Event:
    role: str  # a MESSAGE_ROLE, or "compaction"
    content: Any
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_message(self) -> dict[str, Any]:
        """Render as a chat-completion message dict."""
        return {"role": self.role, "content": self.content, **self.extra}


class EventLog:
    """A growable list of events, mirrored line by line to a .jsonl file.

    The list in memory is just a cache of the file. With path=None nothing is
    written, which is handy for experiments and tests.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[Event] = []
        if path and path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self.events.append(Event(**json.loads(line)))

    def append(self, role: str, content: Any, **extra: Any) -> Event:
        event = Event(role=role, content=content, extra=extra)
        self.events.append(event)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Open, append, close: the line is on disk before we return, so a
            # crash can lose at most the event that was being written.
            with self.path.open("a") as file:
                file.write(json.dumps(asdict(event)) + "\n")
        return event

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

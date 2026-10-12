"""Bounded, cache-friendly view of an agent run's state.

The session (state.events.EventLog) is the append-only source of truth.
What actually gets sent to the model each turn is a small *recomputed view*
over that log, built fresh by build():

    [ system + summary ]   <- stable prefix, rarely changes
    [ recent events ]      <- tail, changes every turn

Keeping the prefix stable across turns is the whole point: it's what lets a
provider reuse cached prompt tokens instead of reprocessing everything from
scratch. The tail is left free to change every turn since it's small.

Compaction never deletes anything. It appends a "compaction" event holding
the new summary and the index of the first event it did *not* cover. The view
is then: latest summary + every message after that index. The session keeps
growing; the view stays small.

No framework, no plugins, no processor pipeline - just a log, a summary
string, and a function that concatenates them.
"""

from __future__ import annotations

from typing import Any

from agent.compaction import Summarizer, compact, naive_summarizer
from state.events import MESSAGE_ROLES, Event, EventLog

CHARS_PER_TOKEN = 4  # rough estimate, good enough for budgeting; no tokenizer dependency


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class ConversationContext:
    """A small, provider-agnostic view over a session's event log."""

    def __init__(
        self,
        system: str = "",
        *,
        log: EventLog | None = None,
        tail_size: int = 12,
        compact_after_tokens: int = 6_000,
        summarizer: Summarizer = naive_summarizer,
    ) -> None:
        self.static = system  # the immutable part of the prefix: role, rules, tools
        # Full history, never trimmed. (Not `log or ...`: an empty log is falsy.)
        self.log = log if log is not None else EventLog()
        self.tail_size = tail_size
        self.compact_after_tokens = compact_after_tokens
        self.summarizer = summarizer

    def append(self, role: str, content: Any, **extra: Any) -> None:
        self.log.append(role, content, **extra)
        self._maybe_compact()

    @property
    def summary(self) -> str:
        """The slowly-changing part of the prefix: the latest compaction's summary."""
        compaction = self._latest_compaction()
        return compaction.content if compaction else ""

    def build(self) -> list[dict[str, Any]]:
        """Recompute the prompt from current state. Call this every turn."""
        prefix = self.static
        if self.summary:
            block = f"# Earlier in this conversation\n{self.summary}"
            prefix = f"{prefix}\n\n{block}" if prefix else block

        messages: list[dict[str, Any]] = []
        if prefix:
            messages.append({"role": "system", "content": prefix})
        messages.extend(event.to_message() for event in self._live_messages())
        return messages

    def _latest_compaction(self) -> Event | None:
        for event in reversed(self.log.events):
            if event.role == "compaction":
                return event
        return None

    def _live_messages(self) -> list[Event]:
        """Messages not yet folded into the summary: everything after the last cut."""
        compaction = self._latest_compaction()
        start = compaction.extra["covers_until"] if compaction else 0
        return [event for event in self.log.events[start:] if event.role in MESSAGE_ROLES]

    def _maybe_compact(self) -> None:
        """Fold old messages into the summary once they get too big.

        This only touches messages *older* than the last tail_size, so the most
        recent turns are never summarized away mid-conversation. It also only
        triggers once a token threshold is crossed, so the prefix doesn't
        shift on every single append.
        """
        live = self._live_messages()
        cut = len(live) - self.tail_size
        # Never start the kept tail on a tool result: its assistant message
        # (the one holding the tool call) must stay in the prompt with it, or
        # the provider sees a result for a call that doesn't exist.
        while cut > 0 and live[cut].role == "tool":
            cut -= 1
        if cut <= 0:
            return

        older = live[:cut]
        if sum(estimate_tokens(str(event.content)) for event in older) < self.compact_after_tokens:
            return

        # Store the cut as an index into the full log, so it survives reloading.
        covers_until = self.log.events.index(live[cut])
        self.log.append("compaction", compact(self.summary, older, self.summarizer), covers_until=covers_until)

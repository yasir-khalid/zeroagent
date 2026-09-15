"""Bounded, cache-friendly view of an agent run's state.

The event log (state.events.EventLog) is the append-only source of truth.
What actually gets sent to the model each turn is a small *recomputed view*
over that log, built fresh by build():

    [ system + summary ]   <- stable prefix, rarely changes
    [ recent events ]      <- tail, changes every turn

Keeping the prefix stable across turns is the whole point: it's what lets a
provider reuse cached prompt tokens instead of reprocessing everything from
scratch. The tail is left free to change every turn since it's small.

No framework, no plugins, no processor pipeline - just a log, a summary
string, and a function that concatenates them.
"""

from __future__ import annotations

from typing import Any

from agent.compaction import Summarizer, compact, naive_summarizer
from state.events import EventLog

CHARS_PER_TOKEN = 4  # rough estimate, good enough for budgeting; no tokenizer dependency


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class ConversationContext:
    """A small, provider-agnostic container for chat-completion messages."""

    def __init__(
        self,
        system: str = "",
        *,
        tail_size: int = 12,
        compact_after_tokens: int = 6_000,
        summarizer: Summarizer = naive_summarizer,
    ) -> None:
        self.static = system  # the immutable part of the prefix: role, rules, tools
        self.summary = ""  # the slowly-changing part: what got compacted away
        self.log = EventLog()  # full history, never trimmed
        self.tail_size = tail_size
        self.compact_after_tokens = compact_after_tokens
        self.summarizer = summarizer

    def append(self, role: str, content: Any, **extra: Any) -> None:
        self.log.append(role, content, **extra)
        self._maybe_compact()

    def build(self) -> list[dict[str, Any]]:
        """Recompute the prompt from current state. Call this every turn."""
        prefix = self.static
        if self.summary:
            block = f"# Earlier in this conversation\n{self.summary}"
            prefix = f"{prefix}\n\n{block}" if prefix else block

        messages: list[dict[str, Any]] = []
        if prefix:
            messages.append({"role": "system", "content": prefix})
        messages.extend(event.to_message() for event in self.log.tail(self.tail_size))
        return messages

    def _maybe_compact(self) -> None:
        """Fold old events into the summary once the tail gets too big.

        This only touches events *older* than the tail window, so the most
        recent turns are never summarized away mid-conversation. It also only
        triggers once a token threshold is crossed, so the prefix doesn't
        shift on every single append.
        """
        older = self.log.events[: -self.tail_size] if len(self.log) > self.tail_size else []
        if not older:
            return

        older_tokens = sum(estimate_tokens(str(event.content)) for event in older)
        if older_tokens < self.compact_after_tokens:
            return

        self.summary = compact(self.summary, older, self.summarizer)
        self.log.events = list(self.log.tail(self.tail_size))

"""Turns old events into a short text summary so they can leave the prompt.

Compaction is meant to run rarely, not every turn: folding events into the
summary changes the stable prefix of the prompt, and changing that prefix is
exactly what you want to avoid on every single turn (it defeats prompt
caching). See ConversationContext._maybe_compact for when this gets called.
"""

from __future__ import annotations

from collections.abc import Callable

from state.events import Event

# A summarizer takes the events being folded away and returns a short string.
# The default below is a dependency-free fallback; swap in one that calls an
# LLM if you want real summarization instead of truncation.
Summarizer = Callable[[list[Event]], str]


def naive_summarizer(events: list[Event]) -> str:
    lines = []
    for event in events:
        content = event.content if isinstance(event.content, str) else str(event.content)
        content = content.strip().replace("\n", " ")
        if len(content) > 200:
            content = content[:200] + "…"
        lines.append(f"- {event.role}: {content}")
    return "\n".join(lines)


def compact(previous_summary: str, events: list[Event], summarizer: Summarizer = naive_summarizer) -> str:
    """Fold `events` into `previous_summary`, returning the new summary."""
    new_summary = summarizer(events)
    if not previous_summary:
        return new_summary
    return f"{previous_summary}\n{new_summary}"

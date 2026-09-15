"""Conversation state owned by an agent run."""

from __future__ import annotations

from typing import Any


class ConversationContext:
    """A small, provider-agnostic container for chat-completion messages."""

    def __init__(self, system: str = "") -> None:
        self.messages: list[dict[str, Any]] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def append(self, role: str, content: Any, **extra: Any) -> None:
        """Append one message while leaving provider-specific fields intact."""
        self.messages.append({"role": role, "content": content, **extra})

"""The tool-calling execution loop for ZeroAgent."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from rich.console import Console

from .context import ConversationContext


class OpenAIResponseFinishReason(Enum):
    TOOL_CALLS = "tool_calls"
    STOP = "stop"


@dataclass
class ToolEvent:
    """A small presentation hook for a tool invocation.

    The agent still owns calling tools and recording their results. A terminal
    (or any future UI) can subscribe to these events without becoming part of
    the execution loop.
    """

    phase: str
    name: str
    arguments: str
    result: str | None = None
    failed: bool = False


class Agent:
    """A minimal OpenRouter-backed agent with a bounded tool-calling loop."""

    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: Mapping[str, Callable[..., Any]] | None = None,
        on_tool_event: Callable[[ToolEvent], None] | None = None,
    ) -> None:
        self.model = model
        self.context = ConversationContext(system)
        self.console = Console()
        self.tools = tools or []
        self.tool_handlers = dict(tool_handlers or {})
        self.on_tool_event = on_tool_event
        # Uppercase name is retained for compatibility with the original demo.
        self.AGENT_MAX_TURNS = 10

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def __call__(self, message: str = "") -> str | None:
        if message:
            self.context.append("user", message)
        return self.execute()

    def execute(self) -> str | None:
        """Run until the model responds with text or the turn limit is reached."""
        for _ in range(self.AGENT_MAX_TURNS):
            with self.console.status("Running Agent .."):
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model,
                    tools=self.tools,
                    messages=self.context.build(),
                )

            for choice in response.choices:
                message = choice.message
                extra: dict[str, Any] = {}
                if message.tool_calls:
                    extra["tool_calls"] = [tool_call.model_dump() for tool_call in message.tool_calls]
                self.context.append("assistant", message.content, **extra)

                if (
                    choice.finish_reason == OpenAIResponseFinishReason.TOOL_CALLS.value
                    and message.tool_calls
                ):
                    self._execute_tool_calls(message.tool_calls)
                elif (
                    choice.finish_reason == OpenAIResponseFinishReason.STOP.value
                    and message.content
                ):
                    return message.content

        self.console.print("[yellow]The agent reached its maximum number of turns.[/yellow]")
        return None

    def _execute_tool_calls(self, tool_calls: Any) -> None:
        for tool_call in tool_calls:
            name = tool_call.function.name
            raw_arguments = tool_call.function.arguments
            self._report_tool_event(ToolEvent("started", name, raw_arguments))

            try:
                arguments = json.loads(raw_arguments)
                handler = self.tool_handlers.get(name)
                if handler is None:
                    result: Any = {"error": f"Unknown tool: {name}"}
                    failed = True
                else:
                    result = handler(**arguments)
                    failed = False
            except Exception as error:
                result = {"error": str(error)}
                failed = True

            result_text = json.dumps(result)
            self._report_tool_event(
                ToolEvent("finished", name, raw_arguments, result_text, failed)
            )

            self.context.append("tool", result_text, tool_call_id=tool_call.id)

    def _report_tool_event(self, event: ToolEvent) -> None:
        """Send a tool event to the UI, with a useful plain-terminal fallback."""
        if self.on_tool_event:
            self.on_tool_event(event)
            return
        if event.phase == "started":
            self.console.log(f"⛏  {event.name}({event.arguments})")
        else:
            result = event.result or ""
            preview = result if len(result) <= 300 else f"{result[:300]}…"
            self.console.log(f"   ↳ {preview}")

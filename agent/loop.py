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

from state.events import EventLog

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


@dataclass
class ModelEvent:
    """Sent just before and just after each model call, so a UI can show a spinner."""

    phase: str  # "started" | "finished"


@dataclass
class Usage:
    """Running token and cost totals, as reported by the provider.

    input_tokens is summed over every call, so it grows fast: each turn resends
    the whole context. That repeated cost is exactly what prompt caching (the
    stable prefix in context.py) is meant to cut, and cached_tokens shows how
    much of it the provider actually reused.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    context_tokens: int = 0  # size of the latest prompt + reply: what's in the window now

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        details = usage.prompt_tokens_details
        self.cached_tokens += (details.cached_tokens or 0) if details else 0
        self.cost += getattr(usage, "cost", None) or 0.0  # OpenRouter extension
        self.context_tokens = usage.prompt_tokens + usage.completion_tokens


class Agent:
    """A minimal OpenRouter-backed agent with a bounded tool-calling loop.

    The agent is the *harness*: it calls the model and routes tool calls. It
    keeps no state of its own that matters - everything lives in the session
    log it's handed - so if this process dies, a new Agent given the same log
    carries on where the old one stopped.
    """

    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: Mapping[str, Callable[..., Any]] | None = None,
        on_tool_event: Callable[[ToolEvent], None] | None = None,
        on_model_event: Callable[[ModelEvent], None] | None = None,
        session: EventLog | None = None,
    ) -> None:
        self.model = model
        self.context = ConversationContext(system, log=session)
        self.console = Console()
        self.tools = tools or []
        self.tool_handlers = dict(tool_handlers or {})
        self.on_tool_event = on_tool_event
        self.on_model_event = on_model_event
        self.usage = Usage()
        # Uppercase name is retained for compatibility with the original demo.
        self.AGENT_MAX_TURNS = 10

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def __call__(self, message: str = "") -> str | None:
        self._close_open_tool_calls()
        if message:
            self.context.append("user", message)
        return self.execute()

    def execute(self) -> str | None:
        """Run until the model responds with text or the turn limit is reached."""
        for _ in range(self.AGENT_MAX_TURNS):
            self._report_model_event(ModelEvent("started"))
            try:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model,
                    tools=self.tools,
                    messages=self.context.build(),
                )
            finally:
                self._report_model_event(ModelEvent("finished"))
            self.usage.add(response.usage)

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

    def _close_open_tool_calls(self) -> None:
        """Give every tool call in the session a result before continuing.

        If the harness died (or was interrupted) after the model asked for a
        tool but before the result was logged, the session ends in a call with
        no answer, and providers reject that. We can't know whether the tool
        ran, so we say exactly that rather than guessing or re-running it.
        """
        answered = {event.extra.get("tool_call_id") for event in self.context.log if event.role == "tool"}
        for event in list(self.context.log):
            for call in event.extra.get("tool_calls", []) if event.role == "assistant" else []:
                if call["id"] not in answered:
                    result = {"error": "Interrupted: this tool may or may not have run. Check before retrying."}
                    self.context.append("tool", json.dumps(result), tool_call_id=call["id"])

    def context_window(self) -> int | None:
        """Ask OpenRouter how many tokens this model accepts, or None if unknown."""
        try:
            for model in self.client.models.list():
                if model.id == self.model:
                    return getattr(model, "context_length", None)
        except Exception:
            pass  # purely informational; never block startup on it
        return None

    def _report_model_event(self, event: ModelEvent) -> None:
        if self.on_model_event:
            self.on_model_event(event)

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

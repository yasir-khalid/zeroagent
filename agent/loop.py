"""The tool-calling execution loop for ZeroAgent."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from rich import print
from rich.console import Console

from .context import ConversationContext

try:
    from pyfiglet import figlet_format
except ImportError:  # The banner is cosmetic; do not make it a runtime dependency.
    figlet_format = None


class OpenAIResponseFinishReason(Enum):
    TOOL_CALLS = "tool_calls"
    STOP = "stop"


class Agent:
    """A minimal OpenRouter-backed agent with a bounded tool-calling loop."""

    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.model = model
        self.context = ConversationContext(system)
        # Kept as a public alias for callers of the original prototype.
        self.messages = self.context.messages
        self.console = Console()
        self.tools = tools or []
        self.tool_handlers = dict(tool_handlers or {})
        # Uppercase name is retained for compatibility with the original demo.
        self.AGENT_MAX_TURNS = 10

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        text = figlet_format("ZeroAgent", font="slant") if figlet_format else "ZeroAgent"
        print(f"[red]{text}[/red]")
        self.console.print(self.model, style="bold red")

    def __call__(self, message: str = "") -> str | None:
        if message:
            self.context.append("user", message)

        final_assistant_content = self.execute()
        if final_assistant_content:
            self.context.append("assistant", final_assistant_content)
        return final_assistant_content

    def execute(self) -> str | None:
        """Run until the model responds with text or the turn limit is reached."""
        for _ in range(self.AGENT_MAX_TURNS):
            with self.console.status("Running Agent .."):
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model,
                    tools=self.tools,
                    messages=self.messages,
                )

            for choice in response.choices:
                message = choice.message
                self.messages.append(message)

                if (
                    choice.finish_reason == OpenAIResponseFinishReason.TOOL_CALLS.value
                    and message.tool_calls
                ):
                    self._execute_tool_calls(message.tool_calls)
                elif (
                    choice.finish_reason == OpenAIResponseFinishReason.STOP.value
                    and message.content
                ):
                    self.console.log("Agent processing finished")
                    return message.content

        self.console.log("Agent reached its maximum number of turns")
        return None

    def _execute_tool_calls(self, tool_calls: Any) -> None:
        for tool_call in tool_calls:
            name = tool_call.function.name
            self.console.log(f"⛏ Initiating tool call: `{name}`")
            try:
                arguments = json.loads(tool_call.function.arguments)
                handler = self.tool_handlers.get(name)
                if handler is None:
                    result: Any = {"error": f"Unknown tool: {name}"}
                else:
                    result = handler(**arguments)
            except Exception as error:
                result = {"error": str(error)}

            self.context.append(
                "tool",
                json.dumps(result),
                tool_call_id=tool_call.id,
            )

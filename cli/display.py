"""Rich terminal rendering for the interactive chat.

This module deliberately only knows about presentation. The agent passes it
small tool events, while all model calls and context changes remain in
``agent.loop``.
"""

from __future__ import annotations

import json
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.json import JSON
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from agent.loop import ToolEvent


class TerminalDisplay:
    """Render a compact chat transcript with expandable-looking tool cards."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def banner(self, model: str, art: str) -> None:
        self.console.print(Padding(art, (1, 0), style="bold bright_red"))
        self.console.print(Rule(style="bright_black"))
        self.console.print(
            f"[bold bright_red]ZeroAgent[/]  [dim]•[/]  [cyan]{model}[/]"
        )
        self.console.print("[dim]Ask anything · Enter an empty line or press Ctrl-C to exit[/]\n")

    def user_message(self, message: str) -> None:
        self.console.print()
        self.console.print(Panel(
            Text(message), title="[bold cyan]You[/]", title_align="left",
            border_style="cyan", box=box.ROUNDED, padding=(0, 1),
        ))

    def assistant_message(self, message: str) -> None:
        self.console.print()
        self.console.print("[bold bright_green]ZeroAgent[/]")
        self.console.print(Padding(Markdown(message, code_theme="monokai"), (0, 1)))

    def tool_event(self, event: ToolEvent) -> None:
        if event.phase == "started":
            arguments = _render_json(event.arguments)
            body = Group(
                Text("Running tool", style="bold magenta"),
                Padding(arguments, (1, 0, 0, 0)),
            )
            self.console.print(Panel(
                body, title=f"[bold magenta]⚙ {event.name}[/]", title_align="left",
                border_style="magenta", box=box.ROUNDED, padding=(0, 1),
            ))
            return

        result = _render_json(event.result or "")
        status = "failed" if event.failed else "tool result"
        icon = "✗" if event.failed else "✓"
        color = "red" if event.failed else "green"
        self.console.print(Panel(
            result, title=f"[bold {color}]{icon} {event.name}[/]", title_align="left",
            subtitle=status, subtitle_align="right", border_style=color,
            box=box.ROUNDED, padding=(0, 1),
        ))


def _render_json(value: str) -> Any:
    """Pretty-print JSON and keep an unusually large result from taking over."""
    try:
        data = json.loads(value)
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        if len(formatted) > 4_000:
            return Syntax(
                f"{formatted[:4_000]}\n… output truncated in the terminal …",
                "json",
                word_wrap=True,
            )
        return JSON.from_data(data)
    except (json.JSONDecodeError, TypeError):
        preview = value if len(value) <= 4_000 else f"{value[:4_000]}\n… output truncated in the terminal …"
        return Syntax(preview, "text", word_wrap=True)

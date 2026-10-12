"""Terminal rendering for the interactive chat, styled after Claude Code and pi.

This module deliberately only knows about presentation. The agent sends it
small events (a tool started or finished, a model call started or finished)
and the REPL asks it to draw the input box; all model calls and context
changes stay in ``agent.loop``.

The transcript reads top to bottom like a log:

    > what files are in this repo?

    ⏺ list_files()
      ⎿  39 files
         .gitignore
         … +37 lines

    ⏺ Here's the layout …

and every prompt sits in a box with a two-line footer under it:

    ~/developer/repository/zeroagent (main)
    ↑12k ↓1.1k R8.0k $0.0031 · ctx 3.2k/1.0M 0.3% · deepseek/deepseek-v4.1-flash
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from agent.loop import ModelEvent, ToolEvent, Usage
from state.events import Event

PREVIEW_LINES = 4  # tool output lines shown before "… +N lines"


class TerminalDisplay:
    """Render the transcript, the thinking spinner, and the input box."""

    def __init__(self, console: Console, model: str, context_window: int | None) -> None:
        self.console = console
        self.model = model
        self.context_window = context_window
        self.location = _location(Path.cwd())
        self.live: Live | None = None

    # -- transcript ---------------------------------------------------------

    def welcome(self, session: str) -> None:
        self.console.print(Panel(
            f"[bold]✻ ZeroAgent[/]\n\n[dim]model:[/]   {self.model}\n[dim]cwd:[/]     {self.location}"
            f"\n[dim]session:[/] {session}",
            border_style="bright_red", expand=False, padding=(0, 2),
        ))
        self.console.print("[dim]  Enter an empty line or press Ctrl-C to exit[/]\n")

    def user_message(self, message: str) -> None:
        self.console.print(Text(f"> {message}", style="bright_black"))
        self.console.print()

    def assistant_message(self, message: str) -> None:
        self._bullet("⏺ ", "white", Markdown(message, code_theme="ansi_dark"))
        self.console.print()

    def tool_event(self, event: ToolEvent) -> None:
        if event.phase == "started":
            header = Text.assemble((event.name, "bold"), (f"({_arguments(event.arguments)})", "default"))
            self._bullet("⏺ ", "green", header)
            return

        lines = _result_lines(event.result or "")
        hidden = len(lines) - PREVIEW_LINES
        preview = "\n".join(lines[:PREVIEW_LINES]).expandtabs(4) or "(no output)"
        # One tool-output line per terminal line, cut off rather than wrapped.
        body = Text(preview, style="red" if event.failed else "bright_black", no_wrap=True, overflow="ellipsis")
        if hidden > 0:
            body.append(f"\n… +{hidden} lines", style="dim italic")
        self._bullet("  ⎿  ", "bright_black", body)
        self.console.print()

    def replay(self, events: list[Event]) -> None:
        """Reprint a resumed session from its log, exactly as it happened.

        This shows the *whole session*, including everything compaction folded
        away, which is usually far more than the model will see in its context.
        """
        for event in events:
            if event.role == "user":
                self.user_message(event.content)
            elif event.role == "assistant":
                if event.content:
                    self.assistant_message(event.content)
                for call in event.extra.get("tool_calls", []):
                    self.tool_event(ToolEvent("started", call["function"]["name"], call["function"]["arguments"]))
            elif event.role == "tool":
                failed = event.content.startswith('{"error"')
                self.tool_event(ToolEvent("finished", "", "", event.content, failed))
            elif event.role == "compaction":
                self.console.print(Text(
                    f"✻ Compacted: older messages were folded into a summary (context now starts at event {event.extra['covers_until']})",
                    style="dim italic",
                ))
                self.console.print()
        self.console.print(Rule("resumed", style="bright_black"))
        self.console.print()

    def interrupted(self) -> None:
        self._bullet("  ⎿  ", "bright_black", Text("Interrupted", style="red"))
        self.console.print()

    # -- thinking spinner ---------------------------------------------------

    def model_event(self, event: ModelEvent) -> None:
        """Show a spinner for exactly as long as a model call is in flight."""
        if event.phase == "started":
            self.live = Live(_Thinking(), console=self.console, refresh_per_second=10, transient=True)
            self.live.start()
        elif self.live:
            self.live.stop()
            self.live = None

    # -- input box ------------------------------------------------------------

    def prompt(self, usage: Usage) -> str:
        """Draw the input box and footer, then read one line inside the box.

        A plain input() can only read at the bottom of the screen, so on a real
        terminal we print the whole box and footer first, then move the cursor
        back up into the box with ANSI escape codes. Afterwards the box is
        erased, and the REPL reprints the message as part of the transcript.
        """
        rule = Text("─" * self.console.width, style="bright_black")
        footer = self._footer(usage)
        if not self.console.is_terminal:  # piped input: no cursor tricks
            self.console.print(footer)
            return self.console.input("> ")

        self.console.print(rule)
        self.console.print()  # the line we'll type on
        self.console.print(rule)
        self.console.print(footer)
        footer_height = len(self.console.render_lines(footer, pad=False))
        self.console.file.write(f"\x1b[{footer_height + 2}A\r")  # up to the typing line
        message = self.console.input("[bold]>[/] ")

        typed_rows = max(1, math.ceil((len(message) + 2) / self.console.width))
        self.console.file.write(f"\x1b[{typed_rows + 1}A\r\x1b[J")  # up to the top rule; clear below
        return message

    def _footer(self, usage: Usage) -> Text:
        stats = Text(
            f"↑{_count(usage.input_tokens)} ↓{_count(usage.output_tokens)} "
            f"R{_count(usage.cached_tokens)} ${usage.cost:.4f} · ",
            style="bright_black",
        )
        if self.context_window:
            percent = 100 * usage.context_tokens / self.context_window
            color = "red" if percent > 90 else "yellow" if percent > 70 else "bright_black"
            stats.append(
                f"ctx {_count(usage.context_tokens)}/{_count(self.context_window)} {percent:.1f}%",
                style=color,
            )
        else:
            stats.append(f"ctx {_count(usage.context_tokens)}", style="bright_black")
        stats.append(f" · {self.model}", style="bright_black")
        return Text.assemble(Text(self.location, style="bright_black"), "\n", stats)

    def _bullet(self, bullet: str, color: str, body: Any) -> None:
        """Print `body` after a bullet in a fixed-width gutter, so wrapped lines stay indented."""
        grid = Table.grid(expand=True)
        grid.add_column(width=len(bullet), no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(Text(bullet, style=color), body)
        self.console.print(grid)


class _Thinking:
    """The ✻ spinner. Rich re-renders it ten times a second while Live is running."""

    FRAMES = "·✢✳✶✻✽✻✶✳✢"

    def __init__(self) -> None:
        self.started = time.monotonic()

    def __rich__(self) -> Text:
        elapsed = time.monotonic() - self.started
        frame = self.FRAMES[int(elapsed * 8) % len(self.FRAMES)]
        return Text.assemble(
            (f"{frame} Thinking… ", "bright_red"),
            (f"({elapsed:.0f}s · ctrl-c to interrupt)", "bright_black"),
        )


def _arguments(raw: str) -> str:
    """Render tool arguments compactly: the first value bare, the rest as key: value."""
    try:
        arguments = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return raw
    parts = []
    for index, (key, value) in enumerate(arguments.items()):
        text = value if isinstance(value, str) else json.dumps(value)
        if len(text) > 80:
            text = text[:80] + "…"
        parts.append(text if index == 0 else f"{key}: {text}")
    return ", ".join(parts)


def _result_lines(raw: str) -> list[str]:
    """Turn a tool's JSON result into the lines worth showing a human."""
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return raw.splitlines()

    if isinstance(result, dict):
        if "error" in result:
            return [f"Error: {result['error']}"]
        if "files" in result:
            return [f"{len(result['files'])} files", *result["files"]]
        if "matches" in result:
            return [f"{len(result['matches'])} matches", *result["matches"]]
        if "content" in result:
            return [f"{result.get('total_lines', '?')} lines", *result["content"].splitlines()]
        if "exit_code" in result:
            output = (result["stdout"] + result["stderr"]).splitlines()
            status = [] if result["exit_code"] == 0 else [f"exit code {result['exit_code']}"]
            return status + output
    return json.dumps(result, indent=2, ensure_ascii=False).splitlines()


def _count(tokens: int) -> str:
    """Format a token count the way pi does: 950, 1.5k, 125k, 2.3M."""
    if tokens < 1_000:
        return str(tokens)
    if tokens < 10_000:
        return f"{tokens / 1_000:.1f}k"
    if tokens < 1_000_000:
        return f"{tokens // 1_000}k"
    return f"{tokens / 1_000_000:.1f}M"


def _location(path: Path) -> str:
    """The working directory, shortened with ~, plus the git branch if there is one."""
    try:
        text = f"~/{path.relative_to(Path.home())}"
    except ValueError:
        text = str(path)
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        branch = ""  # git isn't installed
    return f"{text} ({branch})" if branch else text

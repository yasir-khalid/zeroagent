"""Interactive read-eval-print loop for ZeroAgent.

This is presentation only: print a banner, read a line of input, hand it to
the agent, print whatever comes back. All the actual context/tool-calling
logic lives in agent/loop.py - this file just makes it pleasant to talk to
from a terminal. No slash commands, no state of its own.
"""

from __future__ import annotations

from rich.console import Console

from agent.loop import Agent
from cli.display import TerminalDisplay

try:
    from pyfiglet import figlet_format
except ImportError:  # The banner is cosmetic; don't make it a hard dependency.
    figlet_format = None

console = Console()


def _print_banner(display: TerminalDisplay, model: str) -> None:
    text = figlet_format("ZeroAgent", font="slant") if figlet_format else "ZeroAgent"
    display.banner(model, text)


def run(agent: Agent) -> None:
    """Read a message, run it through the agent, print the reply. Repeat."""
    display = TerminalDisplay(console)
    agent.console = console
    agent.on_tool_event = display.tool_event
    _print_banner(display, agent.model)

    while True:
        try:
            message = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not message:
            break

        display.user_message(message)
        response = agent(message)
        if response:
            display.assistant_message(response)
        console.print()

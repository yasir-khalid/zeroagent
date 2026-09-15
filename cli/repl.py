"""Interactive read-eval-print loop for ZeroAgent.

This is presentation only: print a banner, read a line of input, hand it to
the agent, print whatever comes back. All the actual context/tool-calling
logic lives in agent/loop.py - this file just makes it pleasant to talk to
from a terminal. No slash commands, no state of its own.
"""

from __future__ import annotations

from rich.console import Console

from agent.loop import Agent

try:
    from pyfiglet import figlet_format
except ImportError:  # The banner is cosmetic; don't make it a hard dependency.
    figlet_format = None

console = Console()


def _print_banner(model: str) -> None:
    text = figlet_format("ZeroAgent", font="slant") if figlet_format else "ZeroAgent"
    console.print(f"[bold red]{text}[/bold red]")
    console.print(model, style="bold red")
    console.print("Ask something. Empty line or Ctrl-C to quit.\n", style="dim")


def run(agent: Agent) -> None:
    """Read a message, run it through the agent, print the reply. Repeat."""
    _print_banner(agent.model)

    while True:
        try:
            message = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not message:
            break

        response = agent(message)
        if response:
            console.print(response, style="green")
        console.print()

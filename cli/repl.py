"""Interactive read-eval-print loop for ZeroAgent.

This is presentation only: read a line of input, hand it to the agent, print
whatever comes back. All the actual context/tool-calling logic lives in
agent/loop.py, and all the drawing lives in cli/display.py - this file just
connects the two. No slash commands, no state of its own. The one exception
is approve_command, which asks the human before an unsandboxed command runs.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

from agent.loop import Agent
from cli.display import TerminalDisplay

# Rich draws `inline code` as white on grey by default; plain colored text reads better.
console = Console(theme=Theme({"markdown.code": "bold cyan"}))


def approve_command(command: str) -> bool:
    """Ask the human before a command runs unsandboxed on their machine.

    The tool header printed just above already shows the command itself.
    """
    try:
        answer = console.input("  [bright_black]⎿[/]  [bold yellow]Run this on your machine?[/] [dim][y/N][/] ")
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    return answer.strip().lower() in ("y", "yes")


def run(agent: Agent) -> None:
    """Read a message, run it through the agent, print the reply. Repeat."""
    display = TerminalDisplay(console, agent.model, agent.context_window())
    agent.console = console
    agent.on_tool_event = display.tool_event
    agent.on_model_event = display.model_event
    log = agent.context.log
    display.welcome(log.path.stem if log.path else "(not saved)")
    if len(log):
        display.replay(log.events)

    while True:
        try:
            message = display.prompt(agent.usage).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not message:
            break

        display.user_message(message)
        try:
            response = agent(message)
        except KeyboardInterrupt:
            display.interrupted()
            continue
        if response:
            display.assistant_message(response)

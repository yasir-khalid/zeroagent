"""Compatibility entry point for the original ZeroAgent demo."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep `python zeroagent/main.py` working as documented in the README.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.loop import Agent
from zeroagent.tools import getGutenbergBooksTool, search_gutenberg_books


def main() -> None:
    """Run the Gutenberg-search demonstration agent."""
    agent = Agent(
        model="stepfun/step-3.5-flash:free",
        system="You're a helpful librarian who fetches books from the remote Gutendex library",
        tools=[getGutenbergBooksTool],
        tool_handlers={"search_gutenberg_books": search_gutenberg_books},
    )
    agent("Hello how are you?")
    agent("What are the titles of some James Joyce books?")

if __name__ == "__main__":
    main()

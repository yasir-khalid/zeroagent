"""Compatibility entry point for the original ZeroAgent demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep `python zeroagent/main.py` working as documented in the README.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Read KEY=value lines from .env into os.environ. No dependency needed."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(REPO_ROOT / ".env")

from agent.loop import Agent
from cli.repl import run
from runtime.sandbox import DockerSandbox
from tools.git import GIT_TOOL, GitTool
from tools.shell import BASH_TOOL, BashTool
from zeroagent.tools import getGutenbergBooksTool, search_gutenberg_books


def main() -> None:
    """Wire up a small agent plus an isolated shell workspace."""
    sandbox = DockerSandbox()

    bash = BashTool(sandbox)
    git = GitTool(sandbox)
    agent = Agent(
        model="deepseek/deepseek-v4.1-flash",
        system=(
            "You're a helpful assistant. You can search Project Gutenberg, or work in "
            "a disposable isolated Linux workspace using Bash and Git. The workspace "
            "has no host files and no network."
        ),
        tools=[getGutenbergBooksTool, BASH_TOOL, GIT_TOOL],
        tool_handlers={
            "search_gutenberg_books": search_gutenberg_books,
            "run_bash": bash.run_bash,
            "run_git": git.run_git,
        },
    )
    try:
        run(agent)
    finally:
        sandbox.close()


if __name__ == "__main__":
    main()

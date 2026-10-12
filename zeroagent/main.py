"""Compatibility entry point for the original ZeroAgent demo."""

from __future__ import annotations

import argparse
import os
import sys
import time
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

from rich.console import Console

from agent.loop import Agent
from cli.repl import approve_command, run
from runtime.sandbox import DockerSandbox, LocalSandbox, Sandbox
from state.events import EventLog
from tools.files import GREP_TOOL, LIST_FILES_TOOL, READ_FILE_TOOL, FileTools
from tools.git import GIT_TOOL, GitTool
from tools.shell import BASH_TOOL, BashTool
from zeroagent.tools import getGutenbergBooksTool, search_gutenberg_books


def _start_sandbox(project: Path) -> Sandbox:
    """Prefer the isolated Docker sandbox; otherwise run locally with approval."""
    try:
        return DockerSandbox()
    except RuntimeError as error:
        Console().print(
            f"[yellow]⚠ No Docker sandbox: {error}.[/] "
            "[dim]Bash and Git will run on your machine, with approval for each command.[/]\n"
        )
        return LocalSandbox(project, approve=approve_command)


def _open_session(sessions: Path, resume: str | None) -> EventLog:
    """Open a new session file, or an existing one to continue it."""
    if resume is None:
        return EventLog(sessions / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    if resume == "latest":
        existing = sorted(sessions.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)
        if not existing:
            raise SystemExit(f"No sessions to resume in {sessions}.")
        return EventLog(existing[-1])
    path = sessions / f"{resume}.jsonl"
    if not path.exists():
        raise SystemExit(f"No session named {resume} in {sessions}.")
    return EventLog(path)


def main() -> None:
    """Wire up a small agent: file tools for the project, plus Bash and Git."""
    parser = argparse.ArgumentParser(description="ZeroAgent: a coding agent built from scratch.")
    parser.add_argument(
        "--resume", nargs="?", const="latest", metavar="SESSION",
        help="continue a previous session (the most recent one if no name is given)",
    )
    args = parser.parse_args()

    project = Path.cwd()
    session = _open_session(project / ".zeroagent" / "sessions", args.resume)
    sandbox = _start_sandbox(project)
    if args.resume and isinstance(sandbox, DockerSandbox):
        # The old container died with the old process; this one starts empty.
        sandbox.notice = "This is a resumed session with a brand-new sandbox; /workspace is empty."
    files = FileTools(project)
    bash = BashTool(sandbox)
    git = GitTool(sandbox)

    agent = Agent(
        model="deepseek/deepseek-v4.1-flash",
        system=(
            f"You're a helpful assistant working in the project at {project}. "
            "Use list_files, grep, and read_file to explore it; they are read-only "
            f"and see the project files. {sandbox.description} "
            "You can also search Project Gutenberg for books."
        ),
        tools=[LIST_FILES_TOOL, GREP_TOOL, READ_FILE_TOOL, BASH_TOOL, GIT_TOOL, getGutenbergBooksTool],
        tool_handlers={
            "list_files": files.list_files,
            "grep": files.grep,
            "read_file": files.read_file,
            "run_bash": bash.run_bash,
            "run_git": git.run_git,
            "search_gutenberg_books": search_gutenberg_books,
        },
        session=session,
    )
    try:
        run(agent)
    finally:
        sandbox.close()


if __name__ == "__main__":
    main()

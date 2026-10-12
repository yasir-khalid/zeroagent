"""Read-only file tools: list, search, and read files under one root folder.

These run directly on the host, with no sandbox, because they can't change
anything. Two rules keep them safe and cheap:

- Every path is resolved against `root` and rejected if it escapes it, or if
  it touches something in HIDDEN (secrets, VCS internals, installed packages).
  Whatever a tool returns is sent to the model provider, so .env stays out.
- Every result is capped. A tool result lands in the context window, and one
  huge grep can crowd out the whole conversation.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path

HIDDEN = {".git", ".venv", "__pycache__", "node_modules", ".env", ".zeroagent"}
MAX_RESULTS = 200  # files listed or lines matched per call
MAX_LINES = 400  # lines returned by one read_file call


class FileTools:
    """List, grep, and read files inside `root`, and nowhere else."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def list_files(self, pattern: str = "*", path: str = ".") -> dict[str, object]:
        files = [self._relative(file) for file in self._walk(self._resolve(path))]
        files = [file for file in files if fnmatch(file, pattern)]
        return {"files": files[:MAX_RESULTS], "truncated": len(files) > MAX_RESULTS}

    def grep(self, pattern: str, path: str = ".", glob: str = "*") -> dict[str, object]:
        regex = re.compile(pattern)
        matches: list[str] = []
        for file in self._walk(self._resolve(path)):
            relative = self._relative(file)
            if not fnmatch(relative, glob):
                continue
            try:
                lines = file.read_text().splitlines()
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable; nothing useful to match
            for number, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append(f"{relative}:{number}: {line.strip()[:200]}")
                    if len(matches) >= MAX_RESULTS:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def read_file(self, path: str, start_line: int = 1) -> dict[str, object]:
        lines = self._resolve(path).read_text().splitlines()
        start = max(start_line, 1)
        chunk = lines[start - 1 : start - 1 + MAX_LINES]
        # Line numbers let the model cite code and ask for the next chunk.
        numbered = "\n".join(f"{n}\t{line}" for n, line in enumerate(chunk, start=start))
        return {"content": numbered, "total_lines": len(lines), "truncated": start - 1 + MAX_LINES < len(lines)}

    def _resolve(self, path: str) -> Path:
        full = (self.root / path).resolve()
        if not full.is_relative_to(self.root):
            raise ValueError(f"{path} is outside the workspace.")
        if HIDDEN.intersection(full.relative_to(self.root).parts):
            raise ValueError(f"{path} is hidden from the agent.")
        return full

    def _walk(self, start: Path) -> Iterator[Path]:
        if start.is_file():
            yield start
            return
        for directory, subdirectories, files in os.walk(start):
            subdirectories[:] = sorted(d for d in subdirectories if d not in HIDDEN)
            for name in sorted(files):
                if name not in HIDDEN:
                    yield Path(directory) / name

    def _relative(self, file: Path) -> str:
        return str(file.relative_to(self.root))


LIST_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in the project, optionally filtered by a glob such as \"*.py\" or \"agent/*\".",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob matched against the path relative to the project root. Defaults to \"*\"."},
                "path": {"type": "string", "description": "Folder to list, relative to the project root. Defaults to \".\"."},
            },
        },
    },
}

GREP_TOOL = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": "Search project files for a regular expression. Returns matching lines as path:line: text.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression to search for."},
                "path": {"type": "string", "description": "File or folder to search, relative to the project root. Defaults to \".\"."},
                "glob": {"type": "string", "description": "Only search files whose path matches this glob, such as \"*.py\"."},
            },
            "required": ["pattern"],
        },
    },
}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": f"Read a project file with line numbers, up to {MAX_LINES} lines at a time.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the project root."},
                "start_line": {"type": "integer", "minimum": 1, "description": "First line to return. Use it to page through long files."},
            },
            "required": ["path"],
        },
    },
}

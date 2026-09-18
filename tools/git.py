"""Git tool schema and handler for the isolated workspace."""

from __future__ import annotations

import shlex

from runtime.sandbox import Sandbox


class GitTool:
    """Run Git without making the model compose a shell command."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def run_git(self, args: list[str], timeout: int = 30) -> dict[str, object]:
        if not args or not all(isinstance(arg, str) for arg in args):
            return {"error": "args must be a non-empty list of Git arguments."}
        return self.sandbox.run(shlex.join(["git", *args]), timeout).to_dict()


GIT_TOOL = {
    "type": "function",
    "function": {
        "name": "run_git",
        "description": "Run Git in the isolated temporary workspace. There is no network or host repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "args": {"type": "array", "items": {"type": "string"}, "description": "Git arguments, for example [\"status\"] or [\"init\"]."},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["args"],
        },
    },
}

# Future work: expose repository-aware Git operations here.

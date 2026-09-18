"""Bash tool schema and handler for an already-isolated sandbox."""

from __future__ import annotations

from runtime.sandbox import Sandbox


class BashTool:
    """Expose arbitrary shell commands only after isolation has been chosen."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def run_bash(self, command: str, timeout: int = 30) -> dict[str, object]:
        if not command.strip():
            return {"error": "A command is required."}
        return self.sandbox.run(command, timeout).to_dict()


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": "Run a Bash command in the isolated temporary Linux workspace. It has no network or host files.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to run."},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["command"],
        },
    },
}

# Future work: expose allowlisted command execution here.

"""Small sandbox interface and a Docker implementation.

The Docker sandbox is intentionally disposable: it has no host mounts, no
network, a read-only root filesystem, and only temporary writable directories.
It is useful for learning the tool boundary before adding a full VM runtime.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int

    def to_dict(self) -> dict[str, object]:
        return {"stdout": self.stdout, "stderr": self.stderr, "exit_code": self.exit_code}


class Sandbox(Protocol):
    def run(self, command: str, timeout: int = 30) -> CommandResult: ...

    def close(self) -> None: ...


class DockerSandbox:
    """One isolated Linux container, kept alive for the lifetime of an agent."""

    image = "zeroagent-sandbox:latest"

    def __init__(self) -> None:
        if not shutil.which("docker"):
            raise RuntimeError("Docker is required for the local sandbox.")
        self.container_id = self._start()

    def _start(self) -> str:
        command = [
            "docker", "run", "--detach", "--rm",
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "128",
            "--memory", "512m", "--cpus", "1", "--workdir", "/workspace",
            "--tmpfs", "/tmp:exec,nosuid,nodev,size=128m,mode=1777",
            "--tmpfs", "/workspace:exec,nosuid,nodev,size=256m,mode=1777",
            self.image, "sleep", "infinity",
        ]
        try:
            return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip()
            raise RuntimeError(
                f"Could not start {self.image}. Build it with "
                "`docker build -t zeroagent-sandbox:latest runtime`. "
                f"Docker said: {detail}"
            ) from error

    def run(self, command: str, timeout: int = 30) -> CommandResult:
        """Run shell text in the isolated workspace; capture, never stream, output."""
        timeout = min(max(timeout, 1), 60)
        try:
            completed = subprocess.run(
                ["docker", "exec", self.container_id, "bash", "-lc", command],
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return CommandResult("", f"Command exceeded the {timeout}s sandbox limit.", 124)
        return CommandResult(completed.stdout, completed.stderr, completed.returncode)

    def close(self) -> None:
        subprocess.run(["docker", "rm", "--force", self.container_id], capture_output=True)

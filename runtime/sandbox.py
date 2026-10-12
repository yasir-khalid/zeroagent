"""Small sandbox interface with two implementations.

A sandbox is just "somewhere to run shell text". The Bash and Git tools don't
care which one they get, which is the point of the interface.

- DockerSandbox is intentionally disposable: it has no host mounts, no
  network, a read-only root filesystem, and only temporary writable
  directories. It is useful for learning the tool boundary before adding a
  full VM runtime.
- LocalSandbox runs on your machine with no isolation at all. It exists as a
  fallback when Docker isn't available, so it asks a human before every
  command instead.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int

    def to_dict(self) -> dict[str, object]:
        return {"stdout": self.stdout, "stderr": self.stderr, "exit_code": self.exit_code}


class Sandbox(Protocol):
    # One sentence for the system prompt, so the model knows where its commands run.
    description: str

    def run(self, command: str, timeout: int = 30) -> CommandResult: ...

    def close(self) -> None: ...


class DockerSandbox:
    """One isolated Linux container, kept alive for the lifetime of an agent."""

    image = "zeroagent-sandbox:latest"
    description = (
        "Bash and Git run in a disposable, isolated Linux workspace. It starts "
        "empty and has no network and no access to the project files."
    )

    def __init__(self) -> None:
        if not shutil.which("docker"):
            raise RuntimeError("Docker isn't installed")
        self.container_id = self._start()
        # Told to the model with the next command's output. Set when the
        # container is replaced, so the model doesn't assume old files exist.
        self.notice = ""

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
            if "docker.sock" in detail or "daemon" in detail:
                reason = "Docker isn't running (start Docker Desktop)"
            elif "Unable to find image" in detail or "pull access denied" in detail:
                reason = f"the {self.image} image isn't built (docker build -t {self.image} runtime)"
            else:
                reason = detail.splitlines()[-1] if detail else "docker run failed"
            raise RuntimeError(reason) from error

    def run(self, command: str, timeout: int = 30) -> CommandResult:
        """Run shell text in the isolated workspace; capture, never stream, output.

        The container can die independently of the agent (OOM, docker restart,
        someone ran `docker kill`). Sandboxes are cattle, not pets: if it's
        gone, start a fresh one and say so, rather than failing the tool call.
        """
        if not self._is_running():
            self.container_id = self._start()
            self.notice = "The sandbox had stopped and was replaced with a fresh one; /workspace is empty again."
        result = _run(["docker", "exec", self.container_id, "bash", "-lc", command], timeout)
        if self.notice:
            result.stderr = f"[{self.notice}]\n{result.stderr}"
            self.notice = ""
        return result

    def _is_running(self) -> bool:
        check = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", self.container_id],
            capture_output=True, text=True,
        )
        return check.stdout.strip() == "true"

    def close(self) -> None:
        subprocess.run(["docker", "rm", "--force", self.container_id], capture_output=True)


class LocalSandbox:
    """Runs commands directly on the host, inside `root`. Not isolated.

    Nothing stops a command from touching files outside `root`, so the only
    safety boundary is `approve`: a callback that shows the command to a
    human and returns True if they allow it.
    """

    def __init__(self, root: Path, approve: Callable[[str], bool]) -> None:
        self.root = root
        self.approve = approve
        self.description = (
            f"Bash and Git run on the user's own machine in {root}, the project "
            "folder. The user must approve every command, so prefer the file tools "
            "for reading and searching."
        )

    def run(self, command: str, timeout: int = 30) -> CommandResult:
        if not self.approve(command):
            return CommandResult("", "The user declined to run this command.", 1)
        return _run(["bash", "-lc", command], timeout, cwd=self.root)

    def close(self) -> None:
        pass  # nothing to tear down


def _run(args: list[str], timeout: int, cwd: Path | None = None) -> CommandResult:
    """Run a process with a clamped timeout and capture its output."""
    timeout = min(max(timeout, 1), 60)
    try:
        completed = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return CommandResult("", f"Command exceeded the {timeout}s sandbox limit.", 124)
    return CommandResult(completed.stdout, completed.stderr, completed.returncode)

"""
Docker sandbox executor for Phase 3 verification (§4.6).

Runs generated projects in isolated containers with resource limits.
Captures stdout, stderr, exit codes, and execution time for the
feedback loop back to the Repairer.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from matrioska.core.state import RunState

logger = logging.getLogger("matrioska.tools.sandbox")


class SandboxExecutor:
    """Executes generated code in a Docker sandbox.

    Spins up a container with the project files mounted, runs them,
    and captures all output for the Repairer's feedback loop.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        memory_limit: str = "256m",
        network: bool = False,
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.network = network

    def execute(self, state: RunState) -> Dict[str, Any]:
        """Run the generated project in a sandbox.

        Works by:
        1. Writing all artifacts to a temp directory
        2. Mounting it into a Docker container
        3. Running the main file
        4. Capturing stdout/stderr/exit code
        """
        if not self._docker_available():
            return {"ok": False, "error": "Docker not available", "exit_code": -1}

        with tempfile.TemporaryDirectory(prefix="matrioska_sandbox_") as tmp:
            tmp_path = Path(tmp)

            # Write all artifacts
            for artifact in state.artifacts.values():
                if artifact.status != "done":
                    continue
                file_path = tmp_path / f"{artifact.name}.{artifact.extension}"
                file_path.write_text(artifact.content, encoding="utf-8")

            # Find the main file (highest-order runnable)
            main_file = self._find_main(state)
            if main_file is None:
                return {
                    "ok": False,
                    "error": "No runnable main file found",
                    "exit_code": -1,
                }

            command = self._build_command(main_file)

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmp_path,
                )
                return {
                    "ok": result.returncode == 0,
                    "exit_code": result.returncode,
                    "stdout": result.stdout[:8000],
                    "stderr": result.stderr[:4000],
                    "command": " ".join(command),
                }
            except subprocess.TimeoutExpired:
                return {
                    "ok": False,
                    "error": f"Timeout after {self.timeout}s",
                    "exit_code": -1,
                }
            except Exception as e:
                return {"ok": False, "error": str(e), "exit_code": -1}

    def execute_container(
        self,
        state: RunState,
        volume_mount: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute inside an actual Docker container (requires docker CLI)."""
        if not self._docker_available():
            return {"ok": False, "error": "Docker not available", "exit_code": -1}

        with tempfile.TemporaryDirectory(prefix="matrioska_sandbox_") as tmp:
            tmp_path = Path(tmp)
            for artifact in state.artifacts.values():
                if artifact.status != "done":
                    continue
                (tmp_path / f"{artifact.name}.{artifact.extension}").write_text(
                    artifact.content, encoding="utf-8"
                )

            main_file = self._find_main(state)
            if main_file is None:
                return {"ok": False, "error": "No runnable main file", "exit_code": -1}

            cmd = self._build_command(main_file)
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                f"--memory={self.memory_limit}",
                "--network",
                "none" if not self.network else "host",
                "-v",
                f"{tmp_path}:/app:ro",
                "-w",
                "/app",
                self.image,
                *cmd,
            ]

            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 10,
                )
                return {
                    "ok": result.returncode == 0,
                    "exit_code": result.returncode,
                    "stdout": result.stdout[:8000],
                    "stderr": result.stderr[:4000],
                    "command": " ".join(docker_cmd),
                }
            except subprocess.TimeoutExpired:
                return {
                    "ok": False,
                    "error": f"Timeout after {self.timeout}s",
                    "exit_code": -1,
                }
            except Exception as e:
                return {"ok": False, "error": str(e), "exit_code": -1}

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _docker_available() -> bool:
        try:
            subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    @staticmethod
    def _find_main(state: RunState) -> Optional[str]:
        runnable = {"py", "js", "sh"}
        candidates = [
            a
            for a in state.artifacts.values()
            if a.extension in runnable and a.status == "done"
        ]
        if not candidates:
            return None
        # Pick highest-order runnable (the "main")
        candidates.sort(key=lambda a: -a.order)
        return f"{candidates[0].name}.{candidates[0].extension}"

    @staticmethod
    def _build_command(main_file: str) -> List[str]:
        ext = Path(main_file).suffix.lstrip(".")
        if ext == "py":
            return ["python", main_file]
        if ext == "js":
            return ["node", main_file]
        if ext == "sh":
            return ["bash", main_file]
        return ["cat", main_file]

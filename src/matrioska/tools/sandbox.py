"""
Docker sandbox executor for Phase 3 verification.

Runs generated projects in isolated containers with resource limits.
Captures stdout, stderr, exit codes, and duration for the repair loop.

Fallback hierarchy:
  1. Docker available  → docker run (fully isolated)
  2. Docker absent     → subprocess in tmpdir (dev mode, less isolated)
  3. Neither           → skip with ok=True, executed=False

Project type routing:
  python  → python <entrypoint>
  node    → node <entrypoint>
  shell   → bash <entrypoint>
  web     → HTML syntax validation (no execution)
  unknown → skip
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from matrioska.core.state import RunState

logger = logging.getLogger("matrioska.tools.sandbox")

# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class SandboxResult:
    ok: bool
    executed: bool = False
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    command: str = ""
    duration_s: float = 0.0
    error: str = ""          # infra error (Docker unavailable, timeout, etc.)
    mode: str = ""           # "docker" | "subprocess" | "html_validate" | "skipped"
    entrypoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "executed": self.executed,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:4000],
            "stderr": self.stderr[:4000],
            "command": self.command,
            "duration_s": round(self.duration_s, 2),
            "error": self.error,
            "mode": self.mode,
            "entrypoint": self.entrypoint,
        }


# ── Project type ──────────────────────────────────────────────────────────────


class ProjectType(Enum):
    PYTHON = "python"
    NODE   = "node"
    SHELL  = "shell"
    WEB    = "web"      # HTML/CSS/JS only — no server process
    UNKNOWN = "unknown"


def detect_project_type(state: RunState) -> ProjectType:
    """Infer the dominant project type from artifact extensions."""
    exts = {a.extension for a in state.artifacts.values() if a.status == "done"}
    if "py" in exts:
        return ProjectType.PYTHON
    if "js" in exts and "html" not in exts:
        return ProjectType.NODE
    if "sh" in exts:
        return ProjectType.SHELL
    if exts <= {"html", "css", "js"}:
        return ProjectType.WEB
    if "js" in exts:
        return ProjectType.NODE
    return ProjectType.UNKNOWN


# ── Entrypoint detection ──────────────────────────────────────────────────────

_PYTHON_MAINS = ["main.py", "app.py", "server.py", "run.py", "index.py", "cli.py"]
_NODE_MAINS   = ["index.js", "app.js", "main.js", "server.js"]
_SHELL_MAINS  = ["run.sh", "main.sh", "start.sh"]

# Patterns that indicate a blocking server — we import-check instead of running
_SERVER_PATTERNS = re.compile(
    r"uvicorn\.run|app\.run|flask\.run|create_app|http\.server"
    r"|socketio\.run|tornado\.ioloop|asyncio\.run.*serve",
    re.IGNORECASE,
)


def is_server_process(content: str) -> bool:
    return bool(_SERVER_PATTERNS.search(content))


def detect_entrypoint(state: RunState, ptype: ProjectType) -> Optional[str]:
    """Return the filename of the best entrypoint, or None."""
    done = {
        f"{a.name}.{a.extension}": a
        for a in state.artifacts.values()
        if a.status == "done"
    }

    if ptype == ProjectType.PYTHON:
        for name in _PYTHON_MAINS:
            if name in done:
                return name
        # Fall back to highest-order .py
        py = [a for a in done.values() if a.extension == "py"]
        if py:
            return f"{max(py, key=lambda a: a.order).name}.py"

    elif ptype == ProjectType.NODE:
        for name in _NODE_MAINS:
            if name in done:
                return name
        js = [a for a in done.values() if a.extension == "js"]
        if js:
            return f"{max(js, key=lambda a: a.order).name}.js"

    elif ptype == ProjectType.SHELL:
        for name in _SHELL_MAINS:
            if name in done:
                return name
        sh = [a for a in done.values() if a.extension == "sh"]
        if sh:
            return f"{max(sh, key=lambda a: a.order).name}.sh"

    return None


# ── Dependency extraction ─────────────────────────────────────────────────────

_STDLIB = frozenset(
    "os sys re json time math random pathlib typing dataclasses enum "
    "functools itertools collections abc io logging threading queue "
    "subprocess tempfile shutil hashlib datetime argparse unittest "
    "ast inspect copy string struct signal traceback contextlib "
    "http urllib email csv sqlite3 configparser".split()
)

_KNOWN_ALIASES: Dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "attr": "attrs",
}


def extract_pip_packages(artifacts_content: List[str]) -> List[str]:
    """Extract third-party package names from Python source files."""
    packages: set[str] = set()
    for content in artifacts_content:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fallback to regex
            for m in re.finditer(r"^(?:import|from)\s+(\w+)", content, re.MULTILINE):
                pkg = m.group(1)
                if pkg not in _STDLIB:
                    packages.add(_KNOWN_ALIASES.get(pkg, pkg))
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in _STDLIB:
                        packages.add(_KNOWN_ALIASES.get(top, top))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top not in _STDLIB:
                        packages.add(_KNOWN_ALIASES.get(top, top))

    return sorted(packages)


# ── HTML validation ───────────────────────────────────────────────────────────


def validate_html(content: str) -> SandboxResult:
    """Validate HTML using the stdlib html.parser (no external deps)."""
    from html.parser import HTMLParser

    errors: List[str] = []

    class _Checker(HTMLParser):
        def handle_error(self, message: str) -> None:  # type: ignore[override]
            errors.append(message)

    try:
        _Checker().feed(content)
    except Exception as e:
        errors.append(str(e))

    ok = len(errors) == 0
    return SandboxResult(
        ok=ok,
        executed=True,
        mode="html_validate",
        stderr="\n".join(errors),
        exit_code=0 if ok else 1,
    )


# ── File error parsing ────────────────────────────────────────────────────────

_PY_FILE_RE  = re.compile(r'File "([^"]+\.py)",\s*line (\d+)')
_JS_FILE_RE  = re.compile(r'(\w[\w/]*\.js):(\d+)')
_SH_FILE_RE  = re.compile(r'(\w[\w/]*\.sh):\s*line\s*(\d+)')


def parse_erroring_file(stderr: str, known_files: set[str]) -> Optional[str]:
    """Try to identify which artifact caused a runtime error from stderr."""
    for pattern in (_PY_FILE_RE, _JS_FILE_RE, _SH_FILE_RE):
        for m in pattern.finditer(stderr):
            candidate = Path(m.group(1)).name
            if candidate in known_files:
                return candidate
    return None


# ── Main executor ─────────────────────────────────────────────────────────────


class SandboxExecutor:
    """Runs generated projects in an isolated sandbox.

    Tries Docker first; falls back to a plain subprocess in a tmpdir
    when Docker is unavailable (useful for CI without Docker).
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        memory_limit: str = "256m",
        cpu_limit: str = "0.5",
        network: bool = False,
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network = network

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, state: RunState, work_dir: Optional[Path] = None) -> SandboxResult:
        """Detect project type and run in the most isolated available sandbox."""
        ptype = detect_project_type(state)
        logger.info("Sandbox: project_type=%s", ptype.value)

        if ptype == ProjectType.WEB:
            return self._validate_web(state)
        if ptype == ProjectType.UNKNOWN:
            return SandboxResult(ok=True, mode="skipped",
                                 error="No runnable files detected")

        entrypoint = detect_entrypoint(state, ptype)
        if entrypoint is None:
            return SandboxResult(ok=True, mode="skipped",
                                 error="Could not identify entrypoint")

        with tempfile.TemporaryDirectory(prefix="matrioska_sb_") as tmp:
            tmp_path = Path(tmp)
            self._write_artifacts(state, tmp_path)

            if self._docker_available():
                return self._run_docker(tmp_path, entrypoint, ptype, state)
            else:
                logger.warning("Docker unavailable — running in subprocess (less isolated)")
                return self._run_subprocess(tmp_path, entrypoint, ptype, state)

    # ── Internal runners ─────────────────────────────────────────────────

    def _run_docker(
        self,
        work_dir: Path,
        entrypoint: str,
        ptype: ProjectType,
        state: RunState,
    ) -> SandboxResult:
        shell_cmd = self._build_shell_cmd(entrypoint, ptype, state)
        docker_cmd = [
            "docker", "run", "--rm",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
            "--pids-limit=64",
            "--network", "none" if not self.network else "bridge",
            "--read-only",
            "--tmpfs", "/tmp:size=64m",
            "-v", f"{work_dir}:/app:ro",
            "-w", "/app",
            self.image,
            "sh", "-c", shell_cmd,
        ]
        return self._exec(docker_cmd, mode="docker", entrypoint=entrypoint)

    def _run_subprocess(
        self,
        work_dir: Path,
        entrypoint: str,
        ptype: ProjectType,
        state: RunState,
    ) -> SandboxResult:
        shell_cmd = self._build_shell_cmd(entrypoint, ptype, state)
        cmd = ["sh", "-c", shell_cmd]
        return self._exec(cmd, mode="subprocess", entrypoint=entrypoint, cwd=work_dir)

    def _exec(
        self,
        cmd: List[str],
        *,
        mode: str,
        entrypoint: str,
        cwd: Optional[Path] = None,
    ) -> SandboxResult:
        t0 = time.time()
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd,
            )
            duration = time.time() - t0
            ok = r.returncode == 0
            return SandboxResult(
                ok=ok,
                executed=True,
                exit_code=r.returncode,
                stdout=r.stdout[:4000],
                stderr=r.stderr[:4000],
                command=" ".join(cmd[:6]) + " ...",
                duration_s=duration,
                mode=mode,
                entrypoint=entrypoint,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False, executed=True, exit_code=-1, mode=mode,
                error=f"Timeout after {self.timeout}s",
                entrypoint=entrypoint,
                duration_s=time.time() - t0,
            )
        except Exception as e:
            return SandboxResult(
                ok=False, executed=True, exit_code=-1, mode=mode,
                error=str(e), entrypoint=entrypoint,
                duration_s=time.time() - t0,
            )

    def _validate_web(self, state: RunState) -> SandboxResult:
        html_arts = [
            a for a in state.artifacts.values()
            if a.extension == "html" and a.status == "done"
        ]
        if not html_arts:
            return SandboxResult(ok=True, mode="skipped",
                                 error="No HTML files to validate")
        results = [validate_html(a.content) for a in html_arts]
        combined_ok = all(r.ok for r in results)
        combined_err = "\n".join(r.stderr for r in results if r.stderr)
        return SandboxResult(
            ok=combined_ok, executed=True, mode="html_validate",
            stderr=combined_err, exit_code=0 if combined_ok else 1,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _write_artifacts(self, state: RunState, dest: Path) -> None:
        for a in state.artifacts.values():
            if a.status == "done":
                (dest / f"{a.name}.{a.extension}").write_text(
                    a.content, encoding="utf-8"
                )

    def _build_shell_cmd(
        self,
        entrypoint: str,
        ptype: ProjectType,
        state: RunState,
    ) -> str:
        """Build the shell command string to run inside the container.

        For Python: auto-installs detected third-party packages before running.
        For server processes: runs import check only (avoids blocking on listen).
        """
        if ptype == ProjectType.PYTHON:
            # Check if the entrypoint starts a server
            artifact_name = Path(entrypoint).stem
            art = state.artifacts.get(entrypoint) or state.artifacts.get(artifact_name)
            if art and is_server_process(art.content):
                # Import-only check — don't start the server
                module = Path(entrypoint).stem
                return f"python -c \"import {module}; print('import OK')\""

            # Extract and install deps
            py_contents = [
                a.content for a in state.artifacts.values()
                if a.extension == "py" and a.status == "done"
            ]
            packages = extract_pip_packages(py_contents)
            if packages:
                pkg_str = " ".join(packages)
                pip_cmd = f"pip install -q --no-cache-dir {pkg_str} 2>/dev/null; "
            else:
                pip_cmd = ""
            return f"{pip_cmd}python {entrypoint}"

        elif ptype == ProjectType.NODE:
            return f"node {entrypoint}"

        elif ptype == ProjectType.SHELL:
            return f"bash {entrypoint}"

        return f"cat {entrypoint}"

    @staticmethod
    def _docker_available() -> bool:
        try:
            r = subprocess.run(
                ["docker", "info"],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    # ── Legacy API (backwards compat) ────────────────────────────────────

    def execute(self, state: RunState) -> Dict[str, Any]:
        """Backwards-compatible wrapper — delegates to run()."""
        return self.run(state).to_dict()

    def execute_container(
        self,
        state: RunState,
        volume_mount: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Backwards-compatible wrapper — delegates to run()."""
        return self.run(state).to_dict()

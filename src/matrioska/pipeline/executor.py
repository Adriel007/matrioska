"""
Executor: write artifacts to disk, install dependencies, run generated code.

Implements two Claude Code-inspired capabilities:
  - Real execution feedback: run generated .py files, capture stderr as repair signal
  - Dependency installation: detect non-stdlib imports and pip-install them
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from matrioska.core.state import FileArtifact

logger = logging.getLogger("matrioska.pipeline.executor")

# Servers / long-running processes: skip execution (they'd hang)
_SERVER_PATTERNS = re.compile(
    r"(uvicorn|flask\.run|app\.run|gunicorn|tornado|twisted"
    r"|asyncio\.run|serve\(|listen\(|socketserver)",
    re.IGNORECASE,
)

# Third-party package aliases (import name → pip package name)
_IMPORT_ALIASES: Dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "yaml": "pyyaml",
    "toml": "tomli",
    "jose": "python-jose",
    "jwt": "PyJWT",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "sqlalchemy": "sqlalchemy",
    "pydantic": "pydantic",
    "click": "click",
    "rich": "rich",
    "typer": "typer",
    "requests": "requests",
    "numpy": "numpy",
    "pandas": "pandas",
    "pytest": "pytest",
}

# Modules that are stdlib but not in sys.stdlib_module_names on older Python
_EXTRA_STDLIB = {
    "abc", "argparse", "ast", "asyncio", "collections", "contextlib",
    "copy", "csv", "dataclasses", "datetime", "decimal", "difflib",
    "email", "enum", "functools", "glob", "hashlib", "http", "importlib",
    "inspect", "io", "itertools", "json", "logging", "math", "operator",
    "os", "pathlib", "pickle", "platform", "pprint", "queue", "random",
    "re", "shutil", "signal", "socket", "sqlite3", "string", "struct",
    "subprocess", "sys", "tempfile", "textwrap", "threading", "time",
    "traceback", "typing", "unittest", "urllib", "uuid", "warnings",
    "weakref", "xml", "zipfile", "zlib",
}


@dataclass
class ExecutionResult:
    file: str
    ok: bool
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False
    skip_reason: str = ""

    @property
    def error_for_repair(self) -> str:
        """Compact error string suitable for the Repairer prompt."""
        if self.skipped or self.ok:
            return ""
        msg = self.stderr.strip()
        # Keep the most informative part (last few lines of traceback)
        lines = [l for l in msg.splitlines() if l.strip()]
        if len(lines) > 8:
            lines = lines[-8:]
        return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────


def write_artifacts_to_disk(
    artifacts: Dict[str, FileArtifact],
    work_dir: Path,
    only_done: bool = True,
) -> List[Path]:
    """Write artifact content to work_dir so imports resolve during execution."""
    work_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for art in artifacts.values():
        if only_done and art.status != "done":
            continue
        path = work_dir / f"{art.name}.{art.extension}"
        path.write_text(art.content, encoding="utf-8")
        written.append(path)
    return written


def install_missing_deps(
    artifacts: Dict[str, FileArtifact],
    work_dir: Path,
) -> List[str]:
    """Detect non-stdlib imports in generated .py files and pip-install them.

    Returns the list of packages that were attempted to install.
    Best-effort: failures are logged but never crash the pipeline.
    """
    stdlib = _get_stdlib()
    packages: Set[str] = set()

    for art in artifacts.values():
        if art.extension != "py" or not art.content:
            continue
        imports = _extract_imports(art.content)
        for mod in imports:
            root = mod.split(".")[0]
            if root in stdlib or root in sys.modules:
                continue
            pip_name = _IMPORT_ALIASES.get(root, root)
            packages.add(pip_name)

    if not packages:
        return []

    logger.info("Installing %d detected dep(s): %s", len(packages), ", ".join(sorted(packages)))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *sorted(packages)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=work_dir,
        )
        if result.returncode != 0:
            logger.warning("pip install partial failure:\n%s", result.stderr[:500])
    except Exception as e:
        logger.warning("Dependency installation failed: %s", e)

    return sorted(packages)


def execute_artifact(
    artifact: FileArtifact,
    work_dir: Path,
    timeout: int = 8,
) -> ExecutionResult:
    """Try to import a .py artifact and report runtime errors.

    Strategy:
    - If the file looks like a server (uvicorn, flask.run…) → skip (would hang)
    - Otherwise run: python -c "import <module>" in work_dir
    - Capture ImportError, NameError, SyntaxError, etc. as repair signal

    Returns ExecutionResult with stderr ready to feed the Repairer.
    """
    if artifact.extension != "py" or artifact.status != "done":
        return ExecutionResult(file=f"{artifact.name}.{artifact.extension}", ok=True, skipped=True,
                               skip_reason="non-python or not done")

    content = artifact.content or ""

    # Skip servers — they'd hang the pipeline
    if _SERVER_PATTERNS.search(content):
        return ExecutionResult(file=f"{artifact.name}.{artifact.extension}", ok=True, skipped=True,
                               skip_reason="server / long-running process")

    file_path = work_dir / f"{artifact.name}.{artifact.extension}"
    if not file_path.exists():
        return ExecutionResult(file=f"{artifact.name}.{artifact.extension}", ok=True, skipped=True,
                               skip_reason="file not written to disk yet")

    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {artifact.name}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
        )
        ok = result.returncode == 0
        if not ok:
            logger.debug(
                "Execution error in %s:\n%s", f"{artifact.name}.{artifact.extension}", result.stderr[:300]
            )
        return ExecutionResult(
            file=f"{artifact.name}.{artifact.extension}",
            ok=ok,
            returncode=result.returncode,
            stdout=result.stdout[:1000],
            stderr=result.stderr[:2000],
        )
    except subprocess.TimeoutExpired:
        # Timeout likely means a blocking call — treat as ok (server-like)
        return ExecutionResult(file=f"{artifact.name}.{artifact.extension}", ok=True, skipped=True,
                               skip_reason="timeout (likely blocking I/O)")
    except Exception as e:
        return ExecutionResult(file=f"{artifact.name}.{artifact.extension}", ok=False, returncode=1,
                               stderr=str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_stdlib() -> Set[str]:
    if hasattr(sys, "stdlib_module_names"):
        return sys.stdlib_module_names | _EXTRA_STDLIB  # type: ignore[attr-defined]
    return _EXTRA_STDLIB


def _extract_imports(source: str) -> List[str]:
    """Extract top-level module names from Python source via AST."""
    modules: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to regex on broken code
        for m in re.finditer(r"^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", source, re.M):
            modules.append(m.group(1).split(".")[0])
        return modules

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module.split(".")[0])

    return modules

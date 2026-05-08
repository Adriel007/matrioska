"""GPT-Engineer wrapper — runs gpt-engineer via CLI subprocess.

GPT-Engineer reads OPENAI_API_KEY and OPENAI_API_BASE from environment.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_gpt_engineer(
    task_prompt: str,
    model: str,
    api_key: str,
    api_base: str,
    work_dir: Path,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 600,
) -> dict[str, Any]:
    """Run GPT-Engineer on a coding task.

    Writes the task prompt to a `prompt` file (what gpt-engineer expects)
    and runs `gpt-engineer` in the work directory.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # GPT-Engineer reads the task from a file named 'prompt'
    (work_dir / "prompt").write_text(task_prompt)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_API_BASE"] = api_base
    # GPT-Engineer uses these to construct the model
    env["MODEL"] = model

    # Try to find gpt-engineer
    ge_bin = shutil.which("gpt-engineer")
    if not ge_bin:
        ge_cmd = [sys.executable, "-m", "gpt_engineer"]
    else:
        ge_cmd = [ge_bin]

    # gpt-engineer expects a project directory
    cmd = ge_cmd + [str(work_dir)]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(work_dir),
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return {
            "files": _collect_files(work_dir),
            "tokens_used": 0,
            "duration_s": duration,
            "error": "timeout",
            "raw_response": "",
            "orchestrator": "gpt_engineer",
        }
    except FileNotFoundError:
        duration = time.time() - start
        return {
            "files": [],
            "tokens_used": 0,
            "duration_s": duration,
            "error": "gpt_engineer not found",
            "raw_response": "",
            "orchestrator": "gpt_engineer",
        }

    duration = time.time() - start

    return {
        "files": _collect_files(work_dir),
        "tokens_used": 0,
        "duration_s": duration,
        "stdout": result.stdout[-5000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
        "exit_code": result.returncode,
        "raw_response": result.stdout or "",
        "orchestrator": "gpt_engineer",
    }


def _collect_files(work_dir: Path) -> list[Path]:
    """Collect generated files from work dir, excluding metadata."""
    exclude = {".git", "__pycache__", "venv", ".venv", "node_modules"}
    exclude_files = {"prompt", "TASK.md", ".gitignore", "README.md"}

    files = []
    for f in work_dir.rglob("*"):
        if f.is_file() and not any(ex in f.parts for ex in exclude) and f.name not in exclude_files:
            files.append(f)
    return files

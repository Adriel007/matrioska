"""Aider wrapper — runs Aider as a subprocess in non-interactive mode.

Uses the same model/API config as all other orchestrators via
Aider's --model, --openai-api-base, and --openai-api-key flags.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_aider(
    task_prompt: str,
    model: str,
    api_key: str,
    api_base: str,
    work_dir: Path,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 600,
) -> dict[str, Any]:
    """Run Aider on a coding task.

    Uses `aider --message` for non-interactive mode with the same
    Groq API backend as all other orchestrators.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write task to a file so Aider can reference it
    task_file = work_dir / "TASK.md"
    task_file.write_text(task_prompt)

    # Check if aider is available
    aider_bin = shutil.which("aider")
    if not aider_bin:
        # Try running via python -m aider
        aider_cmd = [sys.executable, "-m", "aider"]
    else:
        aider_cmd = [aider_bin]

    # Build Aider command
    # --model uses format: openai/<model_name> for OpenAI-compatible APIs
    aider_model = f"openai/{model}"

    cmd = aider_cmd + [
        "--model", aider_model,
        "--openai-api-base", api_base,
        "--openai-api-key", api_key,
        "--message", task_prompt,
        "--yes",                    # Non-interactive: auto-accept
        "--no-git",                 # Don't require git
        "--map-tokens", "1024",     # Limit context map
    ]

    env = os.environ.copy()
    # Aider also reads these env vars
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_API_BASE"] = api_base

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
            "orchestrator": "aider",
        }

    duration = time.time() - start

    return {
        "files": _collect_files(work_dir),
        "tokens_used": 0,  # Aider doesn't report tokens easily in subprocess mode
        "duration_s": duration,
        "stdout": result.stdout[-5000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
        "exit_code": result.returncode,
        "raw_response": result.stdout or "",
        "orchestrator": "aider",
    }


def _collect_files(work_dir: Path) -> list[Path]:
    """Collect generated files from work dir, excluding metadata."""
    exclude = {".git", ".aider", "__pycache__", ".matrioska_work"}
    exclude_files = {"TASK.md", ".gitignore", ".aider.conf.yml"}

    files = []
    for f in work_dir.rglob("*"):
        if f.is_file() and not any(ex in f.parts for ex in exclude) and f.name not in exclude_files:
            files.append(f)
    return files

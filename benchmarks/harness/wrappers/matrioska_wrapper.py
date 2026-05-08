"""Matrioska v3 wrapper — runs the full Matrioska pipeline on a task."""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any


def run_matrioska(
    task_prompt: str,
    model: str,
    api_key: str,
    api_base: str,
    work_dir: Path,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Run Matrioska v3 on a single task.

    Falls back to a direct single-shot call when the orchestrator produces
    no files (Agentless-style safety net, arXiv:2407.01489).
    """
    os.environ["MATRIOSKA_API_KEY"] = api_key
    os.environ["MATRIOSKA_BASE_URL"] = api_base
    os.environ["MATRIOSKA_PROVIDER"] = "openai"
    os.environ["MATRIOSKA_MODEL"] = model
    os.environ["MATRIOSKA_ARCHITECT_MODEL"] = model
    os.environ["MATRIOSKA_GENERATOR_MODEL"] = model
    os.environ["MATRIOSKA_VALIDATOR_MODEL"] = model
    os.environ["MATRIOSKA_JUDGE_MODEL"] = model
    os.environ["MATRIOSKA_REPAIRER_MODEL"] = model
    os.environ["MATRIOSKA_MAX_TOKENS"] = str(max_tokens)
    os.environ["MATRIOSKA_WORK_DIR"] = str(work_dir / ".matrioska_work")
    os.environ["MATRIOSKA_ENABLE_TOT"] = "false"
    os.environ["MATRIOSKA_ENABLE_REFLEXION"] = "false"
    os.environ["MATRIOSKA_ARCHITECT_CANDIDATES"] = "1"
    os.environ["MATRIOSKA_PARALLEL"] = "false"
    os.environ["MATRIOSKA_ENABLE_SANDBOX"] = "false"
    os.environ["MATRIOSKA_ENABLE_COST_TRACKING"] = "true"
    os.environ["MATRIOSKA_MAX_REPAIRS"] = "1"

    start = time.time()
    error_log = work_dir / ".matrioska_error.txt"

    try:
        from matrioska.core.config import load_config
        from matrioska.pipeline.orchestrator import Matrioska

        config = load_config()
        config.work_dir = Path(work_dir / ".matrioska_work")

        m = Matrioska(config)
        result = m.run(task_prompt)

    except Exception as e:
        tb = traceback.format_exc()
        error_log.write_text(f"ERROR: {type(e).__name__}: {e}\n\n{tb}")
        print(f"\n  [matrioska crash] {type(e).__name__}: {e}")
        print(f"  [traceback saved to {error_log.name}]")

        # Agentless fallback: single-shot direct generation
        print("  [agentless fallback] running direct single-shot...")
        from benchmarks.harness.wrappers.direct import run_direct
        fallback = run_direct(
            task_prompt=task_prompt,
            model=model,
            api_key=api_key,
            api_base=api_base,
            work_dir=work_dir,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        fallback["orchestrator"] = "matrioska_fallback"
        fallback["duration_s"] = round(time.time() - start, 2)
        return fallback

    duration = round(time.time() - start, 2)

    files = []
    artifacts = result.get("artifacts", []) if isinstance(result, dict) else []

    for artifact in artifacts:
        fname = getattr(artifact, "name", None)
        ext = getattr(artifact, "extension", None)
        content = getattr(artifact, "content", None)
        if fname and content:
            filename = f"{fname}.{ext}" if ext else fname
            dest = work_dir / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
            files.append(dest)

    # Agentless fallback: orchestrator ran but produced nothing
    if not files and not error_log.exists():
        status = result.get("status", "unknown") if isinstance(result, dict) else "error"
        error_log.write_text(f"Pipeline ran but produced 0 artifacts. status={status}")
        print(f"\n  [matrioska empty] status={status}, triggering agentless fallback...")
        from benchmarks.harness.wrappers.direct import run_direct
        fallback = run_direct(
            task_prompt=task_prompt,
            model=model,
            api_key=api_key,
            api_base=api_base,
            work_dir=work_dir,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        fallback["orchestrator"] = "matrioska_fallback"
        fallback["duration_s"] = round(time.time() - start, 2)
        return fallback

    token_data = result.get("tokens", {}) if isinstance(result, dict) else {}
    if isinstance(token_data, dict):
        tokens = (
            token_data.get("total_tokens")
            or token_data.get("prompt_tokens", 0) + token_data.get("completion_tokens", 0)
        )
    else:
        tokens = 0

    return {
        "files": files[:20],
        "tokens_used": tokens,
        "duration_s": duration,
        "status": result.get("status", "unknown") if isinstance(result, dict) else "error",
        "orchestrator": "matrioska",
    }

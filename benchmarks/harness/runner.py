"""Unified benchmark runner — runs all orchestrators on all tasks.

Controls for: same model, same API key, same API base, same task prompt,
same work directory structure.

Usage:
    python -m benchmarks.harness.runner --task cli_todo --orchestrator direct
    python -m benchmarks.harness.runner --all --runs 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.tasks.loader import BenchmarkTask, load_all_tasks, load_task
from benchmarks.harness.wrappers.direct import run_direct
from benchmarks.harness.wrappers.direct_plan import run_direct_plan
from benchmarks.harness.wrappers.aider_wrapper import run_aider
from benchmarks.harness.wrappers.gpt_engineer_wrapper import run_gpt_engineer
from benchmarks.harness.wrappers.matrioska_wrapper import run_matrioska


ORCHESTRATORS = {
    "direct": run_direct,
    "direct_plan": run_direct_plan,
    "aider": run_aider,
    "gpt_engineer": run_gpt_engineer,
    "matrioska": run_matrioska,
}


def run_single(
    orchestrator_name: str,
    task: BenchmarkTask,
    config: dict[str, Any],
    results_dir: Path,
    run_id: int = 0,
) -> dict[str, Any]:
    """Run a single orchestrator on a single task.

    Args:
        orchestrator_name: Key in ORCHESTRATORS dict.
        task: The benchmark task to run.
        config: Dict with model, api_key, api_base, temperature, max_tokens.
        results_dir: Where to write output.
        run_id: Which run number (for repeated runs).

    Returns:
        Dict with results (files, tokens, duration, etc.).
    """
    orch_func = ORCHESTRATORS.get(orchestrator_name)
    if orch_func is None:
        raise ValueError(f"Unknown orchestrator: {orchestrator_name}. "
                         f"Options: {list(ORCHESTRATORS.keys())}")

    # Create isolated work directory
    work_dir = results_dir / "work" / f"{task.id}_{orchestrator_name}_{run_id}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    print(f"  [{orchestrator_name}] Running {task.id} (run {run_id})...", end=" ", flush=True)

    try:
        result = orch_func(
            task_prompt=task.prompt,
            model=config["model"],
            api_key=config["api_key"],
            api_base=config["api_base"],
            work_dir=work_dir,
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 4096),
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "orchestrator": orchestrator_name,
            "task_id": task.id,
            "run_id": run_id,
            "files": [],
            "tokens_used": 0,
            "duration_s": 0,
            "error": str(e),
            "error_type": type(e).__name__,
        }

    # Log result
    log_entry = {
        "orchestrator": orchestrator_name,
        "task_id": task.id,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "files": [str(f.relative_to(work_dir)) for f in result.get("files", [])],
        "num_files": len(result.get("files", [])),
        "tokens_used": result.get("tokens_used", 0),
        "duration_s": result.get("duration_s", 0),
        "error": result.get("error"),
        "orchestrator_metadata": {
            k: str(v)[:500] for k, v in result.items()
            if k not in ("files", "tokens_used", "duration_s", "error")
        },
    }

    log_file = results_dir / "logs" / f"{task.id}_{orchestrator_name}_{run_id}.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(log_entry, indent=2, default=str))

    print(f"OK ({len(result.get('files', []))} files, {result.get('tokens_used', 0)} tokens, {result.get('duration_s', 0):.1f}s)")

    return log_entry


def run_benchmark(
    orchestrators: list[str],
    tasks: list[BenchmarkTask],
    config: dict[str, Any],
    results_dir: Path,
    num_runs: int = 3,
) -> list[dict[str, Any]]:
    """Run the full benchmark: all orchestrators × all tasks × num_runs.

    Args:
        orchestrators: List of orchestrator names to run.
        tasks: List of tasks to run.
        config: Dict with model, api_key, api_base.
        results_dir: Base output directory.
        num_runs: Number of repeated runs per task.

    Returns:
        List of all run log entries.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    total = len(orchestrators) * len(tasks) * num_runs
    print(f"\n{'='*60}")
    print(f"Benchmark: {len(orchestrators)} orchestrators × {len(tasks)} tasks × {num_runs} runs = {total} executions")
    print(f"Model: {config['model']}")
    print(f"API Base: {config['api_base']}")
    print(f"Results: {results_dir}")
    print(f"{'='*60}\n")

    for task in tasks:
        print(f"Task: {task.id} ({task.category}/{task.complexity})")
        for orch in orchestrators:
            for run_id in range(num_runs):
                result = run_single(orch, task, config, results_dir, run_id)
                all_results.append(result)
        print()

    # Write summary
    summary = _compute_summary(all_results, tasks, orchestrators, config)
    summary_file = results_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    _print_summary_table(summary)
    print(f"\nFull results: {results_dir}")

    return all_results


def _compute_summary(
    all_results: list[dict],
    tasks: list[BenchmarkTask],
    orchestrators: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute aggregate summary across all runs."""
    summary = {
        "config": {
            "model": config["model"],
            "api_base": config["api_base"],
            "temperature": config.get("temperature", 0.0),
        },
        "num_tasks": len(tasks),
        "num_orchestrators": len(orchestrators),
        "orchestrators": {},
    }

    for orch in orchestrators:
        orch_results = [r for r in all_results if r["orchestrator"] == orch]
        if not orch_results:
            continue

        errors = [r for r in orch_results if r.get("error")]
        success = [r for r in orch_results if not r.get("error")]

        n = len(orch_results)
        summary["orchestrators"][orch] = {
            "total_runs": n,
            "errors": len(errors),
            "avg_files": sum(r["num_files"] for r in orch_results) / n if n else 0,
            "avg_tokens": sum(r["tokens_used"] for r in orch_results) / n if n else 0,
            "avg_duration_s": sum(r["duration_s"] for r in orch_results) / n if n else 0,
            "total_tokens": sum(r["tokens_used"] for r in orch_results),
            "total_duration_s": sum(r["duration_s"] for r in orch_results),
        }

    return summary


def _print_summary_table(summary: dict):
    """Print a formatted summary table."""
    orch_data = summary["orchestrators"]
    if not orch_data:
        print("No results.")
        return

    # Header
    header = f"{'Orchestrator':<18} {'Runs':>5} {'Errors':>6} {'Avg Files':>10} {'Avg Tokens':>11} {'Avg Time':>9} {'Total Tokens':>13}"
    print(header)
    print("-" * len(header))

    for orch, data in sorted(orch_data.items()):
        print(f"{orch:<18} {data['total_runs']:>5} {data['errors']:>6} "
              f"{data['avg_files']:>10.2f} {data['avg_tokens']:>11.0f} "
              f"{data['avg_duration_s']:>8.1f}s {data['total_tokens']:>13}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Matrioska Benchmark Runner")
    p.add_argument("--task", help="Run a specific task (by id)")
    p.add_argument("--orchestrator", choices=list(ORCHESTRATORS.keys()) + ["all"],
                   default="all", help="Which orchestrator to run")
    p.add_argument("--all", action="store_true", help="Run all tasks")
    p.add_argument("--runs", type=int, default=3, help="Number of repeated runs per task (default: 3)")
    p.add_argument("--model", default="llama-3.3-70b-versatile", help="Model name")
    p.add_argument("--api-key", help="API key (or use env var)")
    p.add_argument("--api-base", default="https://api.groq.com/openai/v1", help="API base URL")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per call")
    p.add_argument("--results-dir", default="./benchmarks/results", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve config
    api_key = args.api_key or os.environ.get("MATRIOSKA_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: No API key. Set --api-key or MATRIOSKA_API_KEY env var.")
        sys.exit(1)

    config = {
        "model": args.model,
        "api_key": api_key,
        "api_base": args.api_base,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    # Load tasks
    tasks_root = Path(__file__).parent.parent / "tasks"
    if args.task:
        tasks = [load_task(tasks_root / args.task)]
    elif args.all:
        tasks = load_all_tasks(tasks_root)
    else:
        tasks = load_all_tasks(tasks_root)

    # Resolve orchestrators
    if args.orchestrator == "all":
        orchestrators = [k for k in ORCHESTRATORS]
    else:
        orchestrators = [args.orchestrator]

    # Run
    results_dir = Path(args.results_dir).expanduser().resolve()
    run_benchmark(
        orchestrators=orchestrators,
        tasks=tasks,
        config=config,
        results_dir=results_dir,
        num_runs=args.runs,
    )


if __name__ == "__main__":
    main()

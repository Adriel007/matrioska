#!/usr/bin/env python3
"""POC: Compare orchestration strategies using same model/API."""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from benchmarks.tasks.loader import load_all_tasks
from benchmarks.harness.wrappers.direct import run_direct
from benchmarks.harness.wrappers.direct_plan import run_direct_plan
from benchmarks.harness.wrappers.matrioska_wrapper import run_matrioska
from benchmarks.harness.evaluator import evaluate_run

API_KEY = os.environ["MATRIOSKA_API_KEY"]
API_BASE = os.environ["MATRIOSKA_BASE_URL"]
MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 4096  # Higher for complex multi-file tasks

RESULTS_DIR = Path("./benchmarks/results/poc").resolve()
TASKS_ROOT = Path("./benchmarks/tasks").resolve()

ORCHESTRATORS = [
    ("direct", run_direct),
    ("direct_plan", run_direct_plan),
    ("matrioska", run_matrioska),
]


def main():
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True)

    # Run complex multi-file tasks only (where Matrioska should shine)
    all_tasks = load_all_tasks(TASKS_ROOT)
    complex_ids = {"fullstack_todo", "cli_pipeline", "api_auth", "web_dashboard"}
    tasks = [t for t in all_tasks if t.id in complex_ids]
    print(f"Benchmark: {len(tasks)} tasks x {len(ORCHESTRATORS)} orchestrators")
    print(f"Model: {MODEL}  |  Max tokens: {MAX_TOKENS}")
    print(f"API: {API_BASE}\n")

    config = {
        "model": MODEL, "api_key": API_KEY, "api_base": API_BASE,
        "temperature": 0.0, "max_tokens": MAX_TOKENS,
    }

    all_evaluations = []

    for task in tasks:
        print(f"{'─'*60}")
        print(f"Task: {task.id} ({task.category}/{task.complexity})")
        print(f"{'─'*60}")

        for orch_name, orch_func in ORCHESTRATORS:
            work_dir = RESULTS_DIR / "work" / f"{task.id}_{orch_name}_0"
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True)

            print(f"  [{orch_name:<14}] ", end="", flush=True)

            try:
                result = orch_func(
                    task_prompt=task.prompt,
                    work_dir=work_dir, **config,
                )
                print(f"{len(result.get('files', []))} files, "
                      f"{result.get('tokens_used', 0)} tok, "
                      f"{result.get('duration_s', 0):.1f}s", end="")
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                result = {"files": [], "tokens_used": 0, "duration_s": 0}

            eval_result = evaluate_run(
                run_result={**result, "orchestrator": orch_name,
                            "task_id": task.id, "run_id": 0},
                task_prompt=task.prompt,
                task_contract=task.contract,
                tests_dir=task.test_dir,
                work_dir=work_dir,
            )

            total = eval_result["num_tests_total"]
            passed = eval_result["num_tests_passed"]
            print(f"  →  test={passed}/{total} "
                  f"({eval_result['test_pass_rate']:.0%})  "
                  f"composite={eval_result['composite_score']:.3f}")

            if passed < total:
                output = eval_result.get("test_output", "")
                for line in output.split("\n"):
                    if "AssertionError" in line or "assert" in line:
                        print(f"       fail: {line.strip()[:140]}")
                        break

            all_evaluations.append(eval_result)

    # Aggregate by orchestrator and task
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    # Per-task table
    for task in tasks:
        print(f"\n  Task: {task.id} ({task.category})")
        header = f"  {'Orchestrator':<16} {'Test':>8} {'Syntax':>8} {'Contract':>10} {'Tokens':>8} {'Time':>8} {'Composite':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for ev in all_evaluations:
            if ev["task_id"] != task.id:
                continue
            contract_rate = ev.get("contract_fulfillment", {}).get("fulfillment_rate", 0)
            print(f"  {ev['orchestrator']:<16} {ev['test_pass_rate']:>7.0%} "
                  f"{ev['syntax_pass_rate']:>8.0%} {contract_rate:>9.0%} "
                  f"{ev['tokens_used']:>8} {ev['duration_s']:>7.1f}s "
                  f"{ev['composite_score']:>10.3f}")

    # Overall average
    print(f"\n  OVERALL AVERAGES:")
    for orch_name, _ in ORCHESTRATORS:
        evals = [e for e in all_evaluations if e["orchestrator"] == orch_name]
        if evals:
            n = len(evals)
            avg_test = sum(e["test_pass_rate"] for e in evals) / n
            avg_tok = sum(e["tokens_used"] for e in evals) / n
            avg_time = sum(e["duration_s"] for e in evals) / n
            avg_comp = sum(e["composite_score"] for e in evals) / n
            print(f"  {orch_name:<16} test={avg_test:.0%}  tokens={avg_tok:.0f}  "
                  f"time={avg_time:.1f}s  composite={avg_comp:.3f}")

    # Save
    (RESULTS_DIR / "evaluations.json").write_text(
        json.dumps(all_evaluations, indent=2, default=str))
    print(f"\nDetailed results: {RESULTS_DIR / 'evaluations.json'}")


if __name__ == "__main__":
    main()

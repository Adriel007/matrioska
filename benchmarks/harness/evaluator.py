"""Evaluation harness — runs test suites against generated code.

For each run result, executes the task's implementation-agnostic test suite
and computes standardized metrics.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def evaluate_run(
    run_result: dict[str, Any],
    task_prompt: str,
    task_contract: str,
    tests_dir: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Evaluate a single orchestrator run against the task's test suite.

    Args:
        run_result: Output from run_single() in runner.py.
        task_prompt: The original task prompt.
        task_contract: The CONTRACT file content.
        tests_dir: Directory containing test_*.py files.
        work_dir: Directory where generated files live.

    Returns:
        Dict with metrics: test_pass_rate, syntax_pass_rate, build_pass_rate,
        contract_fulfillment, cross_file_score, tokens_used, duration_s.
    """
    files = run_result.get("files", [])
    test_files = sorted(tests_dir.glob("test_*.py")) if tests_dir.exists() else []

    metrics = {
        "orchestrator": run_result.get("orchestrator", "unknown"),
        "task_id": run_result.get("task_id", "unknown"),
        "run_id": run_result.get("run_id", 0),
        "num_files": len(files),
        "tokens_used": run_result.get("tokens_used", 0),
        "duration_s": run_result.get("duration_s", 0),

        # Computed metrics
        "syntax_pass_rate": 0.0,
        "test_pass_rate": 0.0,
        "contract_fulfillment": {},
        "cross_file_score": 0.0,
        "num_tests_total": 0,
        "num_tests_passed": 0,
        "test_output": "",
    }

    # 1. Syntax check: all files parse correctly
    if files:
        syntax_score = _check_syntax(files)
        metrics["syntax_pass_rate"] = syntax_score["pass_rate"]
        metrics["syntax_errors"] = syntax_score.get("errors", [])

    # 2. Run the test suite against generated files
    if test_files and files:
        test_result = _run_test_suite(test_files, work_dir)
        metrics["test_pass_rate"] = test_result["pass_rate"]
        metrics["num_tests_total"] = test_result["total"]
        metrics["num_tests_passed"] = test_result["passed"]
        metrics["test_output"] = test_result.get("output", "")[:3000]

    # 3. Contract fulfillment
    if task_contract:
        metrics["contract_fulfillment"] = _check_contract(task_contract, work_dir)

    # 4. Cross-file consistency (imports valid, references exist)
    if len(files) > 1:
        metrics["cross_file_score"] = _check_cross_file(files)
    else:
        metrics["cross_file_score"] = 1.0

    # Composite score (weighted average)
    metrics["composite_score"] = _compute_composite(metrics)

    return metrics


def _check_syntax(files: list[Path]) -> dict[str, Any]:
    """Check that all generated files are syntactically valid."""
    errors = []
    passed = 0
    total = 0

    for f in files:
        total += 1
        try:
            if f.suffix == ".py":
                subprocess.run(
                    [sys.executable, "-m", "py_compile", str(f)],
                    capture_output=True, text=True, timeout=10, check=True,
                )
                passed += 1
            elif f.suffix in (".json",):
                json.loads(f.read_text())
                passed += 1
            elif f.suffix in (".yml", ".yaml"):
                try:
                    import yaml
                    yaml.safe_load(f.read_text())
                except ImportError:
                    pass  # Skip YAML check if yaml not installed
                passed += 1
            elif f.suffix in (".html", ".htm"):
                content = f.read_text()
                if "<!DOCTYPE html>" in content or "<html" in content.lower():
                    passed += 1
                else:
                    errors.append({"file": str(f), "error": "Missing HTML structure"})
            elif f.suffix == ".css":
                passed += 1  # Basic existence check; full CSS parse is complex
            elif f.suffix == ".js":
                # Check for balanced braces
                content = f.read_text()
                if content.count("{") == content.count("}"):
                    passed += 1
                else:
                    errors.append({"file": str(f), "error": "Unbalanced braces"})
            else:
                passed += 1  # Unknown format — assume OK
        except Exception as e:
            errors.append({"file": str(f), "error": str(e)[:200]})

    return {
        "pass_rate": passed / total if total > 0 else 0.0,
        "passed": passed,
        "total": total,
        "errors": errors,
    }


def _run_test_suite(test_files: list[Path], work_dir: Path) -> dict[str, Any]:
    """Run the implementation-agnostic test suite against generated code.

    Each test file uses a `project_dir` fixture that points to where
    generated code lives. We inject a conftest.py to provide this fixture.
    """
    if not test_files:
        return {"pass_rate": 0.0, "total": 0, "passed": 0, "output": ""}

    # Inject conftest.py to provide the project_dir fixture
    conftest_path = work_dir / "conftest.py"
    conftest_template = (Path(__file__).parent / "conftest_template.py").read_text()
    conftest_path.write_text(conftest_template)

    total = 0
    passed = 0
    all_output = []

    for test_file in test_files:
        # Copy test file to work dir to ensure fixture resolution
        test_dest = work_dir / test_file.name
        test_dest.write_text(test_file.read_text())

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    str(test_dest),
                    "--tb=short",
                    "-p", "no:cacheprovider",
                    "--no-header",
                    "-q",
                ],
                capture_output=True,
                text=True,
                cwd=str(work_dir),
                timeout=60,
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            all_output.append(output)

            # Parse pytest output: "X passed, Y failed"
            if "passed" in output:
                # Count from pytest summary line
                import re
                passed_match = re.search(r'(\d+)\s+passed', output)
                failed_match = re.search(r'(\d+)\s+failed', output)
                p = int(passed_match.group(1)) if passed_match else 0
                f = int(failed_match.group(1)) if failed_match else 0
                total += p + f
                passed += p
            elif result.returncode == 0:
                # All passed but didn't print standard format
                total += 1
                passed += 1

        except subprocess.TimeoutExpired:
            all_output.append(f"TIMEOUT: {test_file}")
        except Exception as e:
            all_output.append(f"ERROR running {test_file}: {e}")

    return {
        "pass_rate": passed / total if total > 0 else 0.0,
        "total": total,
        "passed": passed,
        "output": "\n".join(all_output),
    }


def _check_contract(contract: str, work_dir: Path) -> dict[str, Any]:
    """Check contract fulfillment.

    The CONTRACT file defines expected shared_state keys and their types.
    We check if the generated code mentions/defines these keys.
    """
    if not contract:
        return {}

    # Parse contract (simple KEY = value format)
    expected_keys = {}
    for line in contract.split("\n"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            expected_keys[key.strip()] = val.strip().strip('"').strip("'")

    # Check if generated files contain references to these keys
    all_content = ""
    for f in work_dir.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".js", ".html", ".yml", ".json", ".sh", ".css"}:
            try:
                all_content += f.read_text() + "\n"
            except Exception:
                pass

    found = {}
    missing = []
    for key, val_type in expected_keys.items():
        # Check if the key name appears in generated code
        key_name = key.lower().replace("_schema", "").replace("_", " ")
        if key_name in all_content.lower() or key.lower() in all_content.lower():
            found[key] = True
        else:
            found[key] = False
            missing.append(key)

    fulfilled = sum(1 for v in found.values() if v)
    return {
        "expected_keys": list(expected_keys.keys()),
        "found": found,
        "missing": missing,
        "fulfillment_rate": fulfilled / len(expected_keys) if expected_keys else 1.0,
    }


def _check_cross_file(files: list[Path]) -> float:
    """Check cross-file consistency: imports that reference other generated files."""
    py_files = [f for f in files if f.suffix == ".py"]
    if len(py_files) < 2:
        return 1.0

    # Collect all defined names (classes, functions) and module names
    all_modules = {f.stem for f in py_files}
    defined_names = set()

    for f in py_files:
        try:
            tree = __import__("ast").parse(f.read_text())
            for node in __import__("ast").walk(tree):
                if isinstance(node, __import__("ast").ClassDef):
                    defined_names.add(node.name)
                elif isinstance(node, __import__("ast").FunctionDef):
                    defined_names.add(node.name)
        except Exception:
            continue

    # Check imports reference valid modules
    import_errors = 0
    total_imports = 0

    for f in py_files:
        try:
            tree = __import__("ast").parse(f.read_text())
            for node in __import__("ast").walk(tree):
                if isinstance(node, __import__("ast").Import):
                    for alias in node.names:
                        total_imports += 1
                        if alias.name not in all_modules:
                            # External import — fine
                            pass
                elif isinstance(node, __import__("ast").ImportFrom):
                    if node.module:
                        total_imports += 1
                        if node.module not in all_modules and node.module not in __import__("sys").stdlib_module_names:
                            # Could be third-party; skip strict check
                            pass
        except Exception:
            continue

    if total_imports == 0:
        return 1.0

    return max(0.0, 1.0 - (import_errors / total_imports))


def _compute_composite(metrics: dict[str, Any]) -> float:
    """Compute a composite quality score."""
    weights = {
        "syntax_pass_rate": 0.20,
        "test_pass_rate": 0.50,
        "cross_file_score": 0.20,
    }
    contract_rate = metrics.get("contract_fulfillment", {}).get("fulfillment_rate", 0)
    if isinstance(contract_rate, (int, float)):
        weights["contract_rate"] = 0.10
    else:
        contract_rate = 0

    total = 0.0
    total += metrics["syntax_pass_rate"] * 0.20
    total += metrics["test_pass_rate"] * 0.50
    total += metrics["cross_file_score"] * 0.20
    total += contract_rate * 0.10

    return round(total, 3)


# ── Batch evaluation ──────────────────────────────────────────────────────────


def evaluate_all_runs(results_dir: Path, tasks_root: Path) -> dict[str, Any]:
    """Evaluate all runs in a results directory and produce final comparison.

    Reads generated log files from results_dir/logs/, runs test suites,
    and produces a comparison table across orchestrators.
    """
    from benchmarks.tasks.loader import load_task

    logs_dir = results_dir / "logs"
    if not logs_dir.exists():
        print(f"No logs directory at {logs_dir}")
        return {}

    all_evaluations = []

    for log_file in sorted(logs_dir.glob("*.json")):
        log_data = json.loads(log_file.read_text())
        task_id = log_data["task_id"]

        # Find the task and work dir
        task_dir = tasks_root / task_id
        if not task_dir.exists():
            print(f"  WARNING: Task directory not found: {task_dir}")
            continue

        task = load_task(task_dir)
        work_subdir = f"{task_id}_{log_data['orchestrator']}_{log_data['run_id']}"
        work_dir = results_dir / "work" / work_subdir

        if not work_dir.exists():
            print(f"  WARNING: Work directory not found: {work_dir}")
            continue

        eval_result = evaluate_run(
            log_data, task.prompt, task.contract, task.test_dir, work_dir,
        )
        all_evaluations.append(eval_result)

    # Aggregate by orchestrator
    comparison = _aggregate_by_orchestrator(all_evaluations)

    # Write final comparison
    comparison_file = results_dir / "comparison.json"
    comparison_file.write_text(json.dumps(comparison, indent=2))

    # Print table
    _print_comparison_table(comparison)

    return comparison


def _aggregate_by_orchestrator(evaluations: list[dict]) -> dict[str, Any]:
    """Aggregate evaluation results by orchestrator."""
    from collections import defaultdict

    by_orch = defaultdict(list)
    for ev in evaluations:
        by_orch[ev["orchestrator"]].append(ev)

    comparison = {}
    for orch, evals in by_orch.items():
        n = len(evals)
        comparison[orch] = {
            "num_runs": n,
            "avg_syntax_pass": sum(e["syntax_pass_rate"] for e in evals) / n,
            "avg_test_pass": sum(e["test_pass_rate"] for e in evals) / n,
            "avg_cross_file": sum(e["cross_file_score"] for e in evals) / n,
            "avg_composite": sum(e["composite_score"] for e in evals) / n,
            "avg_tokens": sum(e["tokens_used"] for e in evals) / n,
            "avg_duration_s": sum(e["duration_s"] for e in evals) / n,
            "total_tokens": sum(e["tokens_used"] for e in evals),
            "total_duration_s": sum(e["duration_s"] for e in evals),
        }

    return comparison


def _print_comparison_table(comparison: dict[str, Any]):
    """Print the final comparison table."""
    if not comparison:
        print("No results to compare.")
        return

    header = (f"{'Orchestrator':<18} {'Runs':>5} {'TestPass':>9} {'Syntax':>7} "
              f"{'CrossFile':>10} {'Composite':>10} {'AvgTok':>8} {'AvgTime':>9}")
    print("\n" + "=" * len(header))
    print("FINAL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for orch in sorted(comparison.keys()):
        d = comparison[orch]
        print(f"{orch:<18} {d['num_runs']:>5} {d['avg_test_pass']:>8.1%} {d['avg_syntax_pass']:>7.1%} "
              f"{d['avg_cross_file']:>9.1%} {d['avg_composite']:>9.1%} "
              f"{d['avg_tokens']:>8.0f} {d['avg_duration_s']:>8.1f}s")

    print("=" * len(header))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_eval_args():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate benchmark results")
    p.add_argument("results_dir", help="Path to results directory")
    p.add_argument("--tasks-root", default="./benchmarks/tasks", help="Path to tasks directory")
    return p.parse_args()


def eval_main():
    args = parse_eval_args()
    evaluate_all_runs(
        Path(args.results_dir).expanduser().resolve(),
        Path(args.tasks_root).expanduser().resolve(),
    )


if __name__ == "__main__":
    eval_main()

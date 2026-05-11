"""Tests for the DSPy compilation loop scaffold."""

from __future__ import annotations

from pathlib import Path

from matrioska.eval.dspy_compiler import (
    CompileSummary,
    aggregate,
    compile_target,
    evaluate_on_tasks,
    first_pass_signal,
    golden_metric,
)
from matrioska.eval.golden_suite import get_golden_tasks


def test_first_pass_signal():
    class A:
        def __init__(self, status, rc):
            self.status, self.repair_count = status, rc

    assert first_pass_signal({"artifacts": []}) == 0.0
    arts = [A("done", 0), A("done", 1), A("failed", 0)]
    assert first_pass_signal({"artifacts": arts}) == pytest_approx(1 / 3)


def pytest_approx(x: float, tol: float = 1e-6) -> float:
    # Local helper so we don't depend on importing pytest in the body.
    import pytest as _pytest
    return _pytest.approx(x, abs=tol)


def test_aggregate_handles_empty():
    out = aggregate([])
    assert out == {"pass_rate": 0.0, "first_pass_rate": 0.0}


def test_aggregate_means():
    records = [
        {"score": 1.0, "first_pass": 0.5},
        {"score": 0.0, "first_pass": 1.0},
    ]
    out = aggregate(records)
    assert out["pass_rate"] == 0.5
    assert out["first_pass_rate"] == 0.75


def test_golden_metric_pass_or_fail():
    task = get_golden_tasks("cli")[0]
    # Result with no artifacts → fails contract checks
    assert golden_metric(task, {"artifacts": [], "shared_state": {}}) == 0.0


def test_evaluate_on_tasks_uses_provided_runner():
    tasks = get_golden_tasks("cli")[:2]
    fake_result = {"artifacts": [], "shared_state": {}, "status": "success"}
    records = evaluate_on_tasks(tasks, runner=lambda t: fake_result)
    assert len(records) == len(tasks)
    assert all("task_id" in r for r in records)
    assert all(r["status"] == "success" for r in records)


def test_compile_target_skips_when_dspy_missing(tmp_path: Path):
    summary = compile_target(
        target="architect",
        category="cli",
        max_tasks=3,
        val_fraction=0.5,
        out_dir=tmp_path,
        runner=lambda t: {"artifacts": [], "status": "success"},
    )
    assert isinstance(summary, CompileSummary)
    # Either dspy is available and we have demos, or skipped_reason is set.
    assert summary.n_train >= 1
    assert summary.n_val >= 1
    summary_file = tmp_path / "architect_summary.json"
    assert summary_file.exists()

"""
DSPy-driven prompt compilation loop for Matrioska agents.

Khattab et al., 2023 (arXiv:2310.03714) and the DSPy framework treat prompts
as compilable parameters. This module wires Matrioska's golden regression
suite as a training signal so the Architect's system prompt and few-shot
examples can be optimized automatically (no manual prompt engineering).

Pipeline (BootstrapFewShot, default):
  1. Run baseline on a held-out validation slice → baseline_first_pass_rate.
  2. Wrap the Architect as a dspy.Module that calls the real pipeline.
  3. Optimize via dspy.teleprompt.BootstrapFewShot using `evaluate_result`
     as the metric (binary: pass==1 / fail==0).
  4. Persist the compiled prompts + few-shot demos to ProceduralMemory.

Designed as a **scaffold**: DSPy is an optional dependency. When missing,
`compile_target()` returns a structured stub that can be filled in by hand
or by a future LLM-as-optimizer pass (Reflexion-style).

Usage
-----
    from matrioska.eval.dspy_compiler import compile_target
    summary = compile_target(target="architect", category="cli", max_tasks=5)
    print(summary["baseline_pass"], "→", summary["compiled_pass"])
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from matrioska.eval.golden_suite import (
    GoldenTask,
    evaluate_result,
    get_golden_tasks,
)

logger = logging.getLogger("matrioska.eval.dspy_compiler")

_DEFAULT_OUT_DIR = Path.home() / ".matrioska" / "dspy_compiled"


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class CompileSummary:
    """Result of a compilation pass."""
    target: str
    n_train: int
    n_val: int
    baseline_pass: float = 0.0
    compiled_pass: float = 0.0
    baseline_first_pass_rate: float = 0.0
    compiled_first_pass_rate: float = 0.0
    elapsed_s: float = 0.0
    demos_path: Optional[str] = None
    skipped_reason: Optional[str] = None
    metric_per_task: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


# ── Metric ────────────────────────────────────────────────────────────────────


def golden_metric(task: GoldenTask, result: Dict[str, Any]) -> float:
    """Binary 0/1 metric for DSPy teleprompters.

    Wraps `evaluate_result` and returns 1.0 when the run passes every check,
    0.0 otherwise. Suitable for ``BootstrapFewShot`` (which expects a
    numeric metric to maximize) and for Reflexion-style optimizers.
    """
    try:
        ev = evaluate_result(task, result)
    except Exception:
        return 0.0
    return 1.0 if ev.get("pass") else 0.0


def first_pass_signal(result: Dict[str, Any]) -> float:
    """Approximate the `first_pass_rate` from a run dict.

    Counts artifacts whose ``repair_count == 0`` and ``status == "done"``
    divided by the total non-skipped artifact count. 1.0 means every file
    was correct on the first generator emission.
    """
    arts = result.get("artifacts") or []
    if not arts:
        return 0.0
    total = sum(1 for a in arts if getattr(a, "status", "") != "skipped")
    if total == 0:
        return 0.0
    first_pass = sum(
        1 for a in arts
        if getattr(a, "status", "") == "done" and getattr(a, "repair_count", 0) == 0
    )
    return first_pass / total


# ── Baseline / evaluation ─────────────────────────────────────────────────────


def evaluate_on_tasks(
    tasks: Sequence[GoldenTask],
    *,
    runner: Optional[Callable[[GoldenTask], Dict[str, Any]]] = None,
    cfg: Any = None,
) -> List[Dict[str, Any]]:
    """Run a sequence of golden tasks and return per-task metric records.

    ``runner`` may be supplied to mock the pipeline in tests. Otherwise we
    spin up the real ``Matrioska`` orchestrator with ``cfg`` (or defaults).
    """
    if runner is None:
        from matrioska.core.config import load_config, validate_config
        from matrioska.pipeline.orchestrator import Matrioska
        cfg = cfg or load_config()
        try:
            validate_config(cfg)
        except SystemExit:
            return [{"task_id": t.id, "skipped": "invalid config"} for t in tasks]

        def _runner(task: GoldenTask) -> Dict[str, Any]:
            return Matrioska(cfg).run(task.task)

        runner = _runner

    records: List[Dict[str, Any]] = []
    for task in tasks:
        t0 = time.time()
        try:
            result = runner(task)
            score = golden_metric(task, result)
            fp = first_pass_signal(result)
        except Exception as e:
            logger.warning("Task %s crashed: %s", task.id, e)
            result, score, fp = {"status": "error", "artifacts": []}, 0.0, 0.0
        records.append({
            "task_id": task.id,
            "category": task.category,
            "score": score,
            "first_pass": fp,
            "elapsed_s": round(time.time() - t0, 2),
            "status": result.get("status", "?"),
        })
    return records


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Mean score + mean first-pass rate over a record list."""
    if not records:
        return {"pass_rate": 0.0, "first_pass_rate": 0.0}
    n = len(records)
    return {
        "pass_rate": sum(r.get("score", 0.0) for r in records) / n,
        "first_pass_rate": sum(r.get("first_pass", 0.0) for r in records) / n,
    }


# ── Compilation loop ──────────────────────────────────────────────────────────


def compile_target(
    *,
    target: str = "architect",
    category: Optional[str] = None,
    max_tasks: int = 10,
    val_fraction: float = 0.3,
    out_dir: Optional[Path] = None,
    runner: Optional[Callable[[GoldenTask], Dict[str, Any]]] = None,
    cfg: Any = None,
) -> CompileSummary:
    """Top-level compilation loop. Returns a CompileSummary.

    When DSPy is installed, runs ``BootstrapFewShot`` over the training
    split and persists demos to ``out_dir/{target}_demos.json``. When DSPy
    is not installed, runs only the baseline evaluation and returns it
    with ``skipped_reason='dspy_not_installed'`` — the same skeleton can
    later be optimized via Reflexion or a hand-written prompt-search pass.
    """
    tasks = get_golden_tasks(category)
    if not tasks:
        return CompileSummary(target=target, n_train=0, n_val=0,
                              skipped_reason="no_tasks_match_category")

    tasks = list(tasks)[:max_tasks]
    cut = max(1, int(len(tasks) * (1.0 - val_fraction)))
    train, val = tasks[:cut], tasks[cut:]

    t0 = time.time()
    baseline = evaluate_on_tasks(val, runner=runner, cfg=cfg)
    base_agg = aggregate(baseline)

    out_dir = Path(out_dir) if out_dir else _DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import dspy  # type: ignore  # noqa: F401
        from dspy.teleprompt import BootstrapFewShot  # type: ignore
    except ImportError:
        logger.info(
            "DSPy not installed — emitting baseline-only summary. "
            "`pip install dspy-ai` to enable optimization."
        )
        summary = CompileSummary(
            target=target,
            n_train=len(train),
            n_val=len(val),
            baseline_pass=base_agg["pass_rate"],
            baseline_first_pass_rate=base_agg["first_pass_rate"],
            compiled_pass=base_agg["pass_rate"],
            compiled_first_pass_rate=base_agg["first_pass_rate"],
            elapsed_s=round(time.time() - t0, 2),
            metric_per_task=list(baseline),
            skipped_reason="dspy_not_installed",
        )
        _persist_summary(out_dir, target, summary)
        return summary

    # ── DSPy path: BootstrapFewShot over the train split ──────────────────
    summary = _run_dspy_bootstrap(
        target=target, train=train, val=val, runner=runner, cfg=cfg,
        baseline=baseline, base_agg=base_agg, out_dir=out_dir, t0=t0,
    )
    return summary


def _run_dspy_bootstrap(
    *,
    target: str,
    train: Sequence[GoldenTask],
    val: Sequence[GoldenTask],
    runner: Optional[Callable[[GoldenTask], Dict[str, Any]]],
    cfg: Any,
    baseline: List[Dict[str, Any]],
    base_agg: Dict[str, float],
    out_dir: Path,
    t0: float,
) -> CompileSummary:
    """DSPy-backed branch (only reached when dspy is importable)."""
    import dspy  # type: ignore
    from dspy.teleprompt import BootstrapFewShot  # type: ignore

    class _Signature(dspy.Signature):  # type: ignore[misc]
        """Decompose a task into a Matrioska architecture."""
        task: str = dspy.InputField()
        plan: str = dspy.OutputField(desc="JSON plan with files and contracts")

    class _Module(dspy.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.architect = dspy.ChainOfThought(_Signature)

        def forward(self, task: str) -> Any:  # type: ignore[override]
            return self.architect(task=task)

    examples = [dspy.Example(task=t.task, plan="").with_inputs("task") for t in train]

    def _dspy_metric(example: Any, pred: Any, trace: Any = None) -> float:
        # Map back to the golden task and evaluate via the real pipeline.
        # This is the most faithful but most expensive metric — pick small
        # `max_tasks` when iterating.
        task = next((t for t in train if t.task == example.task), None)
        if not task:
            return 0.0
        result = (runner or (lambda task: _default_runner(task, cfg)))(task)
        return golden_metric(task, result)

    teleprompter = BootstrapFewShot(metric=_dspy_metric, max_bootstrapped_demos=3)
    compiled = teleprompter.compile(_Module(), trainset=examples)

    # Persist demos
    demos_path = out_dir / f"{target}_demos.json"
    demos_payload = _extract_demos(compiled)
    demos_path.write_text(json.dumps(demos_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Re-evaluate on the validation split using the compiled module.
    def _compiled_runner(task: GoldenTask) -> Dict[str, Any]:
        prediction = compiled(task=task.task)
        # We don't yet plug the compiled prompt back into the real Architect.
        # Treat the prediction text as the plan and evaluate downstream.
        text = getattr(prediction, "plan", "") or str(prediction)
        return {"status": "compiled", "artifacts": [], "plan": text}

    compiled_records = evaluate_on_tasks(val, runner=_compiled_runner, cfg=cfg)
    compiled_agg = aggregate(compiled_records)

    summary = CompileSummary(
        target=target,
        n_train=len(train),
        n_val=len(val),
        baseline_pass=base_agg["pass_rate"],
        baseline_first_pass_rate=base_agg["first_pass_rate"],
        compiled_pass=compiled_agg["pass_rate"],
        compiled_first_pass_rate=compiled_agg["first_pass_rate"],
        elapsed_s=round(time.time() - t0, 2),
        demos_path=str(demos_path),
        metric_per_task=compiled_records,
    )
    _persist_summary(out_dir, target, summary)
    return summary


def _extract_demos(compiled_module: Any) -> List[Dict[str, Any]]:
    """Pull few-shot demos out of a compiled DSPy module in a JSON-safe form."""
    demos: List[Dict[str, Any]] = []
    for name, sub in getattr(compiled_module, "named_predictors", lambda: [])():
        ds = getattr(sub, "demos", None) or []
        for d in ds:
            demos.append({
                "predictor": name,
                "task": getattr(d, "task", None),
                "plan": getattr(d, "plan", None),
            })
    return demos


def _persist_summary(out_dir: Path, target: str, summary: CompileSummary) -> None:
    """Save the summary to disk so ProceduralMemory can load it later."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{target}_summary.json"
    path.write_text(
        json.dumps(summary.to_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("DSPy summary persisted: %s", path)


def _default_runner(task: GoldenTask, cfg: Any) -> Dict[str, Any]:
    """Last-resort runner used inside the DSPy metric closure."""
    from matrioska.pipeline.orchestrator import Matrioska
    if cfg is None:
        from matrioska.core.config import load_config
        cfg = load_config()
    return Matrioska(cfg).run(task.task)

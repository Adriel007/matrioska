"""
Benchmark parallelisation helper.

Runs a matrix of (task, orchestrator) pairs using ProcessPoolExecutor so that
independent combinations do not block each other.  Each worker process is
isolated — no shared mutable state.

Typical usage::

    from matrioska.eval.bench import run_benchmark

    results = run_benchmark(tasks, orchestrators, max_workers=4)

``tasks`` is a list of objects with at least a ``prompt`` (str) attribute.
``orchestrators`` is a list of ``(name: str, callable)`` pairs where each
callable has the signature ``(task_prompt, **config) -> dict``.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("matrioska.eval.bench")


@dataclass
class BenchResult:
    """Result for a single (task, orchestrator) combination."""
    task_id: str
    orchestrator: str
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_s: float = 0.0


def _worker(
    task_prompt: str,
    task_id: str,
    orch_name: str,
    orch_func: Callable[..., Dict[str, Any]],
    config: Dict[str, Any],
) -> BenchResult:
    """Executed inside a worker process — must be importable at module level."""
    t0 = time.perf_counter()
    try:
        result = orch_func(task_prompt=task_prompt, **config)
        return BenchResult(
            task_id=task_id,
            orchestrator=orch_name,
            result=result,
            duration_s=time.perf_counter() - t0,
        )
    except Exception as exc:
        return BenchResult(
            task_id=task_id,
            orchestrator=orch_name,
            error=f"{type(exc).__name__}: {exc}",
            duration_s=time.perf_counter() - t0,
        )


def run_benchmark(
    tasks: List[Any],
    orchestrators: List[Tuple[str, Callable[..., Dict[str, Any]]]],
    config: Optional[Dict[str, Any]] = None,
    max_workers: int = 2,
    *,
    verbose: bool = True,
) -> List[BenchResult]:
    """Run *tasks* x *orchestrators* in parallel via ProcessPoolExecutor.

    Args:
        tasks:         List of task objects.  Each must have ``.id`` (str) and
                       ``.prompt`` (str) attributes (or dict-like equivalents).
        orchestrators: List of ``(name, callable)`` pairs.
        config:        Keyword arguments forwarded to every orchestrator call.
        max_workers:   Number of parallel worker processes.
        verbose:       Print a one-line summary per completed job.

    Returns:
        List of :class:`BenchResult` in completion order.
    """
    config = config or {}
    total = len(tasks) * len(orchestrators)
    if verbose:
        print(f"Benchmark: {len(tasks)} task(s) x {len(orchestrators)} orchestrator(s) "
              f"= {total} jobs  (max_workers={max_workers})")

    futures: Dict[Any, Tuple[str, str]] = {}
    results: List[BenchResult] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for task in tasks:
            task_id = getattr(task, "id", None) or (task.get("id") if isinstance(task, dict) else str(task))
            task_prompt = getattr(task, "prompt", None) or (task.get("prompt") if isinstance(task, dict) else str(task))
            for orch_name, orch_func in orchestrators:
                fut = pool.submit(_worker, task_prompt, task_id, orch_name, orch_func, config)
                futures[fut] = (task_id, orch_name)

        for fut in as_completed(futures):
            task_id, orch_name = futures[fut]
            br: BenchResult = fut.result()
            results.append(br)
            if verbose:
                if br.error:
                    print(f"  [{task_id}] [{orch_name}] ERROR {br.error[:80]} ({br.duration_s:.1f}s)")
                else:
                    n_files = len(br.result.get("files", []))
                    tokens = br.result.get("tokens_used", 0)
                    print(f"  [{task_id}] [{orch_name}] OK files={n_files} tok={tokens} ({br.duration_s:.1f}s)")

    return results


def aggregate(results: List[BenchResult]) -> Dict[str, Any]:
    """Aggregate BenchResults into per-orchestrator averages."""
    from collections import defaultdict

    by_orch: Dict[str, List[BenchResult]] = defaultdict(list)
    for r in results:
        by_orch[r.orchestrator].append(r)

    summary: Dict[str, Any] = {}
    for orch_name, orch_results in by_orch.items():
        n = len(orch_results)
        errors = [r for r in orch_results if r.error]
        ok = [r for r in orch_results if not r.error]
        avg_dur = sum(r.duration_s for r in orch_results) / n if n else 0.0
        avg_tok = sum(r.result.get("tokens_used", 0) for r in ok) / max(len(ok), 1)
        summary[orch_name] = {
            "total": n,
            "errors": len(errors),
            "avg_duration_s": round(avg_dur, 2),
            "avg_tokens": round(avg_tok),
        }
    return summary

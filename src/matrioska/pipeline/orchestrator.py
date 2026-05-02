"""
Top-level Matrioska V3 orchestrator.

Wires together: Config → LLM → Memory → Agents → 3-Phase Pipeline.
Provides both a programmatic API and the foundation for the CLI.

Usage:
    from matrioska import Matrioska, Config, load_config

    cfg = load_config()
    m = Matrioska(cfg)
    result = m.run("Create a Python CLI todo app with SQLite")
    print(result["status"])
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.matrioska.core.config import Config, load_config, validate_config
from src.matrioska.core.events import (
    EventBus,
    JSONLRecorder,
    TokenTracker,
    MetricsCollector,
)
from src.matrioska.core.state import RunState, StateGraph, PipelineStatus
from src.matrioska.llm.client import LLMClient
from src.matrioska.memory.episodic import EpisodicMemory
from src.matrioska.memory.semantic import SemanticMemory
from src.matrioska.memory.procedural import ProceduralMemory
from src.matrioska.pipeline.phase1 import run_phase1
from src.matrioska.pipeline.phase2 import run_phase2
from src.matrioska.pipeline.phase3 import run_phase3

logger = logging.getLogger("matrioska.orchestrator")


class Matrioska:
    """The Matrioska V3 orchestrator — modular monolith with event-driven core.

    Instantiate with a Config (or let load_config() build one) and call run().
    The orchestrator manages the full lifecycle:

    1. Phase 1 — Architecture (with Tree-of-Thoughts voting)
    2. Phase 2 — Generation (DAG-layered, parallel, with Reflexion + Repair)
    3. Phase 3 — Verification (contract validation + sandbox execution)
    """

    def __init__(self, cfg: Optional[Config] = None, *, depth: int = 0):
        self.cfg = cfg or load_config()
        self.depth = depth
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # ── Event system ─────────────────────────────────────────────────
        self.bus = EventBus()
        self.recorder = JSONLRecorder(self.work_dir / "logs")
        self.tokens = TokenTracker()
        self.metrics = MetricsCollector()
        self.bus.on("llm_done", self.tokens)
        self.bus.on("*", self.recorder)

        # ── Persistence ──────────────────────────────────────────────────
        self.graph = StateGraph(self.work_dir)

        # ── Memory system ────────────────────────────────────────────────
        self.episodic = EpisodicMemory(self.work_dir)
        self.semantic = SemanticMemory(self.work_dir)
        self.procedural = ProceduralMemory(self.work_dir)
        self.procedural.ensure_project_memory()

        # ── LLM ──────────────────────────────────────────────────────────
        if not self.cfg.dry_run:
            self.llm = LLMClient(self.cfg, bus=self.bus)
        else:
            self.llm = None  # type: ignore

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, task: str) -> Dict[str, Any]:
        """Execute the full 3-phase pipeline on a task.

        Returns a dict with status, architecture, artifacts, shared_state,
        and metadata suitable for inspection and API consumption.
        """
        if self.cfg.dry_run:
            return self._dry_run(task)

        started = datetime.now(timezone.utc)
        t0 = time.time()
        self._banner(task)

        self.bus.emit_named(
            "run_start",
            task=task,
            depth=self.depth,
            provider=self.cfg.provider,
            model=self.cfg.model,
        )

        # ── Phase 1: Architecture ──────────────────────────────────────
        state = self.graph.new_run(task)

        if not run_phase1(
            state, self.cfg, self.llm, self.episodic, self.procedural, self.bus
        ):
            self._write_note(task, state, started, t0, "failed")
            return self._result(state, "failed")

        self.graph.save_checkpoint(label="after_architecture")

        if self.cfg.plan_only:
            logger.info("--plan_only: stopping after architecture")
            self._write_note(task, state, started, t0, "plan_only")
            return self._result(state, "plan_only")

        # ── Phase 2: Generation ────────────────────────────────────────
        gen_ok = run_phase2(state, self.cfg, self.llm, self.bus, self.depth)
        self.graph.save_checkpoint(label="after_generation")

        # ── Phase 3: Verification ──────────────────────────────────────
        verify_results = run_phase3(state, self.cfg, self.bus, self.metrics)
        self.graph.save_checkpoint(label="after_verification")

        # ── Finalize ───────────────────────────────────────────────────
        status = "success" if verify_results["overall_ok"] and gen_ok else "partial"
        self._write_note(task, state, started, t0, status)

        self.bus.emit_named(
            "run_end",
            status=status,
            **self.tokens.snapshot(),
            duration_s=round(time.time() - t0, 2),
            **self.metrics.snapshot(),
        )

        return self._result(state, status)

    def resume(self) -> Dict[str, Any]:
        """Resume from the latest checkpoint."""
        state = self.graph.load_latest()
        if state is None:
            raise RuntimeError(f"No checkpoints found in {self.work_dir}")

        logger.info("Resuming from checkpoint: %s", state.checkpoint_id)

        if state.status == PipelineStatus.PLANNING:
            raise RuntimeError(
                "Cannot resume from planning state — run was interrupted before architecture"
            )

        started = datetime.now(timezone.utc)
        t0 = time.time()

        if state.status == PipelineStatus.GENERATING:
            gen_ok = run_phase2(state, self.cfg, self.llm, self.bus, self.depth)
            verify_results = run_phase3(state, self.cfg, self.bus, self.metrics)
        elif state.status == PipelineStatus.VERIFYING:
            gen_ok = all(a.status == "done" for a in state.artifacts.values())
            verify_results = run_phase3(state, self.cfg, self.bus, self.metrics)
        else:
            return self._result(state, state.status.value)

        status = "success" if verify_results["overall_ok"] and gen_ok else "partial"
        self._write_note(state.task, state, started, t0, status)
        return self._result(state, status)

    def show(self) -> Dict[str, Any]:
        """Show current state from checkpoints."""
        state = self.graph.load_latest()
        if state is None:
            return {"status": "empty", "work_dir": str(self.work_dir)}

        return {
            "status": state.status.value,
            "project_name": state.project_name,
            "task": state.task,
            "files": (
                [
                    {
                        "name": a.name,
                        "extension": a.extension,
                        "status": a.status,
                        "chars": len(a.content),
                    }
                    for a in state.artifacts.values()
                ]
                if state.artifacts
                else (
                    [
                        {"name": f.name, "extension": f.extension, "order": f.order}
                        for f in state.architecture.files
                    ]
                    if state.architecture
                    else []
                )
            ),
            "shared_state_keys": list(state.shared_state.keys()),
            "checkpoints": self.graph.list_checkpoints(),
            "tokens": self.tokens.snapshot(),
            "work_dir": str(self.work_dir),
        }

    def clean(self) -> None:
        """Remove the work directory and all checkpoints."""
        import shutil

        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
            logger.info("Removed %s", self.work_dir)

    # ── Internals ───────────────────────────────────────────────────────

    def _banner(self, task: str) -> None:
        bar = "=" * 80
        print(f"\n{bar}\n  Matrioska V3\n  {task}\n{bar}")

    def _result(self, state: RunState, status: str) -> Dict[str, Any]:
        return {
            "status": status,
            "project_name": state.project_name,
            "architecture": state.architecture,
            "artifacts": list(state.artifacts.values()),
            "shared_state": state.shared_state,
            "metrics": self.metrics.snapshot(),
            "tokens": self.tokens.snapshot(),
            "checkpoint_id": state.checkpoint_id,
            "work_dir": str(self.work_dir),
        }

    def _write_note(
        self,
        task: str,
        state: RunState,
        started: datetime,
        t0: float,
        status: str,
    ) -> None:
        elapsed = time.time() - t0
        print(f"\n{'=' * 80}")
        print(f"SUMMARY (status={status})")
        print(f"  Project: {state.project_name}")
        print(f"  Files:   {len(state.artifacts)}")
        for a in state.artifacts.values():
            print(
                f"    {a.order}. {a.name}.{a.extension}  [{a.status}]  "
                f"{len(a.content)} chars"
            )
        ts = self.tokens.snapshot()
        print(
            f"  Tokens:  prompt={ts['prompt_tokens']} completion={ts['completion_tokens']}"
        )
        print(f"  Cost:    ~${ts.get('estimated_cost_usd', 0):.4f}")
        print(f"  Time:    {elapsed:.1f}s")
        print(f"  Work:    {self.work_dir}")
        print(f"{'=' * 80}")

        try:
            note_path = self.episodic.write_run_note(
                task=task,
                arch=state.architecture,
                artifacts=list(state.artifacts.values()),
                shared_state=state.shared_state,
                provider=self.cfg.provider,
                model=self.cfg.model,
                started_at=started,
                duration_s=elapsed,
                tokens_prompt=ts["prompt_tokens"],
                tokens_completion=ts["completion_tokens"],
                status=status,
            )
            print(f"Note: {note_path}")

            # Ingest into semantic memory
            if state.architecture:
                tags = [a.extension for a in state.artifacts.values()]
                self.semantic.ingest_run(task, tags, status)
        except Exception as e:
            logger.warning("Failed to write episodic note: %s", e)

    def _dry_run(self, task: str) -> Dict[str, Any]:
        print("[dry-run] Config:")
        for k, v in vars(self.cfg).items():
            show = "***" if "key" in k and v else v
            print(f"  {k}: {show}")
        print(f"[dry-run] Task: {task!r}")
        return {"status": "dry_run", "config": vars(self.cfg), "task": task}


# ── Convenience ─────────────────────────────────────────────────────────────


def run(task: str, **kwargs: Any) -> Dict[str, Any]:
    """Single-function entry point: config overrides as kwargs."""
    cfg = load_config(kwargs if kwargs else None)
    validate_config(cfg)
    return Matrioska(cfg).run(task)

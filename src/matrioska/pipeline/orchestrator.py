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

from matrioska.core.config import Config, ModelSpec, load_config, validate_config
from matrioska.core.events import (
    EventBus,
    JSONLRecorder,
    TokenTracker,
    MetricsCollector,
)
from matrioska.core.state import RunState, StateGraph, PipelineStatus
from matrioska.llm.client import LLMClient
from matrioska.memory.episodic import EpisodicMemory
from matrioska.memory.semantic import SemanticMemory
from matrioska.memory.procedural import ProceduralMemory
from matrioska.pipeline.phase1 import run_phase1
from matrioska.pipeline.phase2 import run_phase2
from matrioska.pipeline.phase3 import run_phase3
from matrioska.pipeline.preflight import run_preflight
from matrioska.pipeline.executor import install_missing_deps, write_artifacts_to_disk

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

        # ── Memory system ─────────────────────────────────────────────────
        # Each tier is optional: failures are logged but never crash the pipeline.
        self.episodic = EpisodicMemory(self.work_dir)
        try:
            self.semantic = SemanticMemory(self.work_dir)
        except Exception as e:
            logger.warning("SemanticMemory unavailable (chromadb/ONNX?): %s", e)
            self.semantic = None  # type: ignore
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

        # ── Connectivity check ──────────────────────────────────────────
        if not self.cfg.dry_run and self.cfg.provider in ("openai", "anthropic"):
            self._check_connectivity()

        # ── Pre-flight: read MATRIOSKA.md + scan existing code ─────────
        project_dir = Path(self.cfg.project_dir) if self.cfg.project_dir else None
        preflight = run_preflight(self.work_dir, project_dir=project_dir)
        if preflight.has_instructions:
            logger.info("Pre-flight: user instructions loaded from %s", preflight.instructions_source)
        if preflight.has_existing_code:
            logger.info("Pre-flight: %s", preflight.existing_summary)

        # ── Phase 1: Architecture ──────────────────────────────────────
        state = self.graph.new_run(task)
        # Inject pre-flight context into the Architect
        self._preflight = preflight

        architect_ctx = preflight.architect_context_block() or ""
        vault_hint = self._retrieve_vault_context(task) if self.cfg.enable_vault else ""
        if vault_hint:
            architect_ctx = (architect_ctx + "\n\n" + vault_hint).strip()

        if not run_phase1(
            state, self.cfg, self.llm, self.episodic, self.procedural, self.bus,
            preflight_context=architect_ctx or None,
        ):
            self._write_note(task, state, started, t0, "failed")
            return self._result(state, "failed")

        self.graph.save_checkpoint(label="after_architecture")

        if self.cfg.plan_only:
            logger.info("--plan_only: stopping after architecture")
            self._write_note(task, state, started, t0, "plan_only")
            return self._result(state, "plan_only")

        # ── Interactive plan review ─────────────────────────────────────
        if self.cfg.interactive and not self._confirm_plan(state):
            self._write_note(task, state, started, t0, "aborted")
            return self._result(state, "aborted")

        # ── Phase 2: Generation ────────────────────────────────────────
        gen_ok = run_phase2(state, self.cfg, self.llm, self.bus, self.depth)
        self.graph.save_checkpoint(label="after_generation")

        # ── Dep install + final disk write ─────────────────────────────
        write_artifacts_to_disk(state.artifacts, self.work_dir)
        if self.cfg.install_deps:
            installed = install_missing_deps(state.artifacts, self.work_dir)
            if installed:
                logger.info("Installed deps: %s", ", ".join(installed))

        # ── Phase 3: Verification ──────────────────────────────────────
        if self.cfg.quick:
            logger.info("--quick: skipping Phase 3 verification")
            verify_results = {"overall_ok": True, "skipped": "quick mode"}
        else:
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

    def _check_connectivity(self) -> None:
        """Probe the provider endpoint before running the pipeline.

        Catches invalid API keys, bad base URLs, and unavailable models early
        with actionable error messages before tokens are spent on architecture.
        """
        models = [
            self.cfg.effective_architect,
            self.cfg.effective_generator,
            self.cfg.effective_validator,
            self.cfg.effective_judge,
            self.cfg.effective_repairer,
        ]
        tested = set()

        for spec in models:
            key = (spec.provider, spec.base_url, spec.model)
            if key in tested:
                continue
            tested.add(key)

            try:
                probe_spec = ModelSpec(
                    provider=spec.provider,
                    model=spec.model,
                    base_url=spec.base_url,
                    api_key=spec.api_key,
                    max_tokens=5,
                    temperature=0.0,
                )
                self.llm.chat(
                    messages=[{"role": "user", "content": "OK"}],
                    model_spec=probe_spec,
                    system="Reply with exactly: OK",
                )
            except Exception as e:
                msg = str(e).lower()
                if "401" in msg or "unauthorized" in msg or "invalid api key" in msg:
                    raise RuntimeError(
                        f"API key rejected by {spec.provider} ({spec.base_url}). "
                        f"Check MATRIOSKA_API_KEY or --api-key.\n  Error: {e}"
                    ) from e
                if "404" in msg or "not found" in msg or "model" in msg:
                    raise RuntimeError(
                        f"Model '{spec.model}' not found on {spec.provider} ({spec.base_url}). "
                        f"Check MATRIOSKA_MODEL or --model.\n  Error: {e}"
                    ) from e
                if "connection" in msg or "refused" in msg or "timeout" in msg or "resolve" in msg:
                    raise RuntimeError(
                        f"Cannot reach {spec.provider} at {spec.base_url}. "
                        f"Check MATRIOSKA_BASE_URL, network, or firewall.\n  Error: {e}"
                    ) from e
                if "429" in msg or "rate limit" in msg:
                    logger.warning(
                        "Rate limited during connectivity check — proceeding anyway"
                    )
                    return
                raise RuntimeError(
                    f"Connectivity check failed for {spec.provider}/{spec.model}.\n"
                    f"  Error: {e}"
                ) from e

            logger.debug(
                "Connectivity OK: %s/%s", spec.provider, spec.model
            )

    def _confirm_plan(self, state: RunState) -> bool:
        """Show the proposed architecture and ask the user to approve."""
        arch = state.architecture
        if not arch:
            return True
        print(f"\n── Plan for: {arch.project_name} ──")
        for f in arch.files:
            badges = []
            if f.shared_state_reads:
                badges.append(f"reads={f.shared_state_reads}")
            if f.shared_state_writes:
                badges.append(f"writes={f.shared_state_writes}")
            if f.complex:
                badges.append("[COMPLEX]")
            extra = "  " + " ".join(badges) if badges else ""
            print(f"  {f.order}. {f.name}.{f.extension}{extra}")
        print()
        try:
            raw = input("Continue with this plan? [Y/n/q]: ").strip().lower()
        except EOFError:
            return True
        if raw in ("", "y", "yes"):
            return True
        if raw in ("q", "quit", "abort"):
            print("Aborted.")
            return False
        print("Cancelled (use --task-file to refine and rerun).")
        return False

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
            if state.architecture and self.semantic is not None:
                tags = [a.extension for a in state.artifacts.values()]
                self.semantic.ingest_run(task, tags, status)
        except Exception as e:
            logger.warning("Failed to write episodic note: %s", e)

        # Compile knowledge into global Obsidian vault (Karpathy LLM Wiki pattern)
        if self.cfg.enable_vault and state.architecture and status != "aborted":
            try:
                self._compile_into_vault(task, state, status)
            except Exception as e:
                logger.warning("Vault compile failed: %s", e)

    def _retrieve_vault_context(self, task: str, k: int = 5) -> str:
        """Retrieve concept/bug notes from the global vault relevant to the task.

        Returns a compact context block to inject into the Architect.
        Silently no-ops if the vault is empty or unavailable.
        """
        try:
            from matrioska.memory.vault import GlobalVault, default_vault_dir
            root = Path(self.cfg.vault_dir).expanduser() if self.cfg.vault_dir else default_vault_dir()
            if not root.exists():
                return ""
            vault = GlobalVault(root)
            results = vault.search(task, scope="global", k=k)
        except Exception as e:
            logger.debug("Vault retrieval skipped: %s", e)
            return ""

        if not results:
            return ""

        lines = ["RELEVANT VAULT KNOWLEDGE (Karpathy LLM Wiki, dedup'd):"]
        for r in results:
            lines.append(f"  • [{r['kind']}] {r['title']}: {r['snippet'][:160]}")
        return "\n".join(lines)

    def _compile_into_vault(self, task: str, state: "RunState", status: str) -> None:
        """Upsert per-run knowledge into the global Obsidian vault."""
        from matrioska.memory.vault import (
            GlobalVault, default_vault_dir,
            derive_tags, extract_lessons_and_bugs,
        )
        from pathlib import Path as _P

        root = _P(self.cfg.vault_dir).expanduser() if self.cfg.vault_dir else default_vault_dir()
        vault = GlobalVault(root)

        artifacts = list(state.artifacts.values())
        files_meta = [
            {
                "name": a.name,
                "extension": a.extension,
                "status": a.status,
                "repair_count": a.repair_count,
            }
            for a in artifacts
        ]
        tags = derive_tags(task, artifacts)
        lessons, bugs = extract_lessons_and_bugs(artifacts)

        touched = vault.compile_from_run(
            task=task,
            project_name=state.project_name or "untitled",
            files=files_meta,
            shared_state=state.shared_state,
            status=status,
            tags=tags,
            lessons=lessons,
            bugs=bugs,
            provider=self.cfg.provider,
            model=self.cfg.model,
        )
        if touched:
            logger.info("Vault: upserted %d note(s) under %s", len(touched), vault.root)

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

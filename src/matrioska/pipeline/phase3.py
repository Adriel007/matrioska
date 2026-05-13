"""
Phase 3: Verification & Integration — execute, validate contracts, sandbox repair.

Quality gates (in order):
  1. Contract validation   — every file fulfilled its shared_state_writes
  2. Cross-file check      — every shared_state_reads has a declared writer
  3. Sandbox execution     — run in Docker (or subprocess fallback)
  4. Sandbox repair loop   — feed stderr back to Repairer, re-run (max N times)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from matrioska.core.config import Config
from matrioska.core.contracts import ContractValidator
from matrioska.core.events import EventBus, MetricsCollector
from matrioska.core.state import RunState, PipelineStatus
from matrioska.llm.client import LLMClient

logger = logging.getLogger("matrioska.pipeline.phase3")


def run_phase3(
    state: RunState,
    cfg: Config,
    bus: Optional[EventBus] = None,
    metrics: Optional[MetricsCollector] = None,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Execute Phase 3: Verification & Integration."""
    logger.info("=== Phase 3: Verification ===")
    state.status = PipelineStatus.VERIFYING

    results: Dict[str, Any] = {
        "contract_validation": None,
        "cross_file_check": None,
        "sandbox_execution": None,
        "overall_ok": True,
    }

    # 1. Contract validation per-file
    contract_results = _validate_all_contracts(state)
    results["contract_validation"] = contract_results
    if not contract_results["all_ok"]:
        results["overall_ok"] = False
        logger.warning("Contract violations: %s", contract_results.get("violations", []))

    # 2. Cross-file consistency
    if state.contracts:
        cross = ContractValidator.validate_cross_file(
            state.contracts, state.shared_state
        )
        results["cross_file_check"] = {
            "ok": cross.ok,
            "violations": cross.violations,
            "warnings": cross.warnings,
        }
        if not cross.ok:
            results["overall_ok"] = False
            logger.warning("Cross-file issues: %s", cross.violations)

    # 3 + 4. Sandbox execution + repair loop
    if cfg.enable_sandbox:
        sandbox_result = _run_sandbox_with_repair(state, cfg, llm, bus)
        results["sandbox_execution"] = sandbox_result
        if not sandbox_result.get("ok", True):
            results["overall_ok"] = False

    # Metrics
    if metrics is not None:
        for contract_ok in contract_results.get("per_file", {}).values():
            metrics.record_contract(contract_ok)
        for artifact in state.artifacts.values():
            metrics.record_pass(artifact.repair_count == 0)
            if artifact.repair_count > 0:
                metrics.record_repair(artifact.status == "done")

    # Status transition
    if results["overall_ok"]:
        state.status = PipelineStatus.DONE
        state.log("Phase 3 OK: all checks passed")
        logger.info("Phase 3: all checks passed")
    else:
        state.status = PipelineStatus.FAILED
        state.log(f"Phase 3 FAILED: {results}")
        logger.warning("Phase 3: issues found")

    if bus:
        bus.emit_named("phase3_done", overall_ok=results["overall_ok"])

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_all_contracts(state: RunState) -> Dict[str, Any]:
    per_file: Dict[str, bool] = {}
    violations: List[str] = []
    for contract in state.contracts:
        if contract.file not in state.artifacts:
            violations.append(f"{contract.file}: declared but not generated")
            per_file[contract.file] = False
            continue
        result = ContractValidator.validate_writes(contract, state.shared_state)
        per_file[contract.file] = result.ok
        violations.extend(result.violations)
    return {
        "all_ok": all(per_file.values()) if per_file else True,
        "per_file": per_file,
        "violations": violations,
    }


def _run_sandbox_with_repair(
    state: RunState,
    cfg: Config,
    llm: Optional[LLMClient],
    bus: Optional[EventBus],
) -> Dict[str, Any]:
    """Run sandbox; if it fails, repair the offending file and retry.

    Repair iterations: cfg.sandbox_max_repairs (default 2).
    Each iteration:
      - parse stderr for the erroring filename
      - call Repairer with stderr as error signal
      - update artifact in state
      - re-run sandbox
    """
    from matrioska.tools.sandbox import SandboxExecutor, parse_erroring_file

    executor = SandboxExecutor(
        image=cfg.sandbox_image,
        timeout=cfg.sandbox_timeout,
        memory_limit="256m",
    )

    max_repairs = getattr(cfg, "sandbox_max_repairs", 2)
    known_files = {
        f"{a.name}.{a.extension}"
        for a in state.artifacts.values()
        if a.status == "done"
    }

    if bus:
        bus.emit_named("sandbox_started", image=cfg.sandbox_image)

    result = executor.run(state)
    result_dict = result.to_dict()

    if bus:
        bus.emit_named(
            "sandbox_result",
            ok=result.ok,
            mode=result.mode,
            exit_code=result.exit_code,
            duration_s=result.duration_s,
            entrypoint=result.entrypoint,
            stderr=result.stderr[:300],
        )

    if result.ok or not result.executed or not llm:
        return result_dict

    # ── Repair loop ───────────────────────────────────────────────────────
    for attempt in range(max_repairs):
        logger.info(
            "Sandbox failed (exit=%d) — repair attempt %d/%d",
            result.exit_code, attempt + 1, max_repairs,
        )

        erroring_file = parse_erroring_file(result.stderr, known_files)
        if erroring_file is None:
            erroring_file = result.entrypoint  # fallback: repair entrypoint

        if not erroring_file:
            logger.warning("Cannot identify erroring file — skipping sandbox repair")
            break

        artifact_key = erroring_file
        artifact = state.artifacts.get(artifact_key)
        if artifact is None:
            # Try without extension
            artifact_key = Path(erroring_file).stem
            artifact = state.artifacts.get(artifact_key)
        if artifact is None:
            logger.warning("Artifact %r not found in state — skipping repair", erroring_file)
            break

        if bus:
            bus.emit_named(
                "sandbox_repair_start",
                attempt=attempt + 1,
                file=erroring_file,
                stderr=result.stderr[:300],
            )

        repaired = _repair_artifact(
            artifact_key=artifact_key,
            state=state,
            stderr=result.stderr,
            cfg=cfg,
            llm=llm,
            bus=bus,
        )

        if not repaired:
            logger.warning("Sandbox repair produced no output for %s", erroring_file)
            break

        if bus:
            bus.emit_named("sandbox_repair_done", attempt=attempt + 1, file=erroring_file)

        # Re-run sandbox with updated artifacts
        result = executor.run(state)
        result_dict = result.to_dict()
        result_dict["sandbox_repair_attempts"] = attempt + 1

        if bus:
            bus.emit_named(
                "sandbox_result",
                ok=result.ok,
                mode=result.mode,
                exit_code=result.exit_code,
                attempt=attempt + 1,
                stderr=result.stderr[:300],
            )

        if result.ok:
            logger.info("Sandbox passed after %d repair(s)", attempt + 1)
            break

    return result_dict


def _repair_artifact(
    artifact_key: str,
    state: RunState,
    stderr: str,
    cfg: Config,
    llm: LLMClient,
    bus: Optional[EventBus],
) -> bool:
    """Call Repairer on a single artifact using sandbox stderr as the error signal.

    Updates the artifact content in state in-place.
    Returns True if repair produced non-empty content.
    """
    from matrioska.agents.repairer import RepairerAgent

    artifact = state.artifacts.get(artifact_key)
    if artifact is None:
        return False

    # Build a minimal FileSpec so the Repairer has context
    from matrioska.core.state import FileSpec
    spec = FileSpec(
        name=artifact.name,
        extension=artifact.extension,
        order=artifact.order,
        shared_state_reads=artifact.shared_state_reads or [],
        shared_state_writes=artifact.shared_state_writes or [],
    )

    repairer = RepairerAgent(cfg=cfg, llm=llm, bus=bus)
    errors = [f"[sandbox stderr]\n{stderr}"]

    try:
        repaired_content = repairer.repair(
            spec=spec,
            previous_content=artifact.content,
            errors=errors,
            shared_context=state.shared_state,
        )
    except Exception as e:
        logger.warning("Repairer raised during sandbox repair: %s", e)
        return False

    if not repaired_content:
        return False

    artifact.content = repaired_content
    artifact.repair_count += 1
    return True

"""
Phase 3: Verification & Integration — execute, validate contracts, cross-file check.

The final quality gate before delivering output:
  1. Contract validation (did every file fulfill its writes?)
  2. Cross-file consistency check (do all reads have a writer?)
  3. Sandbox execution (optional, for runnable projects)
  4. Replan on failure (returns to Phase 1 or 2)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.matrioska.core.config import Config
from src.matrioska.core.contracts import ContractValidator, ContractValidationResult
from src.matrioska.core.events import EventBus, MetricsCollector
from src.matrioska.core.state import RunState, PipelineStatus

logger = logging.getLogger("matrioska.pipeline.phase3")


def run_phase3(
    state: RunState,
    cfg: Config,
    bus: Optional[EventBus] = None,
    metrics: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    """Execute Phase 3: Verification & Integration.

    Returns a dict with verification results.
    """
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
        logger.warning(
            "Contract violations found: %s", contract_results.get("violations", [])
        )

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
            logger.warning("Cross-file consistency issues: %s", cross.violations)

    # 3. Sandbox execution (optional)
    if cfg.enable_sandbox:
        sandbox_result = _run_sandbox(state, cfg)
        results["sandbox_execution"] = sandbox_result
        if not sandbox_result.get("ok", True):
            results["overall_ok"] = False

    # Record metrics
    if metrics is not None:
        for contract_ok in contract_results.get("per_file", {}).values():
            metrics.record_contract(contract_ok)
        for artifact in state.artifacts.values():
            metrics.record_pass(artifact.repair_count == 0)
            if artifact.repair_count > 0:
                metrics.record_repair(artifact.status == "done")

    # Transition
    if results["overall_ok"]:
        state.status = PipelineStatus.DONE
        state.log("Phase 3 OK: all checks passed")
        logger.info("Phase 3: All checks passed")
    else:
        state.status = PipelineStatus.FAILED
        state.log(f"Phase 3 FAILED: {results}")
        logger.warning("Phase 3: Issues found — see verification results")

    if bus:
        bus.emit_named("phase3_done", overall_ok=results["overall_ok"])

    return results


def _validate_all_contracts(state: RunState) -> Dict[str, Any]:
    """Validate contracts for all generated files."""
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


def _run_sandbox(state: RunState, cfg: Config) -> Dict[str, Any]:
    """Run the generated project in a Docker sandbox.

    This is a scaffold — full Docker sandbox implementation is in tools/sandbox.py.
    For now, returns a placeholder.
    """
    logger.info("Sandbox execution requested (scaffold — see tools/sandbox.py)")

    # Detect runnable files
    runnable_extensions = {"py", "js", "sh", "ts"}
    runnable = [
        a
        for a in state.artifacts.values()
        if a.extension in runnable_extensions and a.status == "done"
    ]

    if not runnable:
        return {"ok": True, "executed": False, "reason": "No runnable files found"}

    # Scaffold: in full implementation, spin up Docker container and run
    try:
        from src.matrioska.tools.sandbox import SandboxExecutor

        executor = SandboxExecutor(
            image=cfg.sandbox_image,
            timeout=cfg.sandbox_timeout,
        )
        result = executor.execute(state)
        return {"ok": result["exit_code"] == 0, "executed": True, **result}
    except ImportError:
        logger.debug("SandboxExecutor not available (docker not installed?)")
        return {"ok": True, "executed": False, "reason": "Sandbox not available"}
    except Exception as e:
        logger.warning("Sandbox execution failed: %s", e)
        return {"ok": False, "executed": True, "error": str(e)}

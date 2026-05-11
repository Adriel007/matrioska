"""
Phase 2: Generation — DAG-layered parallel file generation.

For each layer of the topological DAG, files are generated in parallel
(ThreadPoolExecutor).  Each file goes through:
  Generate → Validate → (fail) Repair → (fail) Mark failed

With optional Reflexion loop: after generation, a Reflector reviews the
output and feeds insights back to future generators in the same run.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from matrioska.core.config import Config
from matrioska.core.events import EventBus
from matrioska.core.state import (
    Architecture,
    FileArtifact,
    FileSpec,
    RunState,
    PipelineStatus,
)
from matrioska.agents.generator import GeneratorAgent
from matrioska.agents.validator import ValidatorAgent
from matrioska.agents.repairer import RepairerAgent
from matrioska.agents.reflector import ReflectorAgent
from matrioska.agents.test_designer import TestDesignerAgent, run_tests_inline
from matrioska.llm.client import LLMClient
from matrioska.pipeline.graph import compute_layers
from matrioska.pipeline.executor import execute_artifact, write_artifacts_to_disk

logger = logging.getLogger("matrioska.pipeline.phase2")


def run_phase2(
    state: RunState,
    cfg: Config,
    llm: LLMClient,
    bus: Optional[EventBus] = None,
    depth: int = 0,
) -> bool:
    """Execute Phase 2: Generate all files layer by layer.

    Modifies state in place. Returns True if all non-complex files generated OK.
    """
    arch = state.architecture
    if arch is None:
        logger.error("No architecture in state — run Phase 1 first")
        return False

    logger.info("=== Phase 2: Generation ===")
    state.status = PipelineStatus.GENERATING

    layers = compute_layers(arch.files)
    logger.info("Computed %d DAG layer(s)", len(layers))

    generator = GeneratorAgent(cfg, llm, bus=bus)
    validator = ValidatorAgent(cfg, bus=bus)
    repairer = RepairerAgent(cfg, llm, bus=bus)
    reflector = ReflectorAgent(cfg, llm, bus=bus) if cfg.enable_reflexion else None
    test_designer = TestDesignerAgent(cfg, llm, bus=bus) if cfg.enable_test_design else None

    for layer_idx, layer in enumerate(layers, 1):
        logger.info(
            "Layer %d/%d: %s",
            layer_idx,
            len(layers),
            [f"{f.name}.{f.extension}" for f in layer],
        )

        rate_limit_hits = 0
        total_files = len(layer)

        if cfg.parallel and len(layer) > 1:
            with ThreadPoolExecutor(max_workers=min(len(layer), 6)) as pool:
                futures = {
                    pool.submit(
                        _generate_file,
                        f,
                        state,
                        cfg,
                        llm,
                        generator,
                        validator,
                        repairer,
                        reflector,
                        test_designer,
                        bus,
                        depth,
                    ): f
                    for f in layer
                }
                for fut in as_completed(futures):
                    artifact = fut.result()
                    _finalize_artifact(artifact, state, futures[fut])
                    if artifact.status == "failed" and "rate limit" in artifact.content.lower():
                        rate_limit_hits += 1
        else:
            for f in layer:
                artifact = _generate_file(
                    f,
                    state,
                    cfg,
                    llm,
                    generator,
                    validator,
                    repairer,
                    reflector,
                    test_designer,
                    bus,
                    depth,
                )
                _finalize_artifact(artifact, state, f)
                if artifact.status == "failed" and "rate limit" in artifact.content.lower():
                    rate_limit_hits += 1

        # Layer-level retry: if all files failed due to rate limits, retry layer
        if rate_limit_hits == total_files and total_files > 0 and layer_idx < len(layers):
            import time as _time
            backoff = min(30, 2 ** layer_idx)
            logger.warning(
                "Layer %d: all %d files hit rate limit, retrying layer after %ds…",
                layer_idx, total_files, backoff,
            )
            _time.sleep(backoff)
            for f in layer:
                artifact = _generate_file(
                    f, state, cfg, llm,
                    generator, validator, repairer, reflector, test_designer, bus, depth,
                )
                _finalize_artifact(artifact, state, f)

    state.status = PipelineStatus.VERIFYING
    success = all(
        a.status == "done"
        for a in state.artifacts.values()
        if not any(
            arch_file.complex
            for arch_file in arch.files
            if arch_file.filename == f"{a.name}.{a.extension}"
        )
    )

    if bus:
        bus.emit_named(
            "phase2_done", num_artifacts=len(state.artifacts), all_success=success
        )

    return success


def _finalize_artifact(artifact: FileArtifact, state: RunState, spec: FileSpec) -> None:
    """Add artifact to state and auto-populate shared_state for failed files.

    Non-blocking: even if a file fails, downstream files that depend on its
    shared_state_writes can still be generated using placeholder values.
    """
    state.add_artifact(artifact)

    if artifact.status == "failed":
        for k in spec.shared_state_writes:
            if k not in state.shared_state:
                state.shared_state[k] = f"__auto__{k}__"
                logger.debug(
                    "  auto-populated %s (failed artifact placeholder)", k
                )


def _generate_file(
    spec: FileSpec,
    state: RunState,
    cfg: Config,
    llm: LLMClient,
    generator: GeneratorAgent,
    validator: ValidatorAgent,
    repairer: RepairerAgent,
    reflector: Optional[ReflectorAgent],
    test_designer: Optional["TestDesignerAgent"],
    bus: Optional[EventBus],
    depth: int,
) -> FileArtifact:
    """Generate, validate, and optionally repair a single file."""
    t0 = time.time()

    # Gather shared context from reads
    context = {
        k: state.shared_state[k]
        for k in spec.shared_state_reads
        if k in state.shared_state
    }

    # Inject reflection insights if available
    if reflector and cfg.enable_reflexion:
        reflection_hint = reflector.render_for_prompt()
        if reflection_hint:
            spec.details += f"\n{reflection_hint}"

    # Surgical editing (incremental mode): inject existing file content so
    # the generator can edit-in-place instead of writing from scratch
    if cfg.incremental:
        existing_path = cfg.work_dir / f"{spec.name}.{spec.extension}"
        if existing_path.exists():
            existing_content = existing_path.read_text(encoding="utf-8", errors="ignore")
            context["_existing_content"] = existing_content
            spec.details += (
                f"\n\nEXISTING FILE CONTENT (modify only what the task requires — "
                f"do NOT rewrite unchanged parts):\n```{spec.extension}\n"
                f"{existing_content[:4000]}\n```"
            )
            logger.info("  Incremental: injecting existing %s (%d chars)", spec.filename, len(existing_content))

    # Handle complex files via nested Matrioska
    if spec.complex and depth < cfg.max_depth:
        logger.info(
            "  >> spawning sub-agent (depth=%d) for %s", depth + 1, spec.filename
        )
        return _nested_generate(spec, cfg, llm, bus, depth)

    # AlphaCodium + AgentCoder: design tests BEFORE generating (arXiv:2401.08500 + 2312.13010).
    # Tests derived from the contract (blind to implementation) give the Generator
    # a concrete interface target and the Repairer executable ground-truth signal.
    designer_tests = ""
    if test_designer is not None:
        designer_tests = test_designer.design_tests(spec, context)
        if designer_tests:
            spec.details += (
                f"\n\nCONTRACT TESTS (your implementation must pass these):\n"
                f"```python\n{designer_tests}\n```"
            )
            logger.info("  AlphaCodium: injected %d chars of tests for %s", len(designer_tests), spec.filename)

    content = ""
    updates: Dict[str, Any] = {}

    contract = next((c for c in state.contracts if c.file == spec.filename), None)
    last_errors: list[str] = []
    last_test_failures: list[str] = []

    for attempt in range(cfg.max_repairs + 1):
        if attempt == 0:
            content, updates = generator.generate(spec, context, state.artifacts)
        else:
            # Repair mode — pass validation errors + test failures (AlphaCodium signal)
            content = repairer.repair(
                spec, content, last_errors, context,
                test_failures=last_test_failures if last_test_failures else None,
            )

        if not content:
            last_errors = ["(empty response)"]
            continue

        # Validate
        result = validator.validate(
            content, spec.extension, contract, state.shared_state
        )

        if result.ok:
            # AlphaCodium: run designer tests inline as smoke check.
            last_test_failures = []
            if designer_tests and spec.extension == "py" and attempt < cfg.max_repairs:
                tests_ok, last_test_failures = _run_designer_tests(
                    spec.name, content, designer_tests
                )
                if not tests_ok:
                    logger.info(
                        "  AlphaCodium test smoke check: %d failure(s) in %s — repair",
                        len(last_test_failures), spec.filename,
                    )
                    last_errors = [f"Interface test failed: {f}" for f in last_test_failures]
                    continue

            # Auto-populate shared_state
            for k in spec.shared_state_writes:
                if k not in state.shared_state:
                    state.shared_state[k] = updates.get(k, f"__auto__{k}__")
            state.update_shared_state(updates)

            artifact = FileArtifact(
                name=spec.name,
                extension=spec.extension,
                order=spec.order,
                content=content,
                shared_state_updates=updates,
                status="done",
                repair_count=attempt,
                generation_tokens=0,
            )

            # Write to disk so downstream files can import this one
            _write_artifact_to_disk(artifact, cfg)

            # Real execution feedback (Claude Code-inspired)
            if cfg.execute_feedback and spec.extension == "py" and attempt < cfg.max_repairs:
                exec_result = execute_artifact(artifact, cfg.work_dir)
                if not exec_result.ok and not exec_result.skipped:
                    err = exec_result.error_for_repair
                    logger.info(
                        "  Execution error in %s — queuing repair:\n    %s",
                        spec.filename, err[:200],
                    )
                    last_errors = [f"Runtime error: {err}"]
                    last_test_failures = []
                    continue  # trigger repair with real stderr

            state.log(
                f"Generated {spec.filename} (attempt {attempt + 1}, "
                f"{len(content)} chars, {time.time() - t0:.1f}s)"
            )

            # Reflexion
            if reflector and cfg.enable_reflexion:
                reflection = reflector.reflect(artifact, spec, state.shared_state)
                if reflection.get("should_repair"):
                    logger.info(
                        "Reflector suggests repair for %s (score=%s)",
                        spec.filename,
                        reflection.get("score"),
                    )
                    # Even if reflector suggests repair, we keep the artifact
                    # but log the suggestion for future runs.

            if bus:
                bus.emit_named(
                    "file_generated",
                    file=spec.filename,
                    status="done",
                    attempts=attempt + 1,
                    chars=len(content),
                    elapsed_s=round(time.time() - t0, 2),
                )

            return artifact

        logger.warning(
            "  ! validation failed (%s, attempt %d): %s",
            spec.filename,
            attempt + 1,
            result.syntax_error or result.contract_violations,
        )
        # Capture actual errors for the repairer
        last_test_failures = []
        last_errors = []
        if result.syntax_error:
            last_errors.append(f"Syntax error: {result.syntax_error}")
        if result.contract_violations:
            if isinstance(result.contract_violations, list):
                last_errors.extend(str(v) for v in result.contract_violations)
            else:
                last_errors.append(f"Contract violation: {result.contract_violations}")

    # Exhausted repairs
    logger.error("  X %s FAILED after %d attempts", spec.filename, cfg.max_repairs + 1)
    return FileArtifact(
        name=spec.name,
        extension=spec.extension,
        order=spec.order,
        content=content
        or f"# Generation failed after {cfg.max_repairs + 1} attempts\n",
        status="failed",
        repair_count=cfg.max_repairs,
    )


def _run_designer_tests(
    module_name: str,
    code: str,
    tests_code: str,
) -> "tuple[bool, list[str]]":
    """Write code to a tempdir and run designer tests against it."""
    import tempfile
    import os

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, f"{module_name}.py")
            with open(module_path, "w", encoding="utf-8") as f:
                f.write(code)
            return run_tests_inline(tests_code, tmpdir)
    except Exception as e:
        return False, [f"test runner error: {e}"]


def _write_artifact_to_disk(artifact: FileArtifact, cfg: Config) -> None:
    """Write a done artifact to work_dir so subsequent files can import it."""
    try:
        path = cfg.work_dir / f"{artifact.name}.{artifact.extension}"
        path.write_text(artifact.content, encoding="utf-8")
    except Exception as e:
        logger.debug("Could not write %s to disk: %s", artifact.filename, e)


def _nested_generate(
    spec: FileSpec,
    cfg: Config,
    llm: LLMClient,
    bus: Optional[EventBus],
    depth: int,
) -> FileArtifact:
    """Handle complex files via recursive Matrioska invocation."""
    from matrioska.pipeline.orchestrator import Matrioska

    sub_cfg = Config(**{k: v for k, v in vars(cfg).items()})
    sub_cfg.work_dir = cfg.work_dir / "matrioska_artifacts" / f"{spec.name}.nested"
    sub_cfg.work_dir.mkdir(parents=True, exist_ok=True)
    sub_cfg.max_depth = cfg.max_depth
    sub_cfg.plan_only = False

    sub = Matrioska(sub_cfg, depth=depth + 1)
    task = f"{spec.content}\n\nRequirements:\n{spec.details}"
    result = sub.run(task)

    sub_arts: List[FileArtifact] = result.get("artifacts", []) or []

    if sub_arts and all(a.extension == spec.extension for a in sub_arts):
        if spec.extension in ("js", "css", "ts"):
            merged = "\n\n".join(
                f"/* ==== {a.name}.{a.extension} ==== */\n{a.content}" for a in sub_arts
            )
        else:
            merged = "\n\n".join(
                f"# ==== {a.name}.{a.extension} ====\n{a.content}" for a in sub_arts
            )
    else:
        lines = [f"# Nested artifacts for {spec.name}"]
        for a in sub_arts:
            lines.append(f"- {a.name}.{a.extension} ({len(a.content)} chars)")
        merged = "\n".join(lines)

    return FileArtifact(
        name=spec.name,
        extension=spec.extension,
        order=spec.order,
        content=merged,
        status="done" if result.get("status") in ("success", "plan_only") else "failed",
    )

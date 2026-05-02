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

from src.matrioska.core.config import Config
from src.matrioska.core.events import EventBus
from src.matrioska.core.state import (
    Architecture,
    FileArtifact,
    FileSpec,
    RunState,
    PipelineStatus,
)
from src.matrioska.agents.generator import GeneratorAgent
from src.matrioska.agents.validator import ValidatorAgent
from src.matrioska.agents.repairer import RepairerAgent
from src.matrioska.agents.reflector import ReflectorAgent
from src.matrioska.llm.client import LLMClient
from src.matrioska.pipeline.graph import compute_layers

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

    for layer_idx, layer in enumerate(layers, 1):
        logger.info(
            "Layer %d/%d: %s",
            layer_idx,
            len(layers),
            [f"{f.name}.{f.extension}" for f in layer],
        )

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
                        bus,
                        depth,
                    ): f
                    for f in layer
                }
                for fut in as_completed(futures):
                    artifact = fut.result()
                    state.add_artifact(artifact)
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
                    bus,
                    depth,
                )
                state.add_artifact(artifact)

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


def _generate_file(
    spec: FileSpec,
    state: RunState,
    cfg: Config,
    llm: LLMClient,
    generator: GeneratorAgent,
    validator: ValidatorAgent,
    repairer: RepairerAgent,
    reflector: Optional[ReflectorAgent],
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

    # Handle complex files via nested Matrioska
    if spec.complex and depth < cfg.max_depth:
        logger.info(
            "  >> spawning sub-agent (depth=%d) for %s", depth + 1, spec.filename
        )
        return _nested_generate(spec, cfg, llm, bus, depth)

    content = ""
    updates: Dict[str, Any] = {}

    for attempt in range(cfg.max_repairs + 1):
        if attempt == 0:
            content, updates = generator.generate(spec, context, state.artifacts)
        else:
            # Repair mode
            errors = [f"Attempt {attempt}: validation failed"]
            content = repairer.repair(spec, content, errors, context)

        if not content:
            continue

        # Validate
        contract = next((c for c in state.contracts if c.file == spec.filename), None)
        result = validator.validate(
            content, spec.extension, contract, state.shared_state
        )

        if result.ok:
            state.update_shared_state(updates)
            state.log(
                f"Generated {spec.filename} (attempt {attempt + 1}, "
                f"{len(content)} chars, {time.time() - t0:.1f}s)"
            )

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


def _nested_generate(
    spec: FileSpec,
    cfg: Config,
    llm: LLMClient,
    bus: Optional[EventBus],
    depth: int,
) -> FileArtifact:
    """Handle complex files via recursive Matrioska invocation."""
    from src.matrioska.pipeline.orchestrator import Matrioska

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

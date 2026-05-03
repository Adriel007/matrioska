"""
Phase 1: Architecture — decompose task into FileSpecs.

With Tree-of-Thoughts: N parallel Architect calls at high temperature
produce diverse plans, then a Judge evaluates and selects the best.
"""

from __future__ import annotations

import logging
from typing import Optional

from matrioska.core.config import Config
from matrioska.core.events import EventBus
from matrioska.core.state import Architecture, RunState, PipelineStatus
from matrioska.agents.architect import ArchitectAgent
from matrioska.llm.client import LLMClient
from matrioska.memory.episodic import EpisodicMemory
from matrioska.memory.procedural import ProceduralMemory

logger = logging.getLogger("matrioska.pipeline.phase1")


def run_phase1(
    state: RunState,
    cfg: Config,
    llm: LLMClient,
    episodic: Optional[EpisodicMemory] = None,
    procedural: Optional[ProceduralMemory] = None,
    bus: Optional[EventBus] = None,
) -> bool:
    """Execute Phase 1: Architecture planning.

    Modifies state in place. Returns True if a valid architecture was produced.
    """
    logger.info("=== Phase 1: Architecture ===")
    state.status = PipelineStatus.PLANNING

    architect = ArchitectAgent(
        cfg=cfg,
        llm=llm,
        episodic=episodic,
        procedural=procedural,
        bus=bus,
    )

    # Retrieve relevant past runs for context
    if episodic:
        past_notes = episodic.retrieve(state.task, k=cfg.retrieve_k)
        if past_notes:
            state.log(f"Retrieved {len(past_notes)} relevant past runs")

    arch = architect.plan(state.task)

    if arch is None:
        logger.error("Architecture phase failed — all candidates were invalid")
        state.status = PipelineStatus.FAILED
        state.log("Phase 1 FAILED: no valid architecture produced")
        return False

    state.architecture = arch
    state.project_name = arch.project_name
    state.contracts = arch.to_contracts()
    state.status = PipelineStatus.GENERATING

    state.log(f"Phase 1 OK: {len(arch.files)} files in {arch.project_name}")
    logger.info("Architecture: %s with %d files", arch.project_name, len(arch.files))
    for f in arch.files:
        logger.info(
            "  %d. %s.%s  reads=%s writes=%s%s",
            f.order,
            f.name,
            f.extension,
            f.shared_state_reads,
            f.shared_state_writes,
            " [COMPLEX]" if f.complex else "",
        )

    if bus:
        bus.emit_named(
            "phase1_done", project_name=arch.project_name, num_files=len(arch.files)
        )

    return True

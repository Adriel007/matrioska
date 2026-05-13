"""
Multi-planner: hierarchical task decomposition before code generation.

Flow:
  1. MetaPlanner identifies N self-contained sub-domains (2-4) from the task.
  2. For each sub-domain, ArchitectAgent generates a scoped FileSpec plan,
     aware of the shared_state interface agreed by previous sub-plans.
  3. The resulting FileSpec lists are merged into a single Architecture.

Compared to Tree-of-Thoughts (N plans for the same scope → Judge votes),
multi-planning is hierarchical: the task itself is split into sub-scopes
first, then each sub-scope gets a dedicated architect. This prevents the
"everything in one huge plan" anti-pattern for complex multi-component tasks.

Enabled via: cfg.enable_multi_plan = True  (or MATRIOSKA_ENABLE_MULTI_PLAN=true)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from matrioska.core.config import Config
from matrioska.core.events import EventBus
from matrioska.core.state import Architecture, FileSpec
from matrioska.core.text_utils import parse_json_safe
from matrioska.llm.client import LLMClient
from matrioska.memory.episodic import EpisodicMemory
from matrioska.memory.procedural import ProceduralMemory

logger = logging.getLogger("matrioska.agents.multi_planner")

# ── Meta-planner prompt & schema ──────────────────────────────────────────────

_META_PROMPT = """You are a software architecture decomposer.
Given a software task, identify 2-4 self-contained sub-domains that can each be
implemented independently and then integrated via a shared_state whiteboard.

Output strict JSON (no prose, no fences):
{
  "subproblems": [
    {
      "id": "short_snake_case_id",
      "label": "Human-readable label",
      "scope": "Specific description of files and functionality for this sub-domain"
    }
  ],
  "shared_interface": {
    "state_key": "What this shared_state key represents and its expected type/format"
  }
}

Rules:
- Use 2 sub-domains for simple 2-component tasks; 3-4 for complex/multi-layer tasks.
- shared_interface keys are the ONLY shared_state contract crossing sub-domains.
- Each scope must be concrete enough for an LLM architect to produce file names.
- Order sub-domains so each depends only on earlier ones (data layer before API layer, etc.).
- Output ONLY the JSON object."""

_META_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "subproblems": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id":    {"type": "string"},
                    "label": {"type": "string"},
                    "scope": {"type": "string"},
                },
                "required": ["id", "label", "scope"],
            },
        },
        "shared_interface": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["subproblems", "shared_interface"],
}


# ── MultiPlanner ──────────────────────────────────────────────────────────────


class MultiPlanner:
    """Hierarchical multi-planning: meta-decompose → N scoped architects → merge."""

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        bus: Optional[EventBus] = None,
        episodic: Optional[EpisodicMemory] = None,
        procedural: Optional[ProceduralMemory] = None,
        preflight_context: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.bus = bus
        self.episodic = episodic
        self.procedural = procedural
        self.preflight_context = preflight_context

    def plan(self, task: str) -> Optional[Architecture]:
        """Full multi-planning pipeline. Falls back to single architect on failure."""
        subproblems, shared_interface = self._identify_subproblems(task)

        if len(subproblems) < 2:
            logger.warning("Meta-planner returned < 2 sub-problems; falling back to single architect")
            return self._fallback(task)

        self._emit(
            "multi_plan_start",
            task=task[:120],
            n_subproblems=len(subproblems),
            subproblems=[sp["id"] for sp in subproblems],
        )
        logger.info(
            "Multi-planning: %d sub-domains — %s",
            len(subproblems),
            ", ".join(sp["id"] for sp in subproblems),
        )

        # Sequential planning: each sub-planner sees accumulated writes
        accumulated_writes: Dict[str, str] = dict(shared_interface)
        all_files: List[FileSpec] = []

        for sp in subproblems:
            scoped = self._build_scoped_task(task, sp, shared_interface, accumulated_writes)
            arch = self._plan_subproblem(scoped, sp["id"])

            if arch is None:
                logger.warning("Sub-plan '%s' failed — continuing without it", sp["id"])
                continue

            all_files.extend(arch.files)
            for f in arch.files:
                for key in f.shared_state_writes:
                    if key not in accumulated_writes:
                        accumulated_writes[key] = f"{f.name}.{f.extension}"

            self._emit(
                "multi_plan_subproblem_done",
                id=sp["id"],
                label=sp["label"],
                n_files=len(arch.files),
            )
            logger.info("  [%s] %d files", sp["id"], len(arch.files))

        if not all_files:
            logger.error("All sub-plans failed; falling back to single architect")
            return self._fallback(task)

        merged = self._merge(task, all_files)
        self._emit("multi_plan_done", n_files=len(merged.files), project=merged.project_name)
        return merged

    # ── Internal ─────────────────────────────────────────────────────────────

    def _identify_subproblems(
        self, task: str
    ) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        spec = self.cfg.effective_architect
        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": task}],
                model_spec=spec,
                system=_META_PROMPT,
                json_schema=_META_SCHEMA,
                json_mode=True,
            )
            data = parse_json_safe(resp.text)
            return data.get("subproblems", []), data.get("shared_interface", {})
        except Exception as e:
            logger.error("Meta-planner LLM call failed: %s", e)
            return [], {}

    def _build_scoped_task(
        self,
        task: str,
        subproblem: Dict[str, str],
        shared_interface: Dict[str, str],
        accumulated_writes: Dict[str, str],
    ) -> str:
        iface_lines = "\n".join(
            f"  {k}: {v}" for k, v in shared_interface.items()
        ) or "  (none)"

        available = {
            k: v for k, v in accumulated_writes.items()
            if k not in shared_interface
        }
        avail_lines = (
            "\n".join(f"  {k} → written by {v}" for k, v in available.items())
            if available else ""
        )

        parts = [
            f"FULL TASK: {task}",
            f"\nYOUR SCOPE — {subproblem['label']}:\n{subproblem['scope']}",
            f"\nSHARED INTERFACE (cross-domain shared_state keys):\n{iface_lines}",
        ]
        if avail_lines:
            parts.append(
                f"\nALREADY AVAILABLE (written by earlier sub-plans, use in shared_state_reads):\n{avail_lines}"
            )
        parts.append(
            "\nGenerate files ONLY for your scope. "
            "Use shared_state_reads for keys produced by other scopes. "
            "Declare your own outputs in shared_state_writes."
        )
        return "\n".join(parts)

    def _plan_subproblem(self, scoped_task: str, sp_id: str) -> Optional[Architecture]:
        from matrioska.agents.architect import ArchitectAgent
        agent = ArchitectAgent(
            cfg=self.cfg,
            llm=self.llm,
            episodic=self.episodic,
            procedural=self.procedural,
            bus=self.bus,
            preflight_context=self.preflight_context,
        )
        return agent._plan_single(scoped_task)

    def _merge(self, task: str, files: List[FileSpec]) -> Architecture:
        # Deduplicate by name.ext — last writer wins
        seen: Dict[str, FileSpec] = {}
        for f in files:
            seen[f"{f.name}.{f.extension}"] = f
        merged = list(seen.values())
        # Stable re-numbering preserving relative order
        for i, f in enumerate(merged, 1):
            f.order = i

        project_name = re.sub(
            r"[^a-z0-9_]", "",
            "_".join(task.lower().split()[:5]),
        ) or "multi_plan_project"

        return Architecture(project_name=project_name, files=merged)

    def _fallback(self, task: str) -> Optional[Architecture]:
        from matrioska.agents.architect import ArchitectAgent
        agent = ArchitectAgent(
            cfg=self.cfg, llm=self.llm,
            episodic=self.episodic, procedural=self.procedural,
            bus=self.bus, preflight_context=self.preflight_context,
        )
        return agent.plan(task)

    def _emit(self, name: str, **data: Any) -> None:
        if self.bus:
            self.bus.emit_named(name, **data)

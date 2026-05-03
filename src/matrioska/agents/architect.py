"""
Architect agent — decomposes a task into FileSpecs with shared_state contracts.

Uses Tree-of-Thoughts branching: N parallel calls with high temperature
produce diverse architectures, then a Judge picks the best one.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from matrioska.core.config import Config, ModelSpec
from matrioska.core.events import EventBus
from matrioska.core.state import Architecture, FileSpec
from matrioska.llm.client import LLMClient
from matrioska.memory.episodic import EpisodicMemory, RunNote
from matrioska.memory.procedural import ProceduralMemory

logger = logging.getLogger("matrioska.agents.architect")

ARCHITECT_SYSTEM_PROMPT = """You are Matrioska Architect. Decompose the user's task into a minimal set of independent FILES that coordinate via a shared_state whiteboard.

OUTPUT FORMAT (strict JSON, no prose outside JSON):
{
  "project_name": "snake_case_name",
  "files": [
    {
      "name": "filename_without_extension",
      "extension": "py|html|css|js|json|md|...",
      "order": 1,
      "complex": false,
      "shared_state_reads":  ["keys this file consumes"],
      "shared_state_writes": ["keys this file defines — concrete, reusable IDs, routes, schemas, class names"],
      "content": "DETAILED prompt for a coding AI to generate the COMPLETE file",
      "details": "concise functional + non-functional requirements"
    }
  ]
}

RULES:
  1. Root object MUST contain "project_name" (string) and "files" (array).
  2. "order" is a positive integer; lower = generated first.
  3. "shared_state_reads" MUST be a subset of keys written by earlier files.
  4. "shared_state_writes" should list concrete, reusable keys, NOT file contents.
  5. Mark "complex": true ONLY if a single file itself warrants nested decomposition.
  6. Return ONLY the JSON object. No markdown, no code fences, no commentary.

{project_memory_section}
{past_runs_section}

NOW PROCESS THIS REQUEST:"""

ARCHITECTURE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "project_name": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "extension": {"type": "string"},
                    "order": {"type": "integer", "minimum": 1},
                    "complex": {"type": "boolean"},
                    "shared_state_reads": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "shared_state_writes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "content": {"type": "string"},
                    "details": {"type": "string"},
                },
                "required": [
                    "name",
                    "extension",
                    "order",
                    "shared_state_reads",
                    "shared_state_writes",
                    "content",
                    "details",
                    "complex",
                ],
            },
        },
    },
    "required": ["project_name", "files"],
}


class ArchitectAgent:
    """Generates Architecture plans from a task description.

    With Tree-of-Thoughts enabled (cfg.enable_tot=True), generates
    N diverse candidates and uses a Judge to select the best.
    """

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        episodic: Optional[EpisodicMemory] = None,
        procedural: Optional[ProceduralMemory] = None,
        bus: Optional[EventBus] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.episodic = episodic
        self.procedural = procedural
        self.bus = bus

    def plan(self, task: str) -> Optional[Architecture]:
        """Generate an architecture plan (with optional ToT voting)."""
        if self.cfg.enable_tot and self.cfg.architect_candidates > 1:
            return self._plan_with_tot(task)
        return self._plan_single(task)

    def plan_candidates(self, task: str, n: int) -> List[Architecture]:
        """Generate N diverse architecture candidates in parallel."""
        import concurrent.futures

        candidates: List[Architecture] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
            futures = {
                pool.submit(self._plan_single, task, seed=i): i for i in range(n)
            }
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                if result is not None:
                    candidates.append(result)

        logger.info("Generated %d/%d valid architecture candidates", len(candidates), n)
        return candidates

    # ── Internal ─────────────────────────────────────────────────────────

    def _plan_single(self, task: str, seed: int = 0) -> Optional[Architecture]:
        """Generate a single architecture plan."""
        spec = self.cfg.effective_architect

        # Build system prompt with memory
        prompt = self._build_system_prompt(seed)

        self._emit("agent_call", agent="architect", model=spec.model)
        t0 = time.time()

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": task}],
                model_spec=spec,
                system=prompt,
                json_schema=ARCHITECTURE_JSON_SCHEMA,
                json_mode=True,
            )
        except Exception as e:
            logger.error("Architect call failed: %s", e)
            return None

        elapsed = time.time() - t0
        self._emit(
            "agent_done",
            agent="architect",
            model=spec.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            elapsed_s=round(elapsed, 2),
        )

        return self._parse_response(resp.text, task)

    def _build_system_prompt(self, seed: int = 0) -> str:
        """Build the system prompt with memory context."""
        memory_parts = []
        if self.procedural:
            pm = self.procedural.read_project_memory()
            if pm:
                memory_parts.append(f"PROJECT MEMORY (MATRIOSKA.md):\n{pm}")

        if self.episodic:
            notes = self.episodic.retrieve_with_embeddings("", k=self.cfg.retrieve_k)
            if notes:
                rendered = self.episodic.render_for_prompt(notes)
                memory_parts.append(rendered)

        memory_section = "\n\n".join(memory_parts)

        prompt = ARCHITECT_SYSTEM_PROMPT.replace(
            "{project_memory_section}", memory_section
        ).replace("{past_runs_section}", "")

        if seed > 0:
            prompt += f"\n\nDIVERSITY SEED: {seed}. Produce a different decomposition than you might otherwise default to."

        return prompt

    def _parse_response(self, raw: str, task: str) -> Optional[Architecture]:
        """Parse the LLM response into an Architecture."""
        if not raw.strip():
            return None

        try:
            data = json.loads(raw)
        except Exception:
            try:
                from json_repair import repair_json

                data = json.loads(repair_json(raw))
            except Exception as e:
                logger.error("Architect JSON unparseable: %s", e)
                return None

        if "instructs" in data and "files" in data.get("instructs", {}):
            files_data = data["instructs"]["files"]
            project_name = (
                data.get("project_name") or f"Project_{len(files_data)}_files"
            )
        elif "files" in data:
            files_data = data["files"]
            project_name = (
                data.get("project_name") or f"Project_{len(files_data)}_files"
            )
        else:
            logger.error("Architect payload missing 'files' key")
            return None

        files = []
        for i, fd in enumerate(files_data, 1):
            files.append(
                FileSpec(
                    name=str(fd.get("name", f"file_{i}")),
                    extension=str(fd.get("extension", "txt")).lstrip("."),
                    order=int(fd.get("order", i)),
                    shared_state_reads=list(fd.get("shared_state_reads") or []),
                    shared_state_writes=list(fd.get("shared_state_writes") or []),
                    content=str(fd.get("content", "")),
                    details=str(fd.get("details", "")),
                    complex=bool(fd.get("complex", False)),
                )
            )
        files.sort(key=lambda x: x.order)
        return Architecture(project_name=project_name, files=files)

    def _plan_with_tot(self, task: str) -> Optional[Architecture]:
        """Tree-of-Thoughts: N architects → Judge → best plan."""
        logger.info(
            "Tree-of-Thoughts: generating %d candidates", self.cfg.architect_candidates
        )
        candidates = self.plan_candidates(task, self.cfg.architect_candidates)

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Judge evaluates each candidate
        from matrioska.agents.judge import JudgeAgent

        judge = JudgeAgent(self.cfg, self.llm, bus=self.bus)

        best = judge.evaluate_architectures(task, candidates)
        if best is None:
            logger.warning(
                "Judge could not determine best architecture; using first candidate"
            )
            return candidates[0]

        return best

    def _emit(self, name: str, **data: Any) -> None:
        if self.bus:
            self.bus.emit_named(name, **data)

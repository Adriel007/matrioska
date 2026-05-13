"""
Reflector agent — Reflexion loop post-generation (§4.4.2).

After each file is generated and validated, the Reflector evaluates:
  1. Does the code fulfil its shared_state_writes contract?
  2. Are there edge cases not covered?
  3. Is the code idiomatic for the language/ecosystem?

The output is stored as episodic memory and feeds future Generators
in the same run via the shared_state reflection key.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional


logger = logging.getLogger("matrioska.agents.reflector")

REFLECTOR_SYSTEM_PROMPT = """You are Matrioska Reflector. Review a generated file and provide constructive feedback.

EVALUATE:
  1. CONTRACT FULFILLMENT: Does the file write all keys it promised?
  2. EDGE CASES: What inputs/scenarios might break this code?
  3. IDIOMATICITY: Is the code idiomatic for its language/ecosystem?
  4. SIMPLICITY: Could this be simpler without losing functionality?

OUTPUT FORMAT (JSON):
{
  "score": 1-10,
  "contract_ok": true/false,
  "edge_cases": ["edge case 1", "edge case 2"],
  "idiom_issues": ["issue 1"],
  "suggestions": ["concrete improvement 1"],
  "should_repair": true/false
}

Return ONLY the JSON object."""


class ReflectorAgent:
    """Meta-cognitive agent that reviews generator output.

    Implements the Reflexion loop (Shinn et al., 2023) — verbal reflection
    on output that feeds back into future generations.
    """

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        bus: Optional[EventBus] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.bus = bus
        self._reflections: Dict[str, List[Dict[str, Any]]] = {}

    def reflect(
        self,
        artifact: FileArtifact,
        spec: FileSpec,
        shared_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a reflection on a generated file.

        Returns a dict with score, issues, and suggestions.
        """
        if not self.cfg.enable_reflexion:
            return {"score": 7, "should_repair": False}

        ms = self.cfg.effective_judge  # Use judge-quality model for reflection
        prompt = (
            f"FILE: {spec.name}.{spec.extension}\n\n"
            f"REQUIREMENTS:\n{spec.details}\n\n"
            f"DECLARED WRITES: {spec.shared_state_writes}\n"
            f"DECLARED READS: {spec.shared_state_reads}\n\n"
            f"CODE:\n```{spec.extension}\n{artifact.content[:5000]}\n```\n\n"
            f"SHARED STATE KEYS WRITTEN: {list(artifact.shared_state_updates.keys())}"
        )

        self._emit("agent_call", agent="reflector", model=ms.model)
        t0 = time.time()

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model_spec=ms,
                system=REFLECTOR_SYSTEM_PROMPT,
                json_mode=True,
            )
        except Exception as e:
            logger.warning("Reflector call failed: %s", e)
            return {"score": 5, "should_repair": False}

        elapsed = time.time() - t0
        self._emit(
            "agent_done",
            agent="reflector",
            model=ms.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            elapsed_s=round(elapsed, 2),
        )

        try:
            result = parse_json_safe(resp.text)
        except ValueError:
            return {"score": 5, "should_repair": False}

        key = spec.filename
        if key not in self._reflections:
            self._reflections[key] = []
        self._reflections[key].append(result)

        return result

    def get_reflections_for(self, filename: str) -> List[Dict[str, Any]]:
        return self._reflections.get(filename, [])

    def render_for_prompt(self) -> str:
        """Render all reflections as guidance for future generators."""
        if not self._reflections:
            return ""

        lines = ["\n## Reflections from earlier files in this run"]
        for filename, refs in self._reflections.items():
            if refs:
                last = refs[-1]
                lines.append(
                    f"- {filename}: score={last.get('score', '?')}/10. "
                    f"Edge cases: {last.get('edge_cases', [])}"
                )
        return "\n".join(lines)

    def _emit(self, name: str, **data: Any) -> None:
        if self.bus:
            self.bus.emit_named(name, **data)
from matrioska.core.text_utils import parse_json_safe
from matrioska.llm.client import LLMClient

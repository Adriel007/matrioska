"""
Judge agent — evaluates competing architecture plans.

Used in Tree-of-Thoughts voting (§4.4.1) and multi-agent debate (§4.4.3).
Evaluates architectures on: completeness, minimality, consistency, feasibility.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional


logger = logging.getLogger("matrioska.agents.judge")

JUDGE_SYSTEM_PROMPT = """You are Matrioska Judge. Evaluate N architecture plans for a coding task and select the best one.

EVALUATION CRITERIA:
  1. COMPLETENESS: Does the plan cover ALL requirements in the task?
  2. MINIMALITY: Are there redundant files or over-decomposition?
  3. CONSISTENCY: Are shared_state contracts well-defined (reads ⊆ previous writes)?
  4. FEASIBILITY: Are dependencies realistic? Can each file be generated independently?

OUTPUT FORMAT (strict JSON):
{
  "evaluations": [
    {
      "index": 0,
      "completeness": 1-10,
      "minimality": 1-10,
      "consistency": 1-10,
      "feasibility": 1-10,
      "rationale": "1-2 sentences explaining the scores"
    }
  ],
  "winner_index": 0,
  "reasoning": "Why this plan is the best choice"
}

Return ONLY the JSON object. No markdown, no code fences."""


class JudgeAgent:
    """Evaluates and ranks architecture candidates."""

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        bus: Optional[EventBus] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.bus = bus

    def evaluate_architectures(
        self, task: str, candidates: List[Architecture]
    ) -> Optional[Architecture]:
        """Evaluate N architecture candidates and return the best one."""
        if len(candidates) <= 1:
            return candidates[0] if candidates else None

        spec = self.cfg.effective_judge
        summary = self._format_candidates(task, candidates)

        self._emit("agent_call", agent="judge", model=spec.model)
        t0 = time.time()

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": summary}],
                model_spec=spec,
                system=JUDGE_SYSTEM_PROMPT,
                json_mode=True,
            )
        except Exception as e:
            logger.error("Judge call failed: %s", e)
            return candidates[0]  # fallback to first

        elapsed = time.time() - t0
        self._emit(
            "agent_done",
            agent="judge",
            model=spec.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            elapsed_s=round(elapsed, 2),
        )

        winner_idx = self._parse_judgment(resp.text, len(candidates))
        if winner_idx is None or winner_idx >= len(candidates):
            return candidates[0]

        logger.info(
            "Judge selected architecture %d/%d", winner_idx + 1, len(candidates)
        )
        return candidates[winner_idx]

    def evaluate_files(
        self, task: str, files: List[str], outputs: List[str]
    ) -> Optional[int]:
        """For multi-agent debate: evaluate competing file versions.

        Returns the index of the best version, or None if tie.
        """
        spec = self.cfg.effective_judge
        candidates_text = "\n\n".join(
            f"### Candidate {i}\n```\n{output}\n```" for i, output in enumerate(outputs)
        )

        prompt = (
            f"Task: {task}\n\n"
            f"File to evaluate: {files[0] if files else 'unknown'}\n\n"
            f"{candidates_text}\n\n"
            f'Which candidate is best? Reply with JSON: {{"best_index": <int>, "reason": "..."}}'
        )

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model_spec=spec,
                json_mode=True,
            )
            data = parse_json_safe(resp.text)
            return int(data.get("best_index", 0))
        except Exception as e:
            logger.warning("File evaluation failed: %s", e)
            return None

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _format_candidates(task: str, candidates: List[Architecture]) -> str:
        parts = [f"TASK: {task}\n"]
        for i, arch in enumerate(candidates):
            parts.append(f"--- CANDIDATE {i} ---")
            parts.append(f"Project: {arch.project_name}")
            parts.append(f"Files ({len(arch.files)}):")
            for f in arch.files:
                parts.append(
                    f"  {f.order}. {f.filename} | reads={f.shared_state_reads} | "
                    f"writes={f.shared_state_writes} | complex={f.complex}"
                )
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _parse_judgment(raw: str, n_candidates: int) -> Optional[int]:
        try:
            data = parse_json_safe(raw)
            idx = int(data.get("winner_index", 0))
            if 0 <= idx < n_candidates:
                return idx
        except Exception:
            pass

        # Last-resort: find the first number that looks like an index
        import re

        nums = re.findall(r"\b(\d)\b", raw)
        for n_str in nums:
            idx = int(n_str)
            if 0 <= idx < n_candidates:
                return idx
        return None

    def _emit(self, name: str, **data: Any) -> None:
        if self.bus:
            self.bus.emit_named(name, **data)
from matrioska.core.text_utils import parse_json_safe
from matrioska.llm.client import LLMClient

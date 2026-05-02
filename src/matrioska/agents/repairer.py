"""
Repairer agent — fixes broken code using real error feedback.

Unlike the Generator which creates from scratch, the Repairer receives
the previous content + validation errors and produces a corrected version.
This implements the Repair loop from §4.4.2.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.matrioska.core.config import Config, ModelSpec
from src.matrioska.core.events import EventBus
from src.matrioska.core.state import FileSpec
from src.matrioska.llm.client import LLMClient

logger = logging.getLogger("matrioska.agents.repairer")

REPAIRER_SYSTEM_PROMPT = """You are Matrioska Repairer. Fix a broken file based on validation feedback.

You receive:
  1. The original generation prompt
  2. The previous file content that failed validation
  3. The validation error(s)

Your job: produce the COMPLETE CORRECTED file. Do not explain the fix.
Emit only the corrected content. If tool use is available, call `finish`.
Otherwise, output raw content + optional SHARED_STATE_UPDATE block.
"""


class RepairerAgent:
    """Fixes code that failed validation using error feedback.

    Run when the Generator's output fails syntax or contract checks.
    The repairer sees the error and produces a corrected version.
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

    def repair(
        self,
        spec: FileSpec,
        previous_content: str,
        errors: List[str],
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Produce a repaired version of the file content.

        Args:
            spec: The original file specification.
            previous_content: The content that failed validation.
            errors: List of validation error messages.
            shared_context: Current shared state for context.

        Returns:
            Corrected file content, or empty string on failure.
        """
        ms = self.cfg.effective_repairer

        error_text = "\n".join(f"- {e}" for e in errors)
        prompt = (
            f"FILE: {spec.name}.{spec.extension}\n\n"
            f"ORIGINAL PROMPT:\n{spec.content}\n\n"
            f"FAILED CONTENT:\n```{spec.extension}\n{previous_content[:6000]}\n```\n\n"
            f"VALIDATION ERRORS:\n{error_text}\n\n"
            f"Produce the COMPLETE CORRECTED file content. "
            f"Fix ALL errors listed above. Do not truncate. Do not explain."
        )

        if shared_context:
            ctx_str = "\n".join(
                f"- {k}: {v}"
                for k, v in shared_context.items()
                if not k.startswith("_")
            )
            prompt += f"\n\nSHARED STATE:\n{ctx_str}"

        self._emit("agent_call", agent="repairer", model=ms.model)
        t0 = time.time()

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model_spec=ms,
                system=REPAIRER_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error("Repairer call failed: %s", e)
            return ""

        elapsed = time.time() - t0
        self._emit(
            "agent_done",
            agent="repairer",
            model=ms.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            elapsed_s=round(elapsed, 2),
        )

        # Extract content from response
        content = resp.text
        if resp.tool_calls:
            for tc in resp.tool_calls:
                if tc.name == "finish":
                    content = str(tc.arguments.get("content", ""))
                    break

        logger.info("Repairer produced %d chars for %s", len(content), spec.filename)
        return content.strip()

    def _emit(self, name: str, **data: Any) -> None:
        if self.bus:
            self.bus.emit_named(name, **data)

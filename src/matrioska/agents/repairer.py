"""
Repairer agent — fixes broken code using real error feedback.

Two modes (SWE-agent ACI inspiration, arXiv:2405.15793):

  ACI mode (use_aci_repair=True, default):
    Asks the model for a TARGETED PATCH — a list of hunks (start_line, end_line,
    new_content) applied surgically.  Preserves invariants, avoids cross-file
    drift from full-file rewrites.  Errors with a precise line number trigger
    this mode automatically.

  Full-file mode (use_aci_repair=False or fallback):
    Classic "produce the corrected complete file" approach.  Used when the
    error doesn't map to specific lines or ACI patch fails to apply.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from matrioska.core.config import Config, ModelSpec
from matrioska.core.events import EventBus
from matrioska.core.state import FileSpec
from matrioska.core.text_utils import strip_fences, sanitize_output
from matrioska.llm.client import LLMClient
from matrioska.llm.circuit import is_small_model

logger = logging.getLogger("matrioska.agents.repairer")

REPAIRER_SYSTEM_FULL = """You are Matrioska Repairer. Fix a broken file based on validation feedback.

You receive:
  1. The original file specification (what this file must contain/do)
  2. The previous file content that failed validation
  3. The specific validation error(s) — parse errors, missing imports, etc.

CRITICAL RULES:
- Output ONLY the complete corrected file content, no explanations
- The output must be SYNTACTICALLY VALID for the target language
- Fix ALL errors listed in the validation feedback
- Do NOT truncate — produce the entire file
- Do NOT add new features or refactor — only fix the errors
- Preserve the file's original structure unless the error requires restructuring
- If the error is a Python syntax error at line N, verify lines N-2 to N+2 carefully
- If tool use is available, call `finish` with the corrected content
"""

# ── Solução B: sistema compacto para modelos pequenos ────────────────────────
#
# Modelos 8b têm duas limitações que causam 413:
#   1. Janela de contexto efetiva menor (o request HTTP cresce muito)
#   2. O system prompt longo confunde o modelo sobre o formato de output
#
# Solução: system prompt de 1 linha + código truncado ao redor do erro.
# Sem shared_state no prompt, sem DETAILS — só o essencial para corrigir.

REPAIRER_SYSTEM_COMPACT = (
    "Fix the broken code. Return ONLY the corrected complete file, no explanation, no fences."
)

# Máximo de chars do arquivo inteiro antes de truncar para repair compacto
_COMPACT_CONTENT_LIMIT = 2500
# Linhas de contexto ao redor do erro no modo compacto
_COMPACT_CONTEXT_LINES = 20

REPAIRER_SYSTEM_ACI = """You are Matrioska Repairer (ACI mode). Fix ONLY the broken parts of a file.

You receive the file content with line numbers and specific errors.
Your job: output a JSON array of targeted patch hunks to apply.

OUTPUT FORMAT — a JSON array, each hunk:
{
  "start_line": <1-indexed line where replacement begins>,
  "end_line":   <1-indexed line where replacement ends (inclusive)>,
  "new_content": "<replacement lines as a string, newline-separated>"
}

RULES:
- Output ONLY the JSON array, nothing else.
- Replace the minimum number of lines needed to fix each error.
- If an error requires adding lines: set end_line = start_line - 1 (insertion before start_line).
- Preserve indentation exactly.
- Each hunk must fix at least one error from the list.
- Hunks must be non-overlapping and sorted by start_line descending (apply bottom-up).
- If you cannot express the fix as targeted hunks, output an empty array [] to trigger full-file fallback.
"""


class RepairerAgent:
    """Fixes code that failed validation using error feedback.

    Tries ACI patch mode first (SWE-agent style), falls back to full-file repair.
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
        test_failures: Optional[List[str]] = None,
    ) -> str:
        """Produce a repaired version of the file content.

        Args:
            spec: The original file specification.
            previous_content: The content that failed validation.
            errors: Validation error messages (syntax/contract).
            shared_context: Current shared state for context.
            test_failures: Test failure messages from TestDesigner (AlphaCodium signal).

        Returns:
            Corrected file content, or empty string on failure.
        """
        all_errors = list(errors)
        if test_failures:
            all_errors += [f"[test] {f}" for f in test_failures]

        ms = self.cfg.effective_repairer

        # Solução B: modelos pequenos usam modo compacto para evitar 413
        if is_small_model(ms.model):
            logger.debug("Small model (%s) — compact repair mode", ms.model)
            return self._compact_repair(spec, previous_content, all_errors)

        if self.cfg.use_aci_repair and self._has_line_errors(all_errors):
            patched = self._aci_repair(spec, previous_content, all_errors, shared_context)
            if patched:
                return patched
            logger.info("ACI repair produced no hunks for %s — falling back to full-file", spec.filename)

        return self._full_repair(spec, previous_content, all_errors, shared_context)

    # ── Compact repair (Solução B) ────────────────────────────────────────

    def _compact_repair(
        self,
        spec: FileSpec,
        content: str,
        errors: List[str],
    ) -> str:
        """Minimal-context repair for small models.

        Sends only: error summary + code snippet around the error line.
        No shared_state, no DETAILS, no long system prompt.
        Prevents 413 Payload Too Large and model confusion.
        """
        ms = self.cfg.effective_repairer
        error_summary = "; ".join(e[:120] for e in errors[:3])
        snippet = _extract_error_context(content, errors, _COMPACT_CONTEXT_LINES)

        # If the snippet covers the whole file already, and it's still small, use it
        prompt = (
            f"Language: {spec.extension}\n"
            f"Error: {error_summary}\n\n"
            f"Code:\n{snippet}\n\n"
            f"Return ONLY the complete corrected file content."
        )

        self._emit("agent_call", agent="repairer_compact", model=ms.model)
        t0 = time.time()

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model_spec=ms,
                system=REPAIRER_SYSTEM_COMPACT,
            )
        except Exception as e:
            logger.error("Compact repair failed for %s: %s", spec.filename, e)
            return ""

        self._emit(
            "agent_done",
            agent="repairer_compact",
            elapsed_s=round(time.time() - t0, 2),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )

        result = sanitize_output(resp.text)
        logger.info("Compact repair: %d chars for %s", len(result), spec.filename)
        return result

    # ── ACI mode ──────────────────────────────────────────────────────────

    def _aci_repair(
        self,
        spec: FileSpec,
        content: str,
        errors: List[str],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Ask the model for targeted line-range patches, apply them."""
        ms = self.cfg.effective_repairer
        numbered = _number_lines(content)

        error_text = "\n".join(f"- {e}" for e in errors)
        prompt = (
            f"FILE: {spec.name}.{spec.extension}\n\n"
            f"CONTENT (line numbers for reference):\n{numbered}\n\n"
            f"ERRORS TO FIX:\n{error_text}\n\n"
            f"Output a JSON array of patch hunks. "
            f"Empty array [] if you need a full rewrite."
        )
        if context:
            ctx_str = "\n".join(f"- {k}: {v}" for k, v in context.items() if not k.startswith("_"))
            prompt += f"\n\nSHARED STATE:\n{ctx_str}"

        self._emit("agent_call", agent="repairer_aci", model=ms.model)
        t0 = time.time()

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model_spec=ms,
                system=REPAIRER_SYSTEM_ACI,
                json_mode=True,
            )
        except Exception as e:
            logger.error("ACI repairer call failed: %s", e)
            return ""

        self._emit("agent_done", agent="repairer_aci", elapsed_s=round(time.time() - t0, 2))

        hunks = _parse_hunks(resp.text)
        if not hunks:
            return ""

        try:
            patched = _apply_hunks(content, hunks)
            logger.info("ACI repair: applied %d hunk(s) to %s", len(hunks), spec.filename)
            return patched
        except Exception as e:
            logger.warning("ACI hunk application failed: %s", e)
            return ""

    # ── Full-file mode ────────────────────────────────────────────────────

    def _full_repair(
        self,
        spec: FileSpec,
        previous_content: str,
        errors: List[str],
        context: Optional[Dict[str, Any]],
    ) -> str:
        ms = self.cfg.effective_repairer

        error_text = "\n".join(f"- {e}" for e in errors)
        prompt = (
            f"FILE: {spec.name}.{spec.extension}\n\n"
            f"ORIGINAL SPECIFICATION:\n{spec.content}\n\n"
            f"DETAILS:\n{spec.details}\n\n"
            f"SHARED STATE READS: {', '.join(spec.shared_state_reads) if spec.shared_state_reads else '(none)'}\n"
            f"SHARED STATE WRITES: {', '.join(spec.shared_state_writes) if spec.shared_state_writes else '(none)'}\n\n"
            f"FAILED CONTENT:\n```{spec.extension}\n{previous_content[:6000]}\n```\n\n"
            f"VALIDATION ERRORS:\n{error_text}\n\n"
            f"Produce the COMPLETE CORRECTED file content. "
            f"Fix ALL errors listed above. Do not truncate. Do not explain."
        )

        if context:
            ctx_str = "\n".join(
                f"- {k}: {v}"
                for k, v in context.items()
                if not k.startswith("_")
            )
            prompt += f"\n\nSHARED STATE:\n{ctx_str}"

        self._emit("agent_call", agent="repairer", model=ms.model)
        t0 = time.time()

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model_spec=ms,
                system=REPAIRER_SYSTEM_FULL,
            )
        except Exception as e:
            logger.error("Repairer call failed: %s", e)
            return ""

        self._emit(
            "agent_done",
            agent="repairer",
            model=ms.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            elapsed_s=round(time.time() - t0, 2),
        )

        content = strip_fences(resp.text)
        if resp.tool_calls:
            for tc in resp.tool_calls:
                if tc.name == "finish":
                    content = strip_fences(str(tc.arguments.get("content", "")))
                    break

        logger.info("Full repair produced %d chars for %s", len(content), spec.filename)
        return content.strip()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _has_line_errors(errors: List[str]) -> bool:
        """True if any error references a specific line number."""
        line_pattern = re.compile(r'line\s+\d+|:\d+:', re.IGNORECASE)
        return any(line_pattern.search(e) for e in errors)

    def _emit(self, name: str, **data: Any) -> None:
        if self.bus:
            self.bus.emit_named(name, **data)


# ── Patch helpers ─────────────────────────────────────────────────────────────


def _extract_error_context(
    content: str, errors: List[str], context_lines: int = 20
) -> str:
    """Return the code snippet most relevant to the errors.

    If the errors reference specific line numbers, returns those lines ±
    context_lines.  Otherwise returns the whole content (truncated to
    _COMPACT_CONTENT_LIMIT chars) so the model has enough to work with.
    """
    lines = content.splitlines()

    error_line_nums: List[int] = []
    for err in errors:
        m = re.search(r'line\s+(\d+)', err, re.IGNORECASE)
        if m:
            error_line_nums.append(int(m.group(1)) - 1)  # 0-indexed

    if not error_line_nums:
        # No line reference — truncate whole content
        if len(content) <= _COMPACT_CONTENT_LIMIT:
            return _number_lines(content)
        return _number_lines(content[: _COMPACT_CONTENT_LIMIT]) + "\n... (truncated)"

    center = error_line_nums[0]
    start = max(0, center - context_lines)
    end = min(len(lines), center + context_lines + 1)

    width = len(str(end))
    snippet_lines = [f"{i + 1:{width}}| {lines[i]}" for i in range(start, end)]
    header = f"(lines {start + 1}–{end} of {len(lines)} total)\n"
    return header + "\n".join(snippet_lines)


def _number_lines(content: str) -> str:
    lines = content.splitlines()
    width = len(str(len(lines)))
    return "\n".join(f"{i + 1:{width}}| {line}" for i, line in enumerate(lines))


def _parse_hunks(text: str) -> List[Dict[str, Any]]:
    """Parse JSON hunk array from model response."""
    text = text.strip()
    # Strip any accidental fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        hunks = json.loads(text)
        if not isinstance(hunks, list):
            return []
        valid = []
        for h in hunks:
            if (
                isinstance(h, dict)
                and isinstance(h.get("start_line"), int)
                and isinstance(h.get("end_line"), int)
                and isinstance(h.get("new_content"), str)
            ):
                valid.append(h)
        return valid
    except Exception:
        try:
            from json_repair import repair_json
            hunks = json.loads(repair_json(text))
            return hunks if isinstance(hunks, list) else []
        except Exception:
            return []


def _apply_hunks(content: str, hunks: List[Dict[str, Any]]) -> str:
    """Apply patch hunks bottom-up so line numbers stay valid."""
    lines = content.splitlines(keepends=True)
    # Sort descending by start_line
    for hunk in sorted(hunks, key=lambda h: h["start_line"], reverse=True):
        start = hunk["start_line"] - 1   # convert to 0-indexed
        end = hunk["end_line"]            # exclusive in slice
        new_text = hunk["new_content"]
        new_lines = [l if l.endswith("\n") else l + "\n" for l in new_text.splitlines()]
        lines[start:end] = new_lines
    return "".join(lines)

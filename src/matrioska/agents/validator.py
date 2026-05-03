"""
Validator agent — validates generated file syntax and contracts.

Uses cheap/fast models (Haiku 4.5 / GPT-4o-mini) plus deterministic
parsers (AST, HTML, JSON).  This is the first line of defense in the
Generate → Validate → Repair cycle.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from matrioska.core.config import Config, ModelSpec
from matrioska.core.contracts import (
    FileContract,
    ContractValidator,
    ContractValidationResult,
)
from matrioska.core.events import EventBus
from matrioska.llm.client import LLMClient

logger = logging.getLogger("matrioska.agents.validator")


@dataclass
class ValidationResult:
    ok: bool
    syntax_ok: bool = True
    syntax_error: str = ""
    contract_ok: bool = True
    contract_violations: List[str] = field(default_factory=list)
    semantic_warnings: List[str] = field(default_factory=list)
    llm_feedback: str = ""


_TRUNCATION_MARKERS = (
    "... (truncated)",
    "# ... rest of code",
    "// ... rest of code",
    "/* ... */",
    "# TODO: continue",
    "// TODO: continue",
    "<!-- TODO: continue -->",
)


class _SafeHTMLParser(HTMLParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


class ValidatorAgent:
    """Validates generated code — syntax + contracts + optional LLM review."""

    def __init__(
        self,
        cfg: Config,
        llm: Optional[LLMClient] = None,
        bus: Optional[EventBus] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.bus = bus

    def validate(
        self,
        content: str,
        extension: str,
        contract: Optional[FileContract] = None,
        shared_state: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Run full validation: syntax + contract + semantic heuristics.

        Returns a ValidationResult with all issues found.
        """
        result = ValidationResult(ok=True)

        # 1. Deterministic syntax check
        syntax = self._check_syntax(content, extension)
        if not syntax.ok:
            result.syntax_ok = False
            result.syntax_error = syntax.syntax_error
            result.ok = False

        # 2. Content quality heuristics
        for m in _TRUNCATION_MARKERS:
            if m in content:
                result.semantic_warnings.append(f"Truncation marker found: {m!r}")
                result.ok = False

        if not content or not content.strip():
            result.syntax_ok = False
            result.syntax_error = "empty content"
            result.ok = False

        # 3. Contract validation (warnings only — full validation in Phase 3)
        if contract is not None and shared_state is not None:
            cr = ContractValidator.validate_writes(contract, shared_state)
            if not cr.ok:
                result.contract_ok = False
                result.contract_violations = cr.violations
                result.semantic_warnings.extend(cr.violations)
            result.semantic_warnings.extend(cr.warnings)

        # 4. LLM semantic review (optional, for complex files)
        if self.llm is not None and not result.ok:
            result.llm_feedback = self._llm_review(content, extension, result)

        self._emit(
            "validation_done",
            file=contract.file if contract else "unknown",
            ok=result.ok,
            syntax_ok=result.syntax_ok,
            contract_ok=result.contract_ok,
        )

        return result

    def _emit(self, name: str, **data: Any) -> None:
        if self.bus:
            self.bus.emit_named(name, **data)

    # ── Syntactic validation ─────────────────────────────────────────────

    @staticmethod
    def _check_syntax(content: str, extension: str) -> ValidationResult:
        ext = extension.lower().lstrip(".")
        try:
            if ext == "py":
                ast.parse(content)
            elif ext == "json":
                try:
                    json.loads(content)
                except Exception:
                    from json_repair import repair_json

                    json.loads(repair_json(content))
            elif ext in ("yml", "yaml"):
                try:
                    import yaml

                    yaml.safe_load(content)
                except ImportError:
                    pass
            elif ext in ("html", "htm"):
                _SafeHTMLParser().feed(content)
            elif ext in ("md", "txt", "rst"):
                pass  # No syntax check for prose
            elif ext in ("js", "mjs", "cjs"):
                _check_js_syntax(content)
            elif ext in ("ts", "tsx", "jsx"):
                pass  # Skip TSX/JSX without compiler
        except Exception as e:
            return ValidationResult(ok=False, syntax_ok=False, syntax_error=str(e))
        return ValidationResult(ok=True, syntax_ok=True)

    # ── LLM review ───────────────────────────────────────────────────────

    def _llm_review(
        self, content: str, extension: str, current: ValidationResult
    ) -> str:
        """Ask a cheap LLM to review the code and suggest fixes."""
        if self.llm is None:
            return ""
        spec = self.cfg.effective_validator
        try:
            resp = self.llm.chat(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Review this {extension} code for correctness.\n\n"
                            f"Known issues:\n- Syntax: {current.syntax_error}\n"
                            f"- Warnings: {current.semantic_warnings}\n\n"
                            f"Code:\n```{extension}\n{content[:4000]}\n```\n\n"
                            f"List any additional issues (1-2 sentences max)."
                        ),
                    }
                ],
                model_spec=spec,
            )
            return resp.text.strip()
        except Exception as e:
            logger.debug("LLM review failed: %s", e)
            return ""


def _check_js_syntax(code: str) -> None:
    """Basic JavaScript syntax checks without Node.js."""
    # Check balanced braces
    stack: List[str] = []
    pairs = {"{": "}", "(": ")", "[": "]"}
    for i, ch in enumerate(code):
        if ch in pairs:
            stack.append(ch)
        elif ch in pairs.values():
            if not stack:
                raise ValueError(f"Unexpected '{ch}' at position {i}")
            expected = pairs[stack.pop()]
            if ch != expected:
                raise ValueError(
                    f"Mismatched '{stack[-1] if stack else '?'}' with '{ch}' at {i}"
                )
    if stack:
        raise ValueError(f"Unclosed {stack[-1]}")

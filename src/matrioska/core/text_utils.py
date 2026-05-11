"""Shared text utilities for stripping markdown fences and cleaning LLM output."""
from __future__ import annotations

import re

# ── Solução C: sanitizer robusto ─────────────────────────────────────────────
#
# Modelos pequenos cometem três padrões de erro no output:
#   1. Envolvem o código com ```python ... ``` (deveria ser código puro)
#   2. Fecham uma fence no meio do output e depois chamam finish() como texto
#   3. Escrevem `finish("filename")` ou finish(...) literal após o código
#
# O sanitize_output cobre todos os casos. strip_fences é mantido como alias.

_LANG_FENCE_RE = re.compile(
    r'^```[a-zA-Z0-9_.+\-]*\n(.*?)(?:\n```[^\n]*)?$',
    re.DOTALL,
)

# Trailing ``` possibly followed by text like "finish(...)" or whitespace
_TRAILING_FENCE_AND_NOISE_RE = re.compile(
    r'\n```[a-zA-Z0-9_.\-]*[ \t]*\n.*$',
    re.DOTALL,
)

# finish(...) literal as the very last expression in the text
_FINISH_CALL_RE = re.compile(
    r'\n?[ \t]*`?finish\s*\([^)]{0,200}\)`?\s*$',
    re.DOTALL,
)

# Inline finish wrapped in backticks anywhere: `finish("foo")`
_INLINE_FINISH_RE = re.compile(r'`finish\s*\([^)]*\)`')


def sanitize_output(text: str) -> str:
    """Strip all known LLM output artifacts from generated code.

    Handles:
    - Wrapped fences:  ```python\\n<code>\\n```
    - Trailing fence:  code ends with \\n``` followed by noise
    - finish() literal: model wrote finish() as Python text instead of tool call
    - Inline finish in backticks: `finish("file")`
    """
    t = text.strip()
    if not t:
        return t

    # Case 1 — entire output is wrapped in a code fence
    if t.startswith("```"):
        m = _LANG_FENCE_RE.match(t)
        if m:
            return m.group(1).strip()
        # Fence opened but never properly closed — strip the opening line
        t = re.sub(r'^```[a-zA-Z0-9_.\-]*\n?', '', t)

    # Case 2 — code ends with a closing fence + trailing noise
    t = _TRAILING_FENCE_AND_NOISE_RE.sub("", t)

    # Case 3 — finish(...) literal at the end of the output
    t = _FINISH_CALL_RE.sub("", t)

    # Case 4 — inline `finish(...)` anywhere
    t = _INLINE_FINISH_RE.sub("", t)

    return t.strip()


def strip_fences(text: str) -> str:
    """Backwards-compatible alias for sanitize_output."""
    return sanitize_output(text)

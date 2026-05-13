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

# Thinking-model output: <think>...</think> block before the actual content.
# Nemotron, Trinity (arcee), QwQ and similar reasoning models emit this.
_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by thinking models."""
    return _THINK_BLOCK_RE.sub('', text).strip()


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


def parse_json_safe(text: str) -> dict:
    """Parse JSON from LLM output with thinking-strip + json_repair fallback.

    Centralises the try-JSON / try-json_repair / fail pattern that was
    duplicated across architect, generator, repairer, and validator agents.
    Also strips <think>...</think> blocks emitted by reasoning models before
    attempting any parse.
    Returns the parsed dict, or raises ValueError with a descriptive message.
    """
    import json
    text = strip_thinking(text).strip()
    if not text:
        raise ValueError("Empty response from LLM")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting a JSON object/array from mixed text (model added prose)
    m = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    try:
        from json_repair import repair_json
        return json.loads(repair_json(text))
    except Exception as e:
        raise ValueError(f"JSON unparseable even after repair: {e}") from e

"""Shared type aliases to reduce Dict[str, Any] proliferation.

Import these in place of raw ``Dict[str, Any]`` / ``List[Dict[str, Any]]``
annotations to make call-site intent explicit without introducing runtime
overhead (all aliases resolve to standard generics at runtime).
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from matrioska.core.state import FileArtifact  # noqa: F401 — used in ArtifactMap

# ── Pipeline types ─────────────────────────────────────────────────────────────

SharedState = Dict[str, Any]
"""The shared whiteboard passed between pipeline phases and agents.

Keys are written by generators (e.g. "db_path", "router_prefix") and
read by downstream files that declared them in FileSpec.reads.
"""

ArtifactMap = Dict[str, "FileArtifact"]
"""Mapping from artifact name to its FileArtifact instance.

Used in Phase 2 context injection so generators can read peer artifacts.
"""

EventData = Dict[str, Any]
"""Payload dict for EventBus events (e.g. llm_done, phase1_done)."""

ConfigOverrides = Dict[str, Any]
"""CLI / programmatic overrides passed to load_config().

Keys are Config field names; values are already-typed (not raw strings).
"""

# ── LLM types ─────────────────────────────────────────────────────────────────

Messages = List[Dict[str, Any]]
"""OpenAI-style message list: [{"role": "user"|"assistant"|"system", "content": str}]."""

ToolSpec = Dict[str, Any]
"""MCP / OpenAI tool descriptor in the function-calling schema format."""

JsonSchema = Dict[str, Any]
"""A JSON Schema dict (Draft-07 compatible), used for structured outputs."""

# ── Memory types ──────────────────────────────────────────────────────────────

RunNoteDict = Dict[str, Any]
"""Serialised RunNote as stored in episodic memory JSONL files."""

VaultSearchResult = Dict[str, Any]
"""Single result from GlobalVault.search().

Keys: title (str), path (str), score (float), snippet (str), scope (str),
      kind (str — "project"|"concept"|"bug").
"""

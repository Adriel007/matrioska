"""
Tool dispatcher — executes tool calls from agents.

Agents (Generator, Repairer) are given read-only tools to inspect
the shared_state and previously generated artifacts.  The dispatcher
resolves tool names to actual operations.

The `finish` tool is handled specially by each agent — it signals
completion and is never dispatched here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


class ToolDispatcher:
    """Dispatches tool calls to their implementations.

    Read-only tools (read_file, list_artifacts, read_shared_state)
    give agents visibility into the project state without allowing
    them to modify it directly.
    """

    def __init__(
        self,
        artifacts_dir: Path,
        shared_state: Optional[Dict[str, Any]] = None,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.shared_state = shared_state or {}

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}

        try:
            return handler(arguments)
        except Exception as e:
            return {"error": f"Tool '{name}' failed: {e}"}

    # ── Tool implementations ────────────────────────────────────────────

    def _tool_read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        name = args.get("name")
        extension = args.get("extension")
        if not name or not extension:
            return {"error": "name and extension required"}
        path = self.artifacts_dir / f"{name}.{extension}"
        if not path.exists():
            return {"error": f"Artifact not found: {name}.{extension}"}
        content = path.read_text(encoding="utf-8")
        return {"name": f"{name}.{extension}", "content": content, "chars": len(content)}

    def _tool_list_artifacts(self, _args: Dict[str, Any]) -> Dict[str, Any]:
        files = sorted(
            p.name for p in self.artifacts_dir.iterdir()
            if p.is_file() and p.stat().st_size > 0
        )
        return {"artifacts": files, "count": len(files)}

    def _tool_read_shared_state(self, _args: Dict[str, Any]) -> Dict[str, Any]:
        return {"shared_state": dict(self.shared_state)}

    def _tool_search_docs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "")
        if not query:
            return {"error": "query required"}
        # Placeholder: in full implementation, query web/docs APIs
        return {
            "query": query,
            "results": [],
            "note": "search_docs requires web API integration (scaffold)",
        }

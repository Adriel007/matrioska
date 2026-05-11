"""
Python API and MCP Server for Matrioska V3.

Provides:
  - Programmatic API: `matrioska.run(task)` as a library
  - MCP Server: Expose Matrioska as a tool for Claude Code, Cursor, etc.
  - Structured output for programmatic consumers
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from matrioska.core.config import Config, load_config, validate_config
from matrioska.pipeline.orchestrator import Matrioska, run

logger = logging.getLogger("matrioska.api")

__all__ = ["run", "Matrioska", "create_mcp_server", "run_with_config"]


def run_with_config(task: str, **config_overrides: Any) -> Dict[str, Any]:
    """Run Matrioska with inline config overrides.

    Example:
        result = run_with_config(
            "Create a CLI tool",
            provider="anthropic",
            model="claude-sonnet-4",
            architect_model="claude-opus-4",
        )
    """
    cfg = load_config(config_overrides if config_overrides else None)
    validate_config(cfg)
    return Matrioska(cfg).run(task)


# ── MCP Server ──────────────────────────────────────────────────────────────


MCP_TOOLS = [
    {
        "name": "matrioska_run",
        "description": "Execute a coding task end-to-end using Matrioska V3 multi-agent pipeline. "
        "Decomposes the task into files, generates them with contract validation, "
        "and verifies the output.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The coding task to execute (e.g., 'Create a FastAPI CRUD API for books')",
                },
                "provider": {
                    "type": "string",
                    "enum": ["openai", "anthropic", "ollama"],
                    "description": "LLM provider",
                },
                "model": {
                    "type": "string",
                    "description": "Model to use for generation",
                },
                "plan_only": {
                    "type": "boolean",
                    "description": "Only generate the architecture plan, no code",
                    "default": False,
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "matrioska_show",
        "description": "Show the current state of a Matrioska work directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "work_dir": {
                    "type": "string",
                    "description": "Path to the Matrioska work directory",
                },
            },
        },
    },
    {
        "name": "matrioska_resume",
        "description": "Resume a previously interrupted Matrioska run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "work_dir": {
                    "type": "string",
                    "description": "Path to the Matrioska work directory",
                },
            },
        },
    },
    {
        "name": "vault_search",
        "description": "Search the global Matrioska vault (Obsidian-compatible).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "scope": {"type": "string", "enum": ["local", "global", "linked", "all"], "default": "all"},
                "project": {"type": "string"},
                "k": {"type": "integer", "default": 8},
                "vault_dir": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "vault_get",
        "description": "Fetch a single vault note (path relative to vault root).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rel_path": {"type": "string"},
                "vault_dir": {"type": "string"},
            },
            "required": ["rel_path"],
        },
    },
    {
        "name": "vault_list",
        "description": "List all projects currently tracked in the vault.",
        "inputSchema": {
            "type": "object",
            "properties": {"vault_dir": {"type": "string"}},
        },
    },
    {
        "name": "vault_doctor",
        "description": "Vault health report (orphan notes, stale, broken wikilinks).",
        "inputSchema": {
            "type": "object",
            "properties": {"vault_dir": {"type": "string"}},
        },
    },
    {
        "name": "vault_related",
        "description": "Find projects reachable via [[wikilinks]] from a starting project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "max_hops": {"type": "integer", "default": 2},
                "vault_dir": {"type": "string"},
            },
            "required": ["project"],
        },
    },
]


async def create_mcp_server(port: int = 9020) -> None:
    """Start a stdio MCP server exposing Matrioska as a tool suite.

    Tools registered:
      - matrioska_run / matrioska_show / matrioska_resume  → pipeline ops
      - vault_search / vault_get / vault_list / vault_doctor / vault_graph
        / vault_related  → global Obsidian vault read-side

    The vault tools let external agents (Claude Code, Cursor, Windsurf) reuse
    Matrioska's accumulated knowledge without duplicating the compilation
    step.  Writes are deliberately not exposed — only the orchestrator
    compiles into the vault after a real run.
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError:
        logger.error("MCP library not installed. Run: pip install mcp")
        raise

    from pathlib import Path
    from matrioska.memory.vault import GlobalVault, default_vault_dir

    server = Server("matrioska")

    def _vault(vault_dir: str = "") -> GlobalVault:
        root = Path(vault_dir).expanduser() if vault_dir else default_vault_dir()
        return GlobalVault(root)

    # ── Pipeline tools ──────────────────────────────────────────────────

    @server.tool()
    async def matrioska_run(
        task: str, provider: str = "openai", model: str = "", plan_only: bool = False
    ) -> str:
        cfg = Config(provider=provider)
        if model:
            cfg.model = model
        cfg.plan_only = plan_only
        validate_config(cfg)

        m = Matrioska(cfg)
        result = m.run(task)
        return json.dumps(
            {
                "status": result["status"],
                "project_name": result["project_name"],
                "files": [
                    {"name": a.name, "extension": a.extension, "status": a.status}
                    for a in result.get("artifacts", [])
                ],
                "tokens": result.get("tokens", {}),
                "work_dir": result.get("work_dir", ""),
            },
            indent=2,
            ensure_ascii=False,
        )

    @server.tool()
    async def matrioska_show(work_dir: str = "./matrioska_work") -> str:
        cfg = Config(work_dir=Path(work_dir))
        m = Matrioska(cfg)
        info = m.show()
        return json.dumps(info, indent=2, ensure_ascii=False, default=str)

    @server.tool()
    async def matrioska_resume(work_dir: str = "./matrioska_work") -> str:
        cfg = Config(work_dir=Path(work_dir))
        m = Matrioska(cfg)
        result = m.resume()
        return json.dumps(
            {
                "status": result["status"],
                "project_name": result["project_name"],
                "files": [
                    {"name": a.name, "extension": a.extension, "status": a.status}
                    for a in result.get("artifacts", [])
                ],
            },
            indent=2,
            ensure_ascii=False,
        )

    # ── Vault tools (read-side; writes happen in the orchestrator) ──────

    @server.tool()
    async def vault_search(
        query: str,
        scope: str = "all",
        project: str = "",
        k: int = 8,
        vault_dir: str = "",
    ) -> str:
        """Search the global vault. Scope: local | global | linked | all."""
        results = _vault(vault_dir).search(
            query, scope=scope, project=project or None, k=k,
        )
        return json.dumps(results, indent=2, ensure_ascii=False)

    @server.tool()
    async def vault_get(rel_path: str, vault_dir: str = "") -> str:
        """Fetch a single note by path relative to the vault root."""
        note = _vault(vault_dir).get_note(rel_path)
        if note is None:
            return json.dumps({"error": "not found", "rel_path": rel_path})
        return json.dumps({
            "path": str(note.path),
            "title": note.title,
            "tags": note.tags,
            "frontmatter": note.frontmatter,
            "wikilinks": note.wikilinks,
            "body": note.body,
        }, indent=2, ensure_ascii=False, default=str)

    @server.tool()
    async def vault_list(vault_dir: str = "") -> str:
        """List projects tracked in the vault with note counts and last-run."""
        return json.dumps(_vault(vault_dir).list_projects(), indent=2, ensure_ascii=False)

    @server.tool()
    async def vault_doctor(vault_dir: str = "") -> str:
        """Vault health report: orphans, stale notes, broken wikilinks."""
        return json.dumps(_vault(vault_dir).doctor(), indent=2, ensure_ascii=False)

    @server.tool()
    async def vault_graph(vault_dir: str = "") -> str:
        """Export the wikilink graph as a Mermaid flowchart string."""
        return _vault(vault_dir).export_graph_mermaid()

    @server.tool()
    async def vault_related(project: str, max_hops: int = 2, vault_dir: str = "") -> str:
        """Find projects reachable via [[wikilinks]] from a starting project."""
        vault = _vault(vault_dir)
        related = sorted(vault._linked_projects(project, max_hops=max_hops))
        return json.dumps({"project": project, "max_hops": max_hops, "related": related})

    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())

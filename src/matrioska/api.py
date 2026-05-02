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

from src.matrioska.core.config import Config, load_config, validate_config
from src.matrioska.pipeline.orchestrator import Matrioska, run

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
]


async def create_mcp_server(port: int = 9020) -> None:
    """Start a minimal MCP server exposing Matrioska as a tool.

    This is a scaffold — full MCP integration uses the `mcp` library
    for proper protocol handling.
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError:
        logger.error("MCP library not installed. Run: pip install mcp")
        raise

    server = Server("matrioska")

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
        from pathlib import Path

        cfg = Config(work_dir=Path(work_dir))
        m = Matrioska(cfg)
        info = m.show()
        return json.dumps(info, indent=2, ensure_ascii=False, default=str)

    @server.tool()
    async def matrioska_resume(work_dir: str = "./matrioska_work") -> str:
        from pathlib import Path

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

    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())

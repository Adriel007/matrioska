"""Validate the MCP tool descriptor registry (cheap, no mcp lib needed)."""

from __future__ import annotations

from matrioska.api import MCP_TOOLS


def test_vault_tools_present():
    names = {t["name"] for t in MCP_TOOLS}
    assert "vault_search" in names
    assert "vault_get" in names
    assert "vault_list" in names
    assert "vault_doctor" in names
    assert "vault_related" in names


def test_pipeline_tools_present():
    names = {t["name"] for t in MCP_TOOLS}
    assert "matrioska_run" in names
    assert "matrioska_show" in names
    assert "matrioska_resume" in names


def test_every_tool_has_required_fields():
    for t in MCP_TOOLS:
        assert "name" in t
        assert "description" in t
        assert "inputSchema" in t
        assert t["inputSchema"]["type"] == "object"


def test_vault_search_schema_lists_scopes():
    tool = next(t for t in MCP_TOOLS if t["name"] == "vault_search")
    scopes = tool["inputSchema"]["properties"]["scope"]["enum"]
    assert set(scopes) == {"local", "global", "linked", "all"}

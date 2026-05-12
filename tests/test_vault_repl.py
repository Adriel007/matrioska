"""Tests for vault drill-down commands added to the REPL (/vault subcommands)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from matrioska.cli.repl import Repl, COMMANDS, HELP
from matrioska.core.config import Config


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def repl(tmp_path: Path) -> Repl:
    cfg = Config(work_dir=tmp_path, provider="openai", api_key="sk-test")
    r = Repl(cfg)
    r._console = None
    r._printed: List[str] = []
    r._print = lambda msg: r._printed.append(str(msg))  # type: ignore[assignment]
    return r


@pytest.fixture()
def vault(tmp_path: Path):
    """Real GlobalVault with a pre-compiled project for integration-style tests."""
    from matrioska.memory.vault import GlobalVault

    v = GlobalVault(tmp_path / "vault")
    v.compile_from_run(
        task="Build a FastAPI CRUD API with SQLite",
        project_name="book_api",
        files=[
            {"name": "main", "extension": "py", "status": "done", "repair_count": 0},
            {"name": "db", "extension": "py", "status": "done", "repair_count": 1},
        ],
        shared_state={"app_routes": ["/books"]},
        status="success",
        tags=["fastapi", "sqlite", "python"],
        lessons=["Repaired db.py in 1 attempt"],
        bugs=[],
    )
    return v


# ── Helper: run vault subcommand via dispatch ──────────────────────────────────


def _dispatch_vault(repl: Repl, subargs: str, vault_mock) -> List[str]:
    """Patch GlobalVault constructor and dispatch /vault <subargs>."""
    with patch("matrioska.memory.vault.GlobalVault", return_value=vault_mock), \
         patch("matrioska.memory.vault.default_vault_dir", return_value=Path("/tmp/vault")):
        repl._printed.clear()
        repl._dispatch(f"/vault {subargs}".strip())
    return list(repl._printed)


# ── Tests: /vault (no subcommand) ─────────────────────────────────────────────


def test_vault_no_args_shows_root(repl: Repl, vault):
    """Without subcommands the vault root and project listing are shown."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    vault_mock.list_projects.return_value = [
        {"name": "book_api", "notes": 4, "last_run": "2026-01-01T00:00:00"}
    ]
    lines = _dispatch_vault(repl, "", vault_mock)
    text = "\n".join(lines)
    assert "vault" in text.lower()
    assert "book_api" in text


# ── Tests: /vault search ──────────────────────────────────────────────────────


def test_vault_search_basic(repl: Repl):
    """search with a query returns results from vault.search()."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    vault_mock.search.return_value = [
        {
            "score": 4.0,
            "kind": "concept",
            "path": "concepts/fastapi.md",
            "title": "fastapi",
            "snippet": "FastAPI service with routes",
            "project": None,
        }
    ]
    lines = _dispatch_vault(repl, "search fastapi", vault_mock)
    text = "\n".join(lines)
    assert "fastapi" in text
    vault_mock.search.assert_called_once_with("fastapi", scope="all", k=8)


def test_vault_search_scope_flag(repl: Repl):
    """--scope flag is parsed and forwarded to vault.search."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    vault_mock.search.return_value = []
    lines = _dispatch_vault(repl, "search sqlite --scope global", vault_mock)
    vault_mock.search.assert_called_once_with("sqlite", scope="global", k=8)
    text = "\n".join(lines)
    assert "no results" in text.lower()


def test_vault_search_invalid_scope(repl: Repl):
    """Invalid --scope prints an error."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    lines = _dispatch_vault(repl, "search foo --scope badscope", vault_mock)
    text = "\n".join(lines)
    assert "invalid scope" in text.lower()


def test_vault_search_no_query(repl: Repl):
    """Search without a query shows usage hint."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    lines = _dispatch_vault(repl, "search", vault_mock)
    text = "\n".join(lines)
    assert "usage" in text.lower()


# ── Tests: /vault project ─────────────────────────────────────────────────────


def test_vault_project_shows_architecture(repl: Repl, tmp_path: Path, vault):
    """project subcommand reads architecture.md from the vault."""
    repl.cfg.vault_dir = str(vault.root)
    repl._printed.clear()
    repl._dispatch("/vault project book_api")
    text = "\n".join(repl._printed)
    assert "architecture" in text.lower()
    assert "book_api" in text.lower() or "architecture.md" in text.lower()


def test_vault_project_not_found(repl: Repl):
    """project subcommand with unknown name prints 'not found'."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    vault_mock.projects_dir = Path("/tmp/vault/projects")
    lines = _dispatch_vault(repl, "project nonexistent", vault_mock)
    text = "\n".join(lines)
    assert "not found" in text.lower()


def test_vault_project_no_arg(repl: Repl):
    """project without a name prints usage."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    lines = _dispatch_vault(repl, "project", vault_mock)
    text = "\n".join(lines)
    assert "usage" in text.lower()


# ── Tests: /vault concept ─────────────────────────────────────────────────────


def test_vault_concept_found(repl: Repl, tmp_path: Path, vault):
    """concept subcommand reads concepts/fastapi.md from the vault."""
    repl.cfg.vault_dir = str(vault.root)
    repl._printed.clear()
    repl._dispatch("/vault concept fastapi")
    text = "\n".join(repl._printed)
    assert "fastapi" in text.lower()


def test_vault_concept_not_found(repl: Repl, tmp_path: Path):
    """concept with unknown name prints 'not found'."""
    vault_mock = MagicMock()
    vault_mock.root = Path(tmp_path / "vault")
    vault_mock.concepts_dir = tmp_path / "vault" / "concepts"
    vault_mock.concepts_dir.mkdir(parents=True, exist_ok=True)
    lines = _dispatch_vault(repl, "concept unknown_concept", vault_mock)
    text = "\n".join(lines)
    assert "not found" in text.lower()


# ── Tests: /vault related ─────────────────────────────────────────────────────


def test_vault_related_shows_linked(repl: Repl):
    """related subcommand calls _linked_projects and shows results."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    vault_mock.projects_dir = Path("/tmp/vault/projects")
    vault_mock._linked_projects.return_value = {"book_api", "auth_service", "shared_lib"}
    lines = _dispatch_vault(repl, "related book_api", vault_mock)
    text = "\n".join(lines)
    # "book_api" is excluded (it's the seed), the other two should appear
    assert "auth_service" in text or "shared_lib" in text
    vault_mock._linked_projects.assert_called_once_with("book_api", max_hops=2)


def test_vault_related_none_found(repl: Repl):
    """related with no linked projects says '(none found)'."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    vault_mock._linked_projects.return_value = {"book_api"}  # only the seed itself
    lines = _dispatch_vault(repl, "related book_api", vault_mock)
    text = "\n".join(lines)
    assert "none found" in text.lower()


def test_vault_related_no_arg(repl: Repl):
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    lines = _dispatch_vault(repl, "related", vault_mock)
    text = "\n".join(lines)
    assert "usage" in text.lower()


# ── Tests: /vault doctor ──────────────────────────────────────────────────────


def test_vault_doctor_healthy(repl: Repl):
    """doctor subcommand renders health summary."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    vault_mock.doctor.return_value = {
        "projects": 2,
        "concepts": 5,
        "bugs": 0,
        "total_notes": 12,
        "orphans": [],
        "stale": [],
        "broken_links": [],
        "status": "healthy",
    }
    lines = _dispatch_vault(repl, "doctor", vault_mock)
    text = "\n".join(lines)
    assert "healthy" in text.lower()
    assert "projects" in text.lower()


def test_vault_doctor_with_issues(repl: Repl):
    """doctor subcommand shows orphans and broken links when present."""
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    vault_mock.doctor.return_value = {
        "projects": 1,
        "concepts": 2,
        "bugs": 0,
        "total_notes": 5,
        "orphans": ["concepts/stale.md"],
        "stale": [],
        "broken_links": [{"from": "projects/foo/links.md", "target": "bar"}],
        "status": "issues_found",
    }
    lines = _dispatch_vault(repl, "doctor", vault_mock)
    text = "\n".join(lines)
    assert "orphan" in text.lower()
    assert "broken" in text.lower()


# ── Tests: /vault graph ───────────────────────────────────────────────────────


def test_vault_graph_output(repl: Repl):
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    vault_mock.export_graph_mermaid.return_value = "```mermaid\ngraph LR\n  A --> B\n```"
    lines = _dispatch_vault(repl, "graph", vault_mock)
    text = "\n".join(lines)
    assert "mermaid" in text.lower()


# ── Tests: unknown subcommand ─────────────────────────────────────────────────


def test_vault_unknown_subcommand(repl: Repl):
    vault_mock = MagicMock()
    vault_mock.root = Path("/tmp/vault")
    lines = _dispatch_vault(repl, "bogus_cmd", vault_mock)
    text = "\n".join(lines)
    assert "unknown" in text.lower()
    assert "subcommand" in text.lower()

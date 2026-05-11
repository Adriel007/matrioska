"""Smoke tests for the global Obsidian-compatible vault."""

from __future__ import annotations

from pathlib import Path

import pytest

from matrioska.memory.vault import (
    GlobalVault,
    derive_tags,
    extract_lessons_and_bugs,
    default_vault_dir,
)


class _StubArtifact:
    def __init__(self, name, ext, status="done", rc=0):
        self.name = name
        self.extension = ext
        self.status = status
        self.repair_count = rc


@pytest.fixture()
def vault(tmp_path: Path) -> GlobalVault:
    return GlobalVault(tmp_path / "vault")


def test_default_vault_dir_under_home():
    p = default_vault_dir()
    assert p.is_absolute()
    assert "matrioska" in str(p).lower()


def test_compile_creates_expected_layout(vault: GlobalVault):
    arts = [_StubArtifact("main", "py"), _StubArtifact("db", "py", rc=1)]
    touched = vault.compile_from_run(
        task="Build a FastAPI CRUD API with SQLite",
        project_name="book_api",
        files=[{"name": a.name, "extension": a.extension,
                "status": a.status, "repair_count": a.repair_count} for a in arts],
        shared_state={"app_routes": ["/books"]},
        status="success",
        tags=["fastapi", "sqlite"],
        lessons=["Repaired db.py in 1 attempt"],
        bugs=[],
    )
    assert (vault.root / "projects" / "book_api" / "architecture.md").exists()
    assert (vault.root / "concepts" / "fastapi.md").exists()
    assert (vault.root / "concepts" / "sqlite.md").exists()
    assert len(touched) >= 4


def test_compile_dedups_identical_concept_entries(vault: GlobalVault):
    """Two compiles within the same second produce identical concept bullets;
    append_dedup must suppress the duplicate."""
    files = [{"name": "main", "extension": "py", "status": "done", "repair_count": 0}]
    vault.compile_from_run(
        task="First", project_name="proj", files=files,
        shared_state={}, status="success", tags=["python"], lessons=[], bugs=[],
    )
    vault.compile_from_run(
        task="Second", project_name="proj", files=files,
        shared_state={}, status="success", tags=["python"], lessons=[], bugs=[],
    )
    concept_text = (vault.root / "concepts" / "python.md").read_text()
    # Same-second runs produce one dedup'd bullet (not two)
    assert concept_text.count("[[proj/architecture]]") == 1
    # Multiple runs always create a fresh architecture section (timestamped block)
    arch_text = (vault.root / "projects" / "proj" / "architecture.md").read_text()
    assert "### Run " in arch_text


def test_search_returns_ranked_results(vault: GlobalVault):
    vault.compile_from_run(
        task="FastAPI CRUD with SQLite",
        project_name="api", files=[{"name": "main", "extension": "py", "status": "done", "repair_count": 0}],
        shared_state={}, status="success", tags=["fastapi", "sqlite"], lessons=[], bugs=[],
    )
    results = vault.search("sqlite", scope="global", k=5)
    assert results
    assert any(r["title"].startswith("sqlite") for r in results)


def test_doctor_reports_healthy_after_compile(vault: GlobalVault):
    vault.compile_from_run(
        task="x", project_name="p", files=[{"name": "a", "extension": "py", "status": "done", "repair_count": 0}],
        shared_state={}, status="success", tags=["py"], lessons=[], bugs=[],
    )
    report = vault.doctor()
    assert report["projects"] == 1
    assert report["status"] == "healthy"
    assert not report["broken_links"]


def test_derive_tags_picks_up_task_keywords():
    tags = derive_tags("Build a FastAPI service with SQLite and JWT", [])
    assert "fastapi" in tags
    assert "sqlite" in tags
    assert "jwt" in tags


def test_extract_lessons_and_bugs():
    arts = [
        _StubArtifact("a", "py", rc=2, status="done"),
        _StubArtifact("b", "py", status="failed"),
    ]
    lessons, bugs = extract_lessons_and_bugs(arts, repair_log=["SyntaxError: bad token"])
    assert any("a.py" in l for l in lessons)
    assert any("b.py" in b for b in bugs)
    assert any("SyntaxError" in b for b in bugs)


def test_export_graph_mermaid(vault: GlobalVault):
    vault.compile_from_run(
        task="x", project_name="p", files=[{"name": "a", "extension": "py", "status": "done", "repair_count": 0}],
        shared_state={}, status="success", tags=["py", "cli"], lessons=[], bugs=[],
    )
    out = vault.export_graph_mermaid()
    assert out.startswith("```mermaid")
    assert "graph LR" in out

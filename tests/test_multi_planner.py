"""Tests for the multi-planner (Tree-of-Thoughts N-candidate) orchestration.

Phase 1 can generate N parallel Architect plans at high temperature
(`architect_candidates > 1`, `enable_tot=True`) and pick the best via Judge.
These tests exercise the multi-planner logic directly, without real LLM calls.

Related config flags: `architect_candidates`, `enable_tot`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from matrioska.core.config import Config, load_config


# ---------------------------------------------------------------------------
# Config-level tests (no mocking needed)
# ---------------------------------------------------------------------------


def test_config_enable_multi_plan_defaults():
    """Hardcoded Config() dataclass defaults: enable_tot=True, architect_candidates=3.

    Uses Config() directly (no env / .env loading) to test the code defaults.
    load_config() is intentionally NOT used here since a local .env may override
    these values on the developer's machine.
    """
    from matrioska.core.config import Config
    cfg = Config()
    assert cfg.enable_tot is True
    assert cfg.architect_candidates == 3


def test_config_enable_multi_plan_quick_mode_disables_tot():
    """Quick mode must disable Tree-of-Thoughts and reduce candidates to 1."""
    cfg = load_config({"quick": True})
    assert cfg.enable_tot is False
    assert cfg.architect_candidates == 1


def test_config_enable_multi_plan_explicit_override_respected():
    """Explicit architect_candidates overrides the quick-mode default."""
    cfg = load_config({"quick": True, "architect_candidates": 5})
    assert cfg.architect_candidates == 5


def test_config_enable_multi_plan_env_candidates(monkeypatch):
    """MATRIOSKA_ARCHITECT_CANDIDATES env var sets candidate count."""
    monkeypatch.setenv("MATRIOSKA_ARCHITECT_CANDIDATES", "7")
    cfg = load_config({})
    assert cfg.architect_candidates == 7


def test_config_enable_multi_plan_tot_env(monkeypatch):
    """MATRIOSKA_ENABLE_TOT=false disables Tree-of-Thoughts."""
    monkeypatch.setenv("MATRIOSKA_ENABLE_TOT", "false")
    cfg = load_config({})
    assert cfg.enable_tot is False


# ---------------------------------------------------------------------------
# ArchitectAgent multi-candidate logic
# ---------------------------------------------------------------------------


def _import_architect_agent():
    """Import ArchitectAgent, skipping if unavailable."""
    try:
        from matrioska.agents.architect import ArchitectAgent
        return ArchitectAgent
    except ImportError:
        pytest.skip("ArchitectAgent not importable")


def test_multi_planner_single_candidate_no_judge_call(tmp_path: Path):
    """With architect_candidates=1, the Judge should NOT be called."""
    ArchitectAgent = _import_architect_agent()
    cfg = Config(
        work_dir=tmp_path,
        provider="openai",
        api_key="sk-test",
        architect_candidates=1,
        enable_tot=False,
    )

    mock_llm = MagicMock()
    # Architect returns a minimal valid plan
    mock_llm.chat.return_value = MagicMock(
        text='{"project_name": "hello", "files": [{"name": "main", "extension": "py", "order": 1, "complex": false, "shared_state_reads": [], "shared_state_writes": [], "content": "print(1)", "details": "entry"}]}',
        tool_calls=[],
    )

    agent = ArchitectAgent(cfg=cfg, llm=mock_llm)
    arch = agent.plan("write hello world")

    # One plan → no judge needed, so llm.chat called exactly once
    assert mock_llm.chat.call_count == 1, (
        "Judge should not be invoked with a single candidate"
    )


def test_multi_planner_multiple_candidates_calls_llm_n_times(tmp_path: Path):
    """With N candidates, Architect should call LLM N times."""
    ArchitectAgent = _import_architect_agent()
    cfg = Config(
        work_dir=tmp_path,
        provider="openai",
        api_key="sk-test",
        architect_candidates=3,
        enable_tot=True,
    )

    minimal_plan = '{"project_name": "api", "files": [{"name": "app", "extension": "py", "order": 1, "complex": false, "shared_state_reads": [], "shared_state_writes": [], "content": "pass", "details": "entry"}]}'
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(text=minimal_plan, tool_calls=[])

    agent = ArchitectAgent(cfg=cfg, llm=mock_llm)
    agent.plan("build a rest api")

    # N architect calls + 1 judge call = N+1 total, OR N calls (judge may be internal)
    # At minimum N=3 calls must have been made
    assert mock_llm.chat.call_count >= 3, (
        f"Expected ≥3 LLM calls for 3 candidates, got {mock_llm.chat.call_count}"
    )


def test_multi_planner_returns_architecture_object(tmp_path: Path):
    """plan() must return an Architecture-like object with a files list."""
    ArchitectAgent = _import_architect_agent()
    cfg = Config(
        work_dir=tmp_path,
        provider="openai",
        api_key="sk-test",
        architect_candidates=1,
        enable_tot=False,
    )
    minimal_plan = '{"project_name": "cli_app", "files": [{"name": "cli", "extension": "py", "order": 1, "complex": false, "shared_state_reads": [], "shared_state_writes": ["db_path"], "content": "pass", "details": "entry"}]}'
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(text=minimal_plan, tool_calls=[])

    agent = ArchitectAgent(cfg=cfg, llm=mock_llm)
    result = agent.plan("build a CLI app")

    assert result is not None, "plan() must return a non-None result"
    assert hasattr(result, "files"), "Architecture object must have a .files attribute"
    assert len(result.files) >= 1

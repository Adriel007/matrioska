"""Tests for REPL autocompletion, hook system, and multi-planner."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from matrioska.core.config import Config, load_config
from matrioska.cli.repl import COMMANDS, Repl


# ── REPL Autocompletion ───────────────────────────────────────────────────────


def test_build_completer_returns_nested():
    cfg = Config(work_dir=Path("/tmp"), provider="openai", api_key="sk-test")
    repl = Repl(cfg)
    completer = repl._build_completer()
    assert completer is not None


def test_build_completer_covers_all_commands():
    """Every registered command should appear in the completer's options."""
    try:
        from prompt_toolkit.completion import NestedCompleter
    except ImportError:
        pytest.skip("prompt_toolkit not installed")

    cfg = Config(work_dir=Path("/tmp"), provider="openai", api_key="sk-test")
    repl = Repl(cfg)
    completer = repl._build_completer()

    # NestedCompleter stores its map as .options
    for name in COMMANDS:
        assert f"/{name}" in completer.options, f"/{name} missing from completer"


def test_build_completer_vault_has_subcommands():
    try:
        from prompt_toolkit.completion import NestedCompleter
    except ImportError:
        pytest.skip("prompt_toolkit not installed")

    cfg = Config(work_dir=Path("/tmp"), provider="openai", api_key="sk-test")
    repl = Repl(cfg)
    completer = repl._build_completer()
    vault_sub = completer.options.get("/vault")
    assert vault_sub is not None
    assert "search" in vault_sub.options
    assert "list" in vault_sub.options
    assert "doctor" in vault_sub.options


def test_build_completer_effort_has_subcommands():
    try:
        from prompt_toolkit.completion import NestedCompleter
    except ImportError:
        pytest.skip("prompt_toolkit not installed")

    cfg = Config(work_dir=Path("/tmp"), provider="openai", api_key="sk-test")
    repl = Repl(cfg)
    completer = repl._build_completer()
    effort_sub = completer.options.get("/effort")
    assert effort_sub is not None
    assert "low" in effort_sub.options
    assert "medium" in effort_sub.options
    assert "high" in effort_sub.options


# ── Hook system ───────────────────────────────────────────────────────────────


def test_hook_runner_inactive_when_no_dir(tmp_path: Path):
    from matrioska.hooks import HookRunner
    runner = HookRunner(project_dir=tmp_path)
    assert not runner.active


def test_hook_runner_active_when_dir_exists(tmp_path: Path):
    from matrioska.hooks import HookRunner
    hooks_dir = tmp_path / ".matrioska" / "hooks"
    hooks_dir.mkdir(parents=True)
    runner = HookRunner(project_dir=tmp_path)
    assert runner.active


def test_hook_runner_executes_script(tmp_path: Path):
    from matrioska.hooks import HookRunner
    hooks_dir = tmp_path / ".matrioska" / "hooks"
    hooks_dir.mkdir(parents=True)

    output_file = tmp_path / "hook_output.txt"
    script = hooks_dir / "post_generate.sh"
    script.write_text(
        f"#!/usr/bin/env bash\ncat > {output_file}\n"
    )
    script.chmod(0o755)

    runner = HookRunner(project_dir=tmp_path)
    runner.run("post_generate", {"file": "main.py", "status": "done"})

    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert data["file"] == "main.py"


def test_hook_runner_ignores_missing_script(tmp_path: Path):
    from matrioska.hooks import HookRunner
    hooks_dir = tmp_path / ".matrioska" / "hooks"
    hooks_dir.mkdir(parents=True)
    runner = HookRunner(project_dir=tmp_path)
    # Should not raise
    runner.run("pre_generate", {"file": "db.py"})


def test_hook_runner_subscribe_fires_on_event(tmp_path: Path):
    from matrioska.hooks import HookRunner
    from matrioska.core.events import EventBus

    hooks_dir = tmp_path / ".matrioska" / "hooks"
    hooks_dir.mkdir(parents=True)
    fired: list[str] = []

    runner = HookRunner(project_dir=tmp_path)
    # Monkeypatch run to capture calls
    original_run = runner.run
    def _spy(name, ctx):
        fired.append(name)
        return original_run(name, ctx)
    runner.run = _spy  # type: ignore

    bus = EventBus()
    runner.subscribe(bus)

    bus.emit_named("file_generated", file="main.py", status="done", chars=100)
    bus.emit_named("agent_call", agent="generator", model="gpt-4o-mini")
    bus.emit_named("agent_call", agent="repairer", model="gpt-4o-mini")

    assert "post_generate" in fired
    assert "pre_generate" in fired
    assert "pre_repair" in fired


# ── Multi-planner ─────────────────────────────────────────────────────────────


def test_multi_planner_fallback_on_meta_failure():
    """If meta-planner LLM fails, MultiPlanner falls back to ArchitectAgent."""
    from matrioska.agents.multi_planner import MultiPlanner
    from matrioska.llm.client import ChatResponse

    cfg = load_config({"provider": "openai", "api_key": "sk-test",
                       "model": "gpt-4o-mini", "enable_vault": False})

    ARCH = json.dumps({
        "project_name": "hello",
        "files": [{"name": "main", "extension": "py", "order": 1,
                   "shared_state_reads": [], "shared_state_writes": [],
                   "content": "print hi", "details": "entry", "complex": False}]
    })

    def mock_chat(messages, **kw):
        # Both meta-planner and architect return valid JSON
        return ChatResponse(text=ARCH, prompt_tokens=10, completion_tokens=10)

    llm = MagicMock()
    llm.chat.side_effect = mock_chat

    planner = MultiPlanner(cfg=cfg, llm=llm)
    # Force _identify_subproblems to return empty (simulate failure)
    with patch.object(planner, "_identify_subproblems", return_value=([], {})):
        result = planner.plan("hello world")

    assert result is not None
    assert len(result.files) >= 1


def test_multi_planner_merge_deduplicates():
    """Files with the same name.ext should be deduplicated (last wins)."""
    from matrioska.agents.multi_planner import MultiPlanner
    from matrioska.core.state import FileSpec

    cfg = load_config({"provider": "openai", "api_key": "sk-test", "model": "gpt-4o-mini"})
    planner = MultiPlanner(cfg=cfg, llm=MagicMock())

    files = [
        FileSpec(name="main", extension="py", order=1,
                 shared_state_reads=[], shared_state_writes=[], content="v1", details="", complex=False),
        FileSpec(name="db", extension="py", order=2,
                 shared_state_reads=[], shared_state_writes=["db_path"], content="db", details="", complex=False),
        FileSpec(name="main", extension="py", order=3,
                 shared_state_reads=["db_path"], shared_state_writes=[], content="v2", details="", complex=False),
    ]
    merged = planner._merge("test task", files)
    names = [f"{f.name}.{f.extension}" for f in merged.files]
    assert names.count("main.py") == 1
    # Last main.py (v2) should survive
    main_file = next(f for f in merged.files if f.name == "main")
    assert main_file.content == "v2"


def test_multi_planner_scoped_task_includes_interface():
    from matrioska.agents.multi_planner import MultiPlanner

    cfg = load_config({"provider": "openai", "api_key": "sk-test", "model": "gpt-4o-mini"})
    planner = MultiPlanner(cfg=cfg, llm=MagicMock())

    scoped = planner._build_scoped_task(
        task="Build a REST API",
        subproblem={"id": "auth", "label": "Auth", "scope": "Login and JWT"},
        shared_interface={"jwt_secret": "JWT signing key"},
        accumulated_writes={},
    )
    assert "FULL TASK" in scoped
    assert "jwt_secret" in scoped
    assert "Login and JWT" in scoped


def test_config_enable_multi_plan_default_false():
    cfg = Config()
    assert cfg.enable_multi_plan is False


def test_config_enable_multi_plan_loadable():
    import os
    cfg = load_config({"enable_multi_plan": True})
    assert cfg.enable_multi_plan is True

"""Tests for the interactive REPL: command registry, dispatch, slash parsing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from matrioska.cli.repl import COMMANDS, HELP, Repl, ReplSession
from matrioska.core.config import Config


@pytest.fixture()
def repl(tmp_path: Path) -> Repl:
    cfg = Config(work_dir=tmp_path, provider="openai", api_key="sk-test")
    r = Repl(cfg)
    # Capture _print output for assertions.
    r._console = None
    r._printed: list[str] = []
    r._print = lambda msg: r._printed.append(str(msg))  # type: ignore[assignment]  # replacing bound method with lambda for test capture
    return r


def test_all_slash_commands_have_help():
    for name in COMMANDS:
        assert name in HELP, f"missing help for /{name}"
        assert HELP[name], f"empty help for /{name}"


def test_dispatch_help_prints_command_list(repl: Repl):
    repl._dispatch("/help")
    text = "\n".join(repl._printed)
    assert "/help" in text
    assert "/config" in text
    assert "/vault" in text


def test_dispatch_config_lists_all_keys(repl: Repl):
    repl._dispatch("/config")
    text = "\n".join(repl._printed)
    assert "provider" in text
    assert "stream_tokens" in text


def test_unknown_command_message(repl: Repl):
    repl._dispatch("/notacommand")
    text = "\n".join(repl._printed)
    assert "unknown command" in text


def test_plan_toggle(repl: Repl):
    assert repl.session.plan_mode is False
    repl._dispatch("/plan")
    assert repl.session.plan_mode is True
    repl._dispatch("/plan")
    assert repl.session.plan_mode is False


def test_effort_set(repl: Repl):
    repl._dispatch("/effort high")
    assert repl.session.effort == "high"
    repl._dispatch("/effort invalid")
    assert repl.session.effort == "high"  # unchanged


def test_stream_toggle(repl: Repl):
    # Capture initial state and assert that /stream toggles it.
    # Default may be True or False depending on env; we only verify the toggle.
    initial = repl.cfg.stream_tokens
    repl._dispatch("/stream")
    assert repl.cfg.stream_tokens is not initial


def test_shell_prefix_runs_subprocess(repl: Repl, monkeypatch):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw
        return MagicMock(stdout="ok\n", stderr="", returncode=0)

    monkeypatch.setattr("matrioska.cli.repl.subprocess.run", fake_run)
    repl._dispatch("!echo hello")
    assert captured["cmd"] == "echo hello"
    assert captured["kw"]["shell"] is True


def test_history_persists_in_session(repl: Repl, monkeypatch):
    monkeypatch.setattr("matrioska.cli.repl.subprocess.run",
                        lambda *a, **kw: MagicMock(stdout="", stderr="", returncode=0))
    repl.session.history.extend(["task one", "/plan", "!ls"])
    repl._dispatch("/history")
    text = "\n".join(repl._printed)
    assert "task one" in text


def test_effective_cfg_applies_plan_and_effort(repl: Repl):
    repl.session.plan_mode = True
    repl.session.effort = "low"
    eff = repl._effective_cfg_for_run()
    assert eff.plan_only is True
    assert eff.enable_tot is False
    assert eff.architect_candidates == 1


def test_clear_resets_session(repl: Repl):
    repl.session.tokens_session["prompt"] = 100
    repl.session.last_result = {"status": "success"}
    repl._dispatch("/clear")
    assert repl.session.tokens_session["prompt"] == 0
    assert repl.session.last_result is None

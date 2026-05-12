"""Tests for the custom commands feature (.matrioska/commands/*.md)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Set
from unittest.mock import MagicMock, patch

import pytest

from matrioska.cli.repl import COMMANDS, HELP, Repl
from matrioska.core.config import Config


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def commands_dir(work_dir: Path) -> Path:
    d = work_dir / ".matrioska" / "commands"
    d.mkdir(parents=True)
    return d


@pytest.fixture(autouse=True)
def _isolate_commands_registry():
    """Snapshot COMMANDS/HELP before each test and restore after."""
    before_commands: Set[str] = set(COMMANDS.keys())
    before_help: Set[str] = set(HELP.keys())
    yield
    # Remove any custom commands added during this test
    added = set(COMMANDS.keys()) - before_commands
    for key in added:
        COMMANDS.pop(key, None)
        HELP.pop(key, None)


def _make_repl(work_dir: Path, monkeypatch) -> Repl:
    """Build a Repl with cwd patched to work_dir."""
    monkeypatch.chdir(work_dir)
    cfg = Config(work_dir=work_dir, provider="openai", api_key="sk-test")
    r = Repl(cfg)
    r._console = None
    r._printed: List[str] = []
    r._print = lambda msg: r._printed.append(str(msg))  # type: ignore[assignment]
    return r


# ── Tests: command registration ───────────────────────────────────────────────


def test_custom_command_registered(commands_dir: Path, work_dir: Path, monkeypatch):
    """A .md file in .matrioska/commands/ is registered as a slash command."""
    (commands_dir / "my_task.md").write_text("# My Custom Task\nDo something useful.", encoding="utf-8")
    repl = _make_repl(work_dir, monkeypatch)
    assert "my_task" in COMMANDS, "custom command 'my_task' should be registered"
    assert "my_task" in HELP, "custom command 'my_task' should have help text"


def test_custom_command_help_text_from_first_line(commands_dir: Path, work_dir: Path, monkeypatch):
    """The help text is derived from the first line of the .md file (sans leading #)."""
    (commands_dir / "deploy.md").write_text("# Deploy to production\nRun deploy script.", encoding="utf-8")
    _make_repl(work_dir, monkeypatch)
    assert "deploy" in HELP
    assert "Deploy to production" in HELP["deploy"]


def test_custom_command_help_has_custom_prefix(commands_dir: Path, work_dir: Path, monkeypatch):
    """Custom command help is prefixed with '[custom]'."""
    (commands_dir / "lint.md").write_text("# Run linter\n", encoding="utf-8")
    _make_repl(work_dir, monkeypatch)
    assert HELP.get("lint", "").startswith("[custom]")


def test_dashes_in_filename_become_underscores(commands_dir: Path, work_dir: Path, monkeypatch):
    """Filenames with dashes are converted to underscores for the command name."""
    (commands_dir / "run-tests.md").write_text("# Run test suite\n", encoding="utf-8")
    _make_repl(work_dir, monkeypatch)
    assert "run_tests" in COMMANDS


def test_multiple_custom_commands(commands_dir: Path, work_dir: Path, monkeypatch):
    """Multiple .md files each create their own command."""
    (commands_dir / "cmd_a.md").write_text("# Command A\n", encoding="utf-8")
    (commands_dir / "cmd_b.md").write_text("# Command B\n", encoding="utf-8")
    (commands_dir / "cmd_c.md").write_text("# Command C\n", encoding="utf-8")
    _make_repl(work_dir, monkeypatch)
    for name in ("cmd_a", "cmd_b", "cmd_c"):
        assert name in COMMANDS, f"'{name}' not in COMMANDS"


def test_no_commands_dir_no_error(work_dir: Path, monkeypatch):
    """If .matrioska/commands/ doesn't exist, init succeeds silently."""
    monkeypatch.chdir(work_dir)
    cfg = Config(work_dir=work_dir, provider="openai", api_key="sk-test")
    repl = Repl(cfg)  # must not raise
    assert repl is not None


def test_empty_commands_dir_no_error(commands_dir: Path, work_dir: Path, monkeypatch):
    """An empty commands dir is handled gracefully."""
    monkeypatch.chdir(work_dir)
    cfg = Config(work_dir=work_dir, provider="openai", api_key="sk-test")
    repl = Repl(cfg)
    assert repl is not None


def test_empty_file_skipped(commands_dir: Path, work_dir: Path, monkeypatch):
    """A .md file with no content is skipped (not registered)."""
    (commands_dir / "empty_cmd.md").write_text("", encoding="utf-8")
    _make_repl(work_dir, monkeypatch)
    assert "empty_cmd" not in COMMANDS


# ── Tests: command execution ──────────────────────────────────────────────────


def test_custom_command_calls_run_task(commands_dir: Path, work_dir: Path, monkeypatch):
    """Dispatching the custom command calls _run_task with the .md content."""
    task_content = "# Custom task\nBuild a FastAPI service with SQLite."
    (commands_dir / "build_api.md").write_text(task_content, encoding="utf-8")
    repl = _make_repl(work_dir, monkeypatch)

    called_with: List[str] = []

    def _fake_run_task(task: str) -> None:
        called_with.append(task)

    repl._run_task = _fake_run_task  # type: ignore[method-assign]
    repl._dispatch("/build_api")

    assert called_with, "expected _run_task to be called"
    assert called_with[0] == task_content.strip()


def test_custom_command_content_is_full_file(commands_dir: Path, work_dir: Path, monkeypatch):
    """The task prompt is the full file content (not just the first line)."""
    content = "# My Task\nStep 1: do X.\nStep 2: do Y.\nStep 3: verify Z."
    (commands_dir / "multi_step.md").write_text(content, encoding="utf-8")
    repl = _make_repl(work_dir, monkeypatch)

    called_with: List[str] = []
    repl._run_task = lambda t: called_with.append(t)  # type: ignore[method-assign]
    repl._dispatch("/multi_step")

    assert called_with[0] == content.strip()
    assert "Step 2" in called_with[0]
    assert "Step 3" in called_with[0]


def test_custom_command_closure_captures_correct_content(commands_dir: Path, work_dir: Path, monkeypatch):
    """Each custom command closure captures its own content (no late-binding bug)."""
    (commands_dir / "task_x.md").write_text("# Task X\ncontent-x", encoding="utf-8")
    (commands_dir / "task_y.md").write_text("# Task Y\ncontent-y", encoding="utf-8")
    repl = _make_repl(work_dir, monkeypatch)

    results: dict = {}

    def _fake_run(task: str) -> None:
        results["last"] = task

    repl._run_task = _fake_run  # type: ignore[method-assign]

    repl._dispatch("/task_x")
    assert "content-x" in results["last"]

    repl._dispatch("/task_y")
    assert "content-y" in results["last"]


# ── Tests: /help includes custom commands ────────────────────────────────────


def test_help_includes_custom_command(commands_dir: Path, work_dir: Path, monkeypatch):
    """After loading custom commands, /help shows them."""
    (commands_dir / "custom_check.md").write_text("# Check all systems\n", encoding="utf-8")
    repl = _make_repl(work_dir, monkeypatch)
    repl._dispatch("/help")
    text = "\n".join(repl._printed)
    assert "custom_check" in text

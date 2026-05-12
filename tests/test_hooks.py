"""Tests for the hook runner system.

Hooks are shell scripts in `.matrioska/hooks/` executed on pipeline events:
  pre_generate, post_generate, pre_repair, session_start, session_end

Scripts receive event context as JSON on stdin.  The feature is tracked in
TODO.md under "Hook system".

These tests cover the public contract of `HookRunner` (to be implemented in
`matrioska.core.hooks` or similar).  They use `tmp_path` for isolation and
patch subprocess so no real shells are spawned.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


def _import_hook_runner():
    """Import HookRunner, skipping if not yet implemented."""
    try:
        from matrioska.core.hooks import HookRunner
        return HookRunner
    except ImportError:
        pytest.skip("HookRunner not yet implemented in matrioska.core.hooks")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hook_runner_no_hooks_dir_does_not_crash(tmp_path: Path):
    """HookRunner must be instantiable even when the hooks directory is absent."""
    HookRunner = _import_hook_runner()
    runner = HookRunner(hooks_dir=tmp_path / "nonexistent" / "hooks")
    # Fire any event — should be a no-op, not an exception
    runner.fire("session_start", {"task": "hello"})


def test_hook_runner_fires_script_for_matching_event(tmp_path: Path):
    """A script named <event>.sh should be executed when that event fires."""
    HookRunner = _import_hook_runner()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    script = hooks_dir / "pre_generate.sh"
    script.write_text("#!/bin/sh\nread ctx\n")
    script.chmod(0o755)

    runner = HookRunner(hooks_dir=hooks_dir)
    captured: list[dict] = []

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        runner.fire("pre_generate", {"file": "main.py"})

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    # Script path should include pre_generate
    assert "pre_generate" in str(args[0])


def test_hook_runner_passes_context_as_json_on_stdin(tmp_path: Path):
    """Context dict must be serialised to JSON and passed via stdin."""
    HookRunner = _import_hook_runner()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    script = hooks_dir / "post_generate.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)

    runner = HookRunner(hooks_dir=hooks_dir)
    ctx = {"file": "app.py", "status": "success", "tokens": 42}

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        runner.fire("post_generate", ctx)

    _args, kwargs = mock_run.call_args
    stdin_data = kwargs.get("input") or ""
    parsed = json.loads(stdin_data)
    assert parsed["file"] == "app.py"
    assert parsed["status"] == "success"


def test_hook_runner_unknown_event_is_no_op(tmp_path: Path):
    """Firing an event with no matching script must be silent."""
    HookRunner = _import_hook_runner()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    runner = HookRunner(hooks_dir=hooks_dir)
    # No script for "unknown_event" — should not raise
    runner.fire("unknown_event", {})


def test_hook_runner_non_executable_script_is_skipped(tmp_path: Path):
    """A non-executable hook file should not cause a crash (skip or warn)."""
    HookRunner = _import_hook_runner()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    script = hooks_dir / "session_start.sh"
    script.write_text("#!/bin/sh\n")
    # Deliberately NOT chmod +x

    runner = HookRunner(hooks_dir=hooks_dir)
    # Should not raise even if file is not executable
    try:
        runner.fire("session_start", {"task": "test"})
    except PermissionError:
        pytest.fail("HookRunner must not propagate PermissionError for non-executable hooks")


def test_hook_runner_failed_script_does_not_abort_pipeline(tmp_path: Path):
    """A hook that exits non-zero must not raise an exception."""
    HookRunner = _import_hook_runner()
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    script = hooks_dir / "pre_repair.sh"
    script.write_text("#!/bin/sh\nexit 1\n")
    script.chmod(0o755)

    runner = HookRunner(hooks_dir=hooks_dir)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error!")
        # Must not raise — hook failures are non-fatal
        runner.fire("pre_repair", {"file": "broken.py"})

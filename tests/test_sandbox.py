"""Tests for the Phase 3 sandbox executor and repair loop."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from matrioska.core.config import load_config
from matrioska.core.state import FileArtifact, RunState, StateGraph
from matrioska.tools.sandbox import (
    ProjectType,
    SandboxExecutor,
    SandboxResult,
    detect_entrypoint,
    detect_project_type,
    extract_pip_packages,
    is_server_process,
    parse_erroring_file,
    validate_html,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_artifact(name: str, ext: str, content: str, order: int = 1) -> FileArtifact:
    a = FileArtifact(name=name, extension=ext, order=order, content=content)
    a.status = "done"
    return a


def _make_state(artifacts: dict[str, FileArtifact]) -> RunState:
    state = MagicMock(spec=RunState)
    state.artifacts = artifacts
    state.shared_state = {}
    return state


# ── detect_project_type ───────────────────────────────────────────────────────


def test_detect_python():
    state = _make_state({"main": _make_artifact("main", "py", "print('hi')")})
    assert detect_project_type(state) == ProjectType.PYTHON


def test_detect_node():
    state = _make_state({"index": _make_artifact("index", "js", "console.log('hi')")})
    assert detect_project_type(state) == ProjectType.NODE


def test_detect_web():
    state = _make_state({
        "index.html": _make_artifact("index", "html", "<html></html>"),
        "style.css":  _make_artifact("style", "css", "body {}"),
    })
    assert detect_project_type(state) == ProjectType.WEB


def test_detect_shell():
    state = _make_state({"run": _make_artifact("run", "sh", "echo hi")})
    assert detect_project_type(state) == ProjectType.SHELL


# ── detect_entrypoint ─────────────────────────────────────────────────────────


def test_prefers_main_py():
    state = _make_state({
        "utils": _make_artifact("utils", "py", "", order=1),
        "main":  _make_artifact("main",  "py", "", order=2),
    })
    assert detect_entrypoint(state, ProjectType.PYTHON) == "main.py"


def test_fallback_highest_order():
    state = _make_state({
        "models": _make_artifact("models", "py", "", order=1),
        "api":    _make_artifact("api",    "py", "", order=2),
    })
    assert detect_entrypoint(state, ProjectType.PYTHON) == "api.py"


def test_no_entrypoint_for_empty():
    state = _make_state({})
    assert detect_entrypoint(state, ProjectType.PYTHON) is None


# ── is_server_process ─────────────────────────────────────────────────────────


def test_detects_uvicorn():
    assert is_server_process("uvicorn.run(app, host='0.0.0.0')")


def test_detects_flask_run():
    assert is_server_process("app.run(debug=True)")


def test_plain_script_not_server():
    assert not is_server_process("print('hello world')")


# ── extract_pip_packages ──────────────────────────────────────────────────────


def test_extracts_third_party():
    src = "import requests\nfrom flask import Flask\nimport os\nimport sys"
    pkgs = extract_pip_packages([src])
    assert "requests" in pkgs
    assert "flask" in pkgs
    assert "os" not in pkgs
    assert "sys" not in pkgs


def test_known_alias():
    src = "from PIL import Image"
    pkgs = extract_pip_packages([src])
    assert "Pillow" in pkgs
    assert "PIL" not in pkgs


def test_empty_file():
    assert extract_pip_packages([]) == []


# ── parse_erroring_file ───────────────────────────────────────────────────────


def test_parse_python_traceback():
    stderr = 'Traceback...\n  File "main.py", line 42, in foo\nNameError: x'
    result = parse_erroring_file(stderr, {"main.py", "utils.py"})
    assert result == "main.py"


def test_parse_js_error():
    stderr = "/app/index.js:15\n  throw new Error('fail')"
    result = parse_erroring_file(stderr, {"index.js"})
    assert result == "index.js"


def test_unknown_file_returns_none():
    result = parse_erroring_file("some random error", {"main.py"})
    assert result is None


# ── validate_html ─────────────────────────────────────────────────────────────


def test_valid_html_ok():
    r = validate_html("<html><body><p>Hello</p></body></html>")
    assert r.ok
    assert r.mode == "html_validate"


def test_empty_html_ok():
    r = validate_html("")
    assert r.ok


# ── SandboxExecutor — subprocess mode (no Docker) ─────────────────────────────


@pytest.fixture()
def executor():
    return SandboxExecutor(timeout=10)


def test_run_python_success(executor, tmp_path: Path):
    """Python script that exits 0 → ok=True."""
    state = _make_state({
        "main": _make_artifact("main", "py", "print('hello')\n")
    })
    with patch.object(SandboxExecutor, "_docker_available", return_value=False), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="hello\n", stderr=""
        )
        result = executor.run(state)

    assert result.ok
    assert result.executed
    assert result.mode == "subprocess"


def test_run_python_failure(executor):
    state = _make_state({
        "main": _make_artifact("main", "py", "raise RuntimeError('boom')")
    })
    with patch.object(SandboxExecutor, "_docker_available", return_value=False), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="",
            stderr='Traceback...\n  File "main.py", line 1\nRuntimeError: boom'
        )
        result = executor.run(state)

    assert not result.ok
    assert result.exit_code == 1
    assert "RuntimeError" in result.stderr


def test_run_timeout(executor):
    state = _make_state({
        "main": _make_artifact("main", "py", "import time; time.sleep(999)")
    })
    with patch.object(SandboxExecutor, "_docker_available", return_value=False), \
         patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10)):
        result = executor.run(state)

    assert not result.ok
    assert "Timeout" in result.error


def test_run_web_project(executor):
    state = _make_state({
        "index.html": _make_artifact("index", "html",
                                     "<html><body>Game</body></html>"),
        "style.css":  _make_artifact("style", "css", "body { margin: 0 }"),
    })
    result = executor.run(state)
    assert result.mode == "html_validate"
    assert result.executed


def test_run_unknown_project_skipped(executor):
    state = _make_state({
        "README": _make_artifact("README", "md", "# doc")
    })
    result = executor.run(state)
    assert result.mode == "skipped"
    assert result.ok   # unknown → don't fail the pipeline


def test_server_process_import_check(executor):
    """Server process uses import-only check, not full execution."""
    src = "from flask import Flask\napp = Flask(__name__)\napp.run()"
    state = _make_state({"main": _make_artifact("main", "py", src)})
    with patch.object(SandboxExecutor, "_docker_available", return_value=False), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="import OK\n", stderr="")
        result = executor.run(state)

    called_cmd = mock_run.call_args[0][0]
    assert "import" in " ".join(called_cmd)
    assert result.ok


# ── Phase 3 sandbox repair integration ───────────────────────────────────────


def test_sandbox_repair_loop_calls_repairer():
    """When sandbox fails, Repairer is called and artifact updated."""
    from matrioska.pipeline.phase3 import _run_sandbox_with_repair
    from matrioska.core.state import FileArtifact

    art = FileArtifact(name="main", extension="py", order=1,
                       content="raise RuntimeError('bad')")
    art.status = "done"
    state = _make_state({"main": art})
    state.contracts = []

    cfg = load_config({
        "enable_sandbox": True,
        "sandbox_max_repairs": 1,
        "enable_vault": False,
    })

    fail_result = SandboxResult(
        ok=False, executed=True, exit_code=1,
        stderr='File "main.py", line 1\nRuntimeError: bad',
        mode="subprocess", entrypoint="main.py",
    )
    ok_result = SandboxResult(ok=True, executed=True, exit_code=0, mode="subprocess")

    mock_llm = MagicMock()

    with patch("matrioska.tools.sandbox.SandboxExecutor") as mock_executor_cls, \
         patch("matrioska.pipeline.phase3._repair_artifact", return_value=True) as mock_repair:
        mock_executor = MagicMock()
        mock_executor.run.side_effect = [fail_result, ok_result]
        mock_executor_cls.return_value = mock_executor

        result = _run_sandbox_with_repair(state, cfg, mock_llm, bus=None)

    assert mock_repair.called
    assert result["ok"]
    assert result.get("sandbox_repair_attempts") == 1

"""Implementation-agnostic test suite for CLI todo app."""
import json
import os
import subprocess
import sys
from pathlib import Path


def find_todo_script(project_dir: Path) -> Path:
    """Find the main todo script regardless of name."""
    candidates = list(project_dir.glob("todo*.py"))
    if not candidates:
        candidates = list(project_dir.glob("main*.py"))
    if not candidates:
        candidates = list(project_dir.glob("app*.py"))
    if not candidates:
        candidates = list(project_dir.glob("cli*.py"))
    assert candidates, f"No todo script found in {project_dir}"
    return candidates[0]


def test_add_task(project_dir: Path):
    script = find_todo_script(project_dir)
    tasks_file = project_dir / "tasks.json"
    if tasks_file.exists():
        tasks_file.unlink()

    result = subprocess.run(
        [sys.executable, str(script), "add", "Buy groceries"],
        capture_output=True, text=True, cwd=str(project_dir), timeout=10,
    )
    assert result.returncode == 0, f"add failed: {result.stderr}"
    assert tasks_file.exists(), "tasks.json not created"

    data = json.loads(tasks_file.read_text())
    assert isinstance(data, list), "tasks.json must contain a list"
    assert len(data) == 1
    assert data[0]["title"] == "Buy groceries"
    assert data[0]["done"] is False
    assert "id" in data[0]
    assert "created_at" in data[0]


def test_list_tasks(project_dir: Path):
    script = find_todo_script(project_dir)
    tasks_file = project_dir / "tasks.json"

    result = subprocess.run(
        [sys.executable, str(script), "list"],
        capture_output=True, text=True, cwd=str(project_dir), timeout=10,
    )
    assert result.returncode == 0, f"list failed: {result.stderr}"
    # Should output something (even if empty)
    assert len(result.stdout.strip()) > 0 or len(result.stderr.strip()) > 0


def test_mark_done(project_dir: Path):
    script = find_todo_script(project_dir)
    tasks_file = project_dir / "tasks.json"

    result = subprocess.run(
        [sys.executable, str(script), "done", "1"],
        capture_output=True, text=True, cwd=str(project_dir), timeout=10,
    )
    assert result.returncode == 0, f"done failed: {result.stderr}"

    data = json.loads(tasks_file.read_text())
    task = next((t for t in data if t["id"] == 1), None)
    assert task is not None, "task 1 not found"
    assert task["done"] is True, f"task 1 should be done, got {task}"


def test_delete_task(project_dir: Path):
    script = find_todo_script(project_dir)
    tasks_file = project_dir / "tasks.json"

    # Add another task
    subprocess.run(
        [sys.executable, str(script), "add", "Task to delete"],
        capture_output=True, text=True, cwd=str(project_dir), timeout=10,
    )

    result = subprocess.run(
        [sys.executable, str(script), "delete", "2"],
        capture_output=True, text=True, cwd=str(project_dir), timeout=10,
    )
    assert result.returncode == 0, f"delete failed: {result.stderr}"

    data = json.loads(tasks_file.read_text())
    ids = [t["id"] for t in data]
    assert 2 not in ids, "task 2 should be deleted"


def test_empty_list(project_dir: Path):
    script = find_todo_script(project_dir)
    tasks_file = project_dir / "tasks.json"
    if tasks_file.exists():
        tasks_file.unlink()

    result = subprocess.run(
        [sys.executable, str(script), "list"],
        capture_output=True, text=True, cwd=str(project_dir), timeout=10,
    )
    assert result.returncode == 0, f"list on empty failed: {result.stderr}"


def test_python_syntax(project_dir: Path):
    """All .py files must be valid Python."""
    for py_file in project_dir.glob("*.py"):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"Syntax error in {py_file.name}: {result.stderr}"

"""Implementation-agnostic test suite for CSV stats calculator."""
import csv
import subprocess
import sys
from pathlib import Path


def find_script(project_dir: Path) -> Path:
    candidates = list(project_dir.glob("stats*.py"))
    if not candidates:
        candidates = list(project_dir.glob("main*.py"))
    if not candidates:
        candidates = list(project_dir.glob("app*.py"))
    assert candidates, f"No stats script found in {project_dir}"
    return candidates[0]


def make_test_csv(project_dir: Path, rows: list[dict], name: str = "test.csv") -> Path:
    path = project_dir / name
    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return path


def test_basic_stats(project_dir: Path):
    script = find_script(project_dir)
    csv_path = make_test_csv(project_dir, [
        {"name": "Alice", "age": "30", "score": "85"},
        {"name": "Bob", "age": "25", "score": "90"},
        {"name": "Charlie", "age": "35", "score": "78"},
    ])

    result = subprocess.run(
        [sys.executable, str(script), str(csv_path.name)],
        capture_output=True, text=True, cwd=str(project_dir), timeout=10,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output = result.stdout
    assert "age" in output.lower() or "score" in output.lower(), f"No column found in output: {output[:200]}"
    assert "mean" in output.lower(), f"No 'mean' in output: {output[:200]}"
    assert "count" in output.lower(), f"No 'count' in output: {output[:200]}"
    # Should have 3 rows for count
    assert "3" in output, f"Expected count=3 in output: {output[:300]}"


def test_categorical_column(project_dir: Path):
    script = find_script(project_dir)
    csv_path = make_test_csv(project_dir, [
        {"name": "Alice", "color": "red"},
        {"name": "Bob", "color": "blue"},
        {"name": "Charlie", "color": "red"},
    ])

    result = subprocess.run(
        [sys.executable, str(script), str(csv_path.name)],
        capture_output=True, text=True, cwd=str(project_dir), timeout=10,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output = result.stdout
    # Should show value counts for "color"
    assert "red" in output.lower(), f"Expected 'red' in output: {output[:300]}"
    assert "blue" in output.lower(), f"Expected 'blue' in output: {output[:300]}"


def test_empty_file(project_dir: Path):
    script = find_script(project_dir)
    csv_path = make_test_csv(project_dir, [], name="empty.csv")

    result = subprocess.run(
        [sys.executable, str(script), str(csv_path.name)],
        capture_output=True, text=True, cwd=str(project_dir), timeout=10,
    )
    assert result.returncode == 0, f"Empty file should exit 0, got {result.returncode}"
    output = (result.stdout + result.stderr).lower()
    assert "empty" in output, f"Should mention 'empty': {output[:200]}"


def test_missing_file(project_dir: Path):
    script = find_script(project_dir)

    result = subprocess.run(
        [sys.executable, str(script), "nonexistent.csv"],
        capture_output=True, text=True, cwd=str(project_dir), timeout=10,
    )
    assert result.returncode != 0, f"Missing file should exit non-zero, got {result.returncode}"
    output = (result.stdout + result.stderr).lower()
    assert "not found" in output or "missing" in output or "no such" in output, \
        f"Should mention file missing: {output[:200]}"


def test_python_syntax(project_dir: Path):
    for py_file in project_dir.glob("*.py"):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"Syntax error in {py_file.name}: {result.stderr}"

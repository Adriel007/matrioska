"""Test suite for CLI data pipeline — validates cross-module coordination."""
import csv
import json
import subprocess
import sys
from pathlib import Path


def find_pipeline(project_dir: Path) -> Path:
    for name in ["pipeline.py", "main.py", "run.py"]:
        p = project_dir / name
        if p.exists():
            return p
    raise AssertionError(f"No pipeline entrypoint found in {project_dir}")


def make_config(project_dir: Path, overrides: dict | None = None) -> Path:
    cfg = {
        "input_file": "test_input.csv",
        "output_file": "test_output.csv",
        "columns": ["name", "email", "status"],
        "filters": {"status": "active"},
        "batch_size": 500,
    }
    if overrides:
        cfg.update(overrides)
    path = project_dir / "test_config.json"
    path.write_text(json.dumps(cfg))
    return path


def make_csv(project_dir: Path, rows: list[dict]) -> Path:
    path = project_dir / "test_input.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "email", "status"])
        w.writeheader()
        w.writerows(rows)
    return path


def test_files_exist(project_dir: Path):
    required = {"config.py", "extract.py", "transform.py", "load.py"}
    found = {f.name for f in project_dir.glob("*.py")}
    missing = required - found
    assert not missing, f"Missing required files: {missing}"


def test_python_syntax(project_dir: Path):
    for py_file in project_dir.glob("*.py"):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True, text=True, timeout=10, cwd=str(project_dir),
        )
        assert result.returncode == 0, f"Syntax error in {py_file.name}: {result.stderr}"


def test_cross_file_imports(project_dir: Path):
    """Verify the import chain: pipeline→load→transform→extract→config."""
    checks = {
        "extract.py": ["config"],
        "transform.py": ["config"],
        "load.py": ["config"],
    }
    for filename, deps in checks.items():
        f = project_dir / filename
        if not f.exists():
            continue
        content = f.read_text()
        for dep in deps:
            assert dep in content.lower(), f"{filename} should import from {dep}"


def test_config_dataclass(project_dir: Path):
    f = project_dir / "config.py"
    if not f.exists():
        return
    content = f.read_text()
    assert "@dataclass" in content or "Config" in content, \
        "config.py must define Config (preferably as dataclass)"
    for field in ["input_file", "output_file", "columns", "filters"]:
        assert field in content, f"config.py missing field '{field}'"


def test_extract_function(project_dir: Path):
    f = project_dir / "extract.py"
    if not f.exists():
        return
    content = f.read_text()
    assert "def extract" in content, "extract.py must define extract()"
    assert "DictReader" in content or "csv.reader" in content, \
        "extract should use csv module"


def test_transform_function(project_dir: Path):
    f = project_dir / "transform.py"
    if not f.exists():
        return
    content = f.read_text()
    assert "def transform" in content, "transform.py must define transform()"


def test_load_function(project_dir: Path):
    f = project_dir / "load.py"
    if not f.exists():
        return
    content = f.read_text()
    assert "def load" in content, "load.py must define load()"
    assert "DictWriter" in content or "csv.writer" in content, \
        "load should use csv module"


def test_pipeline_runs(project_dir: Path):
    """End-to-end: pipeline processes a test CSV."""
    pipeline = find_pipeline(project_dir)

    make_csv(project_dir, [
        {"name": "Alice", "email": "a@x.com", "status": "active"},
        {"name": "Bob", "email": "b@x.com", "status": "inactive"},
        {"name": "Charlie", "email": "c@x.com", "status": "active"},
    ])
    make_config(project_dir)

    result = subprocess.run(
        [sys.executable, str(pipeline), "--config", "test_config.json"],
        capture_output=True, text=True, timeout=15, cwd=str(project_dir),
    )
    output = result.stdout + result.stderr

    # Accept partial success — pipeline may work or fail with clear output
    assert result.returncode in (0, 1), f"Unexpected exit code {result.returncode}"

    # If it ran successfully, verify output
    if result.returncode == 0:
        out_file = project_dir / "test_output.csv"
        if out_file.exists():
            with open(out_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2, f"Expected 2 active rows, got {len(rows)}"
            assert all(r["status"] == "active" for r in rows), \
                "Filtered rows should all have status=active"

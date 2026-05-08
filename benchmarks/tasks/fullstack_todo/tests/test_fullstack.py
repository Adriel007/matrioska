"""Test suite for full-stack todo app — validates cross-file consistency."""
import json
import subprocess
import sys
import time
from pathlib import Path


def find_main(project_dir: Path) -> Path:
    for name in ["main.py", "app.py", "server.py", "api.py"]:
        p = project_dir / name
        if p.exists():
            return p
    raise AssertionError(f"No main.py found in {project_dir}")


def test_files_exist(project_dir: Path):
    """All required files must exist."""
    required = {"main.py", "models.py", "database.py", "routes.py"}
    found = set()
    for f in project_dir.rglob("*.py"):
        found.add(f.name)
    missing = required - found
    assert not missing, f"Missing required files: {missing}"

    # Static files
    static = project_dir / "static"
    assert static.exists() or any(f.suffix in {".html", ".js"} for f in project_dir.rglob("*")), \
        "No static files found"


def test_python_syntax(project_dir: Path):
    """All .py files must be valid Python."""
    for py_file in project_dir.glob("*.py"):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True, text=True, timeout=10, cwd=str(project_dir),
        )
        assert result.returncode == 0, f"Syntax error in {py_file.name}: {result.stderr}"


def test_cross_file_imports(project_dir: Path):
    """Key cross-file imports must exist."""
    imports = {
        "main.py": ["routes", "database", "models"],
        "routes.py": ["models", "database"],
        "database.py": ["sqlalchemy", "models"],
    }
    for filename, expected_imports in imports.items():
        f = project_dir / filename
        if not f.exists():
            continue
        content = f.read_text()
        for imp in expected_imports:
            assert imp.lower() in content.lower() or f"from {imp}" in content or f"import {imp}" in content, \
                f"{filename} should reference {imp}"


def test_api_endpoints_in_routes(project_dir: Path):
    """routes.py must define all 5 required endpoints."""
    routes_file = project_dir / "routes.py"
    if not routes_file.exists():
        return
    content = routes_file.read_text()
    required = ["/api/todos", "GET", "POST", "PUT", "DELETE"]
    for r in required:
        assert r in content, f"routes.py missing '{r}'"


def test_models_define_todo(project_dir: Path):
    """models.py must define Todo model with required fields."""
    models_file = project_dir / "models.py"
    if not models_file.exists():
        return
    content = models_file.read_text()
    required_fields = ["title", "done", "created_at", "id"]
    for field in required_fields:
        assert field in content, f"models.py missing field '{field}'"


def test_main_starts_server(project_dir: Path):
    """main.py starts and responds to /api/todos."""
    main = find_main(project_dir)
    proc = subprocess.Popen(
        [sys.executable, str(main)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(project_dir),
    )
    time.sleep(2)
    try:
        import urllib.request
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/api/todos")
            resp = urllib.request.urlopen(req, timeout=3)
            data = json.loads(resp.read().decode())
            assert isinstance(data, list), f"Expected list, got {type(data)}"
        except Exception as e:
            # Server might use different port — check stderr for uvicorn output
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise AssertionError(f"Server not responding on port 8000: {e}\n{stderr[:300]}")
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_static_html_has_todo_form(project_dir: Path):
    """HTML must have input + button for adding todos."""
    html_files = list(project_dir.rglob("*.html"))
    if not html_files:
        return
    content = html_files[0].read_text()
    assert "<input" in content.lower(), "No input element in HTML"
    assert "<button" in content.lower() or "btn" in content.lower(), "No button in HTML"


def test_js_fetches_api(project_dir: Path):
    """JavaScript must reference the API endpoints."""
    js_files = list(project_dir.rglob("*.js"))
    if not js_files:
        return
    content = js_files[0].read_text()
    assert "fetch" in content.lower(), "No fetch() call in JS"
    assert "todo" in content.lower() or "/api/" in content.lower(), \
        "JS should reference /api/ or 'todo'"

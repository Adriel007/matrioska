"""Test suite for web dashboard — validates cross-file API+UI coordination."""
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


SERVER = None
BASE = "http://127.0.0.1:8000"


def find_main(project_dir: Path) -> Path:
    for name in ["api.py", "main.py", "app.py", "server.py"]:
        p = project_dir / name
        if p.exists():
            return p
    raise AssertionError(f"No api.py found in {project_dir}")


def teardown_module():
    if SERVER:
        SERVER.kill()
        SERVER.wait(timeout=5)


def start_server(project_dir: Path):
    global SERVER
    main = find_main(project_dir)
    SERVER = subprocess.Popen(
        [sys.executable, str(main)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(project_dir),
    )
    for _ in range(15):
        try:
            urllib.request.urlopen(f"{BASE}/api/metrics", timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    SERVER.kill()
    raise RuntimeError("Server did not start")


def test_files_exist(project_dir: Path):
    required_py = {"api.py", "data.py"}
    found_py = {f.name for f in project_dir.glob("*.py")}
    missing = required_py - found_py
    assert not missing, f"Missing Python files: {missing}"

    static = project_dir / "static"
    if static.exists():
        static_files = list(static.rglob("*"))
        assert static_files, "static/ directory is empty"


def test_python_syntax(project_dir: Path):
    for py_file in project_dir.glob("*.py"):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True, text=True, timeout=10, cwd=str(project_dir),
        )
        assert result.returncode == 0, f"Syntax error in {py_file.name}: {result.stderr}"


def test_cross_file_imports(project_dir: Path):
    """api.py must import from data.py."""
    api_file = project_dir / "api.py"
    data_file = project_dir / "data.py"
    if not api_file.exists() or not data_file.exists():
        return
    api_content = api_file.read_text()
    assert "data" in api_content.lower(), "api.py should import from data module"


def test_data_has_functions(project_dir: Path):
    f = project_dir / "data.py"
    if not f.exists():
        return
    content = f.read_text()
    assert "def generate_metrics" in content, "data.py must define generate_metrics()"
    assert "def compute_summary" in content or "def get_summary" in content, \
        "data.py must define summary function"


def test_api_has_endpoints(project_dir: Path):
    f = project_dir / "api.py"
    if not f.exists():
        return
    content = f.read_text()
    assert "/api/metrics" in content, "api.py missing /api/metrics endpoint"
    assert "/api/summary" in content, "api.py missing /api/summary endpoint"


def test_static_html_has_chartjs(project_dir: Path):
    html_files = list(project_dir.rglob("*.html"))
    if not html_files:
        return
    content = html_files[0].read_text()
    assert "chart.js" in content.lower() or "chartjs" in content.lower(), \
        "HTML should load Chart.js"


def test_js_references_api(project_dir: Path):
    js_files = list(project_dir.rglob("*.js"))
    if not js_files:
        return
    content = js_files[0].read_text()
    assert "fetch" in content.lower(), "JS must use fetch()"
    assert "metric" in content.lower() or "summary" in content.lower() or "/api/" in content.lower(), \
        "JS should reference API endpoints"


def test_server_metrics_endpoint(project_dir: Path):
    """End-to-end: server serves metrics and summary."""
    try:
        start_server(project_dir)

        # Metrics endpoint
        req = urllib.request.Request(f"{BASE}/api/metrics")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        assert isinstance(data, dict)
        for field in ["cpu", "memory"]:
            if field in data:
                assert isinstance(data[field], list), f"{field} should be a list"

        # Summary endpoint
        req = urllib.request.Request(f"{BASE}/api/summary")
        with urllib.request.urlopen(req, timeout=3) as resp:
            summary = json.loads(resp.read().decode())
        assert isinstance(summary, dict)
        assert "total_requests" in summary or "avg_cpu" in summary, \
            "Summary should have metrics"
    finally:
        if SERVER:
            SERVER.kill()
            SERVER.wait(timeout=5)

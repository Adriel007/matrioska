"""Implementation-agnostic test suite for Book API.

Starts the server as a subprocess and tests via HTTP requests.
"""
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


SERVER_PROCESS = None
BASE_URL = "http://127.0.0.1:8765"


def find_main_script(project_dir: Path) -> Path:
    candidates = list(project_dir.glob("main*.py"))
    if not candidates:
        candidates = list(project_dir.glob("app*.py"))
    if not candidates:
        candidates = list(project_dir.glob("server*.py"))
    if not candidates:
        candidates = list(project_dir.glob("api*.py"))
    assert candidates, f"No main script found in {project_dir}"
    return candidates[0]


def http_request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = {}
        try:
            error_body = json.loads(e.read().decode())
        except Exception:
            pass
        return e.code, error_body


def setup_module():
    """Start the server once for all tests."""
    global SERVER_PROCESS, BASE_URL
    project_dir = Path.cwd()  # Set by test runner

    script = find_main_script(project_dir)
    SERVER_PROCESS = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(project_dir),
    )
    # Wait for server
    for _ in range(20):
        try:
            urllib.request.urlopen(f"{BASE_URL}/books", timeout=1)
            break
        except Exception:
            time.sleep(0.3)
    else:
        SERVER_PROCESS.kill()
        raise RuntimeError("Server did not start")


def teardown_module():
    if SERVER_PROCESS:
        SERVER_PROCESS.kill()
        SERVER_PROCESS.wait(timeout=5)


def test_create_book():
    status, data = http_request("POST", "/books", {
        "title": "Test Book", "author": "Author Name", "year": 2024,
    })
    assert status == 201, f"Expected 201, got {status}: {data}"
    assert data["id"] == 1
    assert data["title"] == "Test Book"
    assert data["done" if "done" in data else None] is None  # no done field


def test_list_books():
    status, data = http_request("GET", "/books")
    assert status == 200, f"Expected 200, got {status}: {data}"
    assert isinstance(data, list)
    assert len(data) >= 1


def test_get_book():
    status, data = http_request("GET", "/books/1")
    assert status == 200, f"Expected 200, got {status}: {data}"
    assert data["id"] == 1
    assert data["title"] == "Test Book"


def test_get_nonexistent():
    status, data = http_request("GET", "/books/999")
    assert status == 404, f"Expected 404, got {status}"


def test_update_book():
    status, data = http_request("PUT", "/books/1", {
        "title": "Updated Title", "author": "Author Name", "year": 2025,
    })
    assert status == 200, f"Expected 200, got {status}: {data}"
    assert data["title"] == "Updated Title"
    assert data["year"] == 2025


def test_delete_book():
    status, data = http_request("DELETE", "/books/1")
    assert status == 200, f"Expected 200 (or 204), got {status}"

    # Verify gone
    status, _ = http_request("GET", "/books/1")
    assert status == 404, f"Expected 404 after delete, got {status}"


def test_validation():
    status, data = http_request("POST", "/books", {"title": ""})
    assert status in (400, 422), f"Expected 400/422, got {status}: {data}"

    status, data = http_request("POST", "/books", {"title": "X", "year": -5})
    assert status in (400, 422), f"Expected 400/422 for bad year, got {status}: {data}"


def test_python_syntax(project_dir: Path):
    """All .py files must be valid Python."""
    for py_file in project_dir.glob("*.py"):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"Syntax error in {py_file.name}: {result.stderr}"

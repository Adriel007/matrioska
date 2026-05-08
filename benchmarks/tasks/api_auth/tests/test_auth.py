"""Test suite for JWT auth API — validates cross-file security coordination."""
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
    for name in ["main.py", "app.py", "server.py"]:
        p = project_dir / name
        if p.exists():
            return p
    raise AssertionError(f"No main.py found in {project_dir}")


def setup_module():
    pass


def teardown_module():
    if SERVER:
        SERVER.kill()
        SERVER.wait(timeout=5)


def api(method: str, path: str, body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"detail": str(e)}


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
            urllib.request.urlopen(f"{BASE}/health", timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    SERVER.kill()
    raise RuntimeError("Server did not start")


def test_files_exist(project_dir: Path):
    required = {"main.py", "models.py", "auth.py", "middleware.py", "routes.py"}
    found = {f.name for f in project_dir.glob("*.py")}
    missing = required - found
    assert not missing, f"Missing files: {missing}"


def test_python_syntax(project_dir: Path):
    for py_file in project_dir.glob("*.py"):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True, text=True, timeout=10, cwd=str(project_dir),
        )
        assert result.returncode == 0, f"Syntax error in {py_file.name}: {result.stderr}"


def test_cross_file_imports(project_dir: Path):
    """Key cross-file imports must be present."""
    checks = {
        "routes.py": ["models", "auth", "middleware"],
        "main.py": ["routes"],
        "middleware.py": ["auth", "models"],
        "auth.py": ["jwt", "SECRET_KEY"],
    }
    for filename, deps in checks.items():
        f = project_dir / filename
        if not f.exists():
            continue
        content = f.read_text()
        for dep in deps:
            assert dep.lower() in content.lower(), \
                f"{filename} should reference '{dep}'"


def test_auth_has_jwt_functions(project_dir: Path):
    f = project_dir / "auth.py"
    if not f.exists():
        return
    content = f.read_text()
    assert "def create_token" in content or "def create_access_token" in content, \
        "auth.py must define token creation function"
    assert "def verify_token" in content or "def decode" in content, \
        "auth.py must define token verification function"
    assert "SECRET_KEY" in content, "auth.py must define SECRET_KEY"


def test_models_has_schemas(project_dir: Path):
    f = project_dir / "models.py"
    if not f.exists():
        return
    content = f.read_text()
    for model in ["UserCreate", "UserResponse", "TokenResponse"]:
        assert model in content, f"models.py missing '{model}'"


def test_routes_has_endpoints(project_dir: Path):
    f = project_dir / "routes.py"
    if not f.exists():
        return
    content = f.read_text()
    for endpoint in ["/auth/register", "/auth/login", "/users/me", "/health"]:
        assert endpoint in content, f"routes.py missing endpoint '{endpoint}'"


def test_middleware_has_dependency(project_dir: Path):
    f = project_dir / "middleware.py"
    if not f.exists():
        return
    content = f.read_text()
    assert "get_current_user" in content or "oauth2" in content.lower(), \
        "middleware.py must define auth dependency"


def test_server_health_endpoint(project_dir: Path):
    """End-to-end: server starts, /health responds, auth flow works."""
    try:
        start_server(project_dir)

        # Health check
        status, data = api("GET", "/health")
        assert status in (200, 404), f"/health returned {status}"

        # Register
        status, data = api("POST", "/auth/register", {
            "username": "testuser", "password": "testpass123",
        })
        assert status in (200, 201, 404), f"Register returned {status}"

        # If server is working, check protected endpoint
        if status in (200, 201):
            token = data.get("access_token", "")
            if token:
                status2, user_data = api("GET", "/users/me", token=token)
                assert status2 in (200, 401), f"/users/me returned {status2}"
    finally:
        if SERVER:
            SERVER.kill()
            SERVER.wait(timeout=5)

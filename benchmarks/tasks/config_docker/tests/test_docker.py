"""Implementation-agnostic test suite for Docker Compose config."""
import subprocess
import sys
from pathlib import Path


try:
    import yaml
except ImportError:
    yaml = None


def find_compose_file(project_dir: Path) -> Path:
    candidates = list(project_dir.glob("docker-compose*.yml")) + \
                 list(project_dir.glob("docker-compose*.yaml")) + \
                 list(project_dir.glob("compose*.yml")) + \
                 list(project_dir.glob("compose*.yaml"))
    assert candidates, f"No docker-compose file found in {project_dir}"
    return candidates[0]


def load_yaml(path: Path) -> dict:
    if yaml:
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        # Manual lightweight parse for simple validation
        import json
        result = subprocess.run(
            [sys.executable, "-c", f"import yaml; print(yaml.safe_load(open('{path}')))"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise AssertionError(f"YAML parse error: {result.stderr}")
        return eval(result.stdout)  # dict from repr


def test_compose_file_exists(project_dir: Path):
    f = find_compose_file(project_dir)
    assert f.exists()


def test_compose_valid_yaml(project_dir: Path):
    f = find_compose_file(project_dir)
    data = load_yaml(f)
    assert isinstance(data, dict), f"Top-level should be dict, got {type(data)}"


def test_has_services(project_dir: Path):
    f = find_compose_file(project_dir)
    data = load_yaml(f)
    assert "services" in data, "Missing 'services' key"


def test_has_app_service(project_dir: Path):
    f = find_compose_file(project_dir)
    data = load_yaml(f)
    services = data.get("services", {})
    assert "app" in services, "Missing 'app' service"


def test_has_db_service(project_dir: Path):
    f = find_compose_file(project_dir)
    data = load_yaml(f)
    services = data.get("services", {})
    assert "db" in services, "Missing 'db' service"


def test_db_is_postgres(project_dir: Path):
    f = find_compose_file(project_dir)
    data = load_yaml(f)
    db = data.get("services", {}).get("db", {})
    image = db.get("image", "")
    assert "postgres" in image.lower(), f"Expected PostgreSQL image, got: {image}"


def test_app_port(project_dir: Path):
    f = find_compose_file(project_dir)
    data = load_yaml(f)
    app = data.get("services", {}).get("app", {})
    ports = app.get("ports", [])
    assert ports, "app service has no ports"
    # Check port 8000 is exposed somewhere
    port_strs = [str(p) for p in ports]
    assert any("8000" in p for p in port_strs), f"Port 8000 not found in {port_strs}"


def test_db_volume(project_dir: Path):
    f = find_compose_file(project_dir)
    data = load_yaml(f)
    db = data.get("services", {}).get("db", {})
    # Either service-level volumes or top-level volumes
    has_vol = bool(db.get("volumes")) or bool(data.get("volumes"))
    assert has_vol, "No volumes configured for db persistence"


def test_app_depends_on_db(project_dir: Path):
    f = find_compose_file(project_dir)
    data = load_yaml(f)
    app = data.get("services", {}).get("app", {})
    depends = app.get("depends_on", [])
    assert depends, "app should depend_on db"
    dep_names = [d if isinstance(d, str) else list(d.keys())[0] if isinstance(d, dict) else str(d) for d in depends]
    assert "db" in dep_names or any("db" in str(d) for d in depends), \
        f"app should depend on db, got: {depends}"


def test_network(project_dir: Path):
    f = find_compose_file(project_dir)
    data = load_yaml(f)
    # Either service-level networks or top-level networks
    has_net = bool(data.get("networks"))
    for svc in data.get("services", {}).values():
        if svc.get("networks"):
            has_net = True
    assert has_net, "No networks defined"


def test_dockerignore_exists(project_dir: Path):
    ignore_file = project_dir / ".dockerignore"
    assert ignore_file.exists(), ".dockerignore not found"


def test_dockerignore_content(project_dir: Path):
    ignore_file = project_dir / ".dockerignore"
    content = ignore_file.read_text()
    assert "__pycache__" in content or "pycache" in content, \
        "Should ignore __pycache__"
    assert ".env" in content, "Should ignore .env"
    assert ".git" in content, "Should ignore .git"

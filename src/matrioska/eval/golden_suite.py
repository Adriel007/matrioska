"""
Golden regression suite (§4.8).

A fixed set of 30 tasks with expected architectures and quality bars.
CI runs these on every commit to catch regressions.

The suite doubles as the DSPy training set — successful (task, architecture)
pairs become optimized few-shot examples for the Architect prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoldenTask:
    """A single golden task with expected output shape."""
    id: str
    task: str
    category: str  # cli, web, api, data, config, fullstack
    expected_file_count: tuple[int, int]  # (min, max)
    expected_extensions: List[str]
    min_shared_state_keys: int = 1
    tags: List[str] = field(default_factory=list)
    notes: str = ""


# ── Golden Tasks (30 tasks across 6 categories) ──────────────────────────────

GOLDEN_TASKS: List[GoldenTask] = [
    # ── CLI (5 tasks) ─────────────────────────────────────────────────────
    GoldenTask("cli-01", "Create a Python CLI todo app with SQLite storage",
               "cli", (3, 6), ["py", "json"], 3, ["python", "cli", "sqlite"]),
    GoldenTask("cli-02", "Write a bash script that monitors disk usage and sends alerts",
               "cli", (1, 3), ["sh"], 1, ["bash", "monitoring"]),
    GoldenTask("cli-03", "Create a file renaming utility that supports regex patterns",
               "cli", (2, 4), ["py"], 2, ["python", "cli", "files"]),
    GoldenTask("cli-04", "Build a markdown-to-HTML converter with YAML frontmatter support",
               "cli", (2, 5), ["py", "html"], 2, ["python", "markdown"]),
    GoldenTask("cli-05", "Create a git commit message linter that enforces conventional commits",
               "cli", (2, 4), ["py", "json"], 2, ["python", "git", "cli"]),

    # ── Web (5 tasks) ─────────────────────────────────────────────────────
    GoldenTask("web-01", "Build a responsive landing page with Tailwind CSS",
               "web", (2, 3), ["html", "css"], 1, ["html", "css", "tailwind"]),
    GoldenTask("web-02", "Create a JavaScript image gallery with lightbox",
               "web", (3, 5), ["html", "css", "js"], 2, ["javascript", "frontend"]),
    GoldenTask("web-03", "Build a real-time chat UI with WebSocket support",
               "web", (3, 5), ["html", "css", "js"], 3, ["javascript", "websocket"]),
    GoldenTask("web-04", "Create a dashboard with charts using Chart.js",
               "web", (3, 5), ["html", "css", "js"], 2, ["javascript", "dashboard"]),
    GoldenTask("web-05", "Build an interactive form wizard with validation",
               "web", (3, 5), ["html", "css", "js"], 2, ["javascript", "forms"]),

    # ── API (5 tasks) ─────────────────────────────────────────────────────
    GoldenTask("api-01", "Create a FastAPI REST API for a book inventory",
               "api", (3, 6), ["py", "json"], 4, ["python", "fastapi", "rest"]),
    GoldenTask("api-02", "Build a rate-limited URL shortener API",
               "api", (3, 5), ["py", "json"], 4, ["python", "api", "fastapi"]),
    GoldenTask("api-03", "Create an Express.js API with JWT authentication",
               "api", (4, 8), ["js", "json"], 5, ["javascript", "express", "auth"]),
    GoldenTask("api-04", "Build a file upload API with metadata extraction",
               "api", (3, 6), ["py", "json"], 4, ["python", "api", "files"]),
    GoldenTask("api-05", "Create a GraphQL API for a social media feed",
               "api", (4, 8), ["py", "json", "graphql"], 5, ["python", "graphql"]),

    # ── Data (5 tasks) ────────────────────────────────────────────────────
    GoldenTask("data-01", "Build a CSV to SQLite ETL pipeline with validation",
               "data", (3, 5), ["py", "json"], 3, ["python", "etl", "sqlite"]),
    GoldenTask("data-02", "Create a data anonymization script for GDPR compliance",
               "data", (2, 4), ["py"], 2, ["python", "data", "privacy"]),
    GoldenTask("data-03", "Build a log parser that extracts structured data from Apache logs",
               "data", (2, 4), ["py"], 2, ["python", "parsing", "logs"]),
    GoldenTask("data-04", "Create a JSON Schema validator for API payloads",
               "data", (2, 4), ["py", "json"], 2, ["python", "validation", "json"]),
    GoldenTask("data-05", "Build a simple recommendation engine using collaborative filtering",
               "data", (3, 5), ["py"], 3, ["python", "ml", "data"]),

    # ── Config (5 tasks) ──────────────────────────────────────────────────
    GoldenTask("cfg-01", "Create a Docker Compose setup for a Python web app + PostgreSQL",
               "config", (2, 4), ["yml", "yaml", "conf"], 1, ["docker", "devops"]),
    GoldenTask("cfg-02", "Generate a comprehensive .gitignore for a Python project",
               "config", (1, 2), ["gitignore"], 0, ["git", "config"]),
    GoldenTask("cfg-03", "Create a GitHub Actions CI pipeline for Python linting and testing",
               "config", (1, 2), ["yml"], 1, ["ci", "github-actions"]),
    GoldenTask("cfg-04", "Generate Terraform config for a basic AWS S3 + Lambda setup",
               "config", (2, 4), ["tf", "json"], 2, ["terraform", "aws"]),
    GoldenTask("cfg-05", "Create an nginx configuration for a reverse proxy with SSL",
               "config", (1, 2), ["conf"], 1, ["nginx", "devops"]),

    # ── Full-stack (5 tasks) ──────────────────────────────────────────────
    GoldenTask("fs-01", "Build a full-stack todo app: FastAPI backend + vanilla JS frontend",
               "fullstack", (4, 8), ["py", "html", "js", "css"], 5, ["fullstack", "fastapi"]),
    GoldenTask("fs-02", "Create a blog engine with markdown posts and SQLite",
               "fullstack", (4, 8), ["py", "html", "css", "js"], 5, ["fullstack", "blog"]),
    GoldenTask("fs-03", "Build a polling app with real-time results via WebSocket",
               "fullstack", (5, 10), ["py", "html", "js", "css", "json"], 6, ["fullstack", "websocket"]),
    GoldenTask("fs-04", "Create a file sharing service with expiring links",
               "fullstack", (4, 8), ["py", "html", "js", "css"], 5, ["fullstack", "files"]),
    GoldenTask("fs-05", "Build a personal dashboard with weather, RSS, and bookmarks",
               "fullstack", (4, 8), ["py", "html", "js", "css", "json"], 5, ["fullstack", "dashboard"]),
]


def get_golden_tasks(category: Optional[str] = None) -> List[GoldenTask]:
    """Get golden tasks, optionally filtered by category."""
    if category is None:
        return GOLDEN_TASKS
    return [t for t in GOLDEN_TASKS if t.category == category]


def get_categories() -> List[str]:
    return sorted(set(t.category for t in GOLDEN_TASKS))


def evaluate_result(task: GoldenTask, result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a run result against a golden task's expected shape.

    Returns a dict with pass/fail for each criterion.
    """
    arch = result.get("architecture")
    artifacts = result.get("artifacts", [])

    checks = {}

    # File count
    n_files = len(artifacts)
    min_f, max_f = task.expected_file_count
    checks["file_count"] = {
        "pass": min_f <= n_files <= max_f,
        "expected": f"{min_f}-{max_f}", "actual": n_files,
    }

    # Extensions
    actual_exts = sorted(set(a.extension for a in artifacts if hasattr(a, 'extension')))
    checks["extensions"] = {
        "pass": all(ext in actual_exts for ext in task.expected_extensions),
        "expected": task.expected_extensions, "actual": actual_exts,
    }

    # Shared state keys
    ss = result.get("shared_state", {})
    checks["shared_state"] = {
        "pass": len(ss) >= task.min_shared_state_keys,
        "expected_min": task.min_shared_state_keys, "actual": len(ss),
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {"task_id": task.id, "pass": all_pass, "checks": checks}

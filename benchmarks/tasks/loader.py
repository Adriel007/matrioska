"""Task loader — discovers and loads benchmark tasks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    id: str
    category: str
    prompt: str
    contract: str
    expected_file_count: tuple[int, int]
    expected_extensions: list[str]
    min_shared_state_keys: int
    source_files: list[str]
    complexity: str
    tags: list[str]
    task_dir: Path
    test_dir: Path

    @property
    def test_files(self) -> list[Path]:
        return sorted(self.test_dir.glob("test_*.py"))


def load_task(task_dir: Path) -> BenchmarkTask:
    """Load a single benchmark task from its directory."""
    meta_path = task_dir / "meta.json"
    prompt_path = task_dir / "prompt.md"
    contract_path = task_dir / "CONTRACT"
    test_dir = task_dir / "tests"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {task_dir}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt.md in {task_dir}")

    meta = json.loads(meta_path.read_text())

    return BenchmarkTask(
        id=meta["id"],
        category=meta["category"],
        prompt=prompt_path.read_text().strip(),
        contract=contract_path.read_text().strip() if contract_path.exists() else "",
        expected_file_count=tuple(meta["expected_file_count"]),
        expected_extensions=meta["expected_extensions"],
        min_shared_state_keys=meta.get("min_shared_state_keys", 0),
        source_files=meta.get("source_files", []),
        complexity=meta.get("complexity", "simple"),
        tags=meta.get("tags", []),
        task_dir=task_dir,
        test_dir=test_dir,
    )


def load_all_tasks(tasks_root: Optional[Path] = None) -> list[BenchmarkTask]:
    """Load all benchmark tasks from the tasks directory."""
    if tasks_root is None:
        tasks_root = Path(__file__).parent

    tasks = []
    for task_dir in sorted(tasks_root.iterdir()):
        if task_dir.is_dir() and (task_dir / "meta.json").exists():
            tasks.append(load_task(task_dir))
    return tasks


def get_tasks_by_category(tasks: list[BenchmarkTask], category: str) -> list[BenchmarkTask]:
    return [t for t in tasks if t.category == category]

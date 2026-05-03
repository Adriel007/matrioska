"""
Procedural memory — DSPy-compiled patterns and user preferences.

The longest-term memory tier (§4.5).  Stores:
  - DSPy-compiled prompt templates (few-shot examples that work)
  - Architecture patterns that succeeded
  - User preferences from MATRIOSKA.md (evolving)
  - File-ordering heuristics

This module is a scaffold — full DSPy integration requires a golden
task suite and compilation loop (see eval/golden_suite.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("matrioska.memory.procedural")


@dataclass
class Pattern:
    """A reusable architecture/implementation pattern."""
    name: str
    description: str
    template: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    times_used: int = 0
    tags: List[str] = field(default_factory=list)


class ProceduralMemory:
    """DSPy-compiled procedural memory + user preferences.

    Reads MATRIOSKA.md for user preferences and accumulates successful
    architecture patterns over time.  The DSPy compiler uses these as
    optimized few-shot examples.
    """

    PROJECT_MEMORY_CAP = 8 * 1024  # bytes

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.patterns_file = self.work_dir / "knowledge" / "procedural_patterns.json"
        self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
        self._patterns: Dict[str, Pattern] = {}
        self._load()

    # ── Project memory ───────────────────────────────────────────────────

    @property
    def matrioska_md_path(self) -> Path:
        return self.work_dir / "MATRIOSKA.md"

    def ensure_project_memory(self) -> Path:
        if not self.matrioska_md_path.exists():
            self.matrioska_md_path.write_text(MATRIOSKA_MD_TEMPLATE, encoding="utf-8")
        return self.matrioska_md_path

    def read_project_memory(self) -> str:
        if not self.matrioska_md_path.exists():
            return ""
        data = self.matrioska_md_path.read_text(encoding="utf-8")
        enc = data.encode("utf-8")
        if len(enc) > self.PROJECT_MEMORY_CAP:
            data = enc[:self.PROJECT_MEMORY_CAP].decode("utf-8", errors="ignore")
            data += "\n\n[... truncated ...]"
        return data

    def update_project_memory(self, section: str, content: str) -> None:
        """Add or update a section in MATRIOSKA.md."""
        current = self.matrioska_md_path.read_text(encoding="utf-8") if self.matrioska_md_path.exists() else MATRIOSKA_MD_TEMPLATE
        marker = f"## {section}"
        if marker in current:
            lines = current.split("\n")
            start = next(i for i, l in enumerate(lines) if marker in l)
            end = next((i for i, l in enumerate(lines[start+1:], start+1) if l.startswith("## ")), len(lines))
            new = lines[:start+1] + [f"\n{content}\n"] + lines[end:]
            current = "\n".join(new)
        else:
            current += f"\n\n{marker}\n{content}\n"
        self.matrioska_md_path.write_text(current, encoding="utf-8")

    # ── Patterns ─────────────────────────────────────────────────────────

    def add_pattern(self, pattern: Pattern) -> None:
        self._patterns[pattern.name] = pattern
        self._save()

    def get_patterns_for(self, tags: List[str], min_success: float = 0.5) -> List[Pattern]:
        scored = []
        for p in self._patterns.values():
            if p.success_rate < min_success:
                continue
            overlap = len(set(p.tags) & set(tags))
            if overlap > 0:
                scored.append((overlap, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored]

    def record_pattern_outcome(self, name: str, success: bool) -> None:
        if name not in self._patterns:
            return
        p = self._patterns[name]
        p.times_used += 1
        p.success_rate = (
            p.success_rate * (p.times_used - 1) + (1.0 if success else 0.0)
        ) / p.times_used
        self._save()

    def compile_prompt(self, role: str, base_prompt: str) -> str:
        """Inject the best patterns as few-shot examples into a prompt."""
        patterns = [p for p in self._patterns.values() if p.success_rate >= 0.5]
        if not patterns:
            return base_prompt

        examples = []
        for p in sorted(patterns, key=lambda x: -x.success_rate)[:3]:
            examples.append(
                f"Example pattern: {p.name}\n{p.description}\n"
                f"Template: {json.dumps(p.template, indent=2)}"
            )

        separator = "\n\n---\n\n"
        return f"{base_prompt}{separator}## Proven Patterns\n{separator.join(examples)}"

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.patterns_file.exists():
            try:
                data = json.loads(self.patterns_file.read_text(encoding="utf-8"))
                self._patterns = {k: Pattern(**v) for k, v in data.items()}
            except Exception as e:
                logger.warning("Failed to load procedural patterns: %s", e)

    def _save(self) -> None:
        data = {k: vars(v) for k, v in self._patterns.items()}
        self.patterns_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))


MATRIOSKA_MD_TEMPLATE = """# Matrioska Project Memory

This file is auto-loaded into every agent prompt (capped ~8KB).
Edit it to encode stable preferences that should persist across runs.

## Preferences
- (example) Always use type hints in Python.
- (example) Prefer httpx over requests.

## Conventions
- (example) One class per file when possible.

## Stack
- (example) Backend: FastAPI + SQLite; Frontend: vanilla JS + Tailwind CDN.
"""

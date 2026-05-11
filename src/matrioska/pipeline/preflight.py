"""
Pre-flight: read user instructions and scan existing code before Phase 1.

Implements two Claude Code-inspired capabilities:
  - MATRIOSKA.md / CLAUDE.md injection: user conventions fed to the Architect
  - Codebase scan: existing files in the project dir injected as context,
    enabling incremental tasks ("add feature X to this project")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("matrioska.pipeline.preflight")

_CODE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".rs", ".java", ".rb", ".php",
    ".sql", ".sh", ".yaml", ".yml", ".toml", ".json",
}
_SKIP_DIRS = {
    ".venv", "venv", "node_modules", "__pycache__",
    ".git", ".mypy_cache", "dist", "build", ".pytest_cache",
    "matrioska_checkpoints", "logs", "knowledge",
}
_MAX_FILE_CHARS = 3_000   # per file
_MAX_TOTAL_CHARS = 24_000  # total context budget


@dataclass
class PreflightContext:
    """Results of the pre-flight scan."""
    user_instructions: str = ""          # from MATRIOSKA.md / CLAUDE.md
    instructions_source: str = ""        # path of the file that was read
    existing_files: Dict[str, str] = field(default_factory=dict)  # rel_path → content
    existing_summary: str = ""           # compact listing for prompts

    @property
    def has_instructions(self) -> bool:
        return bool(self.user_instructions)

    @property
    def has_existing_code(self) -> bool:
        return bool(self.existing_files)

    def architect_context_block(self) -> str:
        """Format context for injection into the Architect system prompt."""
        parts: List[str] = []

        if self.user_instructions:
            parts.append(
                f"USER INSTRUCTIONS (from {self.instructions_source}):\n"
                f"{self.user_instructions}"
            )

        if self.existing_files:
            lines = [f"EXISTING CODEBASE ({len(self.existing_files)} files):"]
            for rel, content in self.existing_files.items():
                preview = content[:400].rstrip()
                if len(content) > 400:
                    preview += "\n    ... (truncated)"
                lines.append(f"\n  ── {rel} ──\n{preview}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)


def run_preflight(
    work_dir: Path,
    project_dir: Optional[Path] = None,
) -> PreflightContext:
    """Execute pre-flight: read instructions and scan existing code.

    Search order for MATRIOSKA.md / CLAUDE.md:
      project_dir → work_dir → cwd

    Existing code scan:
      project_dir (if given) or work_dir, skipping internal matrioska dirs.
    """
    ctx = PreflightContext()

    # ── 1. User instructions ──────────────────────────────────────────────
    instruction_candidates = []
    if project_dir and project_dir.exists():
        instruction_candidates.append(project_dir)
    instruction_candidates.append(work_dir)
    instruction_candidates.append(Path.cwd())

    for search_dir in instruction_candidates:
        for name in ("MATRIOSKA.md", "CLAUDE.md"):
            f = search_dir / name
            if f.exists() and f.stat().st_size > 0:
                ctx.user_instructions = f.read_text(encoding="utf-8", errors="ignore")[:6_000]
                ctx.instructions_source = str(f.relative_to(Path.cwd()) if f.is_relative_to(Path.cwd()) else f)
                logger.info("Pre-flight: loaded instructions from %s (%d chars)",
                            ctx.instructions_source, len(ctx.user_instructions))
                break
        if ctx.has_instructions:
            break

    # ── 2. Existing code scan ─────────────────────────────────────────────
    scan_root = project_dir if (project_dir and project_dir.exists()) else work_dir
    if scan_root.exists():
        total_chars = 0
        for path in sorted(scan_root.rglob("*")):
            if total_chars >= _MAX_TOTAL_CHARS:
                break
            if path.suffix not in _CODE_EXTENSIONS:
                continue
            if any(skip in path.parts for skip in _SKIP_DIRS):
                continue
            if not path.is_file() or path.stat().st_size == 0:
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                content = content[:_MAX_FILE_CHARS]
                rel = str(path.relative_to(scan_root))
                ctx.existing_files[rel] = content
                total_chars += len(content)
            except Exception:
                continue

        if ctx.existing_files:
            logger.info(
                "Pre-flight: found %d existing file(s) in %s (%.1fK chars total)",
                len(ctx.existing_files), scan_root,
                total_chars / 1000,
            )

    # ── 3. Compact summary for logging ───────────────────────────────────
    if ctx.existing_files:
        ctx.existing_summary = ", ".join(
            f"{p} ({len(c)} chars)" for p, c in list(ctx.existing_files.items())[:10]
        )
        if len(ctx.existing_files) > 10:
            ctx.existing_summary += f" … (+{len(ctx.existing_files) - 10} more)"

    return ctx

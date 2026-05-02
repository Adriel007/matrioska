"""
Episodic memory — Obsidian-compatible past-run storage with keyword + dense retrieval.

Each run is stored as a Markdown note with YAML frontmatter under
knowledge/runs/.  The retrieval system uses:
  1. Keyword + tag matching (fast, deterministic)
  2. Dense embeddings via ChromaDB (semantic similarity)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.matrioska.core.state import Architecture, FileArtifact

logger = logging.getLogger("matrioska.memory.episodic")

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "for",
    "with",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "is",
    "are",
    "be",
    "or",
    "as",
    "from",
    "that",
    "this",
    "it",
    "create",
    "build",
    "make",
    "generate",
    "add",
}


@dataclass
class RunNote:
    """Metadata for a single past run."""

    path: Path
    frontmatter: Dict[str, Any]
    body: str
    score: float = 0.0


class EpisodicMemory:
    """Obsidian-compatible episodic memory store.

    Reads and writes run notes under knowledge/runs/.md with YAML
    frontmatter.  Retrieval uses keyword+tag scoring with optional
    embedding-based re-ranking.
    """

    def __init__(self, work_dir: Path):
        self.root = Path(work_dir) / "knowledge"
        self.runs_dir = self.root / "runs"
        self.concepts_dir = self.root / "concepts"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.concepts_dir.mkdir(parents=True, exist_ok=True)

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(self, task: str, k: int = 3) -> List[Tuple[Path, Dict[str, Any]]]:
        """Retrieve the k most relevant past runs for a given task."""
        if k <= 0:
            return []

        task_tokens = set(self._tokenize(task))
        scored: List[Tuple[float, Path, Dict[str, Any]]] = []

        for p in sorted(self.runs_dir.glob("*.md")):
            try:
                fm, _body = self._parse_note(p)
            except Exception:
                continue

            tags = fm.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]

            title_tokens = set(self._tokenize(p.stem))
            tag_tokens = set(self._tokenize(" ".join(map(str, tags))))

            score = 2.0 * len(task_tokens & tag_tokens) + 1.0 * len(
                task_tokens & title_tokens
            )
            if score > 0:
                scored.append((score, p, fm))

        scored.sort(key=lambda x: -x[0])
        return [(p, fm) for _s, p, fm in scored[:k]]

    def retrieve_with_embeddings(self, task: str, k: int = 3) -> List[RunNote]:
        """Retrieve with keyword pre-filter + embedding re-ranking."""
        keyword_results = self.retrieve(task, k=max(k * 3, 10))

        if not keyword_results:
            return []

        notes = []
        for p, fm in keyword_results:
            body = p.read_text(encoding="utf-8")
            notes.append(RunNote(path=p, frontmatter=fm, body=body))

        # If ChromaDB is available, re-rank by embedding similarity
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            ef = embedding_functions.DefaultEmbeddingFunction()
            client = chromadb.PersistentClient(path=str(self.root / ".chromadb"))
            collection = client.get_or_create_collection(
                name="episodic_memory",
                embedding_function=ef,
            )

            # Re-rank top results using embeddings
            task_embedding = ef([task])[0] if hasattr(ef, "__call__") else None
            if task_embedding is None:
                return notes[:k]

            for note in notes:
                try:
                    results = collection.query(
                        query_embeddings=[task_embedding],
                        where={"path": str(note.path)},
                        n_results=1,
                    )
                    if results.get("distances") and results["distances"][0]:
                        note.score = 1.0 / (1.0 + results["distances"][0][0])
                except Exception:
                    pass

            notes.sort(key=lambda n: -n.score)
        except ImportError:
            pass  # ChromaDB not available; keyword results are fine
        except Exception as e:
            logger.debug("Embedding re-rank failed: %s", e)

        return notes[:k]

    def render_for_prompt(self, notes: List[RunNote]) -> str:
        """Render retrieved notes as a compact prompt section."""
        if not notes:
            return ""
        lines = ["## Past relevant work (auto-retrieved)"]
        for n in notes:
            tags = n.frontmatter.get("tags") or []
            status = n.frontmatter.get("status", "?")
            lines.append(f"- [[{n.path.stem}]] — status={status}, tags={tags}")
        return "\n".join(lines)

    # ── Writing ──────────────────────────────────────────────────────────

    def write_run_note(
        self,
        *,
        task: str,
        arch: Optional[Architecture],
        artifacts: List[FileArtifact],
        shared_state: Dict[str, Any],
        provider: str = "",
        model: str = "",
        started_at: Optional[datetime] = None,
        duration_s: float = 0.0,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        status: str = "done",
        related: Optional[List[str]] = None,
    ) -> Path:
        """Persist a run as an Obsidian-compatible note."""
        started_at = started_at or datetime.now(timezone.utc)
        slug = _slugify(task)[:60] or "run"
        ts = started_at.strftime("%Y-%m-%dT%H-%M-%S")
        path = self.runs_dir / f"{ts}-{slug}.md"

        tags = _infer_tags(task, artifacts)
        fm = {
            "date": started_at.isoformat(),
            "task_slug": slug,
            "model": model,
            "provider": provider,
            "tags": tags,
            "files_generated": len(artifacts),
            "status": status,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_prompt + tokens_completion,
            "duration_s": round(duration_s, 1),
            "related": [f"[[{r}]]" for r in (related or [])],
        }

        lines = [f"# Task: {task}\n"]
        if arch:
            lines.append(f"**Project:** `{arch.project_name}`\n")
            lines.append("## Architecture")
            for f in arch.files:
                reads = (
                    f", reads={f.shared_state_reads}" if f.shared_state_reads else ""
                )
                writes = (
                    f", writes={f.shared_state_writes}" if f.shared_state_writes else ""
                )
                lines.append(f"{f.order}. `{f.filename}`{reads}{writes}")
            lines.append("")

        lines.append("## Final shared_state")
        lines.append("```json")
        lines.append(json.dumps(shared_state, indent=2, ensure_ascii=False))
        lines.append("```\n")

        lines.append("## Artifacts")
        for a in artifacts:
            lines.append(
                f"- `{a.name}.{a.extension}` — status={a.status}, chars={len(a.content)}"
            )

        fm_text = _render_frontmatter(fm)
        path.write_text(fm_text + "\n" + "\n".join(lines), encoding="utf-8")
        logger.info("Episodic note written: %s", path)
        return path

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]

    def _parse_note(self, path: Path) -> Tuple[Dict[str, Any], str]:
        raw = path.read_text(encoding="utf-8")
        m = _FRONTMATTER_RE.match(raw)
        if not m:
            return {}, raw
        fm_raw = m.group(1)
        body = raw[m.end() :]
        fm = _parse_yaml_frontmatter(fm_raw)
        if not isinstance(fm, dict):
            return {}, body
        return fm, body


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-") or "untitled"


def _infer_tags(task: str, artifacts: List[FileArtifact]) -> List[str]:
    tags = set()
    for a in artifacts:
        tags.add(a.extension)
    for kw in (
        "python",
        "javascript",
        "typescript",
        "react",
        "vue",
        "fastapi",
        "flask",
        "cli",
        "api",
        "dashboard",
        "sqlite",
        "postgres",
        "tailwind",
        "crud",
    ):
        if kw in task.lower():
            tags.add(kw)
    return sorted(tags)


def _parse_yaml_frontmatter(raw: str) -> Dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(raw) or {}
    except ImportError:
        result: Dict[str, Any] = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                v = [x.strip().strip('"') for x in v[1:-1].split(",") if x.strip()]
            result[k.strip()] = v
        return result


def _render_frontmatter(fm: Dict[str, Any]) -> str:
    try:
        import yaml

        body = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
    except ImportError:
        lines = []
        for k, v in fm.items():
            k_str = k.replace(":", "")
            if isinstance(v, list):
                inner = ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in v)
                lines.append(f"{k_str}: [{inner}]")
            elif isinstance(v, str):
                lines.append(f'{k_str}: "{v}"')
            else:
                lines.append(f"{k_str}: {v}")
        body = "\n".join(lines)
    return f"---\n{body}\n---"

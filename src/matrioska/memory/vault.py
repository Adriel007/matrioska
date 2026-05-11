"""
Global Obsidian-compatible vault for Matrioska.

Location: ~/.matrioska/vault/ (overridable via MATRIOSKA_VAULT_DIR).

Structure (Karpathy LLM Wiki Pattern — arXiv:2409.14813 dual-level retrieval):
  projects/<name>/architecture.md   per-project decomposition + decisions
  projects/<name>/patterns.md       patterns observed in that project
  projects/<name>/lessons.md        bugs encountered and how they were fixed
  projects/<name>/links.md          [[wikilinks]] to related projects
  concepts/<tag>.md                 cross-project concept (e.g. sqlite_patterns)
  bugs/<slug>.md                    recurring bug + mitigation
  INDEX.md                          auto-generated tag/wikilink index

Notes are pure Markdown with YAML frontmatter — fully compatible with the
native Obsidian app (Graph View, wikilinks, tags). The compiler is
append-only with deduplication so multiple runs strictly accrete knowledge
instead of overwriting.

Search supports scopes (local | global | linked | all) so the same query
can be answered with project-only or cross-project context — same approach
as LightRAG (EMNLP 2025).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("matrioska.memory.vault")

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_TAG_RE = re.compile(r"(?<!\w)#([a-zA-Z_][a-zA-Z0-9_/-]+)")
_STOPWORDS = {
    "a", "an", "and", "the", "for", "with", "of", "to", "in", "on", "at",
    "by", "is", "are", "be", "or", "as", "from", "that", "this", "it",
    "create", "build", "make", "generate", "add", "implement", "show",
    "use", "using",
}
_BUG_KEYWORDS = (
    "error", "fail", "syntax", "import", "exception", "undefined",
    "missing", "circular", "deadlock", "race", "leak",
)


def default_vault_dir() -> Path:
    """Resolve the default vault location (env > ~/.matrioska/vault)."""
    env_val = os.environ.get("MATRIOSKA_VAULT_DIR", "").strip()
    if env_val:
        return Path(env_val).expanduser()
    return Path.home() / ".matrioska" / "vault"


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class VaultNote:
    """A single Markdown note in the vault."""
    path: Path
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @property
    def title(self) -> str:
        return self.frontmatter.get("title") or self.path.stem

    @property
    def wikilinks(self) -> List[str]:
        return _WIKILINK_RE.findall(self.body)

    @property
    def tags(self) -> List[str]:
        tags_fm = self.frontmatter.get("tags") or []
        if isinstance(tags_fm, str):
            tags_fm = [tags_fm]
        body_tags = _TAG_RE.findall(self.body)
        return sorted({*tags_fm, *body_tags})


# ── Vault ─────────────────────────────────────────────────────────────────────


class GlobalVault:
    """Obsidian-compatible knowledge base shared across all Matrioska projects."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else default_vault_dir()
        self.projects_dir = self.root / "projects"
        self.concepts_dir = self.root / "concepts"
        self.bugs_dir = self.root / "bugs"
        for d in (self.projects_dir, self.concepts_dir, self.bugs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._ensure_index()

    # ── Public: compile-after-run ──────────────────────────────────────────

    def compile_from_run(
        self,
        *,
        task: str,
        project_name: str,
        files: Sequence[Dict[str, Any]],
        shared_state: Dict[str, Any],
        status: str,
        tags: Sequence[str] = (),
        lessons: Sequence[str] = (),
        bugs: Sequence[str] = (),
        provider: str = "",
        model: str = "",
    ) -> List[Path]:
        """Upsert per-run knowledge into the global vault.

        Returns the list of note paths that were created or updated.
        Pure-deterministic merge (no LLM cost): every section is upserted
        idempotently with timestamps so successive runs accrete history
        without overwriting.
        """
        proj_slug = _slugify(project_name or "untitled")
        touched: List[Path] = []
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tags = sorted({_slugify(t) for t in tags if t})

        proj_dir = self.projects_dir / proj_slug
        proj_dir.mkdir(parents=True, exist_ok=True)

        # 1. architecture.md
        arch_path = proj_dir / "architecture.md"
        arch_section = _format_arch_section(now, task, status, files, shared_state, provider, model)
        touched.append(self._append_section(
            arch_path,
            default_title=f"{project_name} — architecture",
            section_id=now,
            section_md=arch_section,
            tags=tags,
        ))

        # 2. patterns.md (extract simple patterns from filenames + tags)
        patterns = _extract_patterns(files, tags, shared_state)
        if patterns:
            pat_path = proj_dir / "patterns.md"
            pat_md = "\n".join(f"- {p}" for p in patterns)
            touched.append(self._append_section(
                pat_path,
                default_title=f"{project_name} — patterns",
                section_id=now,
                section_md=f"### Patterns observed ({now})\n{pat_md}",
                tags=tags,
            ))

        # 3. lessons.md
        if lessons:
            les_path = proj_dir / "lessons.md"
            les_md = "\n".join(f"- {l}" for l in lessons)
            touched.append(self._append_section(
                les_path,
                default_title=f"{project_name} — lessons",
                section_id=now,
                section_md=f"### Lessons ({now})\n{les_md}",
                tags=tags,
            ))

        # 4. links.md — wikilink to each concept touched by this run
        if tags:
            links_path = proj_dir / "links.md"
            links_md = "\n".join(f"- [[{t}]]" for t in tags)
            touched.append(self._append_section(
                links_path,
                default_title=f"{project_name} — links",
                section_id="concepts",
                section_md=f"### Related concepts\n{links_md}",
                tags=tags,
                section_overwrite=True,
            ))

        # 5. concepts/<tag>.md — register the project under each tag
        for tag in tags:
            cpath = self.concepts_dir / f"{tag}.md"
            cmd = f"- [[{proj_slug}/architecture]] — run {now} (status={status})"
            touched.append(self._append_section(
                cpath,
                default_title=f"{tag} — pattern",
                section_id="projects",
                section_md=f"### Projects using this concept\n{cmd}",
                tags=[tag],
                section_append_dedup=True,
            ))

        # 6. bugs/<slug>.md — one note per observed bug pattern
        for b in bugs:
            bslug = _bug_slug(b)
            if not bslug:
                continue
            bpath = self.bugs_dir / f"{bslug}.md"
            bmd = f"- {now}: {b[:300]}\n  - Source: [[{proj_slug}/architecture]]"
            touched.append(self._append_section(
                bpath,
                default_title=f"Bug: {bslug}",
                section_id="occurrences",
                section_md=f"### Occurrences\n{bmd}",
                tags=["bug", *tags],
                section_append_dedup=True,
            ))

        self._update_index()
        return touched

    # ── Public: search ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        scope: str = "all",
        project: Optional[str] = None,
        k: int = 8,
        local_root: Optional[Path] = None,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """Search the vault.

        Scope:
          local   — only `local_root` (defaults to no-op if not given)
          global  — concepts/ + bugs/ (cross-project)
          linked  — `project` + every wikilink-reachable project (BFS, max_hops)
          all     — everything

        Returns ranked dicts: {score, kind, path, title, snippet, project}.
        """
        scope = scope.lower()
        if scope not in {"local", "global", "linked", "all"}:
            raise ValueError(f"unknown scope: {scope}")

        roots: List[Tuple[str, Path]] = []
        if scope in ("all", "global"):
            roots.append(("concept", self.concepts_dir))
            roots.append(("bug", self.bugs_dir))
        if scope == "all":
            roots.append(("project", self.projects_dir))
        if scope == "linked" and project:
            for proj in self._linked_projects(project, max_hops=max_hops):
                roots.append(("project", self.projects_dir / proj))
        if scope == "local" and local_root and local_root.exists():
            roots.append(("local", local_root))

        scored: List[Dict[str, Any]] = []
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []

        for kind, root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.md"):
                try:
                    note = self._read_note(path)
                except Exception:
                    continue
                score, snippet = _score(note, q_tokens)
                if score <= 0:
                    continue
                scored.append({
                    "score": score,
                    "kind": kind,
                    "path": str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path),
                    "title": note.title,
                    "snippet": snippet,
                    "project": _project_of(path, self.projects_dir),
                })

        scored.sort(key=lambda r: -r["score"])
        return scored[:k]

    # ── Public: list & inspect ─────────────────────────────────────────────

    def list_projects(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for proj in sorted(self.projects_dir.iterdir()):
            if not proj.is_dir():
                continue
            notes = list(proj.glob("*.md"))
            arch = proj / "architecture.md"
            last_run = "—"
            if arch.exists():
                try:
                    fm, _ = self._parse(arch.read_text(encoding="utf-8"))
                    last_run = str(fm.get("updated", "—"))[:19]
                except Exception:
                    pass
            out.append({
                "name": proj.name,
                "notes": len(notes),
                "last_run": last_run,
            })
        return out

    def get_note(self, rel_path: str) -> Optional[VaultNote]:
        p = (self.root / rel_path).resolve()
        if not p.is_relative_to(self.root.resolve()):
            return None
        if not p.exists():
            return None
        return self._read_note(p)

    def doctor(self) -> Dict[str, Any]:
        """Return a health report on the vault."""
        all_notes = list(self.root.rglob("*.md"))
        all_notes = [p for p in all_notes if p.name != "INDEX.md"]
        notes = [self._read_note(p) for p in all_notes]

        # Build link graph (note title → set of links)
        title_index: Dict[str, Path] = {}
        for n in notes:
            title_index[n.path.stem] = n.path
            for parent in n.path.parents:
                if parent == self.root:
                    break
                title_index[f"{parent.name}/{n.path.stem}"] = n.path

        orphans: List[str] = []
        broken_links: List[Dict[str, str]] = []
        link_targets: Set[str] = set()
        for n in notes:
            outgoing = n.wikilinks
            link_targets.update(outgoing)
            for tgt in outgoing:
                if tgt not in title_index and tgt.split("/")[-1] not in title_index:
                    broken_links.append({
                        "from": str(n.path.relative_to(self.root)),
                        "target": tgt,
                    })

        for n in notes:
            # Notes inside a project folder are implicitly anchored to their
            # sibling architecture.md, so we don't flag them as orphans.
            try:
                rel = n.path.relative_to(self.projects_dir)
                if len(rel.parts) >= 2:
                    continue
            except ValueError:
                pass
            stem = n.path.stem
            qualified = None
            for parent in n.path.parents:
                if parent == self.root:
                    break
                qualified = f"{parent.name}/{stem}"
                break
            no_outgoing = not n.wikilinks
            no_incoming = stem not in link_targets and (qualified or "") not in link_targets
            if no_outgoing and no_incoming:
                orphans.append(str(n.path.relative_to(self.root)))

        now = datetime.now(timezone.utc)
        stale: List[str] = []
        for n in notes:
            ts = n.frontmatter.get("updated") or n.frontmatter.get("created")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                age_days = (now - dt).days
                if age_days > 30:
                    stale.append(f"{n.path.relative_to(self.root)} ({age_days}d)")
            except (ValueError, TypeError):
                pass

        projects = len(list(self.projects_dir.iterdir())) if self.projects_dir.exists() else 0
        concepts = len(list(self.concepts_dir.glob("*.md"))) if self.concepts_dir.exists() else 0
        bugs = len(list(self.bugs_dir.glob("*.md"))) if self.bugs_dir.exists() else 0

        issues = len(orphans) + len(broken_links)
        status = "healthy" if issues == 0 else "issues_found"
        return {
            "projects": projects,
            "concepts": concepts,
            "bugs": bugs,
            "total_notes": len(notes),
            "orphans": orphans,
            "stale": stale,
            "broken_links": broken_links,
            "status": status,
        }

    def export_graph_mermaid(self) -> str:
        """Export the vault wikilink graph as a Mermaid flowchart.

        Useful for visualizing in any Markdown viewer that renders Mermaid.
        """
        notes = [self._read_note(p) for p in self.root.rglob("*.md") if p.name != "INDEX.md"]
        lines = ["```mermaid", "graph LR"]
        seen: Set[str] = set()

        def _id(name: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]", "_", name)

        for n in notes:
            src = n.path.stem
            for tgt in n.wikilinks:
                tgt_clean = tgt.split("/")[-1]
                edge = (src, tgt_clean)
                if edge in seen:
                    continue
                seen.add(edge)
                lines.append(f"    {_id(src)}[{src}] --> {_id(tgt_clean)}[{tgt_clean}]")
        lines.append("```")
        return "\n".join(lines)

    # ── Internals ──────────────────────────────────────────────────────────

    def _read_note(self, path: Path) -> VaultNote:
        text = path.read_text(encoding="utf-8", errors="ignore")
        fm, body = self._parse(text)
        return VaultNote(path=path, frontmatter=fm, body=body)

    @staticmethod
    def _parse(text: str) -> Tuple[Dict[str, Any], str]:
        m = _FRONTMATTER_RE.match(text)
        if not m:
            return {}, text
        return _parse_yaml(m.group(1)), text[m.end():]

    def _append_section(
        self,
        path: Path,
        *,
        default_title: str,
        section_id: str,
        section_md: str,
        tags: Sequence[str] = (),
        section_overwrite: bool = False,
        section_append_dedup: bool = False,
    ) -> Path:
        """Idempotently upsert a section into a note.

        section_overwrite=True replaces the section if it exists.
        section_append_dedup=True appends a new bullet only if it's not already there.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        marker = f"<!-- section:{section_id} -->"
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            text = path.read_text(encoding="utf-8")
            fm, body = self._parse(text)
            tag_set = set(fm.get("tags") or [])
            tag_set.update(tags)
            fm["tags"] = sorted(tag_set)
            fm["updated"] = now
            fm.setdefault("created", now)
            fm.setdefault("title", default_title)
        else:
            fm = {
                "title": default_title,
                "created": now,
                "updated": now,
                "tags": sorted(set(tags)),
            }
            body = ""

        body = self._merge_section(
            body, marker, section_md,
            overwrite=section_overwrite,
            append_dedup=section_append_dedup,
        )
        path.write_text(_render_fm(fm) + "\n" + body.lstrip("\n"), encoding="utf-8")
        return path

    @staticmethod
    def _merge_section(
        body: str,
        marker: str,
        section_md: str,
        *,
        overwrite: bool = False,
        append_dedup: bool = False,
    ) -> str:
        block = f"\n{marker}\n{section_md}\n{marker}\n"
        if marker not in body:
            return body + block

        start = body.index(marker)
        end = body.index(marker, start + len(marker)) + len(marker)
        existing = body[start:end]

        if overwrite:
            return body[:start] + f"{marker}\n{section_md}\n{marker}" + body[end:]

        if append_dedup:
            # append only bullets not already present
            new_lines = [l for l in section_md.splitlines() if l.strip().startswith("-")]
            kept = [l for l in new_lines if l.strip() not in existing]
            if not kept:
                return body
            stripped_marker_block = existing.rstrip().rstrip(marker).rstrip()
            return (
                body[:start]
                + stripped_marker_block
                + "\n" + "\n".join(kept) + "\n"
                + marker
                + body[end:]
            )

        # default: insert new entry above the closing marker
        stripped = existing.rstrip().rstrip(marker).rstrip()
        return (
            body[:start]
            + stripped
            + "\n\n" + section_md + "\n"
            + marker
            + body[end:]
        )

    def _linked_projects(self, project: str, *, max_hops: int = 2) -> Set[str]:
        """BFS over wikilinks in links.md to collect reachable projects."""
        seen: Set[str] = {project}
        frontier: List[str] = [project]
        for _ in range(max_hops):
            next_frontier: List[str] = []
            for proj in frontier:
                links_path = self.projects_dir / proj / "links.md"
                if not links_path.exists():
                    continue
                text = links_path.read_text(encoding="utf-8", errors="ignore")
                for tgt in _WIKILINK_RE.findall(text):
                    tgt_proj = tgt.split("/")[0]
                    if tgt_proj not in seen and (self.projects_dir / tgt_proj).is_dir():
                        seen.add(tgt_proj)
                        next_frontier.append(tgt_proj)
            frontier = next_frontier
            if not frontier:
                break
        return seen

    def _ensure_index(self) -> None:
        idx = self.root / "INDEX.md"
        if idx.exists():
            return
        idx.write_text(
            "# Matrioska Vault — Index\n\n"
            "Auto-maintained by Matrioska. Open this vault in Obsidian for "
            "Graph View and wikilink navigation.\n\n"
            "<!-- AUTO_INDEX -->\n<!-- /AUTO_INDEX -->\n",
            encoding="utf-8",
        )

    def _update_index(self) -> None:
        idx = self.root / "INDEX.md"
        try:
            text = idx.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        projects = sorted(p.name for p in self.projects_dir.iterdir() if p.is_dir())
        concepts = sorted(p.stem for p in self.concepts_dir.glob("*.md"))
        bugs = sorted(p.stem for p in self.bugs_dir.glob("*.md"))

        block = ["## Projects"]
        for p in projects:
            block.append(f"- [[projects/{p}/architecture|{p}]]")
        block.append("\n## Concepts")
        for c in concepts:
            block.append(f"- [[concepts/{c}|{c}]]")
        block.append("\n## Bugs")
        for b in bugs:
            block.append(f"- [[bugs/{b}|{b}]]")
        new_section = "<!-- AUTO_INDEX -->\n" + "\n".join(block) + "\n<!-- /AUTO_INDEX -->"

        if "<!-- AUTO_INDEX -->" in text and "<!-- /AUTO_INDEX -->" in text:
            text = re.sub(
                r"<!-- AUTO_INDEX -->.*?<!-- /AUTO_INDEX -->",
                new_section,
                text,
                flags=re.DOTALL,
            )
        else:
            text = text.rstrip() + "\n\n" + new_section + "\n"
        idx.write_text(text, encoding="utf-8")


# ── Compiler helpers (knowledge extraction from a run) ────────────────────────


def extract_lessons_and_bugs(
    artifacts: Sequence[Any],
    repair_log: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Extract human-friendly lessons + bug patterns from a finished run.

    Heuristics (deterministic, no LLM):
      - any artifact with repair_count > 0 that ended status=done → lesson
        "Repaired <file> in N attempts"
      - any artifact with status=failed → bug "Failed to generate <file>"
      - any line in repair_log matching common error patterns → bug
    """
    lessons: List[str] = []
    bugs: List[str] = []

    for art in artifacts:
        rc = getattr(art, "repair_count", 0) or 0
        status = getattr(art, "status", "")
        name = f"{getattr(art, 'name', '?')}.{getattr(art, 'extension', '?')}"
        if rc > 0 and status == "done":
            lessons.append(f"Repaired `{name}` in {rc} attempt(s) — generator output needed correction")
        elif status == "failed":
            bugs.append(f"Failed to generate `{name}` after {rc} repair attempt(s)")

    if repair_log:
        for line in repair_log:
            low = line.lower()
            if any(k in low for k in _BUG_KEYWORDS):
                bugs.append(line[:300])

    return lessons, list(dict.fromkeys(bugs))


def derive_tags(task: str, files: Sequence[Any]) -> List[str]:
    """Infer concept tags from the task description + generated file extensions."""
    tags: Set[str] = set()
    for f in files:
        ext = getattr(f, "extension", "") or ""
        if ext:
            tags.add(ext.lower())
    task_low = task.lower()
    for kw in (
        "python", "javascript", "typescript", "react", "vue", "fastapi",
        "flask", "django", "express", "cli", "api", "graphql", "rest",
        "dashboard", "sqlite", "postgres", "mysql", "mongodb", "redis",
        "tailwind", "crud", "auth", "jwt", "oauth", "docker", "kubernetes",
        "terraform", "argparse", "click", "typer", "rich",
    ):
        if kw in task_low:
            tags.add(kw)
    return sorted(tags)


# ── Free helpers ──────────────────────────────────────────────────────────────


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s or "untitled"


def _bug_slug(text: str) -> str:
    for pat in (
        r"([A-Z][a-zA-Z]+Error)",
        r"(invalid syntax|unterminated|unexpected indent|expected)",
        r"(missing[a-z ]+)",
    ):
        m = re.search(pat, text)
        if m:
            return _slugify(m.group(1))[:40]
    return _slugify(text[:40])


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower())
            if t not in _STOPWORDS and len(t) > 2]


def _score(note: VaultNote, q_tokens: Set[str]) -> Tuple[float, str]:
    title_tokens = set(_tokenize(note.title))
    tag_tokens = set(_tokenize(" ".join(note.tags)))
    body_tokens = set(_tokenize(note.body[:4000]))
    score = (
        3.0 * len(q_tokens & title_tokens)
        + 2.0 * len(q_tokens & tag_tokens)
        + 1.0 * len(q_tokens & body_tokens)
    )
    snippet = _snippet(note.body, q_tokens)
    return score, snippet


def _snippet(body: str, q_tokens: Set[str]) -> str:
    body_l = body.lower()
    for tok in q_tokens:
        idx = body_l.find(tok)
        if idx >= 0:
            start = max(0, idx - 40)
            end = min(len(body), idx + 80)
            return body[start:end].replace("\n", " ").strip()
    return body[:120].replace("\n", " ").strip()


def _project_of(path: Path, projects_dir: Path) -> Optional[str]:
    try:
        rel = path.relative_to(projects_dir)
        return rel.parts[0] if rel.parts else None
    except ValueError:
        return None


def _format_arch_section(
    now: str,
    task: str,
    status: str,
    files: Sequence[Dict[str, Any]],
    shared_state: Dict[str, Any],
    provider: str,
    model: str,
) -> str:
    lines = [
        f"### Run {now}",
        f"- **Task:** {task}",
        f"- **Status:** {status}",
        f"- **Provider/model:** {provider} / {model}",
        f"- **Files generated:**",
    ]
    for f in files:
        name = f.get("name") or "?"
        ext = f.get("extension") or ""
        st = f.get("status") or ""
        rc = f.get("repair_count", 0)
        rc_str = f" (repaired ×{rc})" if rc else ""
        lines.append(f"  - `{name}.{ext}` — {st}{rc_str}")
    if shared_state:
        lines.append(f"- **shared_state keys:** {sorted(shared_state.keys())}")
    return "\n".join(lines)


def _extract_patterns(
    files: Sequence[Dict[str, Any]],
    tags: Sequence[str],
    shared_state: Dict[str, Any],
) -> List[str]:
    """Heuristic pattern extraction from a finished run (no LLM cost)."""
    out: List[str] = []
    by_ext: Dict[str, List[str]] = {}
    for f in files:
        by_ext.setdefault(f.get("extension", ""), []).append(f.get("name", ""))
    if by_ext.get("py"):
        out.append(f"Python project with {len(by_ext['py'])} module(s): {by_ext['py']}")
    if by_ext.get("sql"):
        out.append(f"SQL schema in {by_ext['sql']}")
    if "sqlite" in tags:
        out.append("Uses SQLite (Python `sqlite3` stdlib)")
    if "fastapi" in tags:
        out.append("FastAPI service — route definitions in shared_state")
    if "cli" in tags:
        out.append("CLI tool — likely argparse / click / typer")
    if shared_state.get("app_routes"):
        out.append(f"REST routes declared: {shared_state['app_routes']}")
    return out


def _parse_yaml(raw: str) -> Dict[str, Any]:
    try:
        import yaml
        loaded = yaml.safe_load(raw)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        out: Dict[str, Any] = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                v = [x.strip().strip('"').strip("'") for x in v[1:-1].split(",") if x.strip()]
            elif v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            out[k.strip()] = v
        return out


def _render_fm(fm: Dict[str, Any]) -> str:
    try:
        import yaml
        body = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
    except ImportError:
        lines = []
        for k, v in fm.items():
            if isinstance(v, list):
                inner = ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in v)
                lines.append(f"{k}: [{inner}]")
            elif isinstance(v, str):
                lines.append(f'{k}: "{v}"')
            else:
                lines.append(f"{k}: {v}")
        body = "\n".join(lines)
    return f"---\n{body}\n---"

"""
Interactive REPL — `matrioska` with no subcommand opens a prompt session.

Inspired by Claude Code: free-form questions get sent through the orchestrator
(or, via `/btw`, a one-shot LLM call); slash commands inspect or mutate the
session; `!` prefix runs shell commands and echoes the output.

Keybindings (prompt_toolkit when available):
  Enter           submit
  Esc+Enter       newline (multi-line)
  Ctrl+C          cancel running task
  Ctrl+D          exit
  ↑ / ↓           history

Falls back to a basic `input()` loop if prompt_toolkit isn't installed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from matrioska.core.config import Config, load_config

# ── Banner ────────────────────────────────────────────────────────────────────


_BANNER = r"""
╭──────────────────────────────────────────────────────────────╮
│  Matrioska — interactive shell                                │
│  /help for commands · ! prefix for shell · Ctrl+D to exit     │
╰──────────────────────────────────────────────────────────────╯"""


# ── Session ───────────────────────────────────────────────────────────────────


@dataclass
class ReplSession:
    """Mutable per-REPL state. One per `matrioska` invocation."""
    cfg: Config
    history: List[str] = field(default_factory=list)
    last_result: Optional[Dict[str, Any]] = None
    plan_mode: bool = False
    effort: str = "medium"  # low | medium | high
    started_at: float = field(default_factory=time.time)
    tokens_session: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})


# ── Command registry ──────────────────────────────────────────────────────────


CommandFn = Callable[["Repl", List[str]], None]
COMMANDS: Dict[str, CommandFn] = {}
HELP: Dict[str, str] = {}


def command(name: str, help_text: str) -> Callable[[CommandFn], CommandFn]:
    def decorator(fn: CommandFn) -> CommandFn:
        COMMANDS[name] = fn
        HELP[name] = help_text
        return fn
    return decorator


# ── REPL ──────────────────────────────────────────────────────────────────────


class Repl:
    """The interactive shell.

    All slash commands are registered through @command above. `run()` is the
    main loop; everything else is dispatch + glue.
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()
        self.session = ReplSession(cfg=self.cfg)
        self._cancel_event = threading.Event()
        self._console = self._make_console()
        self._stream_buffer: List[str] = []
        self._stream_active = False
        self._load_custom_commands()

    def _load_custom_commands(self) -> None:
        """Scan .matrioska/commands/*.md and register each as a slash command."""
        commands_dir = Path.cwd() / ".matrioska" / "commands"
        if not commands_dir.is_dir():
            return
        for f in commands_dir.glob("*.md"):
            name = f.stem.replace("-", "_")
            try:
                content = f.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if not content:
                continue
            help_text = content.splitlines()[0].lstrip("# ").strip() if content else f.name

            def _make_cmd(task_content: str) -> CommandFn:
                def _cmd(repl: "Repl", args: List[str]) -> None:
                    repl._run_task(task_content)
                return _cmd

            COMMANDS[name] = _make_cmd(content)
            HELP[name] = f"[custom] {help_text}"

    def _make_console(self) -> Any:
        try:
            from rich.console import Console
            return Console()
        except ImportError:
            return None

    def _print(self, msg: str) -> None:
        if self._console:
            self._console.print(msg)
        else:
            print(msg)

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self) -> int:
        self._print(_BANNER)
        self._print_status_line()
        get_input = self._build_input_reader()

        while True:
            try:
                raw = get_input()
            except (KeyboardInterrupt, EOFError):
                self._print("\nbye 👋" if self._supports_emoji() else "\nbye")
                return 0
            if raw is None:
                return 0
            line = raw.strip()
            if not line:
                continue
            self.session.history.append(line)
            try:
                self._dispatch(line)
            except SystemExit:
                raise
            except Exception as e:
                self._print(f"[red]error:[/] {e}" if self._console else f"error: {e}")

    def _dispatch(self, line: str) -> None:
        if line.startswith("!"):
            self._shell(line[1:].strip())
            return
        if line.startswith("/"):
            head, *rest = line[1:].split(maxsplit=1)
            args = rest[0].split() if rest else []
            cmd = COMMANDS.get(head)
            if cmd is None:
                self._print(f"unknown command: /{head}. /help for the list.")
                return
            cmd(self, args)
            return
        # Free-form task → run the pipeline
        self._run_task(line)

    # ── Pipeline integration ───────────────────────────────────────────────

    def _run_task(self, task: str) -> None:
        from matrioska.pipeline.orchestrator import Matrioska
        from matrioska.core.config import validate_config

        cfg = self._effective_cfg_for_run()
        try:
            validate_config(cfg)
        except SystemExit as e:
            self._print(f"config error: {e}")
            return

        # Single subscriber for streamed tokens (renders inline)
        m = Matrioska(cfg)
        if cfg.stream_tokens:
            self._stream_buffer.clear()
            self._stream_active = True
            m.bus.on("llm_token", self._on_token)
            m.bus.on("phase1_done", lambda _e: self._flush_stream())
            m.bus.on("phase2_done", lambda _e: self._flush_stream())

        self._cancel_event.clear()
        self._print(f"\n[dim]→ running task ({len(task)} chars)…[/]" if self._console else "→ running task…")
        worker_result: Dict[str, Any] = {}
        err: List[BaseException] = []

        def _worker() -> None:
            try:
                worker_result.update(m.run(task))
            except BaseException as e:
                err.append(e)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            while t.is_alive():
                t.join(timeout=0.2)
        except KeyboardInterrupt:
            self._cancel_event.set()
            self._print("[yellow]^C — cancel requested (waiting for current LLM call to finish)[/]"
                        if self._console else "^C — cancel requested")
            t.join(timeout=30)
        finally:
            self._stream_active = False
            self._flush_stream()

        if err:
            self._print(f"[red]task failed:[/] {err[0]}" if self._console else f"task failed: {err[0]}")
            return

        self.session.last_result = worker_result
        tok = worker_result.get("tokens", {}) or {}
        self.session.tokens_session["prompt"] += int(tok.get("prompt_tokens", 0))
        self.session.tokens_session["completion"] += int(tok.get("completion_tokens", 0))
        status = worker_result.get("status", "?")
        n_files = len(worker_result.get("artifacts", []) or [])
        cost = tok.get("estimated_cost_usd", 0.0)
        self._print(
            f"\n[bold]{status}[/]  files={n_files}  "
            f"tokens={tok.get('prompt_tokens',0)}p+{tok.get('completion_tokens',0)}c  ~${cost:.4f}"
            if self._console else
            f"\nstatus={status} files={n_files} tokens={tok.get('prompt_tokens',0)}p+{tok.get('completion_tokens',0)}c"
        )

    def _effective_cfg_for_run(self) -> Config:
        """Return a copy of cfg with session-mutable knobs applied."""
        import dataclasses
        cfg = dataclasses.replace(self.cfg)
        if self.session.plan_mode:
            cfg.plan_only = True
        if self.session.effort == "low":
            cfg.enable_tot = False
            cfg.enable_reflexion = False
            cfg.enable_test_design = False
            cfg.architect_candidates = 1
        elif self.session.effort == "high":
            cfg.enable_tot = True
            cfg.enable_reflexion = True
            cfg.enable_test_design = True
            cfg.architect_candidates = 5
        return cfg

    def _on_token(self, ev: Any) -> None:
        """EventBus callback: append streamed delta to the inline buffer."""
        if not self._stream_active:
            return
        delta = getattr(ev, "data", {}).get("delta") if hasattr(ev, "data") else None
        if not delta:
            return
        self._stream_buffer.append(delta)
        # Print incrementally only when we accumulate a reasonable chunk
        joined = "".join(self._stream_buffer)
        if len(joined) >= 80 or "\n" in delta:
            sys.stdout.write(joined)
            sys.stdout.flush()
            self._stream_buffer.clear()

    def _flush_stream(self) -> None:
        if self._stream_buffer:
            sys.stdout.write("".join(self._stream_buffer))
            sys.stdout.flush()
            self._stream_buffer.clear()

    # ── Shell prefix ───────────────────────────────────────────────────────

    def _shell(self, cmd: str) -> None:
        if not cmd:
            self._print("usage: !<shell command>")
            return
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=60
            )
        except subprocess.TimeoutExpired:
            self._print("[red]shell timed out (60s)[/]" if self._console else "shell timed out")
            return
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            self._print(f"[red]{result.stderr.rstrip()}[/]" if self._console else result.stderr.rstrip())
        if result.returncode:
            self._print(f"[dim](exit {result.returncode})[/]" if self._console
                        else f"(exit {result.returncode})")

    # ── Input ──────────────────────────────────────────────────────────────

    def _build_input_reader(self) -> Callable[[], Optional[str]]:
        """Return a callable that reads one (possibly multi-line) input.

        Tries prompt_toolkit first (history, multi-line, key bindings); falls
        back to plain `input()` if it's not installed or stdin isn't a TTY.
        """
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.key_binding import KeyBindings
        except ImportError:
            return self._basic_input

        if not sys.stdin.isatty():
            return self._basic_input

        history_path = Path.home() / ".matrioska" / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        kb = KeyBindings()

        @kb.add("c-d")
        def _exit(event: Any) -> None:
            event.app.exit(exception=EOFError())

        session = PromptSession(
            history=FileHistory(str(history_path)),
            multiline=False,
            key_bindings=kb,
        )

        def _read() -> Optional[str]:
            try:
                return session.prompt("matrioska › ")
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:
                return None

        return _read

    def _basic_input(self) -> Optional[str]:
        try:
            return input("matrioska › ")
        except (EOFError, KeyboardInterrupt):
            raise

    # ── Status helpers ─────────────────────────────────────────────────────

    def _print_status_line(self) -> None:
        cfg = self.cfg
        plan_badge = "[plan-mode]" if self.session.plan_mode else ""
        vault = "vault on" if cfg.enable_vault else "vault off"
        msg = (
            f"  provider={cfg.provider}  model={cfg.model}  "
            f"effort={self.session.effort}  {vault}  {plan_badge}"
        )
        self._print(f"[dim]{msg}[/]" if self._console else msg)

    @staticmethod
    def _supports_emoji() -> bool:
        try:
            "👋".encode(sys.stdout.encoding or "utf-8")
            return True
        except (UnicodeEncodeError, TypeError):
            return False


# ── Slash commands ────────────────────────────────────────────────────────────


@command("help", "Show available commands")
def _cmd_help(repl: Repl, args: List[str]) -> None:
    rows = []
    for name in sorted(COMMANDS):
        rows.append(f"  /{name:<10} — {HELP[name]}")
    rows.append("\n  !<shell>     — run a shell command")
    rows.append("  <free text>  — run as a Matrioska task")
    repl._print("\n".join(rows))


@command("quit", "Exit the REPL (same as Ctrl+D)")
def _cmd_quit(repl: Repl, args: List[str]) -> None:
    raise SystemExit(0)


@command("exit", "Exit the REPL")
def _cmd_exit(repl: Repl, args: List[str]) -> None:
    raise SystemExit(0)


@command("config", "Show current configuration")
def _cmd_config(repl: Repl, args: List[str]) -> None:
    cfg = repl.cfg
    visible = {
        "provider": cfg.provider,
        "base_url": cfg.base_url,
        "model": cfg.model,
        "architect_model": cfg.architect_model or cfg.model,
        "generator_model": cfg.generator_model or cfg.model,
        "max_repairs": cfg.max_repairs,
        "max_depth": cfg.max_depth,
        "parallel": cfg.parallel,
        "enable_tot": cfg.enable_tot,
        "enable_reflexion": cfg.enable_reflexion,
        "enable_test_design": cfg.enable_test_design,
        "quick": cfg.quick,
        "permission_mode": cfg.permission_mode,
        "enable_vault": cfg.enable_vault,
        "stream_tokens": cfg.stream_tokens,
        "work_dir": str(cfg.work_dir),
    }
    width = max(len(k) for k in visible)
    for k, v in visible.items():
        repl._print(f"  {k:<{width}} = {v}")


@command("model", "Show or switch the current model. Usage: /model [name]")
def _cmd_model(repl: Repl, args: List[str]) -> None:
    if not args:
        repl._print(f"  current: {repl.cfg.model}")
        repl._print(f"  override roles via env: MATRIOSKA_GENERATOR_MODEL, _ARCHITECT_MODEL, etc.")
        return
    new_model = args[0]
    repl.cfg.model = new_model
    repl._print(f"  → model set to {new_model}")


@command("usage", "Tokens spent in this session")
def _cmd_usage(repl: Repl, args: List[str]) -> None:
    s = repl.session.tokens_session
    elapsed = time.time() - repl.session.started_at
    repl._print(
        f"  session  prompt={s['prompt']:,}  completion={s['completion']:,}  "
        f"total={s['prompt']+s['completion']:,}  elapsed={elapsed:.0f}s"
    )
    if repl.session.last_result:
        lr = repl.session.last_result.get("tokens", {}) or {}
        repl._print(
            f"  last     prompt={lr.get('prompt_tokens',0):,}  "
            f"completion={lr.get('completion_tokens',0):,}  "
            f"cost~${lr.get('estimated_cost_usd',0):.4f}"
        )


@command("clear", "Reset session token counters and last result")
def _cmd_clear(repl: Repl, args: List[str]) -> None:
    repl.session.tokens_session = {"prompt": 0, "completion": 0}
    repl.session.last_result = None
    repl.session.started_at = time.time()
    repl._print("  session cleared")


@command("plan", "Toggle plan-only mode for subsequent tasks")
def _cmd_plan(repl: Repl, args: List[str]) -> None:
    repl.session.plan_mode = not repl.session.plan_mode
    state = "ON" if repl.session.plan_mode else "OFF"
    repl._print(f"  plan mode → {state}")


@command("effort", "Set reasoning effort: /effort {low|medium|high}")
def _cmd_effort(repl: Repl, args: List[str]) -> None:
    if not args:
        repl._print(f"  current: {repl.session.effort}")
        return
    lvl = args[0].lower()
    if lvl not in {"low", "medium", "high"}:
        repl._print("  must be one of: low, medium, high")
        return
    repl.session.effort = lvl
    repl._print(f"  effort → {lvl}")


@command("memory", "Open MATRIOSKA.md in $EDITOR (or print path)")
def _cmd_memory(repl: Repl, args: List[str]) -> None:
    candidates = [
        Path.cwd() / "MATRIOSKA.md",
        Path(repl.cfg.work_dir) / "MATRIOSKA.md",
    ]
    target = next((p for p in candidates if p.exists()), candidates[0])
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor and shutil.which(editor.split()[0]):
        subprocess.run(f"{editor} {target}", shell=True)
    else:
        repl._print(f"  no $EDITOR set — file path: {target}")
        if target.exists():
            repl._print(target.read_text(encoding="utf-8")[:1200])


@command("init", "Run the interactive .env + MATRIOSKA.md scaffolder")
def _cmd_init(repl: Repl, args: List[str]) -> None:
    from matrioska.cli.init_wizard import run_init
    run_init(cwd=Path.cwd())


@command("diff", "Show generated artifacts from the last task (path + sizes)")
def _cmd_diff(repl: Repl, args: List[str]) -> None:
    res = repl.session.last_result
    if not res:
        repl._print("  no run yet in this session")
        return
    for a in res.get("artifacts", []) or []:
        repl._print(f"  {a.status:8s} {a.name}.{a.extension}  ({len(a.content):,} chars)")


@command("slots", "Status of the API slot pool (cooldowns, failure counts)")
def _cmd_slots(repl: Repl, args: List[str]) -> None:
    try:
        from matrioska.llm.circuit import SlotPool
        pool = SlotPool.from_config(repl.cfg)
        for s in pool.status():
            repl._print(
                f"  {s['label']:<28} ok={s['available']!s:5}  "
                f"failures={s['failures']}  "
                f"cooldown={s['cooldown_remaining']:.0f}s"
            )
    except Exception as e:
        repl._print(f"  slot pool unavailable: {e}")


@command("btw", "One-shot LLM query that does NOT touch the session context")
def _cmd_btw(repl: Repl, args: List[str]) -> None:
    if not args:
        repl._print("  usage: /btw <question>")
        return
    from matrioska.llm.client import LLMClient
    question = " ".join(args)
    llm = LLMClient(repl.cfg)
    try:
        resp = llm.chat(
            messages=[{"role": "user", "content": question}],
            model_spec=repl.cfg.effective_generator,
            system="You are a concise technical assistant. Answer in 1-5 short sentences.",
        )
        repl._print(resp.text.strip())
    except Exception as e:
        repl._print(f"  error: {e}")


@command(
    "vault",
    "Vault ops: search <q> [--scope] | project <n> | concept <n> | related <n> | doctor | list | graph",
)
def _cmd_vault(repl: Repl, args: List[str]) -> None:
    from matrioska.memory.vault import GlobalVault, default_vault_dir

    root = Path(repl.cfg.vault_dir).expanduser() if repl.cfg.vault_dir else default_vault_dir()
    vault = GlobalVault(root)

    # ── no subcommand → project listing ──────────────────────────────────
    if not args:
        repl._print(f"  vault root: [cyan]file://{vault.root}[/]" if repl._console
                    else f"  vault root: file://{vault.root}")
        projects = vault.list_projects()
        if not projects:
            repl._print("  (no projects yet)")
            return
        _vault_print_projects(repl, projects)
        return

    sub = args[0]
    rest = args[1:]

    if sub == "list":
        projects = vault.list_projects()
        if not projects:
            repl._print("  (no projects yet)")
            return
        _vault_print_projects(repl, projects)

    elif sub == "search":
        # Parse --scope flag
        scope = "all"
        query_parts: List[str] = []
        i = 0
        while i < len(rest):
            if rest[i] == "--scope" and i + 1 < len(rest):
                scope = rest[i + 1]
                i += 2
            else:
                query_parts.append(rest[i])
                i += 1
        query = " ".join(query_parts)
        if not query:
            repl._print("  usage: /vault search <query> [--scope local|global|linked|all]")
            return
        valid_scopes = {"local", "global", "linked", "all"}
        if scope not in valid_scopes:
            repl._print(f"  invalid scope '{scope}'. Choose from: {', '.join(sorted(valid_scopes))}")
            return
        results = vault.search(query, scope=scope, k=8)
        if not results:
            repl._print(f"  no results for '{query}' (scope={scope})")
            return
        repl._print(f"\n  [bold]Search:[/] {query}  [dim](scope={scope}, {len(results)} hits)[/]"
                    if repl._console else f"\n  Search: {query}  (scope={scope}, {len(results)} hits)")
        for r in results:
            path_str = str(vault.root / r["path"])
            link = f"file://{path_str}"
            score_badge = f"[{r['score']:.1f}]"
            kind_badge = r["kind"].upper()[:7]
            if repl._console:
                repl._print(
                    f"  {score_badge} [yellow]{kind_badge}[/]  "
                    f"[bold]{r['title']}[/]  [dim link={link}]{link}[/]"
                )
                repl._print(f"      [dim]{r['snippet'][:160]}[/]")
            else:
                repl._print(f"  {score_badge} {kind_badge}  {r['title']}  {link}")
                repl._print(f"      {r['snippet'][:160]}")

    elif sub == "project":
        if not rest:
            repl._print("  usage: /vault project <name>")
            return
        proj_name = rest[0]
        proj_dir = vault.projects_dir / proj_name
        if not proj_dir.is_dir():
            repl._print(f"  project '{proj_name}' not found in vault")
            return
        for note_name in ("architecture", "patterns"):
            note_path = proj_dir / f"{note_name}.md"
            link = f"file://{note_path}"
            if repl._console:
                repl._print(f"\n  [bold cyan]{note_name}.md[/]  [dim link={link}]{link}[/]")
            else:
                repl._print(f"\n  {note_name}.md  {link}")
            if note_path.exists():
                content = note_path.read_text(encoding="utf-8", errors="ignore")
                repl._print(content[:2000])
            else:
                repl._print("  (not found)")

    elif sub == "concept":
        if not rest:
            repl._print("  usage: /vault concept <name>")
            return
        concept_name = rest[0]
        concept_path = vault.concepts_dir / f"{concept_name}.md"
        link = f"file://{concept_path}"
        if not concept_path.exists():
            repl._print(f"  concept '{concept_name}' not found")
            repl._print("  known concepts: " + ", ".join(
                p.stem for p in vault.concepts_dir.glob("*.md")
            ) if vault.concepts_dir.exists() else "  concepts dir empty")
            return
        if repl._console:
            repl._print(f"\n  [bold cyan]{concept_name}.md[/]  [dim link={link}]{link}[/]")
        else:
            repl._print(f"\n  {concept_name}.md  {link}")
        repl._print(concept_path.read_text(encoding="utf-8", errors="ignore")[:2000])

    elif sub == "related":
        if not rest:
            repl._print("  usage: /vault related <project>")
            return
        proj_name = rest[0]
        related = sorted(vault._linked_projects(proj_name, max_hops=2) - {proj_name})
        if repl._console:
            repl._print(f"\n  [bold]Related to[/] [cyan]{proj_name}[/] (via wikilinks, ≤2 hops):")
        else:
            repl._print(f"\n  Related to {proj_name} (via wikilinks, <=2 hops):")
        if not related:
            repl._print("  (none found)")
            return
        for name in related:
            proj_path = vault.projects_dir / name
            link = f"file://{proj_path}"
            if repl._console:
                repl._print(f"    • [cyan]{name}[/]  [dim link={link}]{link}[/]")
            else:
                repl._print(f"    - {name}  {link}")

    elif sub == "doctor":
        d = vault.doctor()
        status_color = "green" if d["status"] == "healthy" else "red"
        if repl._console:
            repl._print(
                f"\n  [bold]Vault Health Report[/]\n"
                f"  projects=[cyan]{d['projects']}[/]  concepts=[cyan]{d['concepts']}[/]  "
                f"bugs=[cyan]{d['bugs']}[/]  total_notes=[cyan]{d['total_notes']}[/]\n"
                f"  status=[{status_color}]{d['status']}[/]"
            )
        else:
            repl._print(
                f"\n  Vault Health: projects={d['projects']} concepts={d['concepts']} "
                f"bugs={d['bugs']} total={d['total_notes']}  status={d['status']}"
            )
        if d["orphans"]:
            repl._print(f"  orphan notes ({len(d['orphans'])}):")
            for o in d["orphans"][:10]:
                repl._print(f"    - {o}")
        if d["broken_links"]:
            repl._print(f"  broken wikilinks ({len(d['broken_links'])}):")
            for bl in d["broken_links"][:10]:
                repl._print(f"    - {bl['from']} → [[{bl['target']}]]")
        if d.get("stale"):
            repl._print(f"  stale notes (>30d): {len(d['stale'])}")

    elif sub == "graph":
        out = vault.export_graph_mermaid()
        repl._print(out[:1500])

    else:
        repl._print(f"  unknown /vault subcommand: '{sub}'")
        repl._print("  subcommands: search | project | concept | related | doctor | list | graph")


def _vault_print_projects(repl: Repl, projects: List[Dict[str, Any]]) -> None:
    """Render vault project listing with Rich if available."""
    try:
        from rich.table import Table
        if repl._console is None:
            raise ImportError
        table = Table(title="Vault Projects", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Notes", justify="right")
        table.add_column("Last Run")
        table.add_column("Link", style="dim")
        vault_root = Path(repl.cfg.vault_dir).expanduser() if repl.cfg.vault_dir else None
        try:
            from matrioska.memory.vault import default_vault_dir
            vault_root = vault_root or default_vault_dir()
        except Exception:
            vault_root = Path.home() / ".matrioska" / "vault"
        for p in projects:
            proj_path = vault_root / "projects" / p["name"]
            table.add_row(
                p["name"],
                str(p["notes"]),
                p["last_run"],
                f"file://{proj_path}",
            )
        repl._console.print(table)
    except (ImportError, Exception):
        for p in projects:
            repl._print(f"  {p['name']:<30} notes={p['notes']}  last={p['last_run']}")


@command("rewind", "Revert to a previous pipeline checkpoint")
def _cmd_rewind(repl: Repl, args: List[str]) -> None:
    from matrioska.core.state import StateGraph

    graph = StateGraph(Path(repl.cfg.work_dir))
    checkpoints = graph.list_checkpoints()
    if not checkpoints:
        repl._print("  no checkpoints found in work_dir")
        return

    # If a specific checkpoint id or index is given, use it; otherwise show the list
    target_id: Optional[str] = None
    if args:
        raw = args[0]
        # Accept numeric index (1-based)
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(checkpoints):
                target_id = checkpoints[idx]["id"]
            else:
                repl._print(f"  index out of range (1–{len(checkpoints)})")
                return
        else:
            # Accept checkpoint id prefix
            matches = [c for c in checkpoints if c["id"].startswith(raw)]
            if not matches:
                repl._print(f"  no checkpoint matching '{raw}'")
                return
            target_id = matches[0]["id"]
    else:
        # No arg — show table and prompt, or fall back to most recent
        repl._print("\n  Available checkpoints:")
        for i, cp in enumerate(checkpoints, 1):
            repl._print(
                f"    {i}. [{cp['id'][:8]}]  {cp['label']:<22}  {cp['status']:<12}  {cp['created_at'][:19]}"
            )
        repl._print("\n  Usage: /rewind <index|id-prefix>  (no arg = load most recent)")
        # Default to most recent (last in list)
        target_id = checkpoints[-1]["id"]
        repl._print(f"  → loading most recent: [{target_id[:8]}] {checkpoints[-1]['label']}")

    try:
        state = graph.load_checkpoint(target_id)
        repl._print(
            f"  rewound to checkpoint [{target_id[:8]}]  "
            f"status={state.status.value}  project={state.project_name or '—'}"
        )
        # Store in session so subsequent /diff can reference it
        repl.session.last_result = {
            "status": state.status.value,
            "project_name": state.project_name,
            "artifacts": list(state.artifacts.values()),
            "shared_state": state.shared_state,
            "tokens": {},
        }
    except FileNotFoundError as exc:
        repl._print(f"  checkpoint not found: {exc}")
    except Exception as exc:
        repl._print(f"  failed to load checkpoint: {exc}")


@command("stream", "Toggle token streaming on/off for the next task")
def _cmd_stream(repl: Repl, args: List[str]) -> None:
    repl.cfg.stream_tokens = not repl.cfg.stream_tokens
    repl._print(f"  stream_tokens → {repl.cfg.stream_tokens}")


@command("history", "Show this session's input history")
def _cmd_history(repl: Repl, args: List[str]) -> None:
    for i, h in enumerate(repl.session.history[-20:], 1):
        repl._print(f"  {i:3d}. {h[:100]}")


# ── Entry point ───────────────────────────────────────────────────────────────


def run_repl(cfg: Optional[Config] = None) -> int:
    """Start the REPL with the given config (or load defaults)."""
    return Repl(cfg).run()

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
        self._fire_hook("session_start", {"provider": self.cfg.provider, "model": self.cfg.model})

        try:
            return self._run_loop(get_input)
        finally:
            self._fire_hook("session_end", {
                "tokens_prompt": self.session.tokens_session["prompt"],
                "tokens_completion": self.session.tokens_session["completion"],
                "history_len": len(self.session.history),
            })

    def _run_loop(self, get_input: Callable[[], Optional[str]]) -> int:
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

    def _build_completer(self) -> Any:
        """Build a NestedCompleter for slash commands and their sub-arguments."""
        try:
            from prompt_toolkit.completion import NestedCompleter
        except ImportError:
            return None

        vault_sub = {"list": None, "search": None, "doctor": None, "graph": None}
        effort_sub = {"low": None, "medium": None, "high": None}

        # Build from registered COMMANDS so it stays in sync automatically
        slash_map: Dict[str, Any] = {}
        sub_args: Dict[str, Any] = {
            "effort": effort_sub,
            "vault": vault_sub,
            "model": None,    # free-form text
            "btw": None,
            "history": None,
        }
        for name in COMMANDS:
            slash_map[f"/{name}"] = sub_args.get(name)

        return NestedCompleter.from_nested_dict(slash_map)

    def _build_input_reader(self) -> Callable[[], Optional[str]]:
        """Return a callable that reads one (possibly multi-line) input.

        Tries prompt_toolkit first (history, multi-line, key bindings,
        slash-command autocompletion); falls back to plain `input()` if
        prompt_toolkit isn't installed or stdin isn't a TTY.

        Keyboard shortcuts (prompt_toolkit):
          TAB / ↓       open / navigate completion menu
          Enter         select completion or submit input
          Esc           dismiss completion menu
          Ctrl+D        exit REPL
          Esc+Enter     insert newline (multi-line input)
        """
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.styles import Style
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

        @kb.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        completer = self._build_completer()

        # Muted style so the completion menu doesn't overwhelm the prompt
        style = Style.from_dict({
            "completion-menu.completion": "bg:#1e1e2e fg:#cdd6f4",
            "completion-menu.completion.current": "bg:#313244 fg:#cba6f7 bold",
            "completion-menu.meta.completion": "bg:#1e1e2e fg:#6c7086",
            "scrollbar.background": "bg:#313244",
            "scrollbar.button": "bg:#585b70",
        })

        session = PromptSession(
            history=FileHistory(str(history_path)),
            multiline=False,
            key_bindings=kb,
            completer=completer,
            complete_while_typing=True,   # show menu as user types /
            style=style,
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

    def _fire_hook(self, name: str, context: Dict[str, Any]) -> None:
        try:
            from matrioska.hooks import HookRunner
            runner = HookRunner(project_dir=Path.cwd())
            runner.run(name, context)
        except Exception:
            pass

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


@command("vault", "Vault ops: /vault search <q> | list | doctor | graph")
def _cmd_vault(repl: Repl, args: List[str]) -> None:
    from matrioska.memory.vault import GlobalVault, default_vault_dir
    root = Path(repl.cfg.vault_dir).expanduser() if repl.cfg.vault_dir else default_vault_dir()
    vault = GlobalVault(root)
    if not args:
        repl._print(f"  vault: {vault.root}")
        for p in vault.list_projects():
            repl._print(f"    {p['name']:<30} notes={p['notes']}  last={p['last_run']}")
        return
    sub = args[0]
    rest = args[1:]
    if sub == "list":
        for p in vault.list_projects():
            repl._print(f"  {p['name']:<30} notes={p['notes']}  last={p['last_run']}")
    elif sub == "search":
        if not rest:
            repl._print("  usage: /vault search <query>")
            return
        for r in vault.search(" ".join(rest), scope="all", k=5):
            repl._print(f"  [{r['score']:.1f}] {r['kind']:7s} {r['title']}")
            repl._print(f"      {r['snippet'][:120]}")
    elif sub == "doctor":
        d = vault.doctor()
        repl._print(
            f"  projects={d['projects']} concepts={d['concepts']} bugs={d['bugs']} "
            f"total={d['total_notes']}  status={d['status']}"
        )
        if d["orphans"]:
            repl._print(f"  orphans: {len(d['orphans'])}")
        if d["broken_links"]:
            repl._print(f"  broken_links: {len(d['broken_links'])}")
    elif sub == "graph":
        out = vault.export_graph_mermaid()
        repl._print(out[:1500])
    else:
        repl._print(f"  unknown /vault subcommand: {sub}")


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

"""
Matrioska CLI Dashboard — Rich-based live cockpit.

Layout:
  ┌─ header ──────────────────────────────────────────┐
  │ task + status + elapsed + key hints                │
  ├─ api_slots ──────┬─ progress ─────────────────────┤
  │ slot table       │ phase progress + file list      │
  │ tokens / cost    │                                 │
  ├─ events ─────────┴─────────────────────────────────┤
  │ scrolling event log                                 │
  └─────────────────────────────────────────────────────┘

  When paused, a pause overlay replaces the events panel:
  ┌─ PAUSED ───────────────────────────────────────────┐
  │ [m] model   [e] effort   [r] max_repairs  [s] stream│
  │ [p/Space] resume         [q] abort                  │
  └─────────────────────────────────────────────────────┘

Keyboard controls (requires a real TTY):
  p / Space   — pause / resume
  q / Esc     — abort (first press shows confirmation, second confirms)
  During pause:
    m          — edit model (drops into input line)
    e          — cycle effort  low → medium → high → low
    r          — cycle max_repairs  1 → 2 → 3 → 5 → 1
    s          — toggle stream_tokens
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ── State ─────────────────────────────────────────────────────────────────────


@dataclass
class SlotState:
    label: str
    provider: str
    model: str
    available: bool = True
    cooldown_remaining: float = 0.0
    failures: int = 0
    calls: int = 0


@dataclass
class FileState:
    name: str
    status: str = "pending"   # pending | generating | done | failed
    chars: int = 0
    attempts: int = 0
    elapsed_s: float = 0.0


@dataclass
class DashboardState:
    task: str = ""
    project_name: str = ""
    pipeline_status: str = "starting"
    phase: int = 0                       # 0=init, 1=arch, 2=gen, 3=verify
    started_at: float = field(default_factory=time.time)

    # API slots
    slots: List[SlotState] = field(default_factory=list)
    total_calls: int = 0

    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Files
    files: List[FileState] = field(default_factory=list)
    current_file: str = ""

    # Events log (newest last). Large buffer — display window computed per render.
    events: Deque[str] = field(default_factory=lambda: deque(maxlen=50))

    # Control
    paused: bool = False
    abort_pending: bool = False   # first q pressed — waiting for confirm
    abort_pending_at: float = 0.0
    abort_requested: bool = False

    # Final
    done: bool = False
    final_status: str = ""

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def add_event(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[dim]{ts}[/dim]  {msg}")

    def update_slot(self, label: str, **kwargs: Any) -> None:
        for s in self.slots:
            if s.label == label:
                for k, v in kwargs.items():
                    setattr(s, k, v)
                return


# ── Keyboard controller ────────────────────────────────────────────────────────


class KeyboardController:
    """Non-blocking raw keyboard reader. No-op when stdin is not a TTY."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._old_settings: Any = None
        self._fd: int = -1
        self._active = False

    def start(self) -> bool:
        if not sys.stdin.isatty():
            return False
        try:
            import termios, tty
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            # setcbreak: chars come through immediately; Ctrl+C still raises SIGINT
            tty.setcbreak(self._fd)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self._active = True
            return True
        except Exception:
            return False

    def stop(self) -> None:
        self._stop.set()
        self._active = False
        if self._old_settings is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def restore_cooked(self) -> None:
        """Temporarily restore cooked mode for console.input() calls."""
        if self._old_settings is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def set_cbreak(self) -> None:
        """Re-enter cbreak mode after restore_cooked()."""
        try:
            import termios, tty
            tty.setcbreak(self._fd)
        except Exception:
            pass

    def _loop(self) -> None:
        import select
        while not self._stop.is_set():
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    self._queue.put(ch)
            except Exception:
                break

    def get(self) -> Optional[str]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def flush(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# ── Config helpers ─────────────────────────────────────────────────────────────

_EFFORT_LEVELS = ["low", "medium", "high"]


def _effort_label(cfg: Any) -> str:
    if not cfg.enable_tot and cfg.architect_candidates <= 1:
        return "low"
    if cfg.enable_tot and cfg.architect_candidates >= 5:
        return "high"
    return "medium"


def _apply_effort(cfg: Any, level: str) -> None:
    if level == "low":
        cfg.enable_tot = False
        cfg.enable_reflexion = False
        cfg.enable_test_design = False
        cfg.architect_candidates = 1
    elif level == "high":
        cfg.enable_tot = True
        cfg.enable_reflexion = True
        cfg.enable_test_design = True
        cfg.architect_candidates = 5
    else:  # medium
        cfg.enable_tot = True
        cfg.enable_reflexion = True
        cfg.enable_test_design = True
        cfg.architect_candidates = 3


def _cycle_effort(state: DashboardState, cfg: Any) -> None:
    current = _effort_label(cfg)
    idx = _EFFORT_LEVELS.index(current)
    next_level = _EFFORT_LEVELS[(idx + 1) % len(_EFFORT_LEVELS)]
    _apply_effort(cfg, next_level)
    state.add_event(f"[cyan]effort → {next_level}[/]")


_REPAIR_CYCLE = [1, 2, 3, 5]


def _cycle_max_repairs(state: DashboardState, cfg: Any) -> None:
    current = cfg.max_repairs
    try:
        idx = _REPAIR_CYCLE.index(current)
    except ValueError:
        idx = 0
    next_val = _REPAIR_CYCLE[(idx + 1) % len(_REPAIR_CYCLE)]
    cfg.max_repairs = next_val
    state.add_event(f"[cyan]max_repairs → {next_val}[/]")


def _edit_model(
    live: Live,
    keyboard: KeyboardController,
    state: DashboardState,
    cfg: Any,
    console: Console,
) -> None:
    """Stop Live, drop to cooked mode, get new model name, resume."""
    keyboard.restore_cooked()
    live.stop()
    try:
        console.print(f"\n[bold]Current model:[/] {cfg.model}")
        new = console.input("[bold cyan]New model[/] (empty = keep): ").strip()
        if new:
            cfg.model = new
            state.add_event(f"[cyan]model → {new}[/]")
        else:
            state.add_event("[dim]model unchanged[/]")
    except (KeyboardInterrupt, EOFError):
        state.add_event("[dim]model edit cancelled[/]")
    finally:
        keyboard.set_cbreak()
        live.start()


# ── Renderer ──────────────────────────────────────────────────────────────────

_PHASE_LABELS = {0: "Init", 1: "Architecture", 2: "Generation", 3: "Verification"}

_STATUS_STYLE = {
    "done": "green",
    "failed": "red",
    "aborted": "yellow",
    "generating": "yellow",
    "pending": "dim",
    "starting": "cyan",
    "partial": "yellow",
    "success": "green",
    "planning": "cyan",
    "paused": "yellow",
}


def _phase_bar(state: DashboardState) -> Text:
    phases = ["Arch", "Gen", "Verify"]
    parts: List[Text] = []
    for i, name in enumerate(phases, 1):
        if i < state.phase:
            parts.append(Text(f"✓ {name}", style="green"))
        elif i == state.phase:
            parts.append(Text(f"⟳ {name}", style="bold yellow"))
        else:
            parts.append(Text(f"○ {name}", style="dim"))
        if i < 3:
            parts.append(Text(" → ", style="dim"))
    result = Text()
    for p in parts:
        result.append_text(p)
    return result


def _render_header(state: DashboardState) -> Panel:
    elapsed = state.elapsed()
    m, s = divmod(int(elapsed), 60)
    elapsed_str = f"{m:02d}:{s:02d}"

    task_preview = state.task[:80] + "…" if len(state.task) > 80 else state.task
    proj = f"[bold cyan]{state.project_name}[/]  " if state.project_name else ""
    disp_status = "paused" if state.paused else state.pipeline_status
    status_style = _STATUS_STYLE.get(disp_status.lower(), "white")

    t = Text()
    t.append(f"{proj}")
    t.append(f"[{disp_status.upper()}]", style=f"bold {status_style}")
    t.append(f"  ⏱ {elapsed_str}", style="dim")

    # Keyboard hints (only in live mode)
    if not state.done:
        if state.paused:
            t.append("  │  [p] resume  [m/e/r/s] config  [q] abort", style="dim yellow")
        elif state.abort_pending:
            t.append("  │  [q] confirm abort  [any] cancel", style="bold red")
        else:
            t.append("  │  [p] pause  [q] abort", style="dim")

    t.append("\n")
    t.append(task_preview, style="italic dim")
    t.append("\n\n")
    t.append_text(_phase_bar(state))

    return Panel(t, title="[bold]MATRIOSKA V3[/]", border_style="blue", padding=(0, 1))


def _render_slots(state: DashboardState) -> Panel:
    tbl = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 0), show_edge=False)
    tbl.add_column("Slot", style="bold", no_wrap=True, min_width=10)
    tbl.add_column("St", justify="center", min_width=4)
    tbl.add_column("CD", justify="right", min_width=5)
    tbl.add_column("N", justify="right", style="dim", min_width=3)

    for s in state.slots:
        if s.failures >= 5:
            indicator = Text("✗", style="red")
        elif not s.available:
            indicator = Text("⏸", style="yellow")
        else:
            indicator = Text("●", style="green")

        cd = s.cooldown_remaining
        cd_text = Text(f"{cd:.0f}s", style="yellow") if cd > 0 else Text("—", style="dim")

        tbl.add_row(s.label, indicator, cd_text, str(s.calls))

    tok = state.prompt_tokens + state.completion_tokens
    cost = state.estimated_cost_usd
    cost_style = "bold green" if cost < 0.05 else "bold yellow"

    tok_text = Text()
    tok_text.append("─" * 22 + "\n", style="dim")
    tok_text.append(f" ↑{state.prompt_tokens:,} ↓{state.completion_tokens:,}\n", style="dim")
    tok_text.append(f" {tok:,} tokens  ", style="bold")
    tok_text.append(f"~${cost:.4f}", style=cost_style)

    return Panel(Group(tbl, tok_text), title="[bold]API SLOTS[/]", border_style="cyan", padding=(0, 1))


def _render_progress(state: DashboardState, max_rows: int = 14) -> Panel:
    lines: List[Text] = []

    if not state.files:
        lines.append(Text("Waiting for architecture…", style="dim italic"))
    else:
        done_files    = [f for f in state.files if f.status == "done"]
        failed_files  = [f for f in state.files if f.status == "failed"]
        active_files  = [f for f in state.files if f.status == "generating"]
        pending_files = [f for f in state.files if f.status == "pending"]

        total  = len(state.files)
        n_done = len(done_files)
        pct    = n_done / total if total else 0

        # Progress bar
        bar_w  = 24
        filled = int(bar_w * pct)
        bar = Text()
        bar.append("█" * filled, style="green")
        bar.append("░" * (bar_w - filled), style="dim")
        bar.append(f"  {n_done}/{total}", style="bold")
        if failed_files:
            bar.append(f"  {len(failed_files)} failed", style="red")
        if pending_files:
            bar.append(f"  {len(pending_files)} pending", style="dim")
        lines.append(bar)
        lines.append(Text(""))

        def _file_row(f: FileState) -> Text:
            icon, style = {
                "done":      ("✓", "green"),
                "failed":    ("✗", "red"),
                "generating":("⟳", "bold yellow"),
                "pending":   ("○", "dim"),
            }.get(f.status, ("?", "dim"))
            row = Text()
            row.append(f"  {icon} ", style=style)
            row.append(f.name, style="bold" if f.status == "generating" else "")
            if f.status == "done":
                row.append(f"  {f.chars:,}c", style="dim")
                if f.attempts > 1:
                    row.append(f"  ({f.attempts}×)", style="dim red")
            elif f.status == "generating":
                row.append("  …", style="yellow blink")
            elif f.status == "failed":
                row.append("  ✗ exhausted", style="red dim")
            return row

        # Priority: active > failed > recent done > pending summary
        budget = max_rows - 2  # subtract bar + blank line
        shown: List[FileState] = []
        shown.extend(active_files)
        shown.extend(failed_files)

        # Fill remaining budget with most-recently-done files
        done_budget = budget - len(shown)
        hidden_done = 0
        if done_budget > 0:
            shown.extend(done_files[-done_budget:])
            hidden_done = max(0, n_done - done_budget)
        else:
            hidden_done = n_done

        for f in shown:
            lines.append(_file_row(f))

        # Summaries for hidden files
        if hidden_done > 0:
            t = Text(f"  ✓ …and {hidden_done} more done", style="dim green")
            lines.append(t)
        if pending_files:
            t = Text(f"  ○ {len(pending_files)} pending", style="dim")
            lines.append(t)

    return Panel(Group(*lines), title="[bold]PROGRESS[/]", border_style="green", padding=(0, 1))


def _render_events(state: DashboardState, max_lines: int = 14) -> Panel:
    # Always show the *newest* max_lines events (auto-scroll to bottom).
    visible = list(state.events)[-max_lines:]
    lines = [Text.from_markup(e) for e in visible]
    if not lines:
        lines = [Text("No events yet.", style="dim italic")]
    hidden = len(state.events) - len(visible)
    if hidden > 0:
        lines.insert(0, Text(f"  ↑ {hidden} older events", style="dim"))
    return Panel(Group(*lines), title="[bold]EVENTS[/]", border_style="dim", padding=(0, 1))


def _render_pause_overlay(state: DashboardState, cfg: Any) -> Panel:
    """Config panel shown when the pipeline is paused."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", min_width=3)
    grid.add_column(style="dim", min_width=14)
    grid.add_column(style="bold")

    effort = _effort_label(cfg)
    stream = "on" if cfg.stream_tokens else "off"

    grid.add_row("m", "model", cfg.model)
    grid.add_row("e", "effort", effort)
    grid.add_row("r", "max repairs", str(cfg.max_repairs))
    grid.add_row("s", "stream tokens", stream)

    hint = Text("\n[p / Space] resume   [q / Esc] abort", style="dim")

    if state.abort_pending:
        title = "[bold red]⚠ Press q again to abort[/]"
        border = "red"
    else:
        title = "[bold yellow]⏸ PAUSED — edit config[/]"
        border = "yellow"

    return Panel(Group(grid, hint), title=title, border_style=border, padding=(0, 2))


def render(state: DashboardState, term_height: int = 40) -> Layout:
    """Render the full cockpit layout.

    Sizes are computed dynamically so every panel gets as much vertical space
    as the terminal offers — events and progress auto-scroll to newest content.
    """
    HEADER_H = 6

    # Middle panel: enough for slots or files, capped so events always visible
    min_middle = max(len(state.slots) + 4, min(len(state.files) + 4, 14), 8)
    max_middle = max(min_middle, term_height - HEADER_H - 10)
    middle_h   = min(min_middle, max_middle)

    # Events: the remainder, min 8
    events_h   = max(8, term_height - HEADER_H - middle_h - 1)

    # Content budgets (subtract borders/padding)
    max_file_rows  = max(4, middle_h - 4)
    max_event_lines = max(4, events_h - 2)

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=HEADER_H),
        Layout(name="middle", size=middle_h),
        Layout(name="events", size=events_h),
    )
    layout["middle"].split_row(
        Layout(name="slots", minimum_size=30, ratio=2),
        Layout(name="progress", ratio=3),
    )

    layout["header"].update(_render_header(state))
    layout["slots"].update(_render_slots(state))
    layout["progress"].update(_render_progress(state, max_rows=max_file_rows))
    layout["events"].update(_render_events(state, max_lines=max_event_lines))

    return layout


# ── EventBus subscriber ───────────────────────────────────────────────────────


class DashboardSubscriber:
    """Translates EventBus events into DashboardState mutations."""

    def __init__(self, state: DashboardState) -> None:
        self._state = state
        self._lock = threading.Lock()

    def __call__(self, event: Any) -> None:
        name = getattr(event, "name", "") or ""
        data = getattr(event, "data", {}) or {}
        self._handle(name, data)

    def _handle(self, name: str, data: Dict[str, Any]) -> None:
        s = self._state
        with self._lock:
            if name == "run_start":
                s.pipeline_status = "starting"
                s.phase = 1
                s.add_event(f"[cyan]Run started[/]  model={data.get('model', '?')}")

            elif name == "phase1_done":
                s.phase = 2
                proj = data.get("project_name", "")
                n = data.get("num_files", 0)
                s.project_name = proj
                s.pipeline_status = "generating"
                s.add_event(f"[green]Architecture[/]  {proj}  {n} files planned")

            elif name == "phase2_done":
                s.phase = 3
                ok = data.get("all_success", False)
                s.pipeline_status = "verifying"
                s.add_event(f"[green]Generation done[/]  success={ok}")

            elif name == "phase3_done":
                ok = data.get("overall_ok", False)
                s.pipeline_status = "success" if ok else "partial"
                s.add_event(f"{'[green]✓ Verified[/]' if ok else '[yellow]⚠ Partial[/]'}")

            elif name == "run_end":
                s.done = True
                s.final_status = data.get("status", "done")

            elif name == "pipeline_paused":
                s.paused = True
                s.pipeline_status = "paused"

            elif name == "pipeline_resumed":
                s.paused = False

            elif name == "file_generated":
                fname = data.get("file", "?")
                st = data.get("status", "?")
                chars = data.get("chars", 0)
                attempts = data.get("attempts", 1)
                elapsed = data.get("elapsed_s", 0)
                for f in s.files:
                    if f.name == fname or f.name + "." in fname or fname.startswith(f.name):
                        f.status = st
                        f.chars = chars
                        f.attempts = attempts
                        f.elapsed_s = elapsed
                        break
                style = "green" if st == "done" else "red"
                repair_note = f"  ({attempts} attempts)" if attempts > 1 else ""
                s.add_event(
                    f"[{style}]{'✓' if st == 'done' else '✗'} {fname}[/]"
                    f"  {chars:,}c  {elapsed:.1f}s{repair_note}"
                )

            elif name == "agent_call":
                agent = data.get("agent", "?")
                if agent in ("generator", "generator_json"):
                    for f in s.files:
                        if f.status == "pending":
                            f.status = "generating"
                            s.current_file = f.name
                            break

            elif name == "llm_done":
                pt = data.get("prompt_tokens", 0)
                ct = data.get("completion_tokens", 0)
                s.prompt_tokens += pt
                s.completion_tokens += ct
                s.total_calls += 1
                actual = data.get("actual_cost_usd")
                if actual is not None:
                    s.estimated_cost_usd += float(actual)
                else:
                    # Rough in-dashboard estimate (overridden by final snapshot)
                    s.estimated_cost_usd += (pt * 0.05 + ct * 0.08) / 1_000_000
                slot_label = data.get("slot", "")
                if slot_label:
                    s.update_slot(slot_label, calls=s.total_calls)

            elif name == "llm_rate_limited":
                slot = data.get("slot", "?")
                retry = data.get("retry_after", 0)
                s.update_slot(slot, available=False, cooldown_remaining=retry)
                s.add_event(f"[yellow]⏸ Rate limit[/]  {slot}  cooldown {retry:.0f}s")

            elif name == "llm_retriable_error":
                slot = data.get("slot", "?")
                err = str(data.get("error", ""))[:60]
                s.add_event(f"[red]⚠ Error[/]  {slot}  {err}")


# ── Key handling ──────────────────────────────────────────────────────────────

_ABORT_CONFIRM_TTL = 3.0   # seconds before abort confirmation expires


def _handle_key(
    ch: str,
    state: DashboardState,
    cfg: Any,
    pause_event: threading.Event,
    abort_event: threading.Event,
    live: Live,
    keyboard: KeyboardController,
    console: Console,
) -> None:
    now = time.time()

    # Cancel stale abort confirmation
    if state.abort_pending and (now - state.abort_pending_at) > _ABORT_CONFIRM_TTL:
        state.abort_pending = False
        state.add_event("[dim]Abort cancelled (timeout)[/]")

    if ch in ("p", " "):
        if state.paused:
            # Resume
            state.paused = False
            state.abort_pending = False
            pause_event.clear()
            state.add_event("[green]▶ Resumed[/]")
        else:
            # Pause
            state.paused = True
            pause_event.set()
            state.add_event("[yellow]⏸ Paused  [p] resume  [m/e/r/s] config  [q] abort[/]")

    elif ch in ("q", "\x1b", "\x03"):   # q, Esc, Ctrl+C
        if state.abort_pending or state.paused:
            # Confirmed abort
            abort_event.set()
            pause_event.clear()   # unblock orchestrator so it can check abort
            state.abort_requested = True
            state.abort_pending = False
            state.add_event("[bold red]✗ Pipeline aborted by user[/]")
        else:
            # First press — request confirmation
            state.abort_pending = True
            state.abort_pending_at = now
            state.add_event("[yellow]⚠ Press q again within 3 s to abort[/]")

    elif state.paused:
        # Config editing only while paused
        if ch == "m":
            _edit_model(live, keyboard, state, cfg, console)
        elif ch == "e":
            _cycle_effort(state, cfg)
        elif ch == "r":
            _cycle_max_repairs(state, cfg)
        elif ch == "s":
            cfg.stream_tokens = not cfg.stream_tokens
            state.add_event(f"[cyan]stream_tokens → {cfg.stream_tokens}[/]")

    elif state.abort_pending:
        # Any key other than q/Esc cancels the confirmation
        state.abort_pending = False
        state.add_event("[dim]Abort cancelled[/]")


# ── Main entry point ──────────────────────────────────────────────────────────


def run_with_dashboard(
    cfg: Any,
    task: str,
    console: Optional[Console] = None,
) -> Dict[str, Any]:
    """Run the Matrioska pipeline with a live Rich dashboard.

    Subscribes to the EventBus, refreshes the layout every 0.1 s,
    and returns the full result dict when done.

    Keyboard controls (requires a real TTY):
      p / Space   pause / resume
      q / Esc     abort (first press = confirm; second press within 3 s = abort)
      m / e / r / s   config edits (model, effort, max_repairs, stream) — pause first
    """
    from matrioska.pipeline.orchestrator import Matrioska, PipelineAborted
    from matrioska.core.events import EventBus

    console = console or Console()
    state = DashboardState(task=task)

    # Control events wired into the orchestrator
    pause_event = threading.Event()
    abort_event = threading.Event()

    m = Matrioska(cfg, pause_event=pause_event, abort_event=abort_event)

    # Initialize slot states from the pool
    try:
        for s in m.llm._slot_pool.status():
            state.slots.append(SlotState(
                label=s["label"],
                provider=s["provider"],
                model=s["model"],
                available=s["available"],
                cooldown_remaining=s["cooldown_remaining"],
            ))
    except Exception:
        pass

    # Subscribe dashboard to EventBus
    subscriber = DashboardSubscriber(state)
    m.bus.on("*", subscriber)

    # Slot cooldown refresh thread (polls pool every 0.5 s)
    _stop_refresh = threading.Event()

    def _refresh_slots() -> None:
        while not _stop_refresh.is_set():
            try:
                for info in m.llm._slot_pool.status():
                    state.update_slot(
                        info["label"],
                        available=info["available"],
                        cooldown_remaining=info["cooldown_remaining"],
                        failures=info["failures"],
                    )
            except Exception:
                pass
            _stop_refresh.wait(0.5)

    refresh_thread = threading.Thread(target=_refresh_slots, daemon=True)
    refresh_thread.start()

    # Pipeline thread
    result_container: Dict[str, Any] = {}
    error_container: Dict[str, Any] = {}

    def _run_pipeline() -> None:
        try:
            result_container["result"] = m.run(task)
        except PipelineAborted:
            state.done = True
            state.pipeline_status = "aborted"
            state.final_status = "aborted"
            state.add_event("[bold yellow]Pipeline stopped[/]")
        except Exception as exc:
            error_container["exc"] = exc
            state.done = True
            state.pipeline_status = "failed"
            state.final_status = "error"
            state.add_event(f"[red bold]✗ Pipeline error: {exc}[/]")

    pipeline_thread = threading.Thread(target=_run_pipeline, daemon=True)

    # Keyboard controller
    keyboard = KeyboardController()
    kb_active = keyboard.start()

    try:
        with Live(
            render(state, console.size.height),
            console=console,
            refresh_per_second=10,
            screen=False,
            vertical_overflow="visible",
        ) as live:
            pipeline_thread.start()

            while pipeline_thread.is_alive():
                # Keyboard input
                if kb_active:
                    while True:
                        ch = keyboard.get()
                        if ch is None:
                            break
                        _handle_key(
                            ch, state, cfg,
                            pause_event, abort_event,
                            live, keyboard, console,
                        )

                # Architecture sync
                if state.phase >= 2 and not state.files:
                    try:
                        arch = m.graph.load_latest()
                        if arch and arch.state.get("architecture"):
                            for f in arch.state["architecture"].get("files", []):
                                state.files.append(FileState(
                                    name=f"{f['name']}.{f['extension']}"
                                ))
                    except Exception:
                        pass

                # Render: recompute layout each frame using current terminal height
                th = console.size.height
                if state.paused or state.abort_pending:
                    live.update(Group(render(state, th), _render_pause_overlay(state, cfg)))
                else:
                    live.update(render(state, th))

                time.sleep(0.1)

            # Final render
            state.done = True
            if "result" in result_container:
                r = result_container["result"]
                state.final_status = r.get("status", "done")
                state.pipeline_status = state.final_status
                tok = r.get("tokens", {})
                if tok:
                    state.prompt_tokens = tok.get("prompt_tokens", state.prompt_tokens)
                    state.completion_tokens = tok.get("completion_tokens", state.completion_tokens)
                    # Prefer actual cost from provider; fall back to estimate
                    final_cost = tok.get("actual_cost_usd") or tok.get("estimated_cost_usd")
                    if final_cost is not None:
                        state.estimated_cost_usd = final_cost

            live.update(render(state, console.size.height))

    except KeyboardInterrupt:
        abort_event.set()
        pause_event.clear()
        state.abort_requested = True
    finally:
        keyboard.stop()
        _stop_refresh.set()

    if state.final_status == "aborted":
        return {"status": "aborted", "artifacts": [], "shared_state": {}}

    if "exc" in error_container:
        raise error_container["exc"]

    return result_container.get("result", {})

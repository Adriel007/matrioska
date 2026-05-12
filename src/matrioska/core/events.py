"""
Event system for observable pipelines.

Every LLM call, tool use, state change, and checkpoint is emitted as
a typed event — enabling OpenTelemetry spans, LangFuse traces, and
real-time progress reporting (§4.8 of the plan).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ── Event Types ──────────────────────────────────────────────────────────────


@dataclass
class Event:
    name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"ts": self.timestamp, "event": self.name, **self.data},
                         ensure_ascii=False, default=str)


# ── Event Bus ────────────────────────────────────────────────────────────────


Handler = Callable[[Event], None]


class EventBus:
    """In-process publish/subscribe event bus.

    Listeners register for event name patterns (glob-style).
    The bus is synchronous — handlers run in the publisher's thread.
    """

    def __init__(self):
        self._handlers: Dict[str, List[Handler]] = {}
        self._lock = threading.Lock()
        self._metrics: Dict[str, int] = {}

    def on(self, event_name: str, handler: Handler) -> None:
        with self._lock:
            self._handlers.setdefault(event_name, []).append(handler)

    def off(self, event_name: str, handler: Handler) -> None:
        with self._lock:
            if event_name in self._handlers:
                self._handlers[event_name].remove(handler)

    def emit(self, event: Event) -> None:
        with self._lock:
            self._metrics[event.name] = self._metrics.get(event.name, 0) + 1
            handlers = (
                list(self._handlers.get(event.name, []))
                + list(self._handlers.get("*", []))
            )
        for h in handlers:
            try:
                h(event)
            except Exception:
                pass  # A handler must not crash the pipeline

    def emit_named(self, name: str, **data: Any) -> None:
        self.emit(Event(name=name, data=data))

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._metrics)


# ── Built-in Handlers ────────────────────────────────────────────────────────


class JSONLRecorder:
    """Records every event to a JSONL file for post-hoc analysis."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        self.path = self.log_dir / f"run-{ts}.jsonl"
        self._lock = threading.Lock()
        self._count = 0

    def __call__(self, event: Event) -> None:
        with self._lock:
            self.path.open("a", encoding="utf-8").write(event.to_json() + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        return self._count


class TokenTracker:
    """Accumulates token usage across all LLM calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_calls = 0
        self.total_cost = 0.0

    def record(self, prompt: int, completion: int, model: str = "") -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_calls += 1
            self.total_cost += _estimate_cost(model, prompt, completion)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "total_calls": self.total_calls,
                "estimated_cost_usd": round(self.total_cost, 4),
            }

    def __call__(self, event: Event) -> None:
        if event.name == "llm_done":
            self.record(
                prompt=event.data.get("prompt_tokens", 0),
                completion=event.data.get("completion_tokens", 0),
                model=event.data.get("model", ""),
            )


def _estimate_cost(model: str, prompt: int, completion: int) -> float:
    # Conservative pricing estimates (USD per 1M tokens)
    PRICING: Dict[str, tuple] = {
        "gpt-4.5": (75, 150), "gpt-4o": (2.5, 10), "gpt-4o-mini": (0.15, 0.6),
        "claude-opus-4": (15, 75), "claude-sonnet-4": (3, 15), "claude-haiku-4.5": (1, 5),
        "llama-3.1": (0.2, 0.2), "llama-3.3": (0.2, 0.2),
    }
    for prefix, (pp, cp) in PRICING.items():
        if prefix in model:
            return (prompt / 1_000_000) * pp + (completion / 1_000_000) * cp
    return 0.0


class MetricsCollector:
    """Collects pipeline metrics for evaluation (§4.8)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.contract_checks: List[bool] = []
        self.first_pass: List[bool] = []
        self.repair_success: List[bool] = []
        self.execution_success: List[bool] = []
        self.file_tokens: Dict[str, int] = {}

    def record_contract(self, ok: bool) -> None:
        with self._lock:
            self.contract_checks.append(ok)

    def record_pass(self, first_try: bool) -> None:
        with self._lock:
            self.first_pass.append(first_try)

    def record_repair(self, repaired: bool) -> None:
        with self._lock:
            self.repair_success.append(repaired)

    def record_execution(self, ok: bool) -> None:
        with self._lock:
            self.execution_success.append(ok)

    def record_file_tokens(self, filename: str, tokens: int) -> None:
        with self._lock:
            self.file_tokens[filename] = tokens

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "contract_fulfillment_rate": _safe_mean(self.contract_checks),
                "first_pass_rate": _safe_mean(self.first_pass),
                "repair_effectiveness": _safe_mean(self.repair_success),
                "execution_success_rate": _safe_mean(self.execution_success),
                "token_efficiency": sum(self.file_tokens.values()) / max(len(self.file_tokens), 1),
                "total_files": len(self.file_tokens),
            }


def _safe_mean(vals: List[bool]) -> float:
    return sum(1 for v in vals if v) / max(len(vals), 1)

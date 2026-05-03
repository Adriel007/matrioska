"""
Circuit breaker & provider failover (§4.2).

Protects against cascading failures when a provider is degraded:
  Closed → (failures ≥ threshold) → Open → (timeout) → HalfOpen → Closed
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-provider circuit breaker.

    After `failure_threshold` consecutive failures, the circuit opens
    for `recovery_timeout` seconds.  One probe call in half-open state
    determines whether to close or re-open.
    """

    failure_threshold: int = 3
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 1

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def before_call(self) -> bool:
        """Check if a call is allowed. Returns False if circuit is open."""
        with self._lock:
            now = time.time()

            if self.state == CircuitState.OPEN:
                if now - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    return False

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    return False
                self.half_open_calls += 1

            return True

    def on_success(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0

    def on_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN


class ProviderRouter:
    """Manages circuit breakers across multiple providers with failover.

    When a provider's circuit is open, the router automatically falls
    back to the next provider in the chain.
    """

    def __init__(self, primary: str, fallbacks: Optional[List[str]] = None):
        self.primary = primary
        self.fallbacks = fallbacks or []
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def _breaker(self, provider: str) -> CircuitBreaker:
        with self._lock:
            if provider not in self._breakers:
                self._breakers[provider] = CircuitBreaker()
            return self._breakers[provider]

    def route(self) -> Optional[str]:
        """Return the first available provider, or None if all are open."""
        for provider in [self.primary, *self.fallbacks]:
            if self._breaker(provider).before_call():
                return provider
        return None

    def mark_success(self, provider: str) -> None:
        self._breaker(provider).on_success()

    def mark_failure(self, provider: str) -> None:
        self._breaker(provider).on_failure()

    @property
    def breacher_states(self) -> Dict[str, str]:
        return {p: b.state.value for p, b in self._breakers.items()}


# ── MoE-style routing by file extension ──────────────────────────────────────


EXTENSION_MODEL_MAP: Dict[str, str] = {
    "py": "claude-sonnet-4",       # Best Python benchmark
    "ts": "claude-sonnet-4",       # Strong TypeScript support
    "tsx": "claude-sonnet-4",
    "js": "claude-sonnet-4",
    "jsx": "claude-sonnet-4",
    "html": "claude-haiku-4.5",    # Cheaper for markup
    "css": "claude-haiku-4.5",
    "json": "claude-haiku-4.5",
    "yaml": "claude-haiku-4.5",
    "yml": "claude-haiku-4.5",
    "md": "claude-haiku-4.5",
    "sql": "gpt-4o",               # Strong SQL generation
}


def route_model_for_extension(extension: str, default: str) -> str:
    """MoE routing: map file extension → best-performing model.

    If the extension is unknown, falls back to the default model.
    """
    ext = extension.lower().lstrip(".")
    return EXTENSION_MODEL_MAP.get(ext, default)

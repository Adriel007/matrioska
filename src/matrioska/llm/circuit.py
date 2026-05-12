"""
Circuit breaker & provider failover (§4.2).

Protects against cascading failures when a provider is degraded:
  Closed → (failures ≥ threshold) → Open → (timeout) → HalfOpen → Closed

SlotPool: multi-key / multi-endpoint rotation with per-slot cooldowns.
  - Multiple API keys for the same endpoint: round-robin, skip keys on 429
  - Extra endpoints (DeepSeek, XAI, Mistral…): fallback when all primary slots fail
  - Retry-After header respected per key
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


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


# ── Slot pool: multi-key / multi-endpoint rotation ───────────────────────────


@dataclass
class APISlot:
    """One (provider, base_url, api_key, model) combination in the pool.

    Tracks its own cooldown (rate limit) and failure count independently
    so a single key being throttled doesn't block the whole pool.
    """

    provider: str
    base_url: str
    api_key: Optional[str]
    model: str
    label: str = ""

    _cooldown_until: float = field(default=0.0, init=False, compare=False, repr=False)
    _failure_count: int = field(default=0, init=False, compare=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, compare=False, repr=False
    )

    @property
    def available(self) -> bool:
        with self._lock:
            return time.time() >= self._cooldown_until and self._failure_count < 5

    @property
    def cooldown_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._cooldown_until - time.time())

    def mark_rate_limited(self, retry_after: float = 60.0) -> None:
        with self._lock:
            self._cooldown_until = time.time() + retry_after
        logger.debug("Slot %s on cooldown for %.0fs", self.label or self.api_key, retry_after)

    def mark_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._cooldown_until = time.time() + min(30.0 * self._failure_count, 300.0)

    def mark_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._cooldown_until = 0.0

    def as_model_spec(self, temperature: float, max_tokens: int, thinking: bool) -> Any:
        from matrioska.core.config import ModelSpec
        return ModelSpec(
            provider=self.provider,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )


import logging
logger = logging.getLogger("matrioska.llm.circuit")


class SlotPool:
    """Round-robin pool of APISlots with per-slot rate-limit cooldowns.

    acquire() immediately returns the next available slot, skipping any
    on cooldown. If all slots are cooling down, returns the soonest-ready
    one together with the seconds to wait — letting the caller decide
    whether to sleep or to raise.
    """

    def __init__(self, slots: List[APISlot]):
        self._slots = slots
        self._idx = 0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: "Any") -> "SlotPool":
        """Build pool from Config: primary keys + extra endpoints."""
        slots: List[APISlot] = []
        primary_provider = cfg.provider
        primary_base = cfg.base_url
        primary_model = cfg.model

        keys = cfg.parsed_api_keys()
        if not keys and cfg.api_key:
            keys = [cfg.api_key]

        for i, key in enumerate(keys):
            slots.append(APISlot(
                provider=primary_provider,
                base_url=primary_base,
                api_key=key,
                model=primary_model,
                label=f"{primary_provider}[{i}]",
            ))

        for ep in cfg.parsed_extra_endpoints():
            slots.append(APISlot(
                provider=ep.get("provider", "openai"),
                base_url=ep.get("base_url", primary_base),
                api_key=ep.get("api_key") or cfg.api_key,
                model=ep.get("model", primary_model),
                label=ep.get("label", ep.get("provider", "extra")),
            ))

        if not slots:
            slots.append(APISlot(
                provider=primary_provider,
                base_url=primary_base,
                api_key=cfg.api_key,
                model=primary_model,
                label=primary_provider,
            ))

        logger.debug("SlotPool initialized with %d slot(s)", len(slots))
        return cls(slots)

    def acquire(self) -> Tuple[Optional[APISlot], float]:
        """Return (slot, wait_seconds).

        wait_seconds == 0 → slot is immediately ready.
        wait_seconds > 0  → all slots on cooldown; this is the soonest one.
        Returns (None, 0) only if all slots have permanent failures.
        """
        with self._lock:
            n = len(self._slots)
            if n == 0:
                return None, 0.0

            for _ in range(n):
                slot = self._slots[self._idx % n]
                self._idx += 1
                if slot.available:
                    return slot, 0.0

            live = [s for s in self._slots if s._failure_count < 5]
            if not live:
                return None, 0.0
            soonest = min(live, key=lambda s: s._cooldown_until)
            return soonest, soonest.cooldown_remaining

    def __len__(self) -> int:
        return len(self._slots)

    def status(self) -> List[Dict[str, Any]]:
        return [
            {
                "label": s.label,
                "provider": s.provider,
                "model": s.model,
                "available": s.available,
                "cooldown_remaining": round(s.cooldown_remaining, 1),
                "failures": s._failure_count,
            }
            for s in self._slots
        ]


def parse_retry_after(response_headers: Dict[str, str], body_text: str = "") -> float:
    """Extract Retry-After seconds from a 429 response.

    Checks (in order): Retry-After header, x-ratelimit-reset-requests,
    then parses 'try again in Xs' patterns from the body.
    """
    for header in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        val = response_headers.get(header)
        if val:
            try:
                return float(val)
            except ValueError:
                pass

    m = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", body_text, re.I)
    if m:
        t = float(m.group(1))
        return t / 1000.0 if m.group(2).lower() == "ms" else t

    m = re.search(r"(\d+(?:\.\d+)?)\s*seconds?", body_text, re.I)
    if m:
        return float(m.group(1))

    return 60.0


# ── Small-model detection ─────────────────────────────────────────────────────

# Models that are known to struggle with OpenAI-style function/tool calling.
# For these, the generator falls back to JSON-schema mode (Solução A).
_SMALL_MODEL_PATTERNS = (
    "8b", "instant", "mini", "e2b", "e4b",
    "small", "lite", "1b", "3b", "1.5b",
    "flash", "nano", "phi-",
)


def is_small_model(model: str) -> bool:
    """True if the model name suggests limited instruction-following for tool use."""
    m = model.lower()
    return any(p in m for p in _SMALL_MODEL_PATTERNS)


# ── MoE-style routing by file extension ──────────────────────────────────────

# NOTE: model names here are provider-specific magic strings.
#   They are NOT validated at startup — a typo will surface only at
#   the first LLM call for a file with that extension.
#   Override via cfg.moe_extension_map (JSON str) or MATRIOSKA_MOE_EXTENSION_MAP.
#   The `isinstance` checks in LLMClient dispatch (provider == "hf", "ollama",
#   "anthropic") are intentional string comparisons, not isinstance hacks;
#   they match the Config.provider field values.

import json as _json

_DEFAULT_EXTENSION_MODEL_MAP: Dict[str, str] = {
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

# Backwards-compatible alias (module-level constant)
EXTENSION_MODEL_MAP: Dict[str, str] = _DEFAULT_EXTENSION_MODEL_MAP


def get_extension_model_map(cfg: "Any") -> Dict[str, str]:
    """Return the effective extension→model map for the given Config.

    If ``cfg.moe_extension_map`` is a non-empty JSON string it is parsed and
    merged *on top of* the hardcoded defaults (per-key override semantics).
    """
    merged: Dict[str, str] = dict(_DEFAULT_EXTENSION_MODEL_MAP)
    raw = getattr(cfg, "moe_extension_map", "")
    if raw and isinstance(raw, str):
        raw = raw.strip()
    if raw:
        try:
            overrides = _json.loads(raw)
            if isinstance(overrides, dict):
                merged.update({str(k): str(v) for k, v in overrides.items()})
        except (_json.JSONDecodeError, ValueError) as exc:
            logger.warning("moe_extension_map JSON parse error: %s", exc)
    return merged


def route_model_for_extension(extension: str, default: str, cfg: "Any" = None) -> str:
    """MoE routing: map file extension → best-performing model.

    If *cfg* is supplied, honours ``cfg.moe_extension_map`` overrides.
    If the extension is unknown, falls back to *default*.
    """
    ext = extension.lower().lstrip(".")
    ext_map = get_extension_model_map(cfg) if cfg is not None else _DEFAULT_EXTENSION_MODEL_MAP
    return ext_map.get(ext, default)

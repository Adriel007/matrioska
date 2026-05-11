"""
Multi-provider LLM client with circuit breaker and failover.

Supports: OpenAI-compatible, Anthropic Messages, Ollama native, HuggingFace local.
Every call goes through the circuit breaker — if the primary provider is
degraded, the router automatically fails over to the next available.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from matrioska.core.config import Config, ModelSpec
from matrioska.core.events import EventBus
from matrioska.llm.circuit import ProviderRouter, SlotPool, APISlot, parse_retry_after

logger = logging.getLogger("matrioska.llm")


# ── Core types ───────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatResponse:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


# ── Standard tool definitions ────────────────────────────────────────────────


STD_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a previously generated artifact verbatim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "extension": {"type": "string"},
                },
                "required": ["name", "extension"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_artifacts",
            "description": "List all generated artifacts so far.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_shared_state",
            "description": "Read the full shared_state whiteboard.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Emit the final file content and optional shared_state updates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "shared_state_updates": {"type": "object"},
                },
                "required": ["content"],
            },
        },
    },
]


# ── Client ───────────────────────────────────────────────────────────────────


class LLMClient:
    """Unified chat interface with circuit breaker + provider failover.

    Each provider path returns a ChatResponse with:
      text          - final assistant text (empty if tool_calls present)
      tool_calls    - list[ToolCall]
      token counts  - best-effort from provider response
    """

    def __init__(self, cfg: Config, bus: Optional[EventBus] = None):
        self.cfg = cfg
        self.bus = bus
        self._slot_pool = SlotPool.from_config(cfg)
        # Keep ProviderRouter for backwards-compat (connectivity check etc.)
        self._router = ProviderRouter(cfg.provider, cfg.failover_providers)
        self._http: Any = None
        self._hf_llm: Any = None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model_spec: Optional[ModelSpec] = None,
        json_mode: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        """Dispatch a chat request through the slot pool.

        Rotates across API keys / extra endpoints automatically:
          - 429 (rate limit): marks the slot with Retry-After cooldown, tries next slot
          - Network / 5xx:    marks slot failure (backoff), tries next slot
          - All slots busy:   waits for the soonest to become available, then retries
        """
        if system:
            messages = [{"role": "system", "content": system}] + messages

        start_time = time.time()
        max_attempts = len(self._slot_pool) * 2 + 3
        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            slot, wait_s = self._slot_pool.acquire()

            if slot is None:
                raise RuntimeError(
                    "All API slots have exceeded failure threshold. "
                    "Check your API keys and provider status."
                )

            if wait_s > 0:
                if attempt == 0:
                    # No slot was immediately available on first try — wait for soonest
                    logger.warning(
                        "All slots on cooldown, waiting %.0fs for %s...",
                        wait_s,
                        slot.label,
                    )
                    time.sleep(wait_s)
                else:
                    # We've rotated through all slots; wait for the soonest
                    logger.warning(
                        "All %d slot(s) on cooldown after %d attempt(s); "
                        "waiting %.0fs for %s...",
                        len(self._slot_pool),
                        attempt,
                        wait_s,
                        slot.label,
                    )
                    time.sleep(wait_s)

            # Build effective spec: use slot's provider/base_url/api_key,
            # but keep the caller's model/temperature/max_tokens if provided.
            if model_spec:
                effective = ModelSpec(
                    provider=slot.provider,
                    base_url=slot.base_url,
                    api_key=slot.api_key,
                    model=model_spec.model,
                    temperature=model_spec.temperature,
                    max_tokens=model_spec.max_tokens,
                    thinking=model_spec.thinking,
                )
            else:
                effective = ModelSpec(
                    provider=slot.provider,
                    base_url=slot.base_url,
                    api_key=slot.api_key,
                    model=slot.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    thinking=self.cfg.thinking,
                )

            try:
                if effective.provider == "hf":
                    resp = self._hf_chat(messages, json_mode, effective)
                elif effective.provider == "ollama":
                    resp = self._ollama_chat(messages, json_mode, effective)
                elif effective.provider == "anthropic":
                    resp = self._anthropic_chat(
                        messages, json_mode, json_schema, tools, effective
                    )
                else:
                    resp = self._openai_compatible_chat(
                        messages, json_mode, json_schema, tools,
                        effective.provider, effective,
                    )

                resp.provider = effective.provider
                resp.model = effective.model
                slot.mark_success()
                self._emit(
                    "llm_done",
                    provider=effective.provider,
                    model=effective.model,
                    slot=slot.label,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    elapsed_s=round(time.time() - start_time, 2),
                )
                return resp

            except _RateLimitError as e:
                last_error = e
                slot.mark_rate_limited(e.retry_after)
                self._emit(
                    "llm_rate_limited",
                    slot=slot.label,
                    retry_after=e.retry_after,
                    attempt=attempt + 1,
                )
                logger.warning(
                    "Rate limit on %s (cooldown %.0fs), rotating to next slot…",
                    slot.label,
                    e.retry_after,
                )
                continue  # Immediately try next slot, no sleep here

            except _RetriableError as e:
                last_error = e
                slot.mark_failure()
                self._emit("llm_retriable_error", slot=slot.label, error=str(e))
                msg = str(e)
                hint = _error_hint(msg, effective)
                logger.warning(
                    "Retriable error on %s (attempt %d): %s%s",
                    slot.label, attempt + 1, msg[:120], hint,
                )
                if attempt < len(self._slot_pool):
                    continue  # Try next slot
                raise RuntimeError(
                    f"LLM call failed ({effective.provider}/{effective.model}): {e}{hint}"
                ) from e

            except Exception as e:
                slot.mark_failure()
                self._emit("llm_fatal_error", slot=slot.label, error=str(e))
                raise

        raise RuntimeError(
            f"LLM call failed after {max_attempts} attempt(s): {last_error}"
        ) from last_error

    # ── Provider implementations ─────────────────────────────────────────

    def _openai_compatible_chat(
        self,
        messages: List[Dict[str, Any]],
        json_mode: bool,
        json_schema: Optional[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        provider: str,
        spec: ModelSpec,
    ) -> ChatResponse:
        import httpx

        url = f"{spec.base_url.rstrip('/')}/chat/completions"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if spec.api_key:
            headers["Authorization"] = f"Bearer {spec.api_key}"

        payload: Dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
        }
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Streaming path: opt-in via cfg.stream_tokens. Disabled when tools
        # or json_schema are requested — chunked tool_calls reassembly is
        # provider-specific and not worth the complexity for current callers.
        stream = bool(getattr(self.cfg, "stream_tokens", False)) and not tools and not json_schema
        if stream:
            try:
                return self._openai_compatible_stream(url, headers, payload, spec)
            except _RateLimitError:
                raise
            except _RetriableError:
                raise
            except Exception as e:
                logger.debug("stream failed (%s); falling back to non-streaming", e)
                # Fall through to non-streaming path below

        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as http:
                r = http.post(url, headers=headers, json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            raise _RetriableError(f"network: {e}")

        if r.status_code == 429:
            retry_after = parse_retry_after(dict(r.headers), r.text)
            raise _RateLimitError(f"HTTP 429: {r.text[:200]}", retry_after=retry_after)

        if r.status_code in (408, 425, 500, 502, 503, 504):
            raise _RetriableError(f"HTTP {r.status_code}: {r.text[:200]}")

        if r.status_code == 401:
            raise _RetriableError(
                f"Authentication failed (401). Check your API key:\n"
                f"  Set MATRIOSKA_API_KEY in .env or pass --api-key.\n"
                f"  Provider: {provider} | Model: {spec.model}"
            )
        if r.status_code == 404:
            raise _RetriableError(
                f"Model not found (404): '{spec.model}' on {provider}.\n"
                f"  Check MATRIOSKA_MODEL in .env or pass --model.\n"
                f"  Verify the model name is correct for this provider."
            )

        if r.status_code == 400 and json_schema:
            logger.info("json_schema rejected; falling back to json_object")
            return self._openai_compatible_chat(
                messages, True, None, tools, provider, spec
            )
        if r.status_code == 400 and tools:
            logger.info("tools rejected; retrying without tools")
            return self._openai_compatible_chat(
                messages, json_mode, None, None, provider, spec
            )

        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {}) or {}
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}

        return ChatResponse(
            text=msg.get("content") or "",
            tool_calls=_extract_tool_calls_openai(msg.get("tool_calls") or []),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            raw=data,
        )

    def _openai_compatible_stream(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        spec: ModelSpec,
    ) -> ChatResponse:
        """SSE streaming path for OpenAI-compatible providers.

        Parses ``data: {...}\\n\\n`` chunks, accumulates ``delta.content`` into
        a single text, and emits a ``llm_token`` event per chunk. Usage
        accounting comes from the final chunk (``stream_options.include_usage``).
        The accumulated ChatResponse is returned exactly like the non-streaming
        path, so callers can ignore the streaming detail entirely.
        """
        import httpx

        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload["stream_options"] = {"include_usage": True}

        parts: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as http:
                with http.stream("POST", url, headers=headers, json=stream_payload) as r:
                    if r.status_code == 429:
                        body = r.read().decode("utf-8", errors="ignore")
                        retry_after = parse_retry_after(dict(r.headers), body)
                        raise _RateLimitError(f"HTTP 429: {body[:200]}", retry_after=retry_after)
                    if r.status_code in (408, 425, 500, 502, 503, 504):
                        body = r.read().decode("utf-8", errors="ignore")
                        raise _RetriableError(f"HTTP {r.status_code}: {body[:200]}")
                    r.raise_for_status()

                    for raw_line in r.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload_str = line[5:].strip()
                        if payload_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue

                        for ch in chunk.get("choices") or []:
                            delta = ch.get("delta") or {}
                            piece = delta.get("content")
                            if piece:
                                parts.append(piece)
                                self._emit(
                                    "llm_token",
                                    provider=spec.provider,
                                    model=spec.model,
                                    delta=piece,
                                )

                        usage = chunk.get("usage") or {}
                        if usage:
                            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens) or prompt_tokens)
                            completion_tokens = int(usage.get("completion_tokens", completion_tokens) or completion_tokens)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            raise _RetriableError(f"network: {e}")

        return ChatResponse(
            text="".join(parts),
            tool_calls=[],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw={},
        )

    def _anthropic_chat(
        self,
        messages: List[Dict[str, Any]],
        json_mode: bool,
        json_schema: Optional[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        spec: ModelSpec,
    ) -> ChatResponse:
        import httpx

        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_asst = [
            m for m in messages if m.get("role") in ("user", "assistant", "tool")
        ]

        anthro_messages: List[Dict[str, Any]] = []
        for m in user_asst:
            if m["role"] == "tool":
                anthro_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.get("tool_call_id", ""),
                                "content": m.get("content", ""),
                            }
                        ],
                    }
                )
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks: List[Dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"] or "{}"),
                        }
                    )
                anthro_messages.append({"role": "assistant", "content": blocks})
            else:
                anthro_messages.append({"role": m["role"], "content": m["content"]})

        url = f"{spec.base_url.rstrip('/')}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": spec.api_key or "",
            "anthropic-version": "2023-06-01",
        }

        payload: Dict[str, Any] = {
            "model": spec.model,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
            "messages": anthro_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {"type": "object"}),
                }
                for t in tools
            ]
        if spec.thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(1024, spec.max_tokens // 4),
            }

        if json_mode:
            # Anthropic doesn't have native JSON mode; we prepend instruction
            prefill = "Here is the JSON response:\n{"
            payload["messages"].append({"role": "assistant", "content": prefill})

        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as http:
                r = http.post(url, headers=headers, json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            raise _RetriableError(f"network: {e}")

        if r.status_code == 429:
            retry_after = parse_retry_after(dict(r.headers), r.text)
            raise _RateLimitError(f"HTTP 429: {r.text[:200]}", retry_after=retry_after)

        if r.status_code in (408, 425, 500, 502, 503, 504):
            raise _RetriableError(f"HTTP {r.status_code}: {r.text[:200]}")
        r.raise_for_status()

        data = r.json()
        text_parts = []
        tool_calls: List[ToolCall] = []
        for block in data.get("content", []):
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id") or uuid.uuid4().hex,
                        name=block.get("name") or "",
                        arguments=block.get("input") or {},
                    )
                )

        if json_mode:
            text = "{" + "".join(text_parts)
        else:
            text = "".join(text_parts)

        usage = data.get("usage", {}) or {}
        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            raw=data,
        )

    def _ollama_chat(
        self,
        messages: List[Dict[str, Any]],
        json_mode: bool,
        spec: ModelSpec,
    ) -> ChatResponse:
        import httpx

        url = f"{spec.base_url.rstrip('/')}/api/chat"
        payload: Dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": spec.temperature,
                "num_predict": spec.max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as http:
                r = http.post(url, json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            raise _RetriableError(f"network: {e}")

        if r.status_code == 429:
            retry_after = parse_retry_after(dict(r.headers), r.text)
            raise _RateLimitError(f"HTTP 429: {r.text[:200]}", retry_after=retry_after)

        if r.status_code in (500, 502, 503, 504):
            raise _RetriableError(f"HTTP {r.status_code}: {r.text[:200]}")
        r.raise_for_status()

        data = r.json()
        msg = data.get("message", {}) or {}
        return ChatResponse(
            text=msg.get("content") or "",
            tool_calls=[],
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            raw=data,
        )

    def _hf_chat(
        self,
        messages: List[Dict[str, Any]],
        json_mode: bool,
        spec: ModelSpec,
    ) -> ChatResponse:
        raise NotImplementedError(
            "HuggingFace local provider is not yet ported to v3. "
            "Use the OpenAI-compatible endpoint (ollama/vllm) instead."
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _emit(self, name: str, **data: Any) -> None:
        if self.bus:
            self.bus.emit_named(name, **data)


def _extract_tool_calls_openai(raw: list) -> List[ToolCall]:
    out = []
    for tc in raw:
        fn = tc.get("function", {}) or {}
        args_str = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_str)
        except Exception:
            try:
                from json_repair import repair_json

                args = json.loads(repair_json(args_str))
            except Exception:
                args = {"_raw": args_str}
        out.append(
            ToolCall(
                id=tc.get("id") or uuid.uuid4().hex,
                name=fn.get("name") or "",
                arguments=args,
            )
        )
    return out


class _RetriableError(RuntimeError):
    """An error that should trigger retry / slot rotation."""
    pass


class _RateLimitError(_RetriableError):
    """HTTP 429: rate limited — rotate to next slot immediately, no sleep."""

    def __init__(self, message: str, retry_after: float = 60.0):
        super().__init__(message)
        self.retry_after = retry_after


def _error_hint(msg: str, spec: ModelSpec) -> str:
    if "401" in msg or "unauthorized" in msg:
        return " → Check MATRIOSKA_API_KEY in .env"
    if "404" in msg or "not found" in msg:
        return f" → Check MATRIOSKA_MODEL (current: {spec.model})"
    if "connection" in msg.lower() or "refused" in msg.lower():
        return f" → Check MATRIOSKA_BASE_URL ({spec.base_url})"
    if "429" in msg or "rate limit" in msg.lower():
        return " → Add more keys via MATRIOSKA_API_KEYS or MATRIOSKA_EXTRA_ENDPOINTS"
    return ""

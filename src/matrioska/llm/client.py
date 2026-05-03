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
from matrioska.llm.circuit import ProviderRouter

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
        """Dispatch a chat request with circuit breaker protection.

        Auto-retries on 429 (rate limit) with exponential backoff + jitter.
        """
        import random

        spec = model_spec or ModelSpec(
            provider=self.cfg.provider,
            model=self.cfg.model,
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )

        if system:
            messages = [{"role": "system", "content": system}] + messages

        provider = self._router.route()
        if provider is None:
            raise RuntimeError("All providers are unavailable (circuits open).")

        start_time = time.time()
        last_error: Optional[Exception] = None

        for attempt in range(4):  # initial + 3 retries
            try:
                if provider == "hf":
                    resp = self._hf_chat(messages, json_mode, spec)
                elif provider == "ollama":
                    resp = self._ollama_chat(messages, json_mode, spec)
                elif provider == "anthropic":
                    resp = self._anthropic_chat(
                        messages, json_mode, json_schema, tools, spec
                    )
                else:
                    resp = self._openai_compatible_chat(
                        messages,
                        json_mode,
                        json_schema,
                        tools,
                        provider,
                        spec,
                    )

                resp.provider = provider
                resp.model = spec.model

                self._router.mark_success(provider)
                self._emit(
                    "llm_done",
                    provider=provider,
                    model=spec.model,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    elapsed_s=round(time.time() - start_time, 2),
                )
                return resp

            except _RetriableError as e:
                last_error = e
                is_429 = "429" in str(e)
                if is_429 and attempt < 3:
                    delay = (2 ** (attempt + 1)) + random.random() * 2
                    logger.warning(
                        "Rate limited (attempt %d/3), retrying in %.1fs...",
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                self._router.mark_failure(provider)
                self._emit("llm_retriable_error", provider=provider, error=str(e))
                raise RuntimeError(f"LLM call failed ({provider}): {e}") from e

            except Exception as e:
                self._router.mark_failure(provider)
                self._emit("llm_fatal_error", provider=provider, error=str(e))
                raise

        raise RuntimeError(f"LLM call failed ({provider}): {last_error}") from last_error

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

        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as http:
                r = http.post(url, headers=headers, json=payload)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            raise _RetriableError(f"network: {e}")

        if r.status_code in (408, 425, 429, 500, 502, 503, 504):
            raise _RetriableError(f"HTTP {r.status_code}: {r.text[:200]}")

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

        if r.status_code in (408, 425, 429, 500, 502, 503, 504):
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

        if r.status_code in (429, 500, 502, 503, 504):
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
    """An error that should trigger circuit breaker and retry logic."""

    pass

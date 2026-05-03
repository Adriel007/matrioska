"""
Generator agent — produces complete file content with tool use.

Uses a balanced cost/quality model (Sonnet 4 / GPT-4o).  The generator
has access to read-only tools (read_file, list_artifacts, read_shared_state)
and emits via the `finish` tool.

For complex files, optionally runs multi-agent debate (§4.4.3).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from matrioska.core.config import Config, ModelSpec
from matrioska.core.events import EventBus
from matrioska.core.state import FileArtifact, FileSpec
from matrioska.llm.client import LLMClient, STD_TOOLS, ChatResponse
from matrioska.llm.circuit import route_model_for_extension

logger = logging.getLogger("matrioska.agents.generator")

GENERATOR_SYSTEM_PROMPT = """You are Matrioska Generator. Produce the COMPLETE content of a single file.

You have access to tools:
  - read_file(name, extension): read a previously generated artifact verbatim
  - list_artifacts(): list every artifact produced so far
  - read_shared_state(): read the full shared_state whiteboard
  - finish(content, shared_state_updates): emit the final file and state updates

When ready, call `finish`. Its "content" argument MUST contain the COMPLETE file
(no ellipses, no TODO-continue). If your model does not support tool calling,
emit the raw file content followed by:

    SHARED_STATE_UPDATE:
    { "key1": "value1", "key2": ["item1", "item2"] }

Only include SHARED_STATE_UPDATE if you define keys other files need.

{project_memory_section}
"""

SHARED_STATE_MARKER = "SHARED_STATE_UPDATE:"


class GeneratorAgent:
    """Generates complete file content with tool-use loop and repair support."""

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        bus: Optional[EventBus] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.bus = bus

    def generate(
        self,
        spec: FileSpec,
        shared_context: Dict[str, Any],
        artifacts: Dict[str, FileArtifact],
        *,
        previous_error: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate content for a single file.

        Returns (content, shared_state_updates).
        """
        gen_spec = self.cfg.effective_generator

        # MoE routing: pick model based on file extension.
        # Only applies to official OpenAI/Anthropic APIs — third-party
        # compatible endpoints (NVIDIA, OpenRouter, etc.) use the
        # configured model directly.
        is_official = (
            gen_spec.provider == "anthropic"
            or (
                gen_spec.provider == "openai"
                and gen_spec.base_url
                and "api.openai.com" in gen_spec.base_url
            )
        )
        model = (
            route_model_for_extension(spec.extension, gen_spec.model)
            if is_official
            else gen_spec.model
        )
        model_spec = ModelSpec(
            provider=gen_spec.provider,
            model=model,
            base_url=gen_spec.base_url,
            api_key=gen_spec.api_key,
            temperature=gen_spec.temperature,
            max_tokens=gen_spec.max_tokens,
        )

        system = GENERATOR_SYSTEM_PROMPT.replace("{project_memory_section}", "")

        if self.cfg.enable_debate and (
            spec.complex or previous_error  # debate when complex or retrying
        ):
            return self._debate_generate(spec, shared_context, model_spec, system)

        return self._single_generate(
            spec, shared_context, model_spec, system, previous_error
        )

    # ── Single generation ────────────────────────────────────────────────

    def _single_generate(
        self,
        spec: FileSpec,
        shared_context: Dict[str, Any],
        model_spec: ModelSpec,
        system: str,
        error_hint: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """Run a single generator agent loop with tool use."""
        user_prompt = self._build_user_prompt(spec, shared_context, error_hint)
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        use_tools = model_spec.provider in ("openai", "anthropic")

        content = ""
        updates: Dict[str, Any] = {}

        resp = self._call_llm(messages, model_spec, system, use_tools)

        # Tool-use loop
        for _ in range(5):  # max 5 tool iterations
            if not resp.tool_calls:
                break

            messages.append(self._format_assistant_tool_msg(resp))
            finished = False

            for tc in resp.tool_calls:
                if tc.name == "finish":
                    content = _strip_fences(str(tc.arguments.get("content", "")))
                    raw_updates = tc.arguments.get("shared_state_updates", {})
                    if isinstance(raw_updates, dict):
                        updates = raw_updates
                    finished = True
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"accepted": True}),
                        }
                    )
                else:
                    result = self._dispatch_tool(
                        tc.name, tc.arguments, spec, shared_context
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False)[:16000],
                        }
                    )

            if finished:
                break

            resp = self._call_llm(messages, model_spec, system, use_tools)

        # Fallback: text mode (no tool support)
        if not content:
            content, updates = _extract_from_text(resp.text)

        return content, updates

    # ── Debate generation (for complex files) ────────────────────────────

    def _debate_generate(
        self,
        spec: FileSpec,
        shared_context: Dict[str, Any],
        model_spec: ModelSpec,
        system: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Multi-agent debate: 2 generators → Judge picks best."""
        logger.info("Multi-agent debate for %s", spec.filename)

        # Run two independent generations
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(
                self._single_generate, spec, shared_context, model_spec, system
            )
            f2 = pool.submit(
                self._single_generate, spec, shared_context, model_spec, system
            )
            result1 = f1.result()
            result2 = f2.result()

        # Use Judge to pick
        from matrioska.agents.judge import JudgeAgent

        judge = JudgeAgent(self.cfg, self.llm, bus=self.bus)
        winner = judge.evaluate_files(
            spec.details,
            [spec.filename],
            [result1[0], result2[0]],
        )

        if winner == 1:
            return result2
        return result1

    # ── Helpers ──────────────────────────────────────────────────────────

    def _build_user_prompt(
        self, spec: FileSpec, context: Dict[str, Any], error_hint: str = ""
    ) -> str:
        context_block = ""
        if context:
            items = "\n".join(
                f"- {k}: {json.dumps(v, ensure_ascii=False)}"
                for k, v in context.items()
            )
            context_block = f"\n\nAVAILABLE SHARED STATE (from earlier files):\n{items}"

        prompt = (
            f"FILE: {spec.name}.{spec.extension}\n\n"
            f"GENERATION INSTRUCTIONS:\n{spec.content}{context_block}\n\n"
            f"REQUIREMENTS:\n{spec.details}\n\n"
            f"Emit the COMPLETE file. Use the `finish` tool if available."
        )

        if error_hint:
            prompt += f"\n\nPREVIOUS VALIDATION ERROR (fix this):\n{error_hint}"

        return prompt

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        model_spec: ModelSpec,
        system: str,
        use_tools: bool,
    ) -> ChatResponse:
        self._emit("agent_call", agent="generator", model=model_spec.model)
        t0 = time.time()
        resp = self.llm.chat(
            messages=messages,
            model_spec=model_spec,
            system=system,
            tools=STD_TOOLS if use_tools else None,
        )
        self._emit(
            "agent_done",
            agent="generator",
            model=model_spec.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            elapsed_s=round(time.time() - t0, 2),
        )
        return resp

    def _dispatch_tool(
        self,
        name: str,
        args: Dict[str, Any],
        spec: FileSpec,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a read-only tool for the generator."""
        if name == "read_file":
            # In v3, the orchestrator manages artifacts centrally
            return {
                "error": "use read_shared_state instead — artifacts are in shared_state"
            }
        if name == "list_artifacts":
            return {"artifacts": list(context.get("_artifacts", []))}
        if name == "read_shared_state":
            return {
                "shared_state": {
                    k: v for k, v in context.items() if not k.startswith("_")
                }
            }
        return {"error": f"unknown tool: {name}"}

    @staticmethod
    def _format_assistant_tool_msg(resp: ChatResponse) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": resp.text or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in resp.tool_calls
            ],
        }

    def _emit(self, name: str, **data: Any) -> None:
        if self.bus:
            self.bus.emit_named(name, **data)


def _strip_fences(text: str) -> str:
    """Strip markdown code fences from model output."""
    t = text.strip()
    for fence in ("```python", "```py", "```sql", "```json", "```yaml", "```", "``"):
        if t.startswith(fence):
            t = t[len(fence):].strip()
            if t.endswith("```"):
                t = t[:-3].strip()
            break
    return t


def _extract_from_text(text: str) -> Tuple[str, Dict[str, Any]]:
    """Extract content and shared_state updates from plain-text response."""
    if not text:
        return "", {}

    cleaned = _strip_fences(text)

    idx = cleaned.find(SHARED_STATE_MARKER)
    if idx < 0:
        return cleaned, {}

    content = cleaned[:idx].strip()
    tail = cleaned[idx + len(SHARED_STATE_MARKER) :].strip()

    updates: Dict[str, Any] = {}
    try:
        updates = json.loads(tail)
    except Exception:
        try:
            from json_repair import repair_json

            updates = json.loads(repair_json(tail))
        except Exception:
            pass

    return content, updates if isinstance(updates, dict) else {}

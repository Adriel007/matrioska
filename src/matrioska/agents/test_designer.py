"""
Test Designer agent — generates contract-derived tests BEFORE code is generated.

Implements the AgentCoder insight (arXiv:2312.13010): the test designer is
intentionally BLIND to the implementation. It derives tests from the FileSpec
contract (shared_state writes, interface requirements, functional description)
not from the generated code, eliminating the "test the bug you wrote" failure mode.

Pairs with AlphaCodium flow (arXiv:2401.08500): these tests are injected into
the Generator prompt so the model implements *against* them, then validated in
the repair loop as executable ground-truth.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from matrioska.core.config import Config, ModelSpec
from matrioska.core.events import EventBus
from matrioska.core.state import FileSpec
from matrioska.llm.client import LLMClient

logger = logging.getLogger("matrioska.agents.test_designer")

DESIGNER_SYSTEM = """You are a Test Designer. Given a file specification and its interface contract, write 3-5 concrete pytest test functions that verify the file fulfills its contract.

CRITICAL RULES:
- You have NOT seen the implementation. Do NOT assume implementation details.
- Tests must be derivable purely from the specification and contract.
- Each test should check ONE behavior: an interface, a function signature, a route, a data structure, a shared_state key.
- Tests must be syntactically valid Python and importable.
- Use simple assertions — no complex fixtures, no network calls, no file I/O.
- Prefer structural tests (does the module export X? does class Y have method Z?) over behavioral tests.
- Output ONLY the raw Python test code — no explanations, no markdown fences.

EXAMPLE output format:
import importlib

def test_module_exports_create_function():
    mod = importlib.import_module("mymodule")
    assert hasattr(mod, "create"), "create function must be exported"

def test_model_has_required_fields():
    from models import User
    import inspect
    sig = inspect.signature(User.__init__)
    assert "email" in sig.parameters
    assert "password_hash" in sig.parameters
"""


class TestDesignerAgent:
    """Generates contract-first tests before the Generator runs.

    The returned test snippet is injected into the Generator's prompt so the
    model writes code that satisfies known interface expectations, then the
    ValidatorAgent can execute the tests as a lightweight smoke check.
    """

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        bus: Optional[EventBus] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.bus = bus

    def design_tests(
        self,
        spec: FileSpec,
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Design tests for a file from its spec/contract alone.

        Returns a string of pytest-compatible test functions, or "" on failure.
        """
        if spec.extension not in ("py",):
            return ""

        prompt = self._build_prompt(spec, shared_context or {})
        ms = self.cfg.effective_validator  # cheap/fast model — same tier as validator

        self._emit("agent_call", agent="test_designer", model=ms.model)
        t0 = time.time()

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model_spec=ms,
                system=DESIGNER_SYSTEM,
            )
        except Exception as e:
            logger.warning("TestDesigner failed for %s: %s", spec.filename, e)
            return ""

        self._emit(
            "agent_done",
            agent="test_designer",
            elapsed_s=round(time.time() - t0, 2),
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )

        tests = self._clean(resp.text)
        if tests:
            logger.info("TestDesigner: %d chars of tests for %s", len(tests), spec.filename)
        return tests

    def _build_prompt(self, spec: FileSpec, context: Dict[str, Any]) -> str:
        writes = spec.shared_state_writes
        reads = spec.shared_state_reads

        contract_block = ""
        if writes:
            contract_block += f"OUTPUTS (shared_state keys this file must define):\n"
            contract_block += "\n".join(f"  - {k}" for k in writes) + "\n"
        if reads:
            contract_block += f"INPUTS (shared_state keys available to this file):\n"
            contract_block += "\n".join(f"  - {k}: {context.get(k, '(not yet set)')}" for k in reads) + "\n"

        return (
            f"FILE: {spec.name}.{spec.extension}\n\n"
            f"FUNCTIONAL DESCRIPTION:\n{spec.content}\n\n"
            f"REQUIREMENTS:\n{spec.details}\n\n"
            f"{contract_block}\n"
            f"Write 3-5 structural/interface tests for {spec.name}.{spec.extension} "
            f"that verify it fulfills the above contract. "
            f"Tests import from '{spec.name}' (without extension)."
        )

    def _clean(self, text: str) -> str:
        """Strip markdown fences if the model forgot the rules."""
        from matrioska.core.text_utils import strip_fences
        return strip_fences(text).strip()

    def _emit(self, name: str, **data: Any) -> None:
        if self.bus:
            self.bus.emit_named(name, **data)


def run_tests_inline(tests_code: str, module_dir: str) -> Tuple[bool, List[str]]:
    """Execute designer tests in-process against generated code.

    Returns (all_passed, list_of_failure_messages).
    Failures are fed into the Repairer as concrete signal (AlphaCodium style).
    """
    import ast
    import sys
    import types

    failures: List[str] = []

    # Quick syntax check on tests themselves
    try:
        ast.parse(tests_code)
    except SyntaxError as e:
        return False, [f"Test code has syntax error: {e}"]

    # Add module_dir to path so imports resolve
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    # Collect and run test functions
    test_module = types.ModuleType("_matrioska_tests")
    try:
        exec(compile(tests_code, "<test_designer>", "exec"), test_module.__dict__)  # noqa: S102
    except Exception as e:
        return False, [f"Failed to load test module: {e}"]

    test_fns = [
        (name, fn)
        for name, fn in test_module.__dict__.items()
        if name.startswith("test_") and callable(fn)
    ]

    for name, fn in test_fns:
        try:
            fn()
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")

    return len(failures) == 0, failures

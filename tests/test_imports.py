"""Verify all modules import cleanly."""

import pytest


def test_core_imports():
    from matrioska.core.config import Config, load_config, validate_config
    from matrioska.core.state import RunState, StateGraph, FileSpec, Architecture, FileArtifact
    from matrioska.core.contracts import FileContract, SharedStateSchema, ContractValidator, StateKeyType
    from matrioska.core.events import EventBus, JSONLRecorder, TokenTracker, MetricsCollector
    assert Config is not None


def test_llm_imports():
    from matrioska.llm.client import LLMClient, ChatResponse, ToolCall
    from matrioska.llm.circuit import CircuitBreaker, ProviderRouter, route_model_for_extension
    assert route_model_for_extension("py", "default") == "claude-sonnet-4"


def test_memory_imports():
    from matrioska.memory.episodic import EpisodicMemory
    from matrioska.memory.semantic import SemanticMemory
    from matrioska.memory.procedural import ProceduralMemory
    assert EpisodicMemory is not None


def test_agents_imports():
    from matrioska.agents.architect import ArchitectAgent
    from matrioska.agents.generator import GeneratorAgent
    from matrioska.agents.validator import ValidatorAgent
    from matrioska.agents.judge import JudgeAgent
    from matrioska.agents.repairer import RepairerAgent
    from matrioska.agents.reflector import ReflectorAgent
    assert ArchitectAgent is not None


def test_pipeline_imports():
    from matrioska.pipeline.graph import compute_layers, validate_dag
    from matrioska.pipeline.phase1 import run_phase1
    from matrioska.pipeline.phase2 import run_phase2
    from matrioska.pipeline.phase3 import run_phase3
    from matrioska.pipeline.orchestrator import Matrioska, run
    assert compute_layers is not None


def test_tools_imports():
    from matrioska.tools.sandbox import SandboxExecutor
    from matrioska.tools.dispatcher import ToolDispatcher
    assert SandboxExecutor is not None


def test_eval_imports():
    from matrioska.eval.metrics import RunMetrics, MetricComparator
    from matrioska.eval.golden_suite import GOLDEN_TASKS, get_golden_tasks, evaluate_result
    assert len(GOLDEN_TASKS) == 30
    assert len(get_golden_tasks("cli")) == 5


def test_config_defaults():
    from matrioska.core.config import Config
    c = Config()
    assert c.provider == "openai"
    assert c.architect_candidates == 3
    assert c.enable_tot is True
    assert c.max_repairs == 2


def test_contract_validation():
    from matrioska.core.contracts import (
        SharedStateSchema, FileContract, ContractValidator, StateKeyType,
    )

    schema = SharedStateSchema(
        key="routes", type=StateKeyType.JSON,
        description="API routes", examples=[],
    )
    contract = FileContract(
        file="main.py",
        reads=[],
        writes=[schema],
    )

    # Should fail: routes key not present
    result = ContractValidator.validate_writes(contract, {})
    assert not result.ok
    assert len(result.violations) == 1

    # Should pass: routes key present
    result = ContractValidator.validate_writes(contract, {"routes": ["/api"]})
    assert result.ok
    assert len(result.violations) == 0


def test_state_checkpoint_roundtrip(tmp_path):
    from matrioska.core.state import StateGraph, RunState

    graph = StateGraph(tmp_path)
    state = graph.new_run("test task")
    state.shared_state["key1"] = "value1"
    cp = graph.save_checkpoint(label="test")

    graph2 = StateGraph(tmp_path)
    loaded = graph2.load_checkpoint(cp.id)
    assert loaded.task == "test task"
    assert loaded.shared_state.get("key1") == "value1"


def test_dag_layering():
    from matrioska.pipeline.graph import compute_layers
    from matrioska.core.state import FileSpec

    files = [
        FileSpec("a", "py", 1, [], ["key_a"]),
        FileSpec("b", "py", 2, ["key_a"], ["key_b"]),
        FileSpec("c", "py", 3, ["key_a", "key_b"], ["key_c"]),
        FileSpec("d", "py", 3, [], ["key_d"]),  # independent
    ]
    layers = compute_layers(files)

    # a and d have no dependencies → layer 0
    layer0_names = {f.name for f in layers[0]}
    assert "a" in layer0_names
    assert "d" in layer0_names

    # b depends on a → layer 1
    assert len(layers) >= 2
    layer1_names = {f.name for f in layers[1]}
    assert "b" in layer1_names

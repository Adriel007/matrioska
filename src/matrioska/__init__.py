"""
Matrioska V3 — Modular monolith LLM orchestrator.

Contract-first multi-agent code generation with state-graph orchestration,
hierarchical memory, sandbox execution, and self-optimizing pipelines.
"""

from matrioska.core.config import Config, load_config
from matrioska.core.state import RunState, SharedState, Checkpoint
from matrioska.core.contracts import (
    FileContract,
    SharedStateSchema,
    ContractValidator,
)
from matrioska.pipeline.orchestrator import Matrioska

__version__ = "3.0.0-alpha"
__all__ = [
    "Config",
    "load_config",
    "RunState",
    "SharedState",
    "Checkpoint",
    "FileContract",
    "SharedStateSchema",
    "ContractValidator",
    "Matrioska",
]

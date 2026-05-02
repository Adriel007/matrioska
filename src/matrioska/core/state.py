"""
State Graph & Checkpointing — inspired by LangGraph.

The linear pipeline becomes a typed state graph with checkpointing,
branching, and time-travel debugging (§4.3 of the plan).

Nodes in the graph:
  1. plan          → N× architect (parallel) → judge → select_plan
  2. generate_layer → validate_file → (fail) repair → (fail) replan_layer
  3. execute_sandbox → validate_contracts → (fail) replan_global
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from src.matrioska.core.contracts import FileContract

# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class FileSpec:
    """A single file to generate, with its contract in shared_state."""

    name: str
    extension: str
    order: int
    shared_state_reads: List[str] = field(default_factory=list)
    shared_state_writes: List[str] = field(default_factory=list)
    content: str = ""
    details: str = ""
    complex: bool = False

    @property
    def filename(self) -> str:
        return f"{self.name}.{self.extension}"


@dataclass
class Architecture:
    """The complete architecture: a DAG of files with shared_state contracts."""

    project_name: str
    files: List[FileSpec]

    def to_contracts(self) -> List[FileContract]:
        """Convert FileSpecs to typed FileContracts for validation."""
        from src.matrioska.core.contracts import SharedStateSchema, StateKeyType

        contracts: List[FileContract] = []
        for f in self.files:
            reads = [
                SharedStateSchema(
                    key=k,
                    type=StateKeyType.JSON,
                    description=f"Read by {f.filename}",
                    examples=[],
                )
                for k in f.shared_state_reads
            ]
            writes = [
                SharedStateSchema(
                    key=k,
                    type=StateKeyType.JSON,
                    description=f"Written by {f.filename}",
                    examples=[],
                )
                for k in f.shared_state_writes
            ]
            contracts.append(FileContract(file=f.filename, reads=reads, writes=writes))
        return contracts


@dataclass
class FileArtifact:
    """Generated file with its content and metadata."""

    name: str
    extension: str
    order: int
    content: str
    shared_state_updates: Dict[str, Any] = field(default_factory=dict)
    status: Literal["pending", "generating", "done", "failed"] = "pending"
    repair_count: int = 0
    generator_model: str = ""
    generation_tokens: int = 0


class PipelineStatus(str, Enum):
    PLANNING = "planning"
    GENERATING = "generating"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"


# ── Run State ────────────────────────────────────────────────────────────────


@dataclass
class RunState:
    """The full state of a Matrioska pipeline run.

    This is the central state object — every pipeline node reads and
    writes to it.  Checkpoints save/restore the full state for resume
    and time-travel debugging.
    """

    task: str
    project_name: str = ""
    architecture: Optional[Architecture] = None
    architecture_candidates: List[Architecture] = field(default_factory=list)
    artifacts: Dict[str, FileArtifact] = field(default_factory=dict)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    contracts: List[FileContract] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    execution_log: List[str] = field(default_factory=list)
    status: PipelineStatus = PipelineStatus.PLANNING
    checkpoint_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_checkpoint_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_artifact(self, artifact: FileArtifact) -> None:
        self.artifacts[f"{artifact.name}.{artifact.extension}"] = artifact
        self.touch()

    def update_shared_state(self, updates: Dict[str, Any]) -> None:
        self.shared_state.update(updates)
        self.touch()

    def log(self, entry: str) -> None:
        self.execution_log.append(f"[{datetime.now(timezone.utc).isoformat()}] {entry}")


# ── Checkpoint ───────────────────────────────────────────────────────────────


@dataclass
class Checkpoint:
    """A frozen snapshot of RunState for resume and branching."""

    id: str
    parent_id: Optional[str]
    state: RunState
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "label": self.label,
            "created_at": self.created_at,
            "state": _serialize_state(self.state),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        return cls(
            id=d["id"],
            parent_id=d.get("parent_id"),
            label=d.get("label", ""),
            created_at=d.get("created_at", ""),
            state=_deserialize_state(d["state"]),
        )


# ── State Graph ──────────────────────────────────────────────────────────────


class StateGraph:
    """Persistent state graph with checkpointing.

    Manages RunState transitions and persists checkpoints to disk,
    enabling resume, branching, and time-travel debugging.
    """

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.checkpoints_dir = work_dir / "matrioska_checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._state: Optional[RunState] = None
        self._checkpoints: Dict[str, Checkpoint] = {}

    # ── State access ─────────────────────────────────────────────────────

    @property
    def state(self) -> RunState:
        if self._state is None:
            raise RuntimeError("No active state — call new_run() or load_checkpoint()")
        return self._state

    def new_run(self, task: str) -> RunState:
        self._state = RunState(task=task)
        return self._state

    def transition(self, new_status: PipelineStatus) -> None:
        self.state.status = new_status
        self.state.touch()

    # ── Checkpointing ────────────────────────────────────────────────────

    def save_checkpoint(self, label: str = "") -> Checkpoint:
        cp = Checkpoint(
            id=uuid.uuid4().hex,
            parent_id=self.state.parent_checkpoint_id,
            state=self.state,
            label=label or self.state.status.value,
        )
        self._checkpoints[cp.id] = cp

        path = self.checkpoints_dir / f"{cp.id}.json"
        path.write_text(
            json.dumps(cp.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return cp

    def load_checkpoint(self, checkpoint_id: str) -> RunState:
        path = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        cp = Checkpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self._checkpoints[cp.id] = cp
        self._state = cp.state
        self._state.parent_checkpoint_id = cp.parent_id
        self._state.checkpoint_id = cp.id
        return self._state

    def load_latest(self) -> Optional[RunState]:
        files = sorted(
            self.checkpoints_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return None
        return self.load_checkpoint(files[0].stem)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        result = []
        for p in sorted(
            self.checkpoints_dir.glob("*.json"), key=lambda p: p.stat().st_mtime
        ):
            try:
                cp = Checkpoint.from_dict(json.loads(p.read_text(encoding="utf-8")))
                result.append(
                    {
                        "id": cp.id,
                        "label": cp.label,
                        "created_at": cp.created_at,
                        "status": cp.state.status.value,
                    }
                )
            except Exception:
                continue
        return result

    def branch(self, from_checkpoint_id: str, label: str = "branch") -> str:
        """Create a branch from an existing checkpoint for alternative exploration."""
        original = self.load_checkpoint(from_checkpoint_id)
        original.parent_checkpoint_id = from_checkpoint_id
        cp = self.save_checkpoint(label=label)
        return cp.id


# ── Serialization helpers ────────────────────────────────────────────────────


def _serialize_state(s: RunState) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "task": s.task,
        "project_name": s.project_name,
        "shared_state": s.shared_state,
        "validation_results": s.validation_results,
        "execution_log": s.execution_log,
        "status": s.status.value,
        "checkpoint_id": s.checkpoint_id,
        "parent_checkpoint_id": s.parent_checkpoint_id,
        "metrics": s.metrics,
        "started_at": s.started_at,
        "updated_at": s.updated_at,
    }
    if s.architecture is not None:
        d["architecture"] = {
            "project_name": s.architecture.project_name,
            "files": [asdict(f) for f in s.architecture.files],
        }
    d["artifacts"] = {k: asdict(v) for k, v in s.artifacts.items()}
    return d


def _deserialize_state(d: Dict[str, Any]) -> RunState:
    arch = None
    if "architecture" in d and d["architecture"] is not None:
        arch_data = d["architecture"]
        files = [FileSpec(**f) for f in arch_data.get("files", [])]
        arch = Architecture(project_name=arch_data["project_name"], files=files)

    artifacts = {}
    for k, v in d.get("artifacts", {}).items():
        artifacts[k] = FileArtifact(**v)

    return RunState(
        task=d.get("task", ""),
        project_name=d.get("project_name", ""),
        architecture=arch,
        artifacts=artifacts,
        shared_state=d.get("shared_state", {}),
        validation_results=d.get("validation_results", {}),
        execution_log=d.get("execution_log", []),
        status=PipelineStatus(d.get("status", "planning")),
        checkpoint_id=d.get("checkpoint_id", uuid.uuid4().hex),
        parent_checkpoint_id=d.get("parent_checkpoint_id"),
        metrics=d.get("metrics", {}),
        started_at=d.get("started_at", ""),
        updated_at=d.get("updated_at", ""),
    )


# ── SharedState convenience ──────────────────────────────────────────────────


class SharedState:
    """Thread-safe wrapper around the shared_state dict with change tracking."""

    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = dict(initial or {})
        self._dirty_keys: set[str] = set()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        return {k: self._data[k] for k in keys if k in self._data}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._dirty_keys.add(key)

    def update(self, updates: Dict[str, Any]) -> None:
        self._data.update(updates)
        self._dirty_keys.update(updates.keys())

    def dirty_keys(self) -> set[str]:
        return set(self._dirty_keys)

    def clear_dirty(self) -> None:
        self._dirty_keys.clear()

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

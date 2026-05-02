"""
Contract-first type system for shared_state.

Every file declares typed read/write contracts that are validated
automatically — implementing process supervision (Lightman et al., 2023).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Literal

# ── Shared State Schema ──────────────────────────────────────────────────────


class StateKeyType(str, Enum):
    """Valid types for shared_state keys."""
    JSON = "json"
    STR_LIST = "list[str]"
    DICT_STR_STR = "dict[str,str]"
    CODE = "code"
    FILE_REF = "file_ref"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


@dataclass
class SharedStateSchema:
    """Typed schema for a single shared_state key.

    Goes beyond Dict[str,Any] — each key has a declared type, semantic
    description, examples, and an optional custom validator.  This is the
    foundation of the contract system described in §4.1 of the plan.
    """
    key: str
    type: StateKeyType
    description: str
    examples: List[Any] = field(default_factory=list)
    validator: Optional[Callable[[Any], bool]] = None

    def validate(self, value: Any) -> bool:
        if self.validator is not None:
            return self.validator(value)
        return _default_validate(self.type, value)


# ── File Contract ────────────────────────────────────────────────────────────


@dataclass
class FileContract:
    """Typed contract a single file must fulfil.

    Declares what shared_state keys the file *reads* (inputs) and what keys
    it *writes* (outputs).  After generation the system checks:

    1. Were all `writes` keys actually populated?
    2. Do the values pass their declared schema?
    3. Are values compatible with what downstream readers expect?
    """
    file: str
    reads: List[SharedStateSchema] = field(default_factory=list)
    writes: List[SharedStateSchema] = field(default_factory=list)

    @property
    def read_keys(self) -> List[str]:
        return [s.key for s in self.reads]

    @property
    def write_keys(self) -> List[str]:
        return [s.key for s in self.writes]


# ── Contract Validation ──────────────────────────────────────────────────────


@dataclass
class ContractValidationResult:
    ok: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ContractValidator:
    """Validates that generated files fulfil their declared contracts.

    Implements *process supervision* — each intermediate step is validated
    against a specification, not just the final output.
    """

    @staticmethod
    def validate_writes(
        contract: FileContract,
        shared_state: Dict[str, Any],
    ) -> ContractValidationResult:
        violations: List[str] = []
        warnings: List[str] = []

        for schema in contract.writes:
            if schema.key not in shared_state:
                violations.append(
                    f"{contract.file}: declared write key '{schema.key}' "
                    f"not found in shared_state after generation"
                )
                continue

            value = shared_state[schema.key]
            if not schema.validate(value):
                violations.append(
                    f"{contract.file}: key '{schema.key}' value {_truncate(value)!r} "
                    f"does not match declared type {schema.type.value}"
                )

        for schema in contract.reads:
            if schema.key not in shared_state:
                warnings.append(
                    f"{contract.file}: declared read key '{schema.key}' "
                    f"not yet available in shared_state (may be written later)"
                )

        return ContractValidationResult(
            ok=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

    @staticmethod
    def validate_cross_file(
        contracts: List[FileContract],
        shared_state: Dict[str, Any],
    ) -> ContractValidationResult:
        """Cross-file consistency: every read key must be written by SOME file."""
        violations: List[str] = []
        all_writes: set[str] = set()
        for c in contracts:
            all_writes.update(c.write_keys)

        for c in contracts:
            for key in c.read_keys:
                if key not in all_writes:
                    violations.append(
                        f"{c.file}: reads key '{key}' that no file declares as write"
                    )

        return ContractValidationResult(ok=len(violations) == 0, violations=violations)


# ── Internal helpers ─────────────────────────────────────────────────────────


def _default_validate(t: StateKeyType, v: Any) -> bool:
    if t == StateKeyType.JSON:
        return isinstance(v, (dict, list, str, int, float, bool, type(None)))
    if t == StateKeyType.STR_LIST:
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    if t == StateKeyType.DICT_STR_STR:
        return isinstance(v, dict) and all(
            isinstance(k, str) and isinstance(val, str) for k, val in v.items()
        )
    if t == StateKeyType.CODE:
        return isinstance(v, str)
    if t == StateKeyType.FILE_REF:
        return isinstance(v, str)
    if t == StateKeyType.INT:
        return isinstance(v, int)
    if t == StateKeyType.FLOAT:
        return isinstance(v, (int, float))
    if t == StateKeyType.BOOL:
        return isinstance(v, bool)
    return True


def _truncate(v: Any, n: int = 80) -> str:
    s = str(v)
    return s if len(s) <= n else s[:n] + "..."

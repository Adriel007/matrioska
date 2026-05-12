"""
Objective metrics for pipeline evaluation (§4.8).

Measures what matters:
  - contract_fulfillment_rate: % of files that fulfill their writes
  - first_pass_rate: % of files that pass validation on first attempt
  - execution_success_rate: % of projects that run without errors
  - repair_effectiveness: % of files that pass after ≤1 repair
  - token_efficiency: tokens per file × complexity
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunMetrics:
    """Metrics collected during a single Matrioska run."""
    contract_fulfillment_rate: float = 0.0
    first_pass_rate: float = 0.0
    execution_success_rate: float = 0.0
    repair_effectiveness: float = 0.0
    token_efficiency: float = 0.0
    total_files: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_s: float = 0.0

    per_file: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_fulfillment_rate": round(self.contract_fulfillment_rate, 3),
            "first_pass_rate": round(self.first_pass_rate, 3),
            "execution_success_rate": round(self.execution_success_rate, 3),
            "repair_effectiveness": round(self.repair_effectiveness, 3),
            "token_efficiency": round(self.token_efficiency, 1),
            "total_files": self.total_files,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "duration_s": round(self.duration_s, 1),
        }


def load_baselines(path: Path) -> Dict[str, Any]:
    """Load baseline metrics from a JSON file.

    Returns an empty dict if the file does not exist or cannot be parsed,
    so callers can safely fall back to hardcoded defaults.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_baselines(metrics: Dict[str, Any], path: Path) -> None:
    """Persist baseline metrics to a JSON file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


class MetricComparator:
    """Compares current metrics against baseline and target values.

    The hardcoded baseline is the MVP (estimated from the plan):
        contract_fulfillment_rate: ~60%
        first_pass_rate: ~50%
        execution_success_rate: 0%
        repair_effectiveness: ~40%

    The target is the v3 goal:
        contract_fulfillment_rate: ≥95%
        first_pass_rate: ≥80%
        execution_success_rate: ≥70%
        repair_effectiveness: ≥75%

    If *baseline_file* is provided and exists, it is loaded and used as the
    baseline in place of the hardcoded values.
    """

    _DEFAULT_BASELINE: Dict[str, float] = {
        "contract_fulfillment_rate": 0.60,
        "first_pass_rate": 0.50,
        "execution_success_rate": 0.0,
        "repair_effectiveness": 0.40,
    }

    # Keep the old class-level name for backwards compatibility
    BASELINE = _DEFAULT_BASELINE

    TARGET: Dict[str, float] = {
        "contract_fulfillment_rate": 0.95,
        "first_pass_rate": 0.80,
        "execution_success_rate": 0.70,
        "repair_effectiveness": 0.75,
    }

    def __init__(self, baseline_file: Optional[Path] = None) -> None:
        if baseline_file is not None:
            loaded = load_baselines(baseline_file)
            # Use loaded data if it has at least one recognised key; otherwise
            # fall back to hardcoded defaults so we never compare against nothing.
            if any(k in loaded for k in self._DEFAULT_BASELINE):
                self._baseline: Dict[str, float] = {
                    k: float(loaded.get(k, self._DEFAULT_BASELINE[k]))
                    for k in self._DEFAULT_BASELINE
                }
            else:
                self._baseline = dict(self._DEFAULT_BASELINE)
        else:
            self._baseline = dict(self._DEFAULT_BASELINE)

    def compare(self, metrics: RunMetrics) -> Dict[str, Any]:
        """Compare run metrics against baseline and target."""
        d = metrics.to_dict()
        result = {"metrics": d, "comparison": {}}

        for key in self._baseline:
            val = d.get(key, 0)
            baseline = self._baseline[key]
            target = self.TARGET[key]
            result["comparison"][key] = {
                "value": val,
                "baseline": baseline,
                "target": target,
                "vs_baseline": _delta_str(val, baseline),
                "vs_target": _delta_str(val, target),
                "meets_target": val >= target,
            }

        return result

    def summary_line(self, metrics: RunMetrics) -> str:
        """Single-line summary for CI output."""
        comp = self.compare(metrics)["comparison"]
        parts = []
        for key in self._baseline:
            meets = "PASS" if comp[key]["meets_target"] else "FAIL"
            parts.append(f"{key}={comp[key]['value']:.2f}({meets})")
        return " | ".join(parts)


def _delta_str(current: float, reference: float) -> str:
    delta = current - reference
    if delta >= 0:
        return f"+{delta:.1%}"
    return f"{delta:.1%}"

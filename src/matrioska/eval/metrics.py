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

from dataclasses import dataclass, field
from typing import Any, Dict, List


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


class MetricComparator:
    """Compares current metrics against baseline and target values.

    The baseline is the MVP (estimated from the plan):
        contract_fulfillment_rate: ~60%
        first_pass_rate: ~50%
        execution_success_rate: 0%
        repair_effectiveness: ~40%

    The target is the v3 goal:
        contract_fulfillment_rate: ≥95%
        first_pass_rate: ≥80%
        execution_success_rate: ≥70%
        repair_effectiveness: ≥75%
    """

    BASELINE = {
        "contract_fulfillment_rate": 0.60,
        "first_pass_rate": 0.50,
        "execution_success_rate": 0.0,
        "repair_effectiveness": 0.40,
    }

    TARGET = {
        "contract_fulfillment_rate": 0.95,
        "first_pass_rate": 0.80,
        "execution_success_rate": 0.70,
        "repair_effectiveness": 0.75,
    }

    @classmethod
    def compare(cls, metrics: RunMetrics) -> Dict[str, Any]:
        """Compare run metrics against baseline and target."""
        d = metrics.to_dict()
        result = {"metrics": d, "comparison": {}}

        for key in cls.BASELINE:
            val = d.get(key, 0)
            baseline = cls.BASELINE[key]
            target = cls.TARGET[key]
            result["comparison"][key] = {
                "value": val,
                "baseline": baseline,
                "target": target,
                "vs_baseline": _delta_str(val, baseline),
                "vs_target": _delta_str(val, target),
                "meets_target": val >= target,
            }

        return result

    @classmethod
    def summary_line(cls, metrics: RunMetrics) -> str:
        """Single-line summary for CI output."""
        comp = cls.compare(metrics)["comparison"]
        parts = []
        for key in cls.BASELINE:
            meets = "PASS" if comp[key]["meets_target"] else "FAIL"
            parts.append(f"{key}={comp[key]['value']:.2f}({meets})")
        return " | ".join(parts)


def _delta_str(current: float, reference: float) -> str:
    delta = current - reference
    if delta >= 0:
        return f"+{delta:.1%}"
    return f"{delta:.1%}"

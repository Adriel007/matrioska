"""
Tests for regression-baseline persistence in MetricComparator (Feature 6).

Covers:
  - load_baselines returns {} for missing file
  - load_baselines round-trips through save_baselines
  - MetricComparator with no baseline_file uses hardcoded defaults
  - MetricComparator with a baseline_file uses loaded values
  - MetricComparator with a non-existent file falls back to hardcoded defaults
  - compare() structure and delta strings
  - summary_line() format
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from matrioska.eval.metrics import (
    MetricComparator,
    RunMetrics,
    load_baselines,
    save_baselines,
)


# ── load_baselines ────────────────────────────────────────────────────────────

class TestLoadBaselines:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_baselines(tmp_path / "nonexistent.json")
        assert result == {}

    def test_valid_json_round_trip(self, tmp_path):
        data = {"contract_fulfillment_rate": 0.85, "first_pass_rate": 0.72}
        p = tmp_path / "baselines.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_baselines(p)
        assert loaded["contract_fulfillment_rate"] == pytest.approx(0.85)
        assert loaded["first_pass_rate"] == pytest.approx(0.72)

    def test_invalid_json_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("this is not json", encoding="utf-8")
        assert load_baselines(p) == {}

    def test_non_dict_json_returns_empty(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_baselines(p) == {}


# ── save_baselines ────────────────────────────────────────────────────────────

class TestSaveBaselines:
    def test_creates_file(self, tmp_path):
        data = {"contract_fulfillment_rate": 0.9}
        p = tmp_path / "out" / "baselines.json"
        save_baselines(data, p)
        assert p.exists()

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "dir" / "baselines.json"
        save_baselines({"x": 1.0}, p)
        assert p.exists()

    def test_round_trip(self, tmp_path):
        data = {
            "contract_fulfillment_rate": 0.77,
            "first_pass_rate": 0.65,
            "execution_success_rate": 0.50,
            "repair_effectiveness": 0.60,
        }
        p = tmp_path / "bl.json"
        save_baselines(data, p)
        loaded = load_baselines(p)
        for k, v in data.items():
            assert loaded[k] == pytest.approx(v)

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "bl.json"
        save_baselines({"a": 1.0}, p)
        save_baselines({"b": 2.0}, p)
        loaded = load_baselines(p)
        assert "b" in loaded
        assert "a" not in loaded


# ── MetricComparator: default (no file) ──────────────────────────────────────

class TestMetricComparatorDefault:
    def test_uses_hardcoded_baseline(self):
        mc = MetricComparator()
        assert mc._baseline == MetricComparator._DEFAULT_BASELINE

    def test_compare_structure(self):
        mc = MetricComparator()
        rm = RunMetrics(
            contract_fulfillment_rate=0.80,
            first_pass_rate=0.70,
            execution_success_rate=0.60,
            repair_effectiveness=0.50,
        )
        result = mc.compare(rm)
        assert "metrics" in result
        assert "comparison" in result
        for key in MetricComparator._DEFAULT_BASELINE:
            assert key in result["comparison"]

    def test_compare_meets_target_true(self):
        mc = MetricComparator()
        rm = RunMetrics(
            contract_fulfillment_rate=0.99,
            first_pass_rate=0.99,
            execution_success_rate=0.99,
            repair_effectiveness=0.99,
        )
        comp = mc.compare(rm)["comparison"]
        for key in mc._baseline:
            assert comp[key]["meets_target"] is True

    def test_compare_meets_target_false(self):
        mc = MetricComparator()
        rm = RunMetrics(
            contract_fulfillment_rate=0.10,
            first_pass_rate=0.10,
            execution_success_rate=0.10,
            repair_effectiveness=0.10,
        )
        comp = mc.compare(rm)["comparison"]
        for key in mc._baseline:
            assert comp[key]["meets_target"] is False

    def test_summary_line_format(self):
        mc = MetricComparator()
        rm = RunMetrics()
        line = mc.summary_line(rm)
        assert "contract_fulfillment_rate" in line
        assert "|" in line

    def test_vs_baseline_positive_delta(self):
        mc = MetricComparator()
        rm = RunMetrics(contract_fulfillment_rate=0.90)  # > 0.60 baseline
        comp = mc.compare(rm)["comparison"]
        assert comp["contract_fulfillment_rate"]["vs_baseline"].startswith("+")

    def test_vs_baseline_negative_delta(self):
        mc = MetricComparator()
        rm = RunMetrics(contract_fulfillment_rate=0.30)  # < 0.60 baseline
        comp = mc.compare(rm)["comparison"]
        assert comp["contract_fulfillment_rate"]["vs_baseline"].startswith("-")


# ── MetricComparator: with baseline_file ─────────────────────────────────────

class TestMetricComparatorWithFile:
    def test_loads_from_file(self, tmp_path):
        data = {
            "contract_fulfillment_rate": 0.80,
            "first_pass_rate": 0.75,
            "execution_success_rate": 0.55,
            "repair_effectiveness": 0.65,
        }
        p = tmp_path / "bl.json"
        save_baselines(data, p)
        mc = MetricComparator(baseline_file=p)
        assert mc._baseline["contract_fulfillment_rate"] == pytest.approx(0.80)
        assert mc._baseline["first_pass_rate"] == pytest.approx(0.75)

    def test_nonexistent_file_falls_back_to_defaults(self, tmp_path):
        mc = MetricComparator(baseline_file=tmp_path / "missing.json")
        assert mc._baseline == MetricComparator._DEFAULT_BASELINE

    def test_partial_file_merges_with_defaults(self, tmp_path):
        """If the file only has some keys, missing keys come from hardcoded defaults."""
        p = tmp_path / "partial.json"
        p.write_text(json.dumps({"contract_fulfillment_rate": 0.88}), encoding="utf-8")
        mc = MetricComparator(baseline_file=p)
        assert mc._baseline["contract_fulfillment_rate"] == pytest.approx(0.88)
        # Missing keys should come from hardcoded defaults
        assert mc._baseline["first_pass_rate"] == pytest.approx(
            MetricComparator._DEFAULT_BASELINE["first_pass_rate"]
        )

    def test_compare_uses_file_baseline(self, tmp_path):
        data = {
            "contract_fulfillment_rate": 0.90,
            "first_pass_rate": 0.85,
            "execution_success_rate": 0.70,
            "repair_effectiveness": 0.80,
        }
        p = tmp_path / "bl.json"
        save_baselines(data, p)
        mc = MetricComparator(baseline_file=p)
        rm = RunMetrics(contract_fulfillment_rate=0.85)
        comp = mc.compare(rm)["comparison"]
        # 0.85 vs baseline 0.90 → negative delta
        assert comp["contract_fulfillment_rate"]["baseline"] == pytest.approx(0.90)
        assert comp["contract_fulfillment_rate"]["vs_baseline"].startswith("-")

    def test_none_baseline_file_is_same_as_no_arg(self):
        mc1 = MetricComparator(baseline_file=None)
        mc2 = MetricComparator()
        assert mc1._baseline == mc2._baseline

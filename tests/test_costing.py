"""
Tests for provider-aware token costing (Feature 1).

Covers:
  - Exact match against _PRICING table
  - Prefix / versioned-suffix match (e.g. "claude-sonnet-4-20250514")
  - Unknown model fallback → 0.0
  - Env-var per-token override
"""

from __future__ import annotations

import os
import importlib

import pytest

from matrioska.core.events import estimate_cost, _PRICING


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cost(model: str, prompt: int = 1_000_000, completion: int = 1_000_000) -> float:
    """Convenience wrapper: 1M prompt + 1M completion tokens."""
    return estimate_cost(model, prompt, completion)


# ── Exact match ───────────────────────────────────────────────────────────────

class TestExactMatch:
    def test_gpt4o(self):
        # 1M prompt @$2.50 + 1M completion @$10.00 = $12.50
        assert _cost("gpt-4o") == pytest.approx(12.50, rel=1e-6)

    def test_gpt4o_mini(self):
        assert _cost("gpt-4o-mini") == pytest.approx(0.75, rel=1e-6)

    def test_claude_opus_4(self):
        # 15 + 75 = 90
        assert _cost("claude-opus-4") == pytest.approx(90.0, rel=1e-6)

    def test_claude_sonnet_4(self):
        # 3 + 15 = 18
        assert _cost("claude-sonnet-4") == pytest.approx(18.0, rel=1e-6)

    def test_llama_3_3_70b(self):
        pp, cp = _PRICING["llama-3.3-70b-versatile"]
        expected = pp + cp  # 1M each
        assert _cost("llama-3.3-70b-versatile") == pytest.approx(expected, rel=1e-6)

    def test_deepseek_coder(self):
        pp, cp = _PRICING["deepseek-coder-v2"]
        assert _cost("deepseek-coder-v2") == pytest.approx(pp + cp, rel=1e-6)

    def test_mistral_large(self):
        pp, cp = _PRICING["mistral-large-latest"]
        assert _cost("mistral-large-latest") == pytest.approx(pp + cp, rel=1e-6)


# ── Prefix / versioned-suffix match ──────────────────────────────────────────

class TestPrefixMatch:
    """Model names with versioned suffixes should match the base prefix key."""

    def test_claude_sonnet_4_versioned(self):
        # "claude-sonnet-4-20250514" starts with "claude-sonnet-4"
        expected = _cost("claude-sonnet-4")
        assert _cost("claude-sonnet-4-20250514") == pytest.approx(expected, rel=1e-6)

    def test_claude_opus_4_versioned(self):
        expected = _cost("claude-opus-4")
        assert _cost("claude-opus-4-20250610") == pytest.approx(expected, rel=1e-6)

    def test_gpt4o_versioned(self):
        expected = _cost("gpt-4o")
        assert _cost("gpt-4o-2025-05-01") == pytest.approx(expected, rel=1e-6)

    def test_haiku_versioned(self):
        expected = _cost("claude-haiku-4.5")
        assert _cost("claude-haiku-4.5-20250307") == pytest.approx(expected, rel=1e-6)


# ── Fallback to 0.0 ──────────────────────────────────────────────────────────

class TestFallback:
    def test_completely_unknown_model(self):
        assert _cost("totally-unknown-model-xyz") == 0.0

    def test_empty_string(self):
        assert _cost("") == 0.0

    def test_partial_nonsense(self):
        assert _cost("imaginary-llm-99b") == 0.0


# ── Token counts ─────────────────────────────────────────────────────────────

class TestTokenCounts:
    def test_zero_tokens(self):
        assert estimate_cost("gpt-4o", 0, 0) == 0.0

    def test_asymmetric_tokens(self):
        # 500k prompt + 0 completion
        result = estimate_cost("gpt-4o", 500_000, 0)
        assert result == pytest.approx(2.50 * 0.5, rel=1e-6)

    def test_completion_only(self):
        result = estimate_cost("gpt-4o", 0, 1_000_000)
        assert result == pytest.approx(10.0, rel=1e-6)


# ── Env-var override ─────────────────────────────────────────────────────────

class TestEnvOverride:
    def test_per_token_prompt_override(self, monkeypatch):
        monkeypatch.setenv("MATRIOSKA_COST_PER_PROMPT_TOKEN", "0.000001")
        monkeypatch.delenv("MATRIOSKA_COST_PER_COMPLETION_TOKEN", raising=False)
        # 1M tokens * $0.000001/token = $1.00
        result = estimate_cost("gpt-4o", 1_000_000, 0)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_per_token_completion_override(self, monkeypatch):
        monkeypatch.delenv("MATRIOSKA_COST_PER_PROMPT_TOKEN", raising=False)
        monkeypatch.setenv("MATRIOSKA_COST_PER_COMPLETION_TOKEN", "0.000002")
        result = estimate_cost("gpt-4o", 0, 500_000)
        assert result == pytest.approx(0.000002 * 500_000, rel=1e-6)

    def test_both_overrides(self, monkeypatch):
        monkeypatch.setenv("MATRIOSKA_COST_PER_PROMPT_TOKEN", "0.000003")
        monkeypatch.setenv("MATRIOSKA_COST_PER_COMPLETION_TOKEN", "0.000006")
        result = estimate_cost("gpt-4o", 100, 200)
        expected = 100 * 0.000003 + 200 * 0.000006
        assert result == pytest.approx(expected, rel=1e-6)

    def test_env_override_ignores_table(self, monkeypatch):
        """Env override should apply even for unknown models."""
        monkeypatch.setenv("MATRIOSKA_COST_PER_PROMPT_TOKEN", "0.000001")
        monkeypatch.delenv("MATRIOSKA_COST_PER_COMPLETION_TOKEN", raising=False)
        assert estimate_cost("totally-unknown-model", 1_000, 0) == pytest.approx(0.001, rel=1e-6)


# ── Pricing table sanity ──────────────────────────────────────────────────────

class TestPricingTableSanity:
    def test_all_values_positive(self):
        for model, (pp, cp) in _PRICING.items():
            assert pp >= 0, f"{model} has negative prompt price"
            assert cp >= 0, f"{model} has negative completion price"

    def test_completion_gte_prompt(self):
        """For most models completion is >= prompt price (sanity check)."""
        for model, (pp, cp) in _PRICING.items():
            # There are a few symmetric models (Groq) where pp == cp
            assert cp >= pp or abs(cp - pp) < 0.01, (
                f"{model}: completion ({cp}) unexpectedly cheaper than prompt ({pp})"
            )

    def test_expected_models_present(self):
        for m in ("gpt-4o", "gpt-4o-mini", "claude-opus-4", "claude-sonnet-4",
                  "llama-3.3-70b-versatile", "deepseek-chat", "mistral-large-latest"):
            assert m in _PRICING, f"Expected model {m!r} missing from _PRICING"

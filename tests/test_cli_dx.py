"""Smoke tests for new CLI DX features: --quick, --mode, init, btw, vault."""

from __future__ import annotations

from pathlib import Path

from matrioska.cli.main import build_parser
from matrioska.core.config import load_config


def test_parser_recognizes_new_subcommands():
    parser = build_parser()
    parser.parse_args(["init", "--dir", "."])
    parser.parse_args(["btw", "what", "is", "RRF"])
    parser.parse_args(["vault", "list"])
    parser.parse_args(["vault", "search", "sqlite", "--scope", "global"])
    parser.parse_args(["vault", "doctor"])
    parser.parse_args(["vault", "graph"])


def test_quick_mode_collapses_features():
    cfg = load_config({"quick": True})
    assert cfg.quick is True
    assert cfg.enable_tot is False
    assert cfg.enable_reflexion is False
    assert cfg.enable_test_design is False
    assert cfg.architect_candidates == 1
    assert cfg.max_repairs == 1


def test_quick_respects_explicit_overrides():
    cfg = load_config({"quick": True, "enable_reflexion": True, "architect_candidates": 3})
    assert cfg.enable_reflexion is True
    assert cfg.architect_candidates == 3


def test_mode_plan_sets_plan_only():
    cfg = load_config({"permission_mode": "plan"})
    assert cfg.permission_mode == "plan"
    assert cfg.plan_only is True


def test_mode_auto_default():
    cfg = load_config({})
    assert cfg.permission_mode == "auto"
    assert cfg.plan_only is False


def test_run_subcommand_supports_new_flags():
    parser = build_parser()
    ns = parser.parse_args(["run", "--task", "x", "--quick", "--mode", "ask"])
    assert ns.quick is True
    assert ns.permission_mode == "ask"


def test_vault_dir_override_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MATRIOSKA_VAULT_DIR", str(tmp_path / "v"))
    from matrioska.memory.vault import default_vault_dir
    assert default_vault_dir() == tmp_path / "v"


def test_init_wizard_renders_env(tmp_path):
    from matrioska.cli.init_wizard import _render_env
    out = _render_env({
        "MATRIOSKA_PROVIDER": "openai",
        "MATRIOSKA_API_KEY": "sk-test",
        "MATRIOSKA_MODEL": "gpt-4o-mini",
    })
    assert "MATRIOSKA_PROVIDER=openai" in out
    assert "MATRIOSKA_API_KEY=sk-test" in out

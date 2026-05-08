"""
Configuration system — CLI > env > .env > defaults.

Multi-model architecture: each agent role gets its own model/provider.
Circuit breaker and provider failover configuration (§4.2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelSpec:
    """Model assignment for a specific agent role."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    thinking: bool = False


@dataclass
class CircuitBreakerConfig:
    """Configuration for provider circuit breaker."""
    failure_threshold: int = 3
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 1


@dataclass
class Config:
    """Central configuration for a Matrioska run.

    Each agent role has its own ModelSpec, enabling the multi-model
    architecture from §4.2 of the plan.
    """

    # ── Provider defaults ─────────────────────────────────────────────────
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"

    # ── Role-specific models (override the default) ───────────────────────
    architect_model: str = ""
    architect_provider: str = ""
    generator_model: str = ""
    generator_provider: str = ""
    validator_model: str = ""
    validator_provider: str = ""
    judge_model: str = ""
    judge_provider: str = ""
    repairer_model: str = ""
    repairer_provider: str = ""

    # ── Pipeline ──────────────────────────────────────────────────────────
    work_dir: Path = field(default_factory=lambda: Path("./matrioska_work"))
    max_tokens: int = 4096
    temperature: float = 0.3
    max_repairs: int = 2
    max_depth: int = 2
    parallel: bool = True
    plan_only: bool = False

    # ── Phase 1: Architecture ─────────────────────────────────────────────
    architect_candidates: int = 3       # N for Tree-of-Thoughts
    architect_temperature: float = 0.7  # Higher for diversity
    enable_tot: bool = True             # Enable Tree-of-Thoughts voting

    # ── Phase 2: Generation ───────────────────────────────────────────────
    enable_reflexion: bool = True       # Enable Reflexion loop
    enable_debate: bool = False         # Multi-agent debate for complex files
    enable_test_design: bool = True     # AlphaCodium+AgentCoder: design tests before generating
    use_aci_repair: bool = True         # SWE-agent ACI: targeted patch repair vs full-file rewrite

    # ── Phase 3: Verification ─────────────────────────────────────────────
    enable_sandbox: bool = False        # Docker sandbox execution
    sandbox_timeout: int = 30           # Max execution seconds
    sandbox_image: str = "python:3.11-slim"

    # ── Memory ────────────────────────────────────────────────────────────
    retrieve_k: int = 3
    embedder_model: str = "text-embedding-3-small"
    enable_graphrag: bool = False

    # ── Circuit breaker ───────────────────────────────────────────────────
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    failover_providers: List[str] = field(default_factory=list)

    # ── Observability ─────────────────────────────────────────────────────
    log_level: str = "INFO"
    enable_otel: bool = False
    otel_endpoint: str = ""
    enable_cost_tracking: bool = True

    # ── Misc ──────────────────────────────────────────────────────────────
    thinking: bool = False
    dry_run: bool = False
    interactive: bool = False

    # ── Derived ───────────────────────────────────────────────────────────

    def get_model_spec(self, role: str) -> ModelSpec:
        """Get the ModelSpec for a given agent role."""
        provider_override = getattr(self, f"{role}_provider", "") or self.provider
        model_override = getattr(self, f"{role}_model", "") or self.model
        url_override = self.base_url
        key_override = self.api_key

        temp = self.temperature
        if role == "architect":
            temp = self.architect_temperature

        return ModelSpec(
            provider=provider_override,
            model=model_override,
            base_url=url_override,
            api_key=key_override,
            temperature=temp,
            max_tokens=self.max_tokens,
            thinking=self.thinking,
        )

    @property
    def effective_architect(self) -> ModelSpec:
        return self.get_model_spec("architect")

    @property
    def effective_generator(self) -> ModelSpec:
        return self.get_model_spec("generator")

    @property
    def effective_validator(self) -> ModelSpec:
        return self.get_model_spec("validator")

    @property
    def effective_judge(self) -> ModelSpec:
        return self.get_model_spec("judge")

    @property
    def effective_repairer(self) -> ModelSpec:
        return self.get_model_spec("repairer")


# ── Config Loading ───────────────────────────────────────────────────────────


_BOOL_FLAGS = {
    "parallel", "plan_only", "thinking", "dry_run", "interactive",
    "enable_tot", "enable_reflexion", "enable_debate",
    "enable_test_design", "use_aci_repair",
    "enable_sandbox", "enable_graphrag", "enable_otel", "enable_cost_tracking",
}
_INT_FIELDS = {"max_tokens", "max_repairs", "max_depth", "retrieve_k",
               "architect_candidates", "sandbox_timeout"}
_FLOAT_FIELDS = {"temperature", "architect_temperature"}
_PATH_FIELDS = {"work_dir"}


def _coerce(field_name: str, value: str) -> Any:
    if field_name in _BOOL_FLAGS:
        return value.strip().lower() in ("1", "true", "yes", "on")
    if field_name in _INT_FIELDS:
        return int(value)
    if field_name in _FLOAT_FIELDS:
        return float(value)
    if field_name in _PATH_FIELDS:
        return Path(value).expanduser()
    return value


def load_config(cli_overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load config with precedence: CLI > env > .env > default.

    Args:
        cli_overrides: Optional dict of CLI-provided values keyed by field name.

    Returns:
        A fully resolved Config instance.
    """
    # Load .env into os.environ (does not override existing vars)
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(override=False)
    except ImportError:
        pass

    cfg = Config()
    cli = cli_overrides or {}

    for field_name in cfg.__dataclass_fields__:
        if field_name in cli and cli[field_name] is not None:
            val = cli[field_name]
            setattr(cfg, field_name, val if not isinstance(val, str) else _coerce(field_name, val))
            continue

        env_key = f"MATRIOSKA_{field_name.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None and env_val != "":
            setattr(cfg, field_name, _coerce(field_name, env_val))

    if isinstance(cfg.work_dir, str):
        cfg.work_dir = Path(cfg.work_dir).expanduser()

    # Provider-specific defaults
    if cfg.provider == "ollama" and cfg.base_url == "https://api.openai.com/v1":
        cfg.base_url = "http://localhost:11434"
    if cfg.provider == "anthropic" and cfg.base_url == "https://api.openai.com/v1":
        cfg.base_url = "https://api.anthropic.com"

    return cfg


def validate_config(cfg: Config) -> None:
    """Validate config; raises SystemExit on fatal issues."""
    if cfg.provider not in ("openai", "ollama", "anthropic", "hf"):
        raise SystemExit(f"Invalid provider: {cfg.provider}")

    local_markers = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    is_local = any(m in cfg.base_url for m in local_markers) or cfg.provider == "hf"
    if not is_local and cfg.provider in ("openai", "anthropic") and not cfg.api_key:
        raise SystemExit(
            f"API key required for provider={cfg.provider}. "
            f"Set --api-key or MATRIOSKA_API_KEY."
        )

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
    provider: str = "openai"                            # env: MATRIOSKA_PROVIDER
    base_url: str = "https://api.openai.com/v1"         # env: MATRIOSKA_BASE_URL
    api_key: Optional[str] = None                       # env: MATRIOSKA_API_KEY
    model: str = "gpt-4o-mini"                          # env: MATRIOSKA_MODEL

    # ── Role-specific models (override the default) ───────────────────────
    architect_model: str = ""                           # env: MATRIOSKA_ARCHITECT_MODEL
    architect_provider: str = ""                        # env: MATRIOSKA_ARCHITECT_PROVIDER
    generator_model: str = ""                           # env: MATRIOSKA_GENERATOR_MODEL
    generator_provider: str = ""                        # env: MATRIOSKA_GENERATOR_PROVIDER
    validator_model: str = ""                           # env: MATRIOSKA_VALIDATOR_MODEL
    validator_provider: str = ""                        # env: MATRIOSKA_VALIDATOR_PROVIDER
    judge_model: str = ""                               # env: MATRIOSKA_JUDGE_MODEL
    judge_provider: str = ""                            # env: MATRIOSKA_JUDGE_PROVIDER
    repairer_model: str = ""                            # env: MATRIOSKA_REPAIRER_MODEL
    repairer_provider: str = ""                         # env: MATRIOSKA_REPAIRER_PROVIDER

    # ── Pipeline ──────────────────────────────────────────────────────────
    work_dir: Path = field(default_factory=lambda: Path("./matrioska_work"))  # env: MATRIOSKA_WORK_DIR
    max_tokens: int = 4096                              # env: MATRIOSKA_MAX_TOKENS
    temperature: float = 0.3                            # env: MATRIOSKA_TEMPERATURE
    max_repairs: int = 2                                # env: MATRIOSKA_MAX_REPAIRS
    max_depth: int = 2                                  # env: MATRIOSKA_MAX_DEPTH
    parallel: bool = True                               # env: MATRIOSKA_PARALLEL
    plan_only: bool = False                             # env: MATRIOSKA_PLAN_ONLY

    # ── Phase 1: Architecture ─────────────────────────────────────────────
    architect_candidates: int = 3       # N for Tree-of-Thoughts  # env: MATRIOSKA_ARCHITECT_CANDIDATES
    architect_temperature: float = 0.7  # Higher for diversity    # env: MATRIOSKA_ARCHITECT_TEMPERATURE
    enable_tot: bool = True             # Enable Tree-of-Thoughts voting  # env: MATRIOSKA_ENABLE_TOT

    # ── Phase 2: Generation ───────────────────────────────────────────────
    enable_reflexion: bool = True       # Enable Reflexion loop              # env: MATRIOSKA_ENABLE_REFLEXION
    enable_debate: bool = False         # Multi-agent debate for complex files  # env: MATRIOSKA_ENABLE_DEBATE
    enable_test_design: bool = True     # AlphaCodium+AgentCoder: design tests before generating  # env: MATRIOSKA_ENABLE_TEST_DESIGN
    use_aci_repair: bool = True         # SWE-agent ACI: targeted patch repair vs full-file rewrite  # env: MATRIOSKA_USE_ACI_REPAIR

    # ── Phase 3: Verification ─────────────────────────────────────────────
    enable_sandbox: bool = False        # Docker sandbox execution  # env: MATRIOSKA_ENABLE_SANDBOX
    sandbox_timeout: int = 30           # Max execution seconds     # env: MATRIOSKA_SANDBOX_TIMEOUT
    sandbox_image: str = "python:3.11-slim"                         # env: MATRIOSKA_SANDBOX_IMAGE

    # ── Memory ────────────────────────────────────────────────────────────
    retrieve_k: int = 3                                 # env: MATRIOSKA_RETRIEVE_K
    embedder_model: str = "text-embedding-3-small"      # env: MATRIOSKA_EMBEDDER_MODEL
    enable_graphrag: bool = False                       # env: MATRIOSKA_ENABLE_GRAPHRAG

    # ── Multi-key / multi-endpoint rotation ──────────────────────────────────
    # api_keys: comma-separated API keys for the primary provider (round-robin)
    # extra_endpoints: JSON array {provider,base_url,api_key,model[,label]}
    #   Use to add DeepSeek, XAI, Mistral, Together, etc. as extra fallback slots.
    # Example .env:
    #   MATRIOSKA_API_KEYS=gsk_key1,gsk_key2,gsk_key3
    #   MATRIOSKA_EXTRA_ENDPOINTS=[{"provider":"deepseek","base_url":"https://api.deepseek.com/v1","api_key":"ds_xxx","model":"deepseek-coder-v2"}]
    api_keys: str = ""                                  # env: MATRIOSKA_API_KEYS
    extra_endpoints: str = ""                           # env: MATRIOSKA_EXTRA_ENDPOINTS

    # ── Circuit breaker ───────────────────────────────────────────────────
    # Note: CircuitBreakerConfig sub-fields are not mapped to individual env vars;
    #   use MATRIOSKA_CIRCUIT_BREAKER_* prefix when implementing sub-field loading.
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    failover_providers: List[str] = field(default_factory=list)     # env: MATRIOSKA_FAILOVER_PROVIDERS (comma-separated)

    # ── Observability ─────────────────────────────────────────────────────
    log_level: str = "INFO"             # env: MATRIOSKA_LOG_LEVEL
    enable_otel: bool = False           # env: MATRIOSKA_ENABLE_OTEL
    otel_endpoint: str = ""             # env: MATRIOSKA_OTEL_ENDPOINT
    enable_cost_tracking: bool = True   # env: MATRIOSKA_ENABLE_COST_TRACKING

    # ── Incremental / surgical mode ───────────────────────────────────────
    # incremental=True: pre-flight scans existing files and injects them into
    #   the Architect context so it generates only what needs to change.
    # project_dir: path to the existing project to read (default: work_dir).
    # execute_feedback=True: after each file is generated, run it in a
    #   subprocess and feed stderr back to the Repairer as repair signal.
    # install_deps=True: after Phase 2, auto-install detected pip packages.
    incremental: bool = False           # env: MATRIOSKA_INCREMENTAL
    project_dir: str = ""               # env: MATRIOSKA_PROJECT_DIR
    execute_feedback: bool = True       # env: MATRIOSKA_EXECUTE_FEEDBACK
    install_deps: bool = True           # env: MATRIOSKA_INSTALL_DEPS

    # ── Quick mode / permissions / vault ──────────────────────────────────
    # quick=True: skip ToT, Reflexion, TestDesign, ACI, and Phase 3. Useful
    #   for rapid iteration during development (sub-30s runs typical).
    # permission_mode: "auto" (default, current behavior), "plan" (alias of
    #   plan_only), "ask" (pause and confirm before each file generation).
    # enable_vault: writes/reads ~/.matrioska/vault (global Obsidian-compatible
    #   knowledge base). Disable for ephemeral CI runs.
    quick: bool = False                 # env: MATRIOSKA_QUICK
    permission_mode: str = "auto"       # auto | plan | ask  # env: MATRIOSKA_PERMISSION_MODE
    enable_vault: bool = True           # env: MATRIOSKA_ENABLE_VAULT
    vault_dir: str = ""                 # default: ~/.matrioska/vault  # env: MATRIOSKA_VAULT_DIR

    # ── Multi-planning ────────────────────────────────────────────────────
    # enable_multi_plan=True: before Phase 1, a MetaPlanner decomposes the task
    # into N sub-domains (2-4). Each sub-domain gets its own scoped ArchitectAgent
    # call, and the results are merged into one Architecture. Better decomposition
    # for complex multi-component tasks; costs +1 LLM call.
    enable_multi_plan: bool = False

    # ── Streaming ─────────────────────────────────────────────────────────
    # stream_tokens=True: openai-compatible chat uses SSE and emits per-chunk
    # `llm_token` events through the event bus. Token deltas are accumulated
    # into ChatResponse.text exactly as the non-streaming path returns.
    stream_tokens: bool = True          # env: MATRIOSKA_STREAM_TOKENS

    # ── MoE extension map override ────────────────────────────────────────
    # JSON string mapping file extension → model name.  Merged with (and
    # overrides) the hardcoded defaults in llm/circuit.py.
    # Example: '{"py": "gpt-4o", "sql": "deepseek-coder-v2"}'
    moe_extension_map: str = ""         # env: MATRIOSKA_MOE_EXTENSION_MAP

    # ── Validation ────────────────────────────────────────────────────────
    # skip_validation=True: skip provider connectivity check on startup.
    skip_validation: bool = False       # env: MATRIOSKA_SKIP_VALIDATION

    # ── Server ────────────────────────────────────────────────────────────
    serve_port: int = 9020               # env: MATRIOSKA_SERVE_PORT

    # ── Misc ──────────────────────────────────────────────────────────────
    thinking: bool = False              # env: MATRIOSKA_THINKING
    dry_run: bool = False               # env: MATRIOSKA_DRY_RUN
    interactive: bool = False           # env: MATRIOSKA_INTERACTIVE

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

    def parsed_api_keys(self) -> List[str]:
        """Return list of API keys from the comma-separated api_keys field."""
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    def parsed_extra_endpoints(self) -> List[Dict[str, Any]]:
        """Parse extra_endpoints JSON string into a list of dicts."""
        raw = self.extra_endpoints.strip()
        if not raw:
            return []
        import json
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return []

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
    "incremental", "execute_feedback", "install_deps",
    "quick", "enable_vault", "stream_tokens", "enable_multi_plan",
    "skip_validation",
}
_INT_FIELDS = {"max_tokens", "max_repairs", "max_depth", "retrieve_k",
               "architect_candidates", "sandbox_timeout", "serve_port"}
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

    # Provider-specific base_url defaults
    _PROVIDER_DEFAULTS: Dict[str, str] = {
        "ollama":    "http://localhost:11434",
        "anthropic": "https://api.anthropic.com",
        "deepseek":  "https://api.deepseek.com/v1",
        "xai":       "https://api.x.ai/v1",
        "mistral":   "https://api.mistral.ai/v1",
        "together":  "https://api.together.xyz/v1",
        "cohere":    "https://api.cohere.ai/v1",
        "groq":      "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "nvidia":    "https://integrate.api.nvidia.com/v1",
    }
    default_url = "https://api.openai.com/v1"
    if cfg.provider in _PROVIDER_DEFAULTS and cfg.base_url == default_url:
        cfg.base_url = _PROVIDER_DEFAULTS[cfg.provider]

    # Quick mode: collapse all "slow" features. Honors explicit CLI/env
    # overrides (only sets a flag if it wasn't explicitly toggled).
    if cfg.quick:
        explicitly_set = set(cli.keys()) if cli else set()
        if "enable_tot" not in explicitly_set and "enable_tot" not in os.environ.get("_MATRIOSKA_EXPLICIT", ""):
            cfg.enable_tot = False
        if "enable_reflexion" not in explicitly_set:
            cfg.enable_reflexion = False
        if "enable_test_design" not in explicitly_set:
            cfg.enable_test_design = False
        if "architect_candidates" not in explicitly_set:
            cfg.architect_candidates = 1
        if "max_repairs" not in explicitly_set:
            cfg.max_repairs = 1

    # plan permission mode → set plan_only flag for orchestrator
    if cfg.permission_mode == "plan":
        cfg.plan_only = True

    return cfg


_VALID_PROVIDERS = {
    "openai", "ollama", "anthropic", "hf",
    "deepseek", "xai", "mistral", "together", "cohere",
    "groq", "openrouter", "nvidia",
}


def validate_config(cfg: Config) -> None:
    """Validate config; raises SystemExit on fatal issues."""
    if cfg.provider not in _VALID_PROVIDERS:
        raise SystemExit(
            f"Invalid provider: {cfg.provider}. "
            f"Valid: {', '.join(sorted(_VALID_PROVIDERS))}"
        )

    local_markers = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    is_local = any(m in cfg.base_url for m in local_markers) or cfg.provider == "hf"
    if not is_local and cfg.provider in ("openai", "anthropic") and not cfg.api_key:
        raise SystemExit(
            f"API key required for provider={cfg.provider}. "
            f"Set --api-key or MATRIOSKA_API_KEY."
        )

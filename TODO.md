# TODO — Matrioska V3

## Pipeline & Generation

- [x] **Multi planning** — `agents/multi_planner.py`. MetaPlanner decomposes the task
  into 2-4 self-contained sub-domains with a shared_interface contract. Each sub-domain
  gets a scoped ArchitectAgent call (sequential, so later sub-planners see accumulated
  shared_state writes). Results merged into one Architecture with deduplicated FileSpecs.
  Enabled via `cfg.enable_multi_plan=True` / `MATRIOSKA_ENABLE_MULTI_PLAN=true`.
  Falls back to single ArchitectAgent if meta-planner returns < 2 sub-problems.

- [x] **Phase 3 sandbox execution** — `tools/sandbox.py::SandboxExecutor.run()`.
  Auto-detects project type (python/node/shell/web/unknown). Docker preferred;
  subprocess fallback when Docker unavailable. Python: auto-installs detected
  third-party deps via pip inside container; server processes (uvicorn, flask.run)
  get import-only check to avoid blocking. Web projects: HTML syntax validation
  via stdlib html.parser. `phase3.py::_run_sandbox_with_repair()` feeds stderr
  back to `RepairerAgent` when exit_code != 0, re-runs sandbox (max
  `cfg.sandbox_max_repairs` iterations). Events: sandbox_started, sandbox_result,
  sandbox_repair_start, sandbox_repair_done. 25 new tests in test_sandbox.py.

## Providers & Models

- [ ] **HuggingFace provider** — `_hf_chat()` still raises `NotImplementedError`.
  Port the v2 HuggingFace integration or support any Transformers pipeline.

- [x] **Provider-aware token costing** — `events.py::estimate_cost()` with `_PRICING`
  table covering OpenAI, Anthropic, Groq, DeepSeek, Mistral, Together, NVIDIA.
  Lookup: exact → prefix (longest wins) → substring fallback. Override via
  `MATRIOSKA_COST_PER_PROMPT_TOKEN` / `MATRIOSKA_COST_PER_COMPLETION_TOKEN` env vars.

- [x] **Configurable MoE extension map** — `circuit.py::get_extension_model_map(cfg)`
  reads `cfg.moe_extension_map` (JSON string) and merges per-key overrides on top of
  `_DEFAULT_EXTENSION_MODEL_MAP`. Set via `MATRIOSKA_MOE_EXTENSION_MAP={"py":"deepseek-v3"}`.
  Falls back to defaults silently on JSON parse error.

- [ ] **Ollama provider test** — The native Ollama path works but is untested since
  v3 migration. Validate streaming and tool-use fallback.

## Memory & Knowledge

### Obsidian-compatible Vault (Local + Global + Cross-project)

*Fundamentação: Karpathy LLM Wiki Pattern (abr/2026, 16M views, 5k★),
LightRAG dual-level graph (EMNLP 2025, arXiv 2409.14813),
GAM Hierarchical Graph-based Agentic Memory (arXiv 2604.12285),
engraph/MCPVault/obsidian-mcp — 24+ MCP servers para vaults Obsidian.*

**A ideia central (Karpathy):** em vez de RAG puro (re-derivar relações a cada
query), "compilar" o conhecimento em Markdown permanente. O vault é o binário: notas
de conceitos, padrões e bugs atualizados incrementalmente após cada run. O Matrioska
já armazena runs em Markdown + YAML frontmatter em `knowledge/runs/` — está a um
passo de ser um vault Obsidian nativo.

#### Arquitetura proposta

```
~/.matrioska/vault/               ← VAULT GLOBAL (Obsidian, compartilhado entre projetos)
├── projects/
│   ├── budget_tracker/
│   │   ├── architecture.md       ← decomposição e decisões do projeto
│   │   ├── patterns.md           ← padrões observados (ex: "usa SQLite com AUTOINCREMENT")
│   │   ├── lessons.md            ← bugs encontrados e como foram resolvidos
│   │   └── links.md              ← [[api_auth]] [[cli_pipeline]] (projetos relacionados)
│   └── api_auth/
│       └── ...
├── concepts/                     ← conhecimento transversal a projetos
│   ├── sqlite_patterns.md        ← "como fazer SQLite bem em Python"
│   ├── fastapi_auth.md           ← padrões de autenticação com FastAPI
│   └── argparse_cli.md           ← padrões de CLI com argparse
├── bugs/
│   ├── common_syntax_errors.md   ← erros frequentes do modelo 8b
│   └── import_circular.md        ← padrão "importação circular entre módulos"
└── INDEX.md                      ← índice do grafo (wikilinks + tags)

./matrioska_work/knowledge/        ← VAULT LOCAL (projeto atual, já existe)
├── runs/                          ← EpisodicMemory (já funciona)
├── concepts/                      ← SemanticMemory (parcial)
└── MATRIOSKA.md                   ← ProceduralMemory (já funciona)
```

- [x] **Drill-down interativo no REPL** — `repl.py::_cmd_vault`. Subcomandos: `search`
  (query + `--scope`), `list`, `project <name>` (abre architecture.md + patterns.md),
  `concept <name>` (abre nota, lista conhecidos se não encontrado), `related <project>`
  (BFS sobre wikilinks, ≤2 hops), `doctor` (relatório de saúde). Resultados com
  Rich markup + links `file://` clicáveis no terminal (Cmd+Click).

### Outros itens de Memory

- [ ] **GraphRAG auto-ingest** — Semantic memory tracks concepts and relationships
  but doesn't auto-extract them from run artifacts. Add LLM-based entity extraction
  on `write_run_note()` to populate the knowledge graph.

- [ ] **ChromaDB lazy download** — First-run downloads 79 MB of ONNX model for
  embeddings. Show a progress bar and consider making embeddings optional
  (keyword-only fallback).

## CLI & Interactive REPL (Claude Code-inspired)

*Executar `matrioska` sem subcomando abre um REPL interativo — prompt simples de
conversa com o agente, mais `/comandos` para controle. Inspirado no Claude Code.*

- [x] **REPL autocompletion & keyboard navigation** — `cli/repl.py::_build_completer()`.
  `NestedCompleter` built from `COMMANDS` registry; `/vault` → `{list,search,doctor,graph}`,
  `/effort` → `{low,medium,high}`, all other commands have bare completion. Added to
  `PromptSession(completer=..., complete_while_typing=True)` with Catppuccin-style menu.
  TAB/↓ opens menu, arrows navigate, Enter selects, Esc dismisses. Also fixed
  `EventBus.emit` wildcard bug: `"*"` handlers were never fired (silent dashboard/recorder).

- [x] **Rewind / checkpoint** — `/rewind` no REPL lista checkpoints disponíveis, aceita
  índice ou prefixo de ID, carrega o estado no `session.last_result` para uso com `/diff`.
  StateGraph.list_checkpoints() + load_checkpoint() já existiam — apenas expostos.

- [x] **Comandos customizados** — `repl.py::_load_custom_commands()`. Escaneia
  `.matrioska/commands/*.md` no cwd ao iniciar o REPL. Cada arquivo vira um slash
  command: nome = stem do arquivo (hifens → underscores), help = primeira linha,
  ação = injeta o conteúdo do arquivo como task no pipeline.

- [x] **Hook system** — `hooks.py::HookRunner`. Searches `.matrioska/hooks/` (project)
  then `~/.matrioska/hooks/` (global). Scripts named `pre_generate`, `post_generate`,
  `pre_repair`, `phase1_done`, `phase2_done`, `run_end`, `session_start`, `session_end`
  (any extension; must be executable). Context passed as JSON via stdin. 10s timeout,
  non-zero exit logged as warning. Wired into orchestrator `__init__` (auto-detects dirs)
  and REPL `run()` / `finally` (session hooks). 14 new tests across 3 features.

## CLI & DX

- [ ] **Model validation on startup** — When `--dry-run` is not set, validate that
  the configured model(s) exist on the provider before spending tokens.

- [x] **Design cockpit** — Dynamic layout sizing based on terminal height. Events panel
  grows to fill available space (deque maxlen=50, displays last N that fit). Progress
  panel smart-truncates: always shows generating/failed files; pending shown as
  summary badge; done files fill remaining budget. Sizes recomputed each render frame.

## MCP Server & API

- [ ] **Full MCP integration** — `matrioska serve` uses the `mcp` library scaffold
  but `mcp` is an optional dependency. Test end-to-end with Claude Desktop and
  verify all 3 tools (run, show, resume) work.

- [x] **Streaming results via API** — `api.py::MatrioskaStreaming.arun()`. Yields
  `{"event": str, "data": dict, "timestamp": float}` dicts via `queue.Queue` +
  sentinel pattern. Intercepts `bus.emit` to capture all events without patching
  subscribers. Final `run_end` event carries full result; `run_error` on exception.
  Usage: `for event in client.arun("task"): ...`

## Eval & Metrics

- [ ] **Benchmark parallelization** — `poc.py` currently runs tasks and orchestrators
  sequentially: `for task in tasks: for orch in ORCHESTRATORS:`. Since tasks are
  independent (no inter-task dependencies), they could run in parallel via
  ProcessPoolExecutor, reducing benchmark time from ~40min to ~10min. Similarly,
  orchestrators on the same task are independent and could run in parallel.
  Caveat: daily API quota is shared, so parallel runs may hit rate limits sooner.
  Recommend: parallelize tasks first (ProcessPoolExecutor with max_workers=2-3),
  then orchestrators if quota permits.

- [ ] **Automated golden suite CI** — The 30-task golden suite exists but requires
  a real API key. Set up a scheduled CI run (e.g., GitHub Actions weekly) that
  posts results to a dashboard.

- [ ] **Execution success metric** — The MVP baseline shows `execution_success_rate:
  0%` because sandbox was never wired. Once sandbox works, measure whether
  generated code actually runs and produces correct output.

- [x] **Regression comparison** — `metrics.py::load_baselines(path)` +
  `save_baselines(metrics, path)`. `MetricComparator(baseline_file=Path(...))` loads
  JSON baselines and falls back to hardcoded MVP defaults if file absent or unrecognised.
  CI can call `save_baselines()` after each run to keep the file updated.

## Observability

- [ ] **OpenTelemetry instrumentation** — `otel_endpoint` config exists but nothing
  is instrumented. Add spans for each pipeline phase, file generation, and LLM
  call, exportable to LangFuse / Grafana.

- [ ] **Cost tracking per provider** — Track actual cost (not estimated) for
  providers that return cost in usage metadata. Fall back to estimation for others.

## Technical Debt

- [ ] **Type annotations audit** — Several `Dict[str, Any]` could be more specific
  (e.g., `SharedState`, `ArtifactMap`). Add type aliases in `core/types.py`.

- [ ] **Test coverage for bug fixes** — Tests exist for imports, config, contracts,
  and DAG layering, but not for the bugs found (generator shadow, missing _emit,
  code fence stripping, rate limit retry). Add integration tests for ACI repair
  and test designer pipeline.

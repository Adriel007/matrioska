# TODO — Matrioska V3

## Pipeline & Generation

- [ ] **Phase 3 sandbox execution** — `execute_container()` exists but untested.
  Wire up the Docker sandbox with volume mounts and capture stdout/stderr/exit code
  as validation signals that feed back into the repair loop.

## Providers & Models

- [ ] **HuggingFace provider** — `_hf_chat()` still raises `NotImplementedError`.
  Port the v2 HuggingFace integration or support any Transformers pipeline.

- [ ] **Provider-aware token costing** — `TokenTracker._estimate_cost()` uses
  hardcoded GPT/Claude pricing. Add a pricing table per provider (OpenAI, Anthropic,
  Groq, NVIDIA, Ollama) or make it configurable via `.env`.

- [ ] **Configurable MoE extension map** — `EXTENSION_MODEL_MAP` is hardcoded in
  `circuit.py`. Load from config so users can define their own per-extension routing
  (e.g., `.py → deepseek-v3.2` on Groq).

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

- [ ] **Drill-down interativo no REPL** — no REPL (`matrioska`), comandos de navegação
  no vault: `/vault search <query>` busca em todos os escopos e mostra ranking,
  `/vault project <nome>` abre overview do projeto, `/vault concept <nome>` abre nota
  do conceito, `/vault related <projeto>` lista projetos com wikilinks comuns.
  Resultado renderizado com Rich — links clicáveis no terminal (Cmd+Click).

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

- [ ] **REPL autocompletion & keyboard navigation** — Auto-complete `/` slash commands
  in the REPL with history-aware suggestions. Support arrow keys, ENTER, and TAB for
  navigation and selection in the completions menu. Integrate with prompt_toolkit's
  `FuzzyCompleter` and `NestedCompleter`. Allow TAB to cycle through suggestions,
  ENTER to select, arrow keys to browse, and Esc to dismiss.

- [ ] **Rewind / checkpoint** — no REPL, `Esc+Esc` ou `/rewind` volta ao último
  checkpoint salvo (desfaz geração da última rodada). StateGraph já tem checkpoints —
  falta expor no REPL.

- [ ] **Comandos customizados** — arquivos `.matrioska/commands/meu-cmd.md` criam
  `/meu-cmd` como slash command no REPL. Conteúdo do arquivo vira prompt injetado.
  Permite criar workflows reutilizáveis por projeto.

- [ ] **Hook system** — `.matrioska/hooks/` com scripts shell executados em eventos:
  `pre_generate` (antes de gerar um arquivo), `post_generate` (após geração),
  `pre_repair` (antes de repair), `session_start`, `session_end`. Scripts recebem
  JSON via stdin com contexto do evento.

## CLI & DX

- [ ] **Model validation on startup** — When `--dry-run` is not set, validate that
  the configured model(s) exist on the provider before spending tokens.

## MCP Server & API

- [ ] **Full MCP integration** — `matrioska serve` uses the `mcp` library scaffold
  but `mcp` is an optional dependency. Test end-to-end with Claude Desktop and
  verify all 3 tools (run, show, resume) work.

- [ ] **Streaming results via API** — The Python API blocks until the pipeline
  completes. Add an async `arun()` that yields events (phase start, file generated,
  validation result) via the event bus.

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

- [ ] **Regression comparison** — `MetricComparator` compares against hardcoded
  baselines. Load baselines from a JSON file so each CI run updates them,
  tracking improvement (or regression) over time.

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
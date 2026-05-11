# TODO — Matrioska V3

## Pipeline & Generation

- [x] **Generator prompt: emit shared_state_updates** — The generator rarely populates
  shared_state_updates via the finish tool, forcing the auto-populate fallback in
  phase2.py. Fixed: system prompt now includes concrete examples of what to emit
  (class names, schemas, routes, URLs) and the user prompt explicitly lists the
  keys the generator must write. Also added shared_state_reads/writes context
  to the repairer prompt.

- [x] **Repairer quality** — On retry, the repairer often produces invalid syntax
  (AST parse fails). Fixed two root causes: (1) the repair loop was passing a
  generic "validation failed" message instead of the actual syntax/contract errors
  from the validator — now `last_errors` captures `result.syntax_error` and
  `result.contract_violations` between attempts; (2) improved the repairer system
  prompt with explicit rules (fix ALL errors, verify surrounding lines for syntax
  errors, preserve structure) and added file spec details + shared_state context
  to the repair prompt.

- [x] **ACI Repairer (SWE-agent, arXiv:2405.15793)** — New dual-mode repairer:
  ACI mode emits targeted JSON patch hunks (start_line/end_line/new_content)
  applied surgically, preserving cross-file invariants. Full-file fallback when
  hunks can't be applied. `use_aci_repair=True` (default). Activated when errors
  reference specific line numbers (regex `line \d+|:\d+:`).

- [x] **Test Designer agent (AgentCoder+AlphaCodium, arXiv:2312.13010+2401.08500)**
  — New `TestDesignerAgent` generates contract-first tests BEFORE code is generated
  (blind to implementation, from FileSpec/contract only). Tests injected into
  Generator prompt as implementation target. After syntax validation passes, tests
  run inline as smoke check; failures feed the Repairer as executable ground-truth
  signal. Configurable via `enable_test_design` (default True).

- [ ] **Phase 3 sandbox execution** — `execute_container()` exists but untested.
  Wire up the Docker sandbox with volume mounts and capture stdout/stderr/exit code
  as validation signals that feed back into the repair loop.

- [x] **Non-blocking generation per file** — A single file failure shouldn't block
  the entire layer. Failed files now get placeholder shared_state artifacts
  (`__auto__<key>__`) via `_finalize_artifact()` so downstream files that depend
  on them can still be generated.

- [x] **Layer-level retry** — When all files in a layer fail due to rate limits,
  retry the entire layer after exponential backoff (2^layer_idx seconds, max 30s).

## Providers & Models

- [ ] **HuggingFace provider** — `_hf_chat()` still raises `NotImplementedError`.
  Port the v2 HuggingFace integration or support any Transformers pipeline.

- [ ] **Provider-aware token costing** — `TokenTracker._estimate_cost()` uses
  hardcoded GPT/Claude pricing. Add a pricing table per provider (OpenAI, Anthropic,
  Groq, NVIDIA, Ollama) or make it configurable via `.env`.

- [ ] **Configurable MoE extension map** — `EXTENSION_MODEL_MAP` is hardcoded in
  `circuit.py`. Load from config so users can define their own per-extension routing
  (e.g., `.py → deepseek-v3.2` on Groq).

- [x] **API connectivity check** — `matrioska run` now probes each configured
  model's provider endpoint before Phase 1. Catches invalid API keys (401),
  unknown models (404), and connection errors early with actionable messages
  pointing to the specific env var to fix (MATRIOSKA_API_KEY, MATRIOSKA_MODEL,
  MATRIOSKA_BASE_URL).

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

- [x] **Vault global em `~/.matrioska/vault/`** — `memory/vault.py::GlobalVault`.
  Diretório `~/.matrioska/vault/` (override via `MATRIOSKA_VAULT_DIR`/`--vault-dir`)
  com layout Obsidian: `projects/<name>/{architecture,patterns,lessons,links}.md`,
  `concepts/<tag>.md`, `bugs/<slug>.md`, `INDEX.md` auto-gerado. YAML frontmatter
  + wikilinks `[[note]]`. Após cada run, `orchestrator._compile_into_vault()` faz
  upsert idempotente (não duplica concept entries no mesmo timestamp).

- [x] **Knowledge compiler (LLM Wiki pattern)** — implementação determinística (sem
  LLM extra). `compile_from_run()`: (1) `derive_tags()` infere concepts do task +
  extensões, (2) `extract_lessons_and_bugs()` lê `repair_count > 0` (lesson) e
  `status == "failed"` (bug), (3) upsert por seção via markers HTML
  (`<!-- section:ID -->`) com modos `append` / `overwrite` / `append_dedup`.
  Stub para futura camada LLM merge — basta substituir `_extract_patterns`.

- [x] **Dual-level retrieval (LightRAG style)** — `GlobalVault.search(query, scope=...)`
  com 4 escopos: `local` (knowledge/runs do projeto atual), `global`
  (concepts + bugs cross-project), `linked` (BFS sobre wikilinks em `links.md`,
  `max_hops=2`), `all`. Ranking: tokens query × {3.0×title + 2.0×tags + 1.0×body}.
  ChromaDB pluggable como re-ranker (não obrigatório).

- [x] **Context scoping por query** — CLI: `matrioska vault search <query> --scope
  {local,global,linked,all}`. Internamente, o orchestrator chama
  `_retrieve_vault_context(task)` com scope=global antes de Phase 1, injetando
  "RELEVANT VAULT KNOWLEDGE" no system prompt do Architect.

- [x] **Cross-project links (wikilinks + grafo)** — `links.md` por projeto contém
  `[[projeto_b]]` wikilinks. `_linked_projects()` faz BFS bounded para scope=linked.
  `matrioska vault graph` exporta Mermaid flowchart de todos os wikilinks
  (compatível com qualquer viewer Markdown ou Obsidian Graph View).

- [x] **MCP server para o vault** — `api.py::create_mcp_server` agora registra
  ferramentas read-side do vault em adição aos pipeline tools: `vault_search`
  (com scope local|global|linked|all), `vault_get`, `vault_list`, `vault_doctor`,
  `vault_graph`, `vault_related`. Writes intencionalmente não expostos — a
  única forma de escrever no vault é via `orchestrator._compile_into_vault`
  após uma run real (mantém o vault como derivada deterministicamente).
  Tool descriptors em `MCP_TOOLS` para listagem por clients.

- [ ] **Drill-down interativo no REPL** — no REPL (`matrioska`), comandos de navegação
  no vault: `/vault search <query>` busca em todos os escopos e mostra ranking,
  `/vault project <nome>` abre overview do projeto, `/vault concept <nome>` abre nota
  do conceito, `/vault related <projeto>` lista projetos com wikilinks comuns.
  Resultado renderizado com Rich — links clicáveis no terminal (Cmd+Click).

- [x] **Vault health & graph visualization** — `matrioska vault doctor` reporta
  total de notas, projetos, concepts, bugs, orphans (notas sem links in/out fora
  de pastas de projeto), stale (> 30 dias via frontmatter `updated`), broken_links
  (wikilinks que apontam para notas inexistentes). Status: `healthy` ou
  `issues_found`. `matrioska vault graph -o graph.md` exporta Mermaid flowchart.

### Outros itens de Memory

- [x] **DSPy compilation loop** — `eval/dspy_compiler.py`. Scaffold completo:
  `compile_target(target, category, max_tasks, val_fraction)` separa golden
  tasks em train/val, roda baseline, e se DSPy estiver instalado executa
  `BootstrapFewShot` com `golden_metric` (binary 0/1 via `evaluate_result`)
  como objetivo. Demos extraídos e persistidos em `~/.matrioska/dspy_compiled/
  {target}_demos.json`. Quando `dspy` não está instalado, retorna baseline
  com `skipped_reason='dspy_not_installed'`. CLI: `matrioska compile --target
  architect --category cli --max-tasks 5`. Faltando: integrar demos compilados
  no `ArchitectAgent` (hot-swap do system prompt) — TBD.

- [ ] **GraphRAG auto-ingest** — Semantic memory tracks concepts and relationships
  but doesn't auto-extract them from run artifacts. Add LLM-based entity extraction
  on `write_run_note()` to populate the knowledge graph.

- [ ] **ChromaDB lazy download** — First-run downloads 79 MB of ONNX model for
  embeddings. Show a progress bar and consider making embeddings optional
  (keyword-only fallback).

## CLI & Interactive REPL (Claude Code-inspired)

*Executar `matrioska` sem subcomando abre um REPL interativo — prompt simples de
conversa com o agente, mais `/comandos` para controle. Inspirado no Claude Code.*

- [x] **REPL interativo** — `cli/repl.py`. `matrioska` sem args abre prompt
  `matrioska › ` com prompt_toolkit (histórico em `~/.matrioska/history`,
  Ctrl+D sai, Ctrl+C cancela task em andamento), fallback `input()` quando
  prompt_toolkit ausente ou stdin não-TTY. Texto livre → `Matrioska(cfg).run(task)`
  em thread separada, com integração de streaming opcional.

- [x] **Slash commands no REPL** — registry via decorator `@command("name", "help")`.
  Implementados: `/help`, `/quit`, `/exit`, `/config`, `/model [name]`, `/usage`,
  `/clear`, `/plan`, `/effort {low,medium,high}`, `/memory`, `/init`, `/diff`,
  `/slots`, `/btw <q>`, `/vault [list|search|doctor|graph]`, `/stream`, `/history`.

- [x] **`!` prefix para shell** — `!<cmd>` no REPL roda `subprocess.run(..., shell=True,
  timeout=60)` e imprime stdout/stderr inline com a sessão. Exit code não-zero
  é exibido em dim.

- [x] **Effort levels** — `/effort {low,medium,high}` no REPL. low=1 candidate,
  no ToT/Reflexion/TestDesign; medium=default cfg; high=5 candidates + ToT
  + Reflexion + TestDesign. Aplicado via `_effective_cfg_for_run()` (não
  mutila a Config base, retorna uma cópia).

- [x] **Plan mode interativo** — `matrioska run --interactive` mostra a arquitetura
  proposta após Phase 1 e pergunta `[Y/n/q]` antes de prosseguir para Phase 2.
  Quando `--mode plan`, retorna logo após Phase 1 sem prompt. Dashboard é
  desabilitado automaticamente quando o flow precisa de stdin.

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

- [x] **`/btw` — pergunta rápida sem histórico** — `matrioska btw "<question>"`
  faz one-shot ao LLM com system prompt "concise technical assistant, 1-5 sentences"
  e renderiza a resposta como Markdown via Rich. Sem episodic note, sem vault write,
  sem pipeline. Quando o REPL existir, vira `/btw <question>` (mesma rota).

- [ ] **Hook system** — `.matrioska/hooks/` com scripts shell executados em eventos:
  `pre_generate` (antes de gerar um arquivo), `post_generate` (após geração),
  `pre_repair` (antes de repair), `session_start`, `session_end`. Scripts recebem
  JSON via stdin com contexto do evento.

- [x] **Permission modes** — `--mode auto|plan|ask`:
  - `auto` (default) — gera tudo sem perguntar
  - `plan` — força `plan_only`, retorna após Phase 1
  - `ask` — em Phase 2, antes de cada layer, prompt `[Y/n/s/q]` por arquivo
    (skip cria placeholder em `shared_state` para downstream não quebrar).
  Dashboard auto-desabilitado em `ask`. Quit aborta com exit 130.

## CLI & DX

- [x] **Progress display** — Rich live dashboard implementado: painel de API slots
  com cooldowns em tempo real, progress bar de arquivos, tokens/custo, log de eventos.
  `matrioska run` usa dashboard por default; `--no-dashboard` para logs simples.

- [x] **`--quick` mode** — `cfg.quick=True` colapsa: `enable_tot=False`,
  `enable_reflexion=False`, `enable_test_design=False`, `architect_candidates=1`,
  `max_repairs=1`, skip Phase 3. Respeita overrides explícitos via CLI/env
  (não sobrescreve flags que o usuário setou diretamente).

- [ ] **Model validation on startup** — When `--dry-run` is not set, validate that
  the configured model(s) exist on the provider before spending tokens.

- [x] **Better error messages** — API errors now include actionable guidance:
  401 → "Check MATRIOSKA_API_KEY in .env", 404 → "Check MATRIOSKA_MODEL
  (current: <model>)", connection refused → "Check MATRIOSKA_BASE_URL",
  429 → "Wait for rate limit reset or switch provider". Both in the LLM
  client retry loop and the startup connectivity check.

- [x] **`matrioska init`** — `cli/init_wizard.py`. Wizard interativo: pick
  provider (10 opções: openai/anthropic/groq/openrouter/deepseek/xai/mistral/
  together/nvidia/ollama), base_url, api_key, model, per-role overrides opcionais,
  multi-key rotation opcional. Detecta `.env` existente e oferece merge/overwrite/
  abort. Gera `.env` agrupado por seções comentadas + opcional `MATRIOSKA.md` scaffold.

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

- [x] **Evaluator `fulfillment_rate` bug** — `len(found) / len(expected_keys)` was
  always 1.0 because `found` dict contained ALL keys (including False values) —
  `len(found) == len(expected_keys)` always. Fixed to `sum(1 for v in found.values() if v) / len(expected_keys)`.

- [x] **Benchmark wrapper crash propagation** — Matrioska wrapper now saves full
  traceback to `.matrioska_error.txt` and triggers Agentless fallback (run_direct)
  when the orchestrator crashes or returns 0 files. Token count fix: uses
  `prompt_tokens + completion_tokens` if `total_tokens` key absent.

- [ ] **Automated golden suite CI** — The 30-task golden suite exists but requires
  a real API key. Set up a scheduled CI run (e.g., GitHub Actions weekly) that
  posts results to a dashboard.

- [ ] **Execution success metric** — The MVP baseline shows `execution_success_rate:
  0%` because sandbox was never wired. Once sandbox works, measure whether
  generated code actually runs and produces correct output.

- [ ] **Regression comparison** — `MetricComparator` compares against hardcoded
  baselines. Load baselines from a JSON file so each CI run updates them,
  tracking improvement (or regression) over time.

## Claude Code-inspired Features

Diferenciais do Claude Code que a Matrioska não tem e valem implementar:

- [x] **Real code execution feedback loop** — Após cada arquivo .py gerado e aprovado
  na validação sintática, roda `python -c "import <module>"` em subprocess com timeout
  de 8s. Se returncode != 0, o stderr vira sinal de repair no próximo attempt. Servidores
  (uvicorn, flask.run, etc.) são detectados e pulados para evitar hang. Config:
  `execute_feedback=True` (default). Arquivos são escritos em disco via `_write_artifact_to_disk`
  logo após geração para que imports resolvam corretamente.

- [x] **Codebase pre-flight** — `pipeline/preflight.py`: antes de Phase 1, escaneia
  `work_dir` (ou `project_dir`) por arquivos existentes (.py, .ts, .js, .go, etc.) e
  constrói um bloco de contexto injetado no Architect. Limite: 24K chars total, 3K por
  arquivo. Ignora .venv, node_modules, __pycache__, .git, etc.

- [x] **Surgical file editing (incremental mode)** — `cfg.incremental=True`: antes de
  gerar um arquivo, verifica se ele já existe em `work_dir`. Se sim, injeta o conteúdo
  existente no prompt do Generator com instrução de modificar apenas o necessário.
  Funciona como "in-place edit" para modelos que suportam JSON mode ou tool use.

- [x] **Real dependency installation** — `pipeline/executor.py`: após Phase 2, detecta
  imports não-stdlib nos .py gerados (via AST parse, fallback regex), e roda `pip install -q`
  em subprocess. Aliases conhecidos: cv2→opencv-python, PIL→Pillow, etc. Config:
  `install_deps=True` (default). Best-effort: falhas são logadas mas não travam a pipeline.

- [x] **CLAUDE.md / MATRIOSKA.md injection** — `pipeline/preflight.py`: antes de Phase 1,
  procura `MATRIOSKA.md` ou `CLAUDE.md` em `project_dir` → `work_dir` → `cwd`. O conteúdo
  é injetado no system prompt do Architect como "USER INSTRUCTIONS", permitindo definir
  convenções, linguagens preferidas, restrições de deps.

- [x] **Incremental task mode** — `cfg.incremental=True` + `cfg.project_dir=<path>`:
  combina pre-flight (injeta código existente no Architect) com surgical editing (injeta
  arquivo existente no Generator). O Architect recebe o codebase atual e pode decidir
  gerar apenas os arquivos que precisam mudar.

- [x] **Streaming output / live progress** — Flag `cfg.stream_tokens=True`
  (CLI: `/stream` no REPL). `_openai_compatible_stream()` faz SSE com
  `stream_options.include_usage`, acumula `delta.content` na ChatResponse.text
  (callers não precisam mudar nada), e emite `llm_token` event por chunk.
  Desabilitado quando há tools/json_schema (não vale a complexidade de remontar
  tool_calls chunked). Fallback automático para non-streaming se SSE quebrar.

## Observability

- [ ] **OpenTelemetry instrumentation** — `otel_endpoint` config exists but nothing
  is instrumented. Add spans for each pipeline phase, file generation, and LLM
  call, exportable to LangFuse / Grafana.

- [ ] **Cost tracking per provider** — Track actual cost (not estimated) for
  providers that return cost in usage metadata. Fall back to estimation for others.

## Technical Debt

- [x] **Deduplicate `_strip_fences`** — Extracted to `core/text_utils.py` as
  `strip_fences()`. Both generator.py and repairer.py now import from there.
  Also added more fence variants (html, css, js, bash, dockerfile, yml).

- [x] **`SystemExit` in orchestrator → `RuntimeError`** — `_check_connectivity()`
  raised `SystemExit` (BaseException) which bypassed the benchmark harness's
  `except Exception`, crashing the process for all subsequent tasks. Changed to
  `RuntimeError`. Also made `SemanticMemory` init optional (ChromaDB failures
  log a warning instead of crashing the pipeline).

- [x] **Generator project_memory_section no-op** — `GENERATOR_SYSTEM_PROMPT` had
  a `{project_memory_section}` placeholder but `.replace("{project_memory_section}", "")`
  always zeroed it. Now reads `ProceduralMemory` and injects actual content.

- [x] **Generator `use_tools` for OpenAI-compat providers** — Groq/OpenRouter/NVIDIA
  were sending tools (provider=="openai") but the `is_official` MoE check excluded
  them from the code path. Now `force_tools=True` for all openai-compat providers;
  the client's HTTP 400 fallback handles rejection gracefully.

- [ ] **Type annotations audit** — Several `Dict[str, Any]` could be more specific
  (e.g., `SharedState`, `ArtifactMap`). Add type aliases in `core/types.py`.

- [ ] **Test coverage for bug fixes** — Tests exist for imports, config, contracts,
  and DAG layering, but not for the bugs found (generator shadow, missing _emit,
  code fence stripping, rate limit retry). Add integration tests for ACI repair
  and test designer pipeline.

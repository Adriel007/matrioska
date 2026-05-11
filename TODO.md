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

- [ ] **Vault global em `~/.matrioska/vault/`** — diretório único compartilhado entre
  todos os projetos Matrioska. Compatível nativamente com Obsidian (Markdown + YAML
  frontmatter + wikilinks `[[NomeDoArquivo]]`). Após cada run, extrair conceitos,
  padrões e bugs e fazer upsert nas notas relevantes do vault global.
  Inspiração: Karpathy LLM Wiki — "compilar em vez de re-derivar".

- [ ] **Knowledge compiler (LLM Wiki pattern)** — após cada run bem-sucedido, um agente
  "compiler" lê os artefatos gerados + erros encontrados e produz/atualiza notas no
  vault global: (1) identifica qual conceito foi usado (ex: "SQLite", "argparse"),
  (2) extrai padrões ("usa AUTOINCREMENT", "separa DB logic em módulo próprio"),
  (3) extrai bugs ("modelo 8b esquece fechar conexão SQLite"), (4) faz merge com a
  nota existente — nunca substitui, apenas enriquece. Abordagem: LLM small/fast
  gera diff da nota, outro LLM faz merge inteligente.

- [ ] **Dual-level retrieval (LightRAG style)** — implementar dois escopos de busca
  distintos, configuráveis por query:
  - **Local**: busca apenas no vault do projeto atual (`./matrioska_work/knowledge/`)
  - **Global**: busca em conceitos transversais (`~/.matrioska/vault/concepts/` + bugs)
  - **Cross-project**: atravessa wikilinks para encontrar projetos com padrões similares
    (ex: "que outros projetos usaram SQLite + argparse?")
  Retrieval: BM25 keyword + ChromaDB embeddings + graph traversal por wikilinks.
  Fusão via RRF (Reciprocal Rank Fusion) — mesmo approach que atinge 95.2% no
  LongMemEval-S (agentmemory, 2025).

- [ ] **Context scoping por query** — flag `--scope local|global|linked|all`:
  - `local`  — só memória do projeto atual (default, mais barato)
  - `global` — conceitos e padrões globais sem cruzar projetos específicos
  - `linked` — projeto atual + projetos que o usuário marcou como relacionados
  - `all`    — vault inteiro (para queries de "o que aprendi sobre SQLite em geral?")
  O Architect recebe o contexto filtrado pelo scope, evitando ruído de projetos
  não relacionados. Granularidade: projeto → módulo → conceito → bug.

- [ ] **Cross-project links (wikilinks + grafo)** — o usuário pode marcar projetos
  como relacionados via `[[projeto_b]]` em `links.md` do projeto A, ou via
  `matrioska link projeto_a projeto_b --reason "mesma stack FastAPI+SQLite"`.
  Quando scope=linked, o retrieval atravessa esses links com BFS limitado a `max_hops`
  (default 2). Dual-level: fine-grained local + high-level global (LightRAG arXiv 2409.14813).

- [ ] **MCP server para o vault** — expor o vault da Matrioska como MCP server para
  que outros agentes (Claude Code, Cursor, Windsurf) possam ler/escrever a memória.
  Ferramentas MCP: `search_vault(query, scope, project?)`, `get_note(path)`,
  `list_project_notes(project)`, `find_related(project, max_hops)`,
  `upsert_concept(name, content)`. Implementar com `mcp` library (já dep opcional).
  Referência: engraph (engraph GitHub), MCPVault (bitbonsai GitHub),
  obsidian-mcp (lstpsche GitHub).

- [ ] **Drill-down interativo no REPL** — no REPL (`matrioska`), comandos de navegação
  no vault: `/vault search <query>` busca em todos os escopos e mostra ranking,
  `/vault project <nome>` abre overview do projeto, `/vault concept <nome>` abre nota
  do conceito, `/vault related <projeto>` lista projetos com wikilinks comuns.
  Resultado renderizado com Rich — links clicáveis no terminal (Cmd+Click).

- [ ] **Vault health & graph visualization** — `matrioska vault doctor` detecta notas
  órfãs (sem wikilinks), conceitos stale (último update > 30 dias sem uso),
  projetos sem lessons.md. `matrioska vault graph` exporta o grafo de wikilinks como
  DOT/Mermaid para visualização. Alternativa: abrir diretamente no Obsidian — o vault
  já é compatível nativamente com o Graph View do Obsidian.

### Outros itens de Memory

- [ ] **DSPy compilation loop** — `ProceduralMemory` is a scaffold. Wire up the
  golden task suite as a DSPy training set: generate → evaluate → compile prompt
  → repeat until `first_pass_rate ≥ 80%`.

- [ ] **GraphRAG auto-ingest** — Semantic memory tracks concepts and relationships
  but doesn't auto-extract them from run artifacts. Add LLM-based entity extraction
  on `write_run_note()` to populate the knowledge graph.

- [ ] **ChromaDB lazy download** — First-run downloads 79 MB of ONNX model for
  embeddings. Show a progress bar and consider making embeddings optional
  (keyword-only fallback).

## CLI & Interactive REPL (Claude Code-inspired)

*Executar `matrioska` sem subcomando abre um REPL interativo — prompt simples de
conversa com o agente, mais `/comandos` para controle. Inspirado no Claude Code.*

- [ ] **REPL interativo** — `matrioska` sem args abre um loop de pergunta-resposta
  com o orquestrador como agente. Suporte a multiline (`\`+Enter), histórico de
  comandos (↑/↓), Ctrl+C para cancelar geração, Ctrl+D para sair. Usar `prompt_toolkit`
  ou `readline` para edição de linha rica.

- [ ] **Slash commands no REPL** — Dispatcher de `/comando` no REPL:
  - `/model [nome]`     — trocar modelo mid-session; sem args mostra picker interativo
  - `/config`           — exibir config atual (provider, model, flags, work_dir)
  - `/usage`            — tokens usados na sessão, custo estimado, rate limit status por slot
  - `/context`          — tamanho do contexto atual vs. limite do modelo
  - `/clear`            — nova sessão, mantém memória do projeto
  - `/plan`             — ativa plan mode (Architect planeja, não gera)
  - `/memory`           — ver e editar MATRIOSKA.md + auto-memory persistente
  - `/init`             — gerar MATRIOSKA.md para o projeto atual com scaffold
  - `/review`           — code review read-only dos últimos artefatos gerados
  - `/diff`             — o que mudou nesta sessão (artefatos gerados vs. existentes)
  - `/compact`          — comprimir histórico de conversa (manter só resumo)
  - `/slots`            — status do SlotPool em tempo real (disponível/cooldown)
  - `/help`             — listar todos os comandos disponíveis

- [ ] **`!` prefix para shell** — no REPL, `!ls -la` executa o comando e injeta o
  output no contexto do agente. Permite inspecionar o projeto sem sair do REPL.

- [ ] **Effort levels** — `/effort low|medium|high` (ou `--effort` no CLI) controla
  o quanto o Architect "pensa": low=1 candidato sem ToT, medium=3 candidatos,
  high=5 candidatos + Judge + Reflexion completo.

- [ ] **Plan mode interativo** — `/plan` no REPL entra em modo onde o agente só planeja
  (mostra arquitetura proposta) e aguarda aprovação antes de gerar. `y` executa,
  `n` cancela, `e` edita o plano antes de executar.

- [ ] **Rewind / checkpoint** — no REPL, `Esc+Esc` ou `/rewind` volta ao último
  checkpoint salvo (desfaz geração da última rodada). StateGraph já tem checkpoints —
  falta expor no REPL.

- [ ] **Comandos customizados** — arquivos `.matrioska/commands/meu-cmd.md` criam
  `/meu-cmd` como slash command no REPL. Conteúdo do arquivo vira prompt injetado.
  Permite criar workflows reutilizáveis por projeto.

- [ ] **`/btw` — pergunta rápida sem histórico** — `/btw o que faz esse arquivo?`
  faz um call ao LLM sem adicionar a pergunta/resposta ao contexto da sessão.
  Útil para consultas pontuais que não devem "poluir" o contexto da task.

- [ ] **Hook system** — `.matrioska/hooks/` com scripts shell executados em eventos:
  `pre_generate` (antes de gerar um arquivo), `post_generate` (após geração),
  `pre_repair` (antes de repair), `session_start`, `session_end`. Scripts recebem
  JSON via stdin com contexto do evento.

- [ ] **Permission modes** — `--mode auto|plan|ask` controla nível de autonomia:
  `ask`=aprova cada arquivo antes de gerar (default), `plan`=só planeja,
  `auto`=gera tudo sem perguntar. Equivalente ao `acceptEdits`/`bypassPermissions`
  do Claude Code.

## CLI & DX

- [x] **Progress display** — Rich live dashboard implementado: painel de API slots
  com cooldowns em tempo real, progress bar de arquivos, tokens/custo, log de eventos.
  `matrioska run` usa dashboard por default; `--no-dashboard` para logs simples.

- [ ] **`--quick` mode** — Skips ToT, reflexion, contract validation in Phase 2,
  and Phase 3 entirely. Useful for rapid iteration during development.

- [ ] **Model validation on startup** — When `--dry-run` is not set, validate that
  the configured model(s) exist on the provider before spending tokens.

- [x] **Better error messages** — API errors now include actionable guidance:
  401 → "Check MATRIOSKA_API_KEY in .env", 404 → "Check MATRIOSKA_MODEL
  (current: <model>)", connection refused → "Check MATRIOSKA_BASE_URL",
  429 → "Wait for rate limit reset or switch provider". Both in the LLM
  client retry loop and the startup connectivity check.

- [ ] **`matrioska init`** — Scaffold a `.env` with interactive prompts for
  provider, API key, and model selection.

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

- [ ] **Streaming output / live progress** — Claude Code mostra o código sendo gerado
  em tempo real. A Matrioska é silenciosa (só logs). Adicionar streaming opcional:
  ao receber tokens do LLM, emitir via EventBus linha a linha. Útil para UX e para
  detectar quando o modelo está divagando antes de completar o arquivo.

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

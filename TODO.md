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

- [ ] **DSPy compilation loop** — `ProceduralMemory` is a scaffold. Wire up the
  golden task suite as a DSPy training set: generate → evaluate → compile prompt
  → repeat until `first_pass_rate ≥ 80%`.

- [ ] **GraphRAG auto-ingest** — Semantic memory tracks concepts and relationships
  but doesn't auto-extract them from run artifacts. Add LLM-based entity extraction
  on `write_run_note()` to populate the knowledge graph.

- [ ] **ChromaDB lazy download** — First-run downloads 79 MB of ONNX model for
  embeddings. Show a progress bar and consider making embeddings optional
  (keyword-only fallback).

## CLI & DX

- [ ] **Progress display** — The current pipeline is silent except for log lines.
  Add a Rich-based live display showing: current phase, files generated/remaining,
  token usage, estimated cost, ETA.

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

- [ ] **Real code execution feedback loop** — O diferencial central: Claude Code executa
  o código gerado, captura stdout/stderr, e usa o erro real como sinal de repair.
  A Matrioska só valida sintaxe (AST parse). Wiring: ao final de cada arquivo gerado,
  rodar em subprocess com timeout; se exitcode != 0, passar stderr para o Repairer como
  `test_failures`. Isso elimina a necessidade do TestDesigner para validação funcional.

- [ ] **Codebase pre-flight** — Claude Code lê o projeto existente antes de começar.
  A Matrioska começa cega. Adicionar uma fase 0 que lê `work_dir` e injeta arquivos
  existentes no shared_state: nomes, assinaturas de função, schemas de DB.
  Crítico para tarefas de incremento (add feature, fix bug) onde o projeto já existe.

- [ ] **Surgical file editing (diff-based)** — Claude Code edita linhas específicas com
  Edit/Write. A Matrioska regenera o arquivo inteiro. O ACI repair vai na direção certa
  mas só é ativado no repair loop. Estender para: se o arquivo já existe em work_dir,
  gerar um diff-patch em vez de arquivo completo.

- [ ] **Real dependency installation** — Claude Code pode rodar `pip install`, `npm i`,
  `go get`. A Matrioska gera código mas não instala deps. Adicionar um Phase 3.5 que
  lê os `import` dos arquivos gerados, detecta pacotes não-stdlib, e roda `pip install`
  antes de executar os testes. Requer sandbox ou venv isolado.

- [ ] **CLAUDE.md / project context injection** — Claude Code lê CLAUDE.md para
  entender convenções do projeto. Adicionar suporte a um arquivo `MATRIOSKA.md` (ou
  ler o CLAUDE.md se presente) e injetar no system prompt do Architect. Permite ao
  usuário definir: linguagem preferida, padrões de código, restrições de deps.

- [ ] **Incremental task mode** — Claude Code trata "add feature X" diferente de
  "build from scratch". A Matrioska sempre recomeça do zero. Adicionar modo
  `--task-type=increment` que: (1) lê arquivos existentes, (2) passa para o Architect
  como contexto, (3) gera apenas os arquivos que precisam mudar. Diferença entre
  phase1 atual e phase1 com diff-awareness.

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

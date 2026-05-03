# TODO — Matrioska V3

## Pipeline & Generation

- [ ] **Generator prompt: emit shared_state_updates** — The generator rarely populates
  `shared_state_updates` via the `finish` tool, forcing the auto-populate fallback in
  phase2.py. The system prompt needs examples of _what_ to emit (class names, schemas,
  routes, URLs) and _why_ downstream files depend on them.

- [ ] **Repairer quality** — On retry, the repairer often produces invalid syntax
  (AST parse fails). The repair prompt should include the full file spec again, not
  just the error diff, and should enforce that output passes syntax checks.

- [ ] **Phase 3 sandbox execution** — `execute_container()` exists but untested.
  Wire up the Docker sandbox with volume mounts and capture stdout/stderr/exit code
  as validation signals that feed back into the repair loop.

- [ ] **Non-blocking generation per file** — A single file failure shouldn't block
  the entire layer. Failed files should get placeholder artifacts so downstream files
  that depend on them (via shared_state) can still be generated.

- [ ] **Layer-level retry** — When all files in a layer fail due to rate limits,
  retry the entire layer after a backoff instead of cascading failures.

## Providers & Models

- [ ] **HuggingFace provider** — `_hf_chat()` still raises `NotImplementedError`.
  Port the v2 HuggingFace integration or support any Transformers pipeline.

- [ ] **Provider-aware token costing** — `TokenTracker._estimate_cost()` uses
  hardcoded GPT/Claude pricing. Add a pricing table per provider (OpenAI, Anthropic,
  Groq, NVIDIA, Ollama) or make it configurable via `.env`.

- [ ] **Configurable MoE extension map** — `EXTENSION_MODEL_MAP` is hardcoded in
  `circuit.py`. Load from config so users can define their own per-extension routing
  (e.g., `.py → deepseek-v3.2` on Groq).

- [ ] **API connectivity check** — `matrioska run` should probe the provider endpoint
  before starting the pipeline, with a clear error message if the API key, base URL,
  or model name is invalid.

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

- [ ] **Better error messages** — Wrap API errors (404 model not found, 401 auth,
  429 rate limit) with actionable guidance: which env var to check, which model
  name to use, how long to wait.

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

- [ ] **Deduplicate `_strip_fences`** — The same code fence stripping function
  exists in both `generator.py` and `repairer.py`. Extract to a shared utility.

- [ ] **Type annotations audit** — Several `Dict[str, Any]` could be more specific
  (e.g., `SharedState`, `ArtifactMap`). Add type aliases in `core/types.py`.

- [ ] **Test coverage for bug fixes** — Tests exist for imports, config, contracts,
  and DAG layering, but not for the bugs found (generator shadow, missing _emit,
  code fence stripping, rate limit retry). Add integration tests.

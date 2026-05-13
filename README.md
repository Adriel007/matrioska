# Matrioska V3

**Contract-first, state-graph multi-agent LLM orchestrator for code generation.**

Matrioska decomposes complex coding tasks into a DAG of files that coordinate through a typed `shared_state` whiteboard. Three pipeline phases — Architecture (with Tree-of-Thoughts + Multi-Planning), Generation (DAG-layered parallel with AlphaCodium, Reflexion, and ACI Repair), and Verification (contract validation + sandbox) — produce complete, validated projects.

## Architecture

```
Task → [Phase 1: MetaPlanner (optional) → N× Architect → Judge → Best Plan]
    → [Phase 2: DAG layers → TestDesign ∥ Generate ∥ Validate ∥ ACIRepair ∥ Reflect]
    → [Phase 3: Contract check → Cross-file → Sandbox → Replan?]
    → Output + Episodic Note + Vault Compilation
```

### Modular Monolith with Event-Driven Core

```
src/matrioska/
├── core/         State graph, typed contracts, events, config, text utils, types
├── llm/          Multi-provider client, circuit breaker, MoE router, slot pool
├── memory/       Episodic → Semantic → Procedural + Obsidian Vault (global)
├── agents/       Architect, MultiPlanner, Generator, TestDesigner, Validator,
│                 Judge, Repairer, Reflector
├── pipeline/     3-phase orchestration with checkpointing, preflight, executor
├── hooks.py      Hook system: .matrioska/hooks/ scripts on pipeline events
├── tools/        Sandbox executor, tool dispatcher
├── eval/         Metrics, golden regression suite (30 tasks), bench.py
├── cli/          Rich dashboard + REPL + init wizard
│                 (run, resume, show, clean, serve, eval, compile, init, btw, vault)
└── api.py        Python API (run + arun streaming) + MCP server
```

## Key Design Decisions

| Dimension | Choice | Rationale |
|-----------|--------|-----------|
| **Coordination** | Typed shared_state contracts | Prevents chain hallucination (MetaGPT insight) |
| **DAG** | Kahn topological layers | Enables intra-layer parallelism |
| **Architecture** | Tree-of-Thoughts (N candidates → Judge voting) | 70pp improvement on reasoning tasks (Yao et al.) |
| **Multi-Planning** | MetaPlanner → N scoped sub-domain architects | Better decomposition for complex multi-component tasks |
| **Test Design** | Contract-first TestDesigner (blind to code) | Eliminates "test-the-bug-you-wrote" (AgentCoder, arXiv:2312.13010) |
| **Generation** | AlphaCodium flow: tests → generate → smoke-check | GPT-4: 19%→44% pass@5 on CodeContests (arXiv:2401.08500) |
| **Validation** | Process supervision (syntax + contracts per-file) | Outperforms outcome-only (Lightman et al.) |
| **Repair** | ACI targeted patch (hunks) + full-file fallback | Preserves cross-file invariants (SWE-agent, arXiv:2405.15793) |
| **Reflection** | Verbal reflection → episodic memory | 91% HumanEval (Shinn et al.) |
| **Memory** | Episodic → Semantic → Procedural + Global Vault | Multi-timescale retrieval + Karpathy LLM Wiki Pattern |
| **Models** | Role-specific (Architect≠Generator≠Validator) | Right capability/cost per role |
| **State** | Checkpointed state graph | Resume, branching, time-travel debug |
| **Execution** | Docker sandbox (optional) | Ground truth feedback (AutoDev-inspired) |
| **Fallback** | Agentless single-shot when orchestrator fails | Safety net (arXiv:2407.01489) |
| **DSPy** | Prompt compilation scaffold | Smaller models via optimized few-shots |

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Interactive setup: pick provider, model, scaffold .env and MATRIOSKA.md
matrioska init

# …or set env vars manually
export MATRIOSKA_PROVIDER=openai
export MATRIOSKA_BASE_URL=https://api.openai.com/v1
export MATRIOSKA_API_KEY=sk-...
export MATRIOSKA_MODEL=gpt-4o-mini

# Run a task
matrioska run --task "Create a FastAPI CRUD API for books with SQLite"

# Rapid iteration: skip ToT, Reflexion, TestDesign, ACI, Phase 3
matrioska run --task "..." --quick

# Multi-planning: meta-decompose task into N sub-domains before coding
matrioska run --task "Build a full-stack blog with auth, API, and React UI" \
    --enable-multi-plan

# Permission modes
matrioska run --task "..." --mode plan         # plan only (return after Phase 1)
matrioska run --task "..." --mode ask          # confirm each file before generation
matrioska run --task "..." --interactive       # review and approve plan before Phase 2

# Skip connectivity check (useful in CI)
matrioska run --task "..." --skip-validation

# One-shot LLM query without touching the pipeline or memory
matrioska btw "what is RRF retrieval fusion?"

# Resume / inspect
matrioska resume
matrioska show

# Interactive REPL (no subcommand) — slash commands, autocompletion, ! shell
matrioska
# matrioska › /help
# matrioska › /effort high            → 5 candidates, ToT + Reflexion + TestDesign
# matrioska › /stream                 → token-by-token via EventBus
# matrioska › Create a FastAPI CRUD API with SQLite
# matrioska › /vault search "sqlite"  → inline vault query with Rich output
# matrioska › /vault project myapp    → view project notes
# matrioska › /rewind                 → restore last checkpoint
# matrioska › !ls matrioska_work
# matrioska › /usage                  → tokens / cost this session

# Global knowledge vault (Obsidian-compatible)
matrioska vault list
matrioska vault search "sqlite patterns" --scope global
matrioska vault doctor
matrioska vault graph -o vault-graph.md

# Evaluation and regression
matrioska eval --category cli
matrioska eval --category cli --save-baseline --baseline-file baselines.json

# DSPy-driven prompt compilation
matrioska compile --target architect --category cli --max-tasks 5
```

### Provider Examples

```bash
# Anthropic (native)
matrioska run --provider anthropic --api-key sk-ant-... \
    --model claude-sonnet-4 --architect-model claude-opus-4 \
    --task "Create a Python CLI todo app"

# Groq (fast inference)
export MATRIOSKA_BASE_URL=https://api.groq.com/openai/v1
export MATRIOSKA_API_KEY=gsk-...
export MATRIOSKA_MODEL=llama-3.3-70b-versatile
matrioska run --task "Create a Python CLI todo app with SQLite"

# OpenRouter (multi-provider gateway)
matrioska run --base-url https://openrouter.ai/api/v1 --api-key sk-or-... \
    --model anthropic/claude-sonnet-4 \
    --task "Build a markdown-to-HTML converter"

# Ollama (local)
matrioska run --provider openai --base-url http://localhost:11434/v1 \
    --api-key ollama --model llama3.1:8b \
    --task "Create a file renaming utility"

# Multi-key rotation (same provider, round-robin)
export MATRIOSKA_API_KEYS=gsk_key1,gsk_key2,gsk_key3

# Extra endpoints (fallback to different providers)
export MATRIOSKA_EXTRA_ENDPOINTS='[{"provider":"deepseek","base_url":"https://api.deepseek.com/v1","api_key":"ds_xxx","model":"deepseek-coder-v2"}]'
```

### Python API

```python
from matrioska import Matrioska, Config, load_config

cfg = load_config({"provider": "openai", "model": "gpt-4o-mini"})
m = Matrioska(cfg)

# Blocking run
result = m.run("Create a CLI todo app with SQLite")
print(result["status"])       # success | partial | failed | aborted
print(result["project_name"]) # snake_case_project_name
for a in result["artifacts"]:
    print(f"  {a.name}.{a.extension} [{a.status}] {len(a.content)} chars")

# Streaming: yield events as they happen
for event in m.arun("Create a CLI todo app"):
    print(event["event"], event["data"])
    if event["event"] == "run_end":
        result = event["data"]
        break
```

## Pipeline Phases

### Phase 1: Architecture

#### Standard (Tree-of-Thoughts)
- N parallel Architect calls (default N=3, temperature=0.7 for diversity)
- Judge evaluates each plan on completeness, minimality, consistency, feasibility
- Best plan selected via voting; structured JSON output (`ARCHITECTURE_JSON_SCHEMA`)

#### Multi-Planning (`enable_multi_plan=True`)
1. **MetaPlanner**: one LLM call identifies 2-4 sub-domains + `shared_interface` keys
2. **Scoped architects**: for each sub-domain, a full ArchitectAgent generates files aware of the shared interface and previously declared keys (sequential, not parallel)
3. **Merge**: deduplicate by `name.ext`, renumber order, produce a unified Architecture
4. Falls back to single architect if meta-planner returns < 2 sub-problems

### Phase 2: Generation (AlphaCodium + AgentCoder flow)

Each file in the DAG goes through:

```
[TestDesigner] → contract-first tests (blind to code)
      ↓
[Generator]   → code that targets those tests
      ↓
[Validator]   → AST syntax + contract checks
      ↓
[TestRunner]  → inline smoke check against designer tests
      ↓ (fail)
[ACIRepairer] → targeted hunk patch or full-file fallback
      ↓
[Reflector]   → verbal reflection → episodic memory
```

1. **TestDesigner** (AgentCoder, arXiv:2312.13010): generates 3-5 structural tests from the file's contract — *without seeing any code*.
2. **AlphaCodium flow** (arXiv:2401.08500): tests injected into the Generator prompt as implementation targets. Failures become concrete repair signal.
3. **ACI Repairer** (SWE-agent, arXiv:2405.15793): dual-mode repair. Line-specific errors get JSON patch hunks applied surgically bottom-up; falls back to full-file otherwise.
4. **Reflexion loop** (Shinn et al., 2023): verbal reflection → episodic memory → 91% HumanEval.

### Phase 3: Verification & Integration

1. **Contract validation** — every file's `shared_state_writes` populated and typed
2. **Cross-file consistency** — every `shared_state_reads` has a declared writer
3. **Sandbox execution** (optional, `--sandbox`) — runs the project in a Docker container (subprocess fallback when Docker unavailable):
   - Auto-detects project type: Python → `pip install` deps + `python main.py`; Node → `node index.js`; Shell → `bash`; Web → HTML syntax validation via stdlib
   - Server processes (`uvicorn.run`, `flask.run`) get an import-only check to avoid blocking
   - Captures stdout/stderr/exit code and duration
4. **Sandbox repair loop** — on non-zero exit, stderr is fed to the Repairer as an error signal; the fixed file is re-executed (up to `sandbox_max_repairs` iterations, default 2)
5. **Replan on failure** — returns to Phase 1 or 2 with error context

## Core Concepts

### Typed Shared State Contracts

```python
from matrioska.core.contracts import SharedStateSchema, FileContract, StateKeyType

routes_schema = SharedStateSchema(
    key="app_routes",
    type=StateKeyType.STR_LIST,
    description="API route paths",
    examples=[["/users", "/books", "/health"]],
)
contract = FileContract(
    file="main.py",
    reads=[],
    writes=[routes_schema],
)
```

### Multi-Model Role Assignment

| Role | Default Model | Why |
|------|--------------|-----|
| Architect | gpt-4o / claude-opus-4 | Strong reasoning for decomposition |
| TestDesigner | gpt-4o-mini / claude-haiku-4.5 | Cheap — structural tests from contract |
| Generator | gpt-4o-mini / claude-sonnet-4 | Balanced cost/quality |
| Validator | gpt-4o-mini / claude-haiku-4.5 | Cheap and fast |
| Judge | gpt-4o / claude-sonnet-4 | Analytical precision |
| Repairer | (same as Generator) | Focused on debugging |

MoE routing (`.py → claude-sonnet-4`, `.sql → gpt-4o`, etc.) applies to official OpenAI/Anthropic APIs. Third-party providers use the configured model directly. Override per role:
```bash
matrioska run --architect-model claude-opus-4 --generator-model claude-sonnet-4 \
    --validator-model claude-haiku-4.5 --task "..."

# Override MoE extension map (JSON)
export MATRIOSKA_MOE_EXTENSION_MAP='{"py": "deepseek-coder-v2", "ts": "gpt-4o"}'
```

### Hierarchical Memory

| Tier | Storage | Retrieval | Retention |
|------|---------|-----------|-----------|
| **Working** | shared_state + architecture + artifacts | Direct access | Current run |
| **Episodic** | knowledge/runs/*.md (Obsidian-compatible) | Keyword + ChromaDB | All runs (per project) |
| **Semantic** | knowledge/concepts/*.md + knowledge_graph.json | Embedding + k-hop graph | Cross-project |
| **Procedural** | knowledge/procedural_patterns.json + MATRIOSKA.md | DSPy-compiled few-shots | Permanent |
| **Global Vault** | `~/.matrioska/vault/` (Markdown + YAML + wikilinks) | local / global / linked / all | Permanent, cross-project |

### Global Obsidian Vault (LLM Wiki Pattern)

Inspired by [LightRAG (EMNLP 2025)](https://arxiv.org/abs/2409.14813) and Karpathy's LLM Wiki pattern, the vault "compiles" run knowledge into permanent Markdown.

```
~/.matrioska/vault/
├── projects/<name>/
│   ├── architecture.md   # decisions + per-run breakdown (timestamped)
│   ├── patterns.md       # detected patterns (heuristic, dedup'd)
│   ├── lessons.md        # repaired files, recurring fixes
│   └── links.md          # [[wikilinks]] to related projects
├── concepts/<tag>.md     # cross-project: sqlite, fastapi, argparse…
├── bugs/<slug>.md        # recurring bug → which projects hit it
└── INDEX.md              # auto-maintained, Obsidian Graph View-ready
```

After every run, `_compile_into_vault()` runs idempotent append-only compilation. Retrieval scopes (`local` / `global` / `linked` / `all`) feed the Architect's context.

The same vault is exposed read-only via the MCP server (`matrioska serve`): tools `vault_search`, `vault_get`, `vault_list`, `vault_doctor`, `vault_graph`, `vault_related`.

## Interactive REPL

```
$ matrioska
╭──────────────────────────────────────────────────────────────╮
│  Matrioska — interactive shell                                │
│  /help for commands · ! prefix for shell · Ctrl+D to exit     │
╰──────────────────────────────────────────────────────────────╯
  provider=openai  model=gpt-4o-mini  effort=medium  vault on

matrioska › /effort high
matrioska › Create a FastAPI CRUD API with SQLite
matrioska › /vault search "sqlite"
matrioska › /vault project my_api
matrioska › /rewind
matrioska › /usage
matrioska › !pytest -x -q
```

**Slash commands**: `/help`, `/config`, `/model`, `/usage`, `/clear`, `/plan`, `/effort`, `/memory`, `/init`, `/diff`, `/slots`, `/btw`, `/vault`, `/stream`, `/rewind`, `/history`, `/quit`, plus any `.matrioska/commands/*.md` as custom commands.

**Autocompletion** (requires `prompt_toolkit`): TAB/↓ opens a completion menu for all `/` commands; `/vault` and `/effort` show sub-command menus. Arrow keys navigate, Enter selects, Esc dismisses.

**Keyboard shortcuts**: Ctrl+D exit, Ctrl+C cancel task, Esc+Enter newline.

### Custom commands

Create `.matrioska/commands/deploy.md` with the task prompt as content. It becomes `/deploy` in the REPL session.

## Dashboard Controls (Live Run)

The Rich dashboard supports interactive control while the pipeline runs:

| Key | Action |
|-----|--------|
| `p` / Space | Pause / Resume |
| `q` (first press) | Show abort confirmation |
| `q` (second press within 3 s) | Abort pipeline |
| `m` *(paused)* | Edit model name inline |
| `e` *(paused)* | Cycle effort: low → medium → high |
| `r` *(paused)* | Cycle max\_repairs: 1 → 2 → 3 → 5 |
| `s` *(paused)* | Toggle stream\_tokens |

Config changes during pause take effect for the next phase.

## Hook System

Scripts in `.matrioska/hooks/` (project) or `~/.matrioska/hooks/` (global) are executed on pipeline events. Any executable file (`*.sh`, `*.py`, etc.) receives the event context as JSON via stdin.

```bash
# .matrioska/hooks/post_generate.sh
#!/usr/bin/env bash
EVENT=$(cat)
FILE=$(echo "$EVENT" | jq -r '.file')
echo "$(date): generated $FILE" >> generation_log.txt
```

Supported hooks: `pre_generate`, `post_generate`, `pre_repair`, `phase1_done`, `phase2_done`, `run_end`, `session_start`, `session_end`. Timeout: 10 s.

## DSPy Prompt Compilation

`matrioska compile --target architect --category cli --max-tasks 10` runs the [DSPy](https://arxiv.org/abs/2310.03714) compilation loop against the golden suite: splits tasks into train/val, runs a baseline, then `BootstrapFewShot` with `golden_metric` as the objective. Compiled few-shot demos persisted at `~/.matrioska/dspy_compiled/{target}_demos.json`.

## Evaluation

### Golden Regression Suite (30 tasks)

| Category | Count | Example |
|----------|-------|---------|
| CLI | 5 | Todo app with SQLite, git linter, markdown converter |
| Web | 5 | Landing page, image gallery, chat UI, dashboard |
| API | 5 | FastAPI CRUD, URL shortener, GraphQL feed |
| Data | 5 | CSV ETL, log parser, recommendation engine |
| Config | 5 | Docker Compose, GitHub Actions, Terraform |
| Full-stack | 5 | Blog engine, polling app, file sharing, dashboard |

### Metrics

| Metric | MVP Baseline | V3 Target |
|--------|-------------|-----------|
| `contract_fulfillment_rate` | ~60% | ≥95% |
| `first_pass_rate` | ~50% | ≥80% |
| `execution_success_rate` | 0% (sandbox was unimplemented) | ≥70% |
| `repair_effectiveness` | ~40% | ≥75% |

Save and compare regression baselines:
```bash
matrioska eval --save-baseline --baseline-file baselines.json
# next run: compare against saved
matrioska eval --baseline-file baselines.json
```

### Benchmark Parallelization

```python
from matrioska.eval.bench import run_benchmark
results = run_benchmark(tasks, orchestrators=["matrioska", "agentless"], max_workers=3)
```

Runs tasks independently via `ProcessPoolExecutor` — reduces 40-task benchmark from ~40 min to ~10 min at `max_workers=3`.

## Configuration Reference

All options: CLI flags, env vars (`MATRIOSKA_<UPPER>`), or `.env` file.

| Flag | Env | Default | Description |
|------|-----|---------|-------------|
| `--provider` | `MATRIOSKA_PROVIDER` | `openai` | `openai`, `anthropic`, `ollama`, `hf` |
| `--model` | `MATRIOSKA_MODEL` | `gpt-4o-mini` | Default model for all roles |
| `--architect-model` | `MATRIOSKA_ARCHITECT_MODEL` | (same as model) | Model for architecture phase |
| `--architect-candidates` | `MATRIOSKA_ARCHITECT_CANDIDATES` | `3` | N for Tree-of-Thoughts |
| `--max-repairs` | `MATRIOSKA_MAX_REPAIRS` | `2` | Max repair attempts per file |
| `--max-depth` | `MATRIOSKA_MAX_DEPTH` | `2` | Max nested sub-agent depth |
| `--parallel` | `MATRIOSKA_PARALLEL` | `true` | Parallel generation within DAG layers |
| `--sandbox` | `MATRIOSKA_ENABLE_SANDBOX` | `false` | Enable Docker sandbox execution |
| _(env only)_ | `MATRIOSKA_SANDBOX_MAX_REPAIRS` | `2` | Max Repairer iterations after sandbox failure |
| _(env only)_ | `MATRIOSKA_SANDBOX_TIMEOUT` | `30` | Max seconds per sandbox run |
| _(env only)_ | `MATRIOSKA_SANDBOX_IMAGE` | `python:3.11-slim` | Docker image for sandbox |
| `--no-reflexion` | `MATRIOSKA_ENABLE_REFLEXION` | `true` | Disable Reflexion loop |
| `--no-tot` | `MATRIOSKA_ENABLE_TOT` | `true` | Disable Tree-of-Thoughts voting |
| `--retrieve-k` | `MATRIOSKA_RETRIEVE_K` | `3` | Past runs to retrieve as context |
| _(env only)_ | `MATRIOSKA_ENABLE_TEST_DESIGN` | `true` | AlphaCodium+AgentCoder test enrichment |
| _(env only)_ | `MATRIOSKA_USE_ACI_REPAIR` | `true` | SWE-agent ACI targeted patch repair |
| _(env only)_ | `MATRIOSKA_ENABLE_MULTI_PLAN` | `false` | Hierarchical meta-decomposition |
| _(env only)_ | `MATRIOSKA_MOE_EXTENSION_MAP` | `""` | JSON override for extension→model routing |
| `--quick` | `MATRIOSKA_QUICK` | `false` | Skip ToT, Reflexion, TestDesign, ACI, Phase 3 |
| `--mode` | `MATRIOSKA_PERMISSION_MODE` | `auto` | `auto` \| `plan` \| `ask` |
| `--no-vault` | `MATRIOSKA_ENABLE_VAULT` | `true` | Disable global Obsidian vault writes/reads |
| `--vault-dir` | `MATRIOSKA_VAULT_DIR` | `~/.matrioska/vault` | Override vault location |
| `--skip-validation` | `MATRIOSKA_SKIP_VALIDATION` | `false` | Skip startup connectivity check |
| _(env only)_ | `MATRIOSKA_STREAM_TOKENS` | `true` | SSE token streaming (or `/stream` in REPL) |
| _(env only)_ | `MATRIOSKA_SERVE_PORT` | `9020` | MCP server port |
| _(env only)_ | `MATRIOSKA_COST_PER_PROMPT_TOKEN` | `""` | Override prompt pricing ($/token) |
| _(env only)_ | `MATRIOSKA_COST_PER_COMPLETION_TOKEN` | `""` | Override completion pricing ($/token) |

### Provider Pricing (built-in)

Matrioska tracks estimated cost per run using a built-in pricing table (18 models across OpenAI, Anthropic, Groq, DeepSeek, Mistral, Together, NVIDIA). Override via env vars or add custom pricing via `MATRIOSKA_COST_PER_PROMPT_TOKEN`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `400 Bad Request` on startup | Model deprecated / renamed, or account restricted | Check MATRIOSKA\_MODEL; if `organization_restricted` error: contact provider support |
| `401 Unauthorized` | Invalid API key | Check `MATRIOSKA_API_KEY` in `.env` |
| `404 Not Found` | Wrong model name | Check `MATRIOSKA_MODEL`; run `matrioska init` to pick from known models |
| All slots on cooldown | Hit rate limits on all keys | Add more keys via `MATRIOSKA_API_KEYS=key1,key2,...` |
| ChromaDB ImportError | Optional dep | `pip install chromadb`; first run downloads ~79 MB (progress bar shown) |
| `mcp` not found | Optional dep | `pip install mcp` or omit `serve` subcommand |
| Dashboard not rendering | Non-TTY / CI | Add `--no-dashboard` flag |
| Port in use | MCP server | `--port <other>` or set `MATRIOSKA_SERVE_PORT` |
| Sandbox skipped silently | Docker not running | Start Docker or set `MATRIOSKA_ENABLE_SANDBOX=false`; subprocess fallback activates automatically |
| Sandbox timeout on server app | App blocks on `listen()` | Matrioska detects `uvicorn.run`/`flask.run` and runs import-only check instead |
| `<tool_call>finish(...)` in generated files | Model without native tool-use | Fixed automatically by `sanitize_output()` — re-run the task |

## Theoretical Foundations

- **MetaGPT** (Hong et al., 2023): SOPs as prompts → typed shared_state contracts
- **Tree of Thoughts** (Yao et al., NeurIPS 2023): N candidates → Judge → voting
- **Reflexion** (Shinn et al., 2023): Verbal reflection → episodic memory → 91% HumanEval
- **Process Supervision** (Lightman et al., 2023): Validate intermediate steps, not just output
- **CRITIC** (Gou et al., 2023): Tool-augmented self-correction
- **CodePlan** (Bairi et al., 2023): Repository-level editing as dependency analysis
- **DSPy** (Khattab et al., 2023): Prompts as compilable parameters
- **AgentCoder** (Huang et al., 2024, arXiv:2312.13010): Blind test designer eliminates test-the-bug bias; 96.3% HumanEval
- **AlphaCodium** (Ridnik et al., 2024, arXiv:2401.08500): Flow engineering with test enrichment; GPT-4 19%→44% CodeContests
- **SWE-agent** (Yang et al., NeurIPS 2024, arXiv:2405.15793): Agent-Computer Interface with targeted edits
- **Agentless** (Xia et al., 2024, arXiv:2407.01489): Deterministic localize→repair→validate as safety net
- **AutoCodeRover** (Zhang et al., ISSTA 2024, arXiv:2404.05427): AST-level context for cross-file repairs
- **LightRAG** (Edge et al., EMNLP 2025, arXiv:2409.14813): Dual-level graph-enhanced RAG
- **LATS** (Zhou et al., ICML 2024, arXiv:2310.04406): MCTS with value function over agent trajectories

## License

MIT

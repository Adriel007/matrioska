# Matrioska V3

**Contract-first, state-graph multi-agent LLM orchestrator for code generation.**

Matrioska decomposes complex coding tasks into a DAG of files that coordinate through a typed `shared_state` whiteboard. Three pipeline phases — Architecture (with Tree-of-Thoughts voting), Generation (DAG-layered parallel with AlphaCodium test enrichment + Reflexion + ACI Repair), and Verification (contract validation + sandbox execution) — produce complete, validated projects.

## Architecture

```
Task → [Phase 1: N× Architect → Judge → Best Plan]
    → [Phase 2: DAG layers → TestDesign ∥ Generate ∥ Validate ∥ ACIRepair ∥ Reflect]
    → [Phase 3: Contract check → Cross-file → Sandbox → Replan?]
    → Output + Episodic Note
```

### Modular Monolith with Event-Driven Core

```
src/matrioska/
├── core/         State graph, typed contracts, events, config
├── llm/          Multi-provider client, circuit breaker, MoE router, slot pool
├── memory/       Episodic → Semantic → Procedural + Obsidian Vault (global)
├── agents/       Architect, Generator, TestDesigner, Validator, Judge, Repairer, Reflector
├── pipeline/     3-phase orchestration with checkpointing, preflight, executor
├── tools/        Sandbox executor, tool dispatcher
├── eval/         Metrics, golden regression suite (30 tasks)
├── cli/          Rich CLI + dashboard + init wizard
│                 (run, resume, show, clean, serve, eval, init, btw, vault)
└── api.py        Python API + MCP server
```

## Key Design Decisions

| Dimension | Choice | Rationale |
|-----------|--------|-----------|
| **Coordination** | Typed shared_state contracts | Prevents chain hallucination (MetaGPT insight) |
| **DAG** | Kahn topological layers | Enables intra-layer parallelism |
| **Architecture** | Tree-of-Thoughts (N candidates → Judge voting) | 70pp improvement on reasoning tasks (Yao et al.) |
| **Test Design** | Contract-first TestDesigner (blind to code) | Eliminates "test-the-bug-you-wrote" (AgentCoder, arXiv:2312.13010) |
| **Generation** | AlphaCodium flow: tests → generate → smoke-check | GPT-4: 19%→44% pass@5 on CodeContests (arXiv:2401.08500) |
| **Validation** | Process supervision (syntax + contracts per-file) | Outperforms outcome-only (Lightman et al.) |
| **Repair** | ACI targeted patch (hunks) + full-file fallback | Preserves cross-file invariants (SWE-agent, arXiv:2405.15793) |
| **Reflection** | Verbal reflection → episodic memory | 91% HumanEval (Shinn et al.) |
| **Memory** | Episodic → Semantic → Procedural | Multi-timescale retrieval |
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

# Permission modes
matrioska run --task "..." --mode plan         # plan only (return after Phase 1)
matrioska run --task "..." --mode ask          # confirm each file before generation
matrioska run --task "..." --interactive       # review and approve plan before Phase 2

# One-shot LLM query without touching the pipeline or memory
matrioska btw "what is RRF retrieval fusion?"

# Resume / inspect
matrioska resume
matrioska show

# Interactive REPL (no subcommand) — slash commands, ! shell, streaming
matrioska
# matrioska › /help
# matrioska › /effort high
# matrioska › Create a CLI todo app with SQLite
# matrioska › /vault search sqlite
# matrioska › !ls

# Global knowledge vault (Obsidian-compatible)
matrioska vault list
matrioska vault search "sqlite patterns" --scope global
matrioska vault doctor
matrioska vault graph -o vault-graph.md

# DSPy-driven prompt compilation against the golden suite
matrioska compile --target architect --category cli --max-tasks 5

# Run golden evaluation suite
matrioska eval --category cli
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
```

### Python API

```python
from matrioska import Matrioska, Config, load_config

cfg = load_config({"provider": "openai", "model": "gpt-4o-mini"})
m = Matrioska(cfg)
result = m.run("Create a CLI todo app with SQLite")

print(result["status"])       # success | partial | failed
print(result["project_name"]) # snake_case_project_name
for a in result["artifacts"]:
    print(f"  {a.name}.{a.extension} [{a.status}] {len(a.content)} chars")
```

## Pipeline Phases

### Phase 1: Architecture (with Tree-of-Thoughts)
- N parallel Architect calls (default N=3, temperature=0.7 for diversity)
- Judge evaluates each plan on completeness, minimality, consistency, feasibility
- Best plan selected via voting; structured JSON output (`ARCHITECTURE_JSON_SCHEMA`)

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

1. **TestDesigner** (AgentCoder, arXiv:2312.13010): generates 3-5 structural/interface tests from the file's contract — *without seeing any code*. Eliminates the bias of testing your own bugs.
2. **AlphaCodium flow** (arXiv:2401.08500): tests injected into the Generator prompt as implementation targets. After syntax validation passes, tests run inline as a smoke check. Failures become concrete repair signal.
3. **ACI Repairer** (SWE-agent, arXiv:2405.15793): dual-mode repair. When errors reference specific line numbers, asks the model for targeted JSON patch hunks (`start_line`, `end_line`, `new_content`) applied surgically bottom-up. Falls back to full-file repair when hunks fail to apply. Preserves cross-file invariants instead of rewriting from scratch.
4. **Reflexion loop** (Shinn et al., 2023): Reflector reviews generated artifacts and feeds verbal insights forward into subsequent generations.
5. Complex files spawn recursive nested Matrioska sub-agents.

### Phase 3: Verification & Integration
- Contract validation: every file's `shared_state_writes` populated and typed
- Cross-file consistency: every `shared_state_reads` has a declared writer
- Optional Docker sandbox execution with captured stdout/stderr
- On failure: replan (returns to Phase 1 or 2 with error context)

> **Note:** During Phase 2, contract violations are non-blocking (warnings only).
> The authoritative contract check runs in Phase 3. This prevents generators
> from stalling on missing `shared_state_updates` — keys declared in
> `shared_state_writes` are auto-populated with placeholders so downstream
> files can reference them.

## Core Concepts

### Typed Shared State Contracts

```python
from matrioska.core.contracts import SharedStateSchema, FileContract, StateKeyType

# Define what a key looks like
routes_schema = SharedStateSchema(
    key="app_routes",
    type=StateKeyType.STR_LIST,
    description="API route paths",
    examples=[["/users", "/books", "/health"]],
)

# File declares its reads and writes
contract = FileContract(
    file="main.py",
    reads=[],            # This file is the entry point
    writes=[routes_schema],  # It produces the routes list
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

MoE routing (`.py → claude-sonnet-4`, `.sql → gpt-4o`, etc.) applies to official OpenAI/Anthropic APIs. Third-party providers (Groq, NVIDIA, Ollama, OpenRouter) use the configured model directly — set `MATRIOSKA_<ROLE>_MODEL` explicitly.

Override per role:
```bash
matrioska run --architect-model claude-opus-4 --generator-model claude-sonnet-4 \
    --validator-model claude-haiku-4.5 --task "..."
```

### Hierarchical Memory

| Tier | Storage | Retrieval | Retention |
|------|---------|-----------|-----------|
| **Working** | shared_state + architecture + artifacts | Direct access | Current run |
| **Episodic** | knowledge/runs/*.md (Obsidian-compatible) | Keyword + ChromaDB embeddings | All runs (per project) |
| **Semantic** | knowledge/concepts/*.md + knowledge_graph.json | Embedding + k-hop graph traversal | Cross-project |
| **Procedural** | knowledge/procedural_patterns.json + MATRIOSKA.md | DSPy-compiled few-shots | Permanent |
| **Global Vault** | `~/.matrioska/vault/` (Markdown + YAML + wikilinks) | Dual-level: local / global / linked / all | Permanent, cross-project |

### Global Obsidian Vault (LLM Wiki Pattern)

Inspired by Karpathy's [LLM Wiki](https://x.com/karpathy/) and [LightRAG (EMNLP 2025)](https://arxiv.org/abs/2409.14813),
the vault "compiles" run knowledge into permanent Markdown instead of re-deriving with RAG every call.

```
~/.matrioska/vault/
├── projects/<name>/
│   ├── architecture.md   # decisions + per-run breakdown (timestamped)
│   ├── patterns.md       # detected patterns (heuristic, dedup'd)
│   ├── lessons.md        # repaired files, recurring fixes
│   └── links.md          # [[wikilinks]] to related projects
├── concepts/<tag>.md     # cross-project: e.g. sqlite, fastapi, argparse
├── bugs/<slug>.md        # recurring bug → which projects hit it
└── INDEX.md              # auto-maintained, Obsidian Graph View-ready
```

After every run, `orchestrator._compile_into_vault()` runs an **idempotent**, **append-only** compiler:
`derive_tags(task, artifacts)` → `extract_lessons_and_bugs()` → `compile_from_run()`. Marker-based section
upserts (`<!-- section:ID -->`) ensure rerunning the same task never duplicates concept bullets.

Retrieval scopes (LightRAG dual-level):
- `local`  — current project (`./matrioska_work/knowledge/`)
- `global` — concepts + bugs across all projects
- `linked` — BFS over `[[wikilinks]]` in `links.md`, `max_hops=2`
- `all`    — everything

Before Phase 1, the Architect receives a `RELEVANT VAULT KNOWLEDGE` block (scope=global, k=5) so
patterns observed in past projects influence new decompositions.

The same vault is exposed via the MCP server (`matrioska serve`): tools `vault_search`,
`vault_get`, `vault_list`, `vault_doctor`, `vault_graph`, `vault_related` let Claude Code,
Cursor, or Windsurf reuse Matrioska's accumulated knowledge directly. Writes are not
exposed — the vault is only mutated by `_compile_into_vault` after a real run, so it
remains a deterministic derivative of the run history.

## Interactive REPL

```
$ matrioska
╭──────────────────────────────────────────────────────────────╮
│  Matrioska — interactive shell                                │
│  /help for commands · ! prefix for shell · Ctrl+D to exit     │
╰──────────────────────────────────────────────────────────────╯
  provider=groq  model=llama-3.3-70b  effort=medium  vault on

matrioska › /help
matrioska › /effort high            # 5 candidates, ToT + Reflexion + TestDesign
matrioska › /stream                 # token-by-token via EventBus
matrioska › Create a FastAPI CRUD API with SQLite
matrioska › /vault search "sqlite"  # query the global vault inline
matrioska › !ls matrioska_work       # shell prefix
matrioska › /usage                  # tokens / cost this session
```

Slash commands: `/help`, `/config`, `/model`, `/usage`, `/clear`, `/plan`, `/effort`,
`/memory`, `/init`, `/diff`, `/slots`, `/btw`, `/vault`, `/stream`, `/history`,
`/quit`. Built on prompt_toolkit when installed (history at `~/.matrioska/history`,
multi-line via Esc+Enter, ↑/↓ for previous commands); falls back to `input()` otherwise.

## DSPy Prompt Compilation

`matrioska compile --target architect --category cli --max-tasks 10` runs the
[DSPy](https://arxiv.org/abs/2310.03714) compilation loop against the golden suite:
splits tasks into train/val, runs a baseline, then `BootstrapFewShot` with
`golden_metric` (binary `evaluate_result.pass`) as the objective. Compiled few-shot
demos persisted at `~/.matrioska/dspy_compiled/{target}_demos.json`. When `dspy-ai`
isn't installed, returns the baseline with `skipped_reason='dspy_not_installed'`
so the same harness works either way.

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
| `execution_success_rate` | 0% | ≥70% |
| `repair_effectiveness` | ~40% | ≥75% |

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
| `--no-reflexion` | `MATRIOSKA_ENABLE_REFLEXION` | `true` | Disable Reflexion loop |
| `--no-tot` | `MATRIOSKA_ENABLE_TOT` | `true` | Disable Tree-of-Thoughts voting |
| `--retrieve-k` | `MATRIOSKA_RETRIEVE_K` | `3` | Past runs to retrieve as context |
| _(env only)_ | `MATRIOSKA_ENABLE_TEST_DESIGN` | `true` | AlphaCodium+AgentCoder test enrichment |
| _(env only)_ | `MATRIOSKA_USE_ACI_REPAIR` | `true` | SWE-agent ACI targeted patch repair |
| `--quick` | `MATRIOSKA_QUICK` | `false` | Skip ToT, Reflexion, TestDesign, ACI, Phase 3 |
| `--mode` | `MATRIOSKA_PERMISSION_MODE` | `auto` | `auto` \| `plan` \| `ask` |
| `--no-vault` | `MATRIOSKA_ENABLE_VAULT` | `true` | Disable global Obsidian vault writes/reads |
| `--vault-dir` | `MATRIOSKA_VAULT_DIR` | `~/.matrioska/vault` | Override vault location |
| _(env only)_ | `MATRIOSKA_STREAM_TOKENS` | `false` | SSE token streaming (or `/stream` in REPL) |

See `.env.example` for the full reference.

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
- **LATS** (Zhou et al., ICML 2024, arXiv:2310.04406): MCTS with value function over agent trajectories

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `400 Bad Request` on startup connectivity check | Model deprecated / renamed on provider | Matrioska warns and proceeds — update `MATRIOSKA_MODEL` if generation also fails |
| `401 Unauthorized` | Invalid API key | Check `MATRIOSKA_API_KEY` in `.env` |
| `404 Not Found` | Wrong model name for this provider | Check `MATRIOSKA_MODEL`; run `matrioska init` to pick from known models |
| All slots on cooldown | Hit rate limits on all keys | Add more API keys via `MATRIOSKA_API_KEYS=key1,key2,...` |
| ChromaDB ImportError | Optional dep not installed | `pip install chromadb` or disable embeddings (keyword-only fallback activates automatically) |
| `mcp` not found | Optional dep | `pip install mcp` or omit `serve` subcommand |
| Dashboard flicker / CI failure | Rich not compatible with non-TTY | Add `--no-dashboard` flag |

## License

MIT

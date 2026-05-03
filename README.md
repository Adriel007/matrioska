# Matrioska V3

**Contract-first, state-graph multi-agent LLM orchestrator for code generation.**

Matrioska decomposes complex coding tasks into a DAG of files that coordinate through a typed `shared_state` whiteboard. Three pipeline phases — Architecture (with Tree-of-Thoughts voting), Generation (DAG-layered parallel with Reflexion + Repair), and Verification (contract validation + sandbox execution) — produce complete, validated projects.

## Architecture

```
Task → [Phase 1: N× Architect → Judge → Best Plan]
    → [Phase 2: DAG layers → Generate ∥ Validate ∥ Repair ∥ Reflect]
    → [Phase 3: Contract check → Cross-file → Sandbox → Replan?]
    → Output + Episodic Note
```

### Modular Monolith with Event-Driven Core

```
src/matrioska/
├── core/         State graph, typed contracts, events, config
├── llm/          Multi-provider client, circuit breaker, MoE router
├── memory/       Episodic → Semantic → Procedural (3-tier)
├── agents/       Architect, Generator, Validator, Judge, Repairer, Reflector
├── pipeline/     3-phase orchestration with checkpointing
├── tools/        Sandbox executor, tool dispatcher
├── eval/         Metrics, golden regression suite (30 tasks)
├── cli/          Rich CLI (run, resume, show, clean, serve, eval)
└── api.py        Python API + MCP server
```

## Key Design Decisions

| Dimension | Choice | Rationale |
|-----------|--------|-----------|
| **Coordination** | Typed shared_state contracts | Prevents chain hallucination (MetaGPT insight) |
| **DAG** | Kahn topological layers | Enables intra-layer parallelism |
| **Architecture** | Tree-of-Thoughts (N candidates → Judge voting) | 70pp improvement on reasoning tasks (Yao et al.) |
| **Validation** | Process supervision (syntax + contracts per-file) | Outperforms outcome-only (Lightman et al.) |
| **Repair** | Real error feedback → Repairer agent | CRITIC-inspired (Gou et al.) |
| **Reflection** | Verbal reflection → episodic memory | 91% HumanEval (Shinn et al.) |
| **Memory** | Episodic → Semantic → Procedural | Multi-timescale retrieval |
| **Models** | Role-specific (Architect≠Generator≠Validator) | Right capability/cost per role |
| **State** | Checkpointed state graph | Resume, branching, time-travel debug |
| **Execution** | Docker sandbox (optional) | Ground truth feedback (AutoDev-inspired) |
| **DSPy** | Prompt compilation scaffold | Smaller models via optimized few-shots |

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Set up your provider (create a .env from .env.example)
export MATRIOSKA_PROVIDER=openai
export MATRIOSKA_BASE_URL=https://api.openai.com/v1
export MATRIOSKA_API_KEY=sk-...
export MATRIOSKA_MODEL=gpt-4o-mini
export MATRIOSKA_ARCHITECT_MODEL=gpt-4o

# Run a task
matrioska run --task "Create a FastAPI CRUD API for books with SQLite"

# Plan only (architecture without code generation)
matrioska run --task "Build a real-time chat app" --plan-only

# Resume interrupted run
matrioska resume

# Show current state
matrioska show

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

### Phase 2: Generation (DAG-layered parallel)
- Files partitioned into topological layers via Kahn's algorithm
- Files within a layer generated in parallel (ThreadPoolExecutor)
- Each file: Generate → Validate (AST/syntax) → (fail) Repair → (fail) Mark failed
- Optional Reflexion loop: Reflector reviews output, feeds insights forward
- Optional Multi-Agent Debate for complex files (2 generators → Judge picks best)
- Complex files spawn recursive nested Matrioska sub-agents

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
| Generator | gpt-4o-mini / claude-sonnet-4 | Balanced cost/quality |
| Validator | gpt-4o-mini / claude-haiku-4.5 | Cheap and fast |
| Judge | gpt-4o / claude-sonnet-4 | Analytical precision |
| Repairer | (same as Generator) | Focused on debugging |

MoE routing (`.py → claude-sonnet-4`, `.sql → gpt-4o`, etc.) only applies to
official OpenAI/Anthropic APIs. Third-party providers (Groq, NVIDIA, Ollama)
use the configured model directly — set `MATRIOSKA_<ROLE>_MODEL` explicitly.

Override per role:
```bash
matrioska run --architect-model claude-opus-4 --generator-model claude-sonnet-4 \
    --validator-model claude-haiku-4.5 --task "..."
```

### Hierarchical Memory

| Tier | Storage | Retrieval | Retention |
|------|---------|-----------|-----------|
| **Working** | shared_state + architecture + artifacts | Direct access | Current run |
| **Episodic** | knowledge/runs/*.md (Obsidian-compatible) | Keyword + ChromaDB embeddings | All runs |
| **Semantic** | knowledge/concepts/*.md + knowledge_graph.json | Embedding + k-hop graph traversal | Cross-project |
| **Procedural** | knowledge/procedural_patterns.json + MATRIOSKA.md | DSPy-compiled few-shots | Permanent |

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
| `--reflexion` | `MATRIOSKA_ENABLE_REFLEXION` | `true` | Enable Reflexion loop |
| `--retrieve-k` | `MATRIOSKA_RETRIEVE_K` | `3` | Past runs to retrieve as context |

See `.env.example` for the full reference.

## Theoretical Foundations

- **MetaGPT** (Hong et al., 2023): SOPs as prompts → our typed shared_state contracts
- **Tree of Thoughts** (Yao et al., NeurIPS 2023): N candidates → Judge → voting
- **Reflexion** (Shinn et al., 2023): Verbal reflection → episodic memory → 91% HumanEval
- **Process Supervision** (Lightman et al., 2023): Validate intermediate steps, not just output
- **CRITIC** (Gou et al., 2023): Tool-augmented self-correction
- **CodePlan** (Bairi et al., 2023): Repository-level editing as dependency analysis
- **DSPy** (Khattab et al., 2023): Prompts as compilable parameters
- **SWE-Agent** (Yang et al., NeurIPS 2024): Agent-Computer Interface
- **More Agents Is All You Need** (Li et al., TMLR 2024): Sampling + voting scales with difficulty

## License

MIT

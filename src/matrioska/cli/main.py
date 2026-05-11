"""
Matrioska V3 CLI — Rich-based terminal interface.

Subcommands:
  run     Execute a task end-to-end
  resume  Resume from the latest checkpoint
  show    Display current state
  clean   Remove work directory
  serve   Start MCP server for Claude Code integration
  eval    Run golden regression suite
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matrioska",
        description="Matrioska V3 — Contract-first multi-agent LLM orchestrator",
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--provider", choices=["openai", "ollama", "anthropic", "hf"])
        sp.add_argument("--base-url", dest="base_url")
        sp.add_argument("--api-key", dest="api_key")
        sp.add_argument("--model")
        sp.add_argument("--architect-model", dest="architect_model")
        sp.add_argument("--generator-model", dest="generator_model")
        sp.add_argument("--validator-model", dest="validator_model")
        sp.add_argument("--judge-model", dest="judge_model")
        sp.add_argument(
            "--work-dir", dest="work_dir", type=lambda s: Path(s).expanduser()
        )
        sp.add_argument("--max-tokens", dest="max_tokens", type=int)
        sp.add_argument("--max-repairs", dest="max_repairs", type=int)
        sp.add_argument("--max-depth", dest="max_depth", type=int)
        sp.add_argument("--temperature", type=float)
        sp.add_argument("--parallel", action="store_true", default=None)
        sp.add_argument("--no-parallel", dest="parallel", action="store_false")
        sp.add_argument("--retrieve-k", dest="retrieve_k", type=int)
        sp.add_argument("--architect-candidates", dest="architect_candidates", type=int)
        sp.add_argument(
            "--no-tot", dest="enable_tot", action="store_false", default=None
        )
        sp.add_argument(
            "--reflexion/--no-reflexion", dest="enable_reflexion", default=None
        )
        sp.add_argument(
            "--sandbox", dest="enable_sandbox", action="store_true", default=None
        )
        sp.add_argument(
            "--log-level",
            dest="log_level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
        sp.add_argument("--dry-run", dest="dry_run", action="store_true", default=None)
        sp.add_argument(
            "--quick", dest="quick", action="store_true", default=None,
            help="Skip ToT, Reflexion, TestDesign, ACI and Phase 3 (rapid iteration mode)",
        )
        sp.add_argument(
            "--mode", dest="permission_mode",
            choices=["auto", "plan", "ask"],
            help="auto=run, plan=show plan only, ask=confirm before each file",
        )
        sp.add_argument(
            "--no-vault", dest="enable_vault", action="store_false", default=None,
            help="Disable global Obsidian vault (~/.matrioska/vault)",
        )
        sp.add_argument("--vault-dir", dest="vault_dir",
                        help="Override global vault location (default: ~/.matrioska/vault)")

    # run
    run_p = sub.add_parser("run", help="Execute a task end-to-end")
    _common(run_p)
    run_p.add_argument("--task")
    run_p.add_argument("--task-file", type=Path)
    run_p.add_argument(
        "--plan-only", dest="plan_only", action="store_true", default=None
    )
    run_p.add_argument("--interactive", action="store_true", default=None)
    run_p.add_argument("--no-dashboard", dest="no_dashboard", action="store_true",
                       help="Disable live dashboard, print plain logs instead")

    # resume
    res_p = sub.add_parser("resume", help="Resume from latest checkpoint")
    _common(res_p)

    # show
    show_p = sub.add_parser("show", help="Display current state")
    show_p.add_argument(
        "--work-dir", dest="work_dir", type=lambda s: Path(s).expanduser()
    )

    # clean
    clean_p = sub.add_parser("clean", help="Remove work directory")
    clean_p.add_argument(
        "--work-dir", dest="work_dir", type=lambda s: Path(s).expanduser()
    )
    clean_p.add_argument("--yes", action="store_true")

    # serve
    serve_p = sub.add_parser(
        "serve", help="Start MCP server for Claude Code integration"
    )
    _common(serve_p)
    serve_p.add_argument("--port", type=int, default=9020)

    # eval
    eval_p = sub.add_parser("eval", help="Run golden regression suite")
    _common(eval_p)
    eval_p.add_argument("--category")
    eval_p.add_argument("--task-id", dest="task_id")

    # compile (DSPy)
    comp_p = sub.add_parser("compile", help="DSPy-driven prompt compilation against golden suite")
    _common(comp_p)
    comp_p.add_argument("--target", default="architect", help="Agent to optimize")
    comp_p.add_argument("--category", help="Only use golden tasks in this category")
    comp_p.add_argument("--max-tasks", dest="max_tasks", type=int, default=10)
    comp_p.add_argument("--val-fraction", dest="val_fraction", type=float, default=0.3)
    comp_p.add_argument("--out-dir", dest="out_dir", type=lambda s: Path(s).expanduser())

    # init
    init_p = sub.add_parser("init", help="Interactive .env + MATRIOSKA.md scaffolder")
    init_p.add_argument("--dir", dest="target_dir", type=lambda s: Path(s).expanduser(),
                        help="Target directory (default: cwd)")

    # btw — one-shot LLM query without polluting any session/context
    btw_p = sub.add_parser("btw", help="One-shot LLM query (no memory, no pipeline)")
    _common(btw_p)
    btw_p.add_argument("question", nargs="+", help="The question to ask")

    # vault — global Obsidian-compatible knowledge base ops
    vault_p = sub.add_parser("vault", help="Global Obsidian vault operations")
    vault_sub = vault_p.add_subparsers(dest="vault_cmd", required=True)
    vault_search = vault_sub.add_parser("search", help="Search the vault")
    vault_search.add_argument("query", nargs="+")
    vault_search.add_argument("--scope", choices=["local", "global", "linked", "all"], default="all")
    vault_search.add_argument("--project")
    vault_search.add_argument("--k", type=int, default=8)
    vault_search.add_argument("--vault-dir", dest="vault_dir")
    vault_search.add_argument("--work-dir", dest="work_dir",
                              type=lambda s: Path(s).expanduser())
    vault_doctor = vault_sub.add_parser("doctor", help="Check vault health (orphan notes, staleness)")
    vault_doctor.add_argument("--vault-dir", dest="vault_dir")
    vault_graph = vault_sub.add_parser("graph", help="Export vault wikilink graph as Mermaid")
    vault_graph.add_argument("--vault-dir", dest="vault_dir")
    vault_graph.add_argument("--output", "-o", type=Path)
    vault_list = vault_sub.add_parser("list", help="List projects in the vault")
    vault_list.add_argument("--vault-dir", dest="vault_dir")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.cmd is None:
        from matrioska.cli.repl import run_repl
        return run_repl()

    if ns.cmd == "run":
        return _cmd_run(ns)
    if ns.cmd == "resume":
        return _cmd_resume(ns)
    if ns.cmd == "show":
        return _cmd_show(ns)
    if ns.cmd == "clean":
        return _cmd_clean(ns)
    if ns.cmd == "serve":
        return _cmd_serve(ns)
    if ns.cmd == "eval":
        return _cmd_eval(ns)
    if ns.cmd == "compile":
        return _cmd_compile(ns)
    if ns.cmd == "init":
        return _cmd_init(ns)
    if ns.cmd == "btw":
        return _cmd_btw(ns)
    if ns.cmd == "vault":
        return _cmd_vault(ns)

    parser.print_help()
    return 2


def _build_cli_overrides(ns: argparse.Namespace) -> Dict[str, Any]:
    """Extract non-None CLI values for load_config."""
    overrides: Dict[str, Any] = {}
    for key in (
        "provider",
        "base_url",
        "api_key",
        "model",
        "architect_model",
        "generator_model",
        "validator_model",
        "judge_model",
        "max_tokens",
        "max_repairs",
        "max_depth",
        "temperature",
        "retrieve_k",
        "architect_candidates",
        "log_level",
        "parallel",
        "plan_only",
        "dry_run",
        "interactive",
        "enable_tot",
        "enable_reflexion",
        "enable_sandbox",
        "quick",
        "permission_mode",
        "enable_vault",
        "vault_dir",
    ):
        val = getattr(ns, key, None)
        if val is not None:
            overrides[key] = val

    work_dir = getattr(ns, "work_dir", None)
    if work_dir is not None:
        overrides["work_dir"] = work_dir

    return overrides


def _cmd_run(ns: argparse.Namespace) -> int:
    from matrioska.core.config import load_config, validate_config

    overrides = _build_cli_overrides(ns)
    cfg = load_config(overrides)
    if not cfg.dry_run:
        validate_config(cfg)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    task = getattr(ns, "task", None)
    if not task and getattr(ns, "task_file", None):
        task = Path(ns.task_file).read_text(encoding="utf-8").strip()
    if not task:
        print("ERROR: --task or --task-file required", file=sys.stderr)
        return 2

    no_dash = getattr(ns, "no_dashboard", False)

    # Interactive flows (ask mode, plan review) can't share stdin with the
    # Live dashboard — disable it automatically.
    if cfg.permission_mode == "ask" or (cfg.interactive and not cfg.plan_only):
        no_dash = True

    if cfg.dry_run or no_dash:
        from matrioska.pipeline.orchestrator import Matrioska
        m = Matrioska(cfg)
        result = m.run(task)
    else:
        import logging
        logging.disable(logging.WARNING)  # silence logs so they don't bleed into dashboard
        try:
            from matrioska.cli.dashboard import run_with_dashboard
            result = run_with_dashboard(cfg, task)
        finally:
            logging.disable(logging.NOTSET)

    return (
        0
        if result.get("status") in ("success", "plan_only", "partial", "dry_run")
        else 1
    )


def _cmd_resume(ns: argparse.Namespace) -> int:
    from matrioska.core.config import load_config, validate_config
    from matrioska.pipeline.orchestrator import Matrioska

    overrides = _build_cli_overrides(ns)
    cfg = load_config(overrides)
    validate_config(cfg)
    m = Matrioska(cfg)
    result = m.resume()
    return 0 if result.get("status") in ("success", "partial") else 1


def _cmd_show(ns: argparse.Namespace) -> int:
    from matrioska.pipeline.orchestrator import Matrioska
    from matrioska.core.config import Config

    work_dir = getattr(ns, "work_dir", None) or Path("./matrioska_work")
    cfg = Config(work_dir=work_dir)
    m = Matrioska(cfg)
    info = m.show()
    _rich_show(info)
    return 0


def _cmd_clean(ns: argparse.Namespace) -> int:
    import shutil

    work_dir = getattr(ns, "work_dir", None) or Path("./matrioska_work")
    work_dir = Path(work_dir) if isinstance(work_dir, str) else work_dir

    if not work_dir.exists():
        print(f"Nothing to clean: {work_dir}")
        return 0
    if not ns.yes:
        print(f"Refusing to delete {work_dir} without --yes", file=sys.stderr)
        return 2
    shutil.rmtree(work_dir)
    print(f"Removed {work_dir}")
    return 0


def _cmd_serve(ns: argparse.Namespace) -> int:
    """Start the Matrioska MCP server."""
    try:
        from matrioska.api import create_mcp_server
        import asyncio

        print(f"Matrioska MCP server starting on port {ns.port}...")
        asyncio.run(create_mcp_server(port=ns.port))
    except ImportError:
        print("MCP not available. Install with: pip install mcp", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nServer stopped.")
    return 0


def _cmd_eval(ns: argparse.Namespace) -> int:
    """Run golden regression suite."""
    from matrioska.eval.golden_suite import get_golden_tasks, evaluate_result
    from matrioska.core.config import load_config, validate_config
    from matrioska.pipeline.orchestrator import Matrioska

    overrides = _build_cli_overrides(ns)
    cfg = load_config(overrides)
    validate_config(cfg)

    category = getattr(ns, "category", None)
    task_id = getattr(ns, "task_id", None)

    if task_id:
        tasks = [t for t in get_golden_tasks() if t.id == task_id]
    else:
        tasks = get_golden_tasks(category)

    if not tasks:
        print("No matching golden tasks found.")
        return 2

    print(f"Running {len(tasks)} golden task(s)...\n")

    passed = 0
    for task in tasks:
        print(f"  [{task.id}] {task.task[:70]}...", end=" ", flush=True)
        try:
            m = Matrioska(cfg)
            result = m.run(task.task)
            evaluation = evaluate_result(task, result)
            if evaluation["pass"]:
                print("PASS")
                passed += 1
            else:
                print("FAIL")
                for check_name, check in evaluation["checks"].items():
                    if not check["pass"]:
                        print(
                            f"    - {check_name}: expected={check.get('expected', '?')} "
                            f"actual={check.get('actual', '?')}"
                        )
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n{passed}/{len(tasks)} passed")

    return 0 if passed == len(tasks) else 1


def _rich_show(info: Dict[str, Any]) -> None:
    """Pretty-print `matrioska show` output with Rich."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        _pretty_print(info)
        return

    console = Console()
    status = info.get("status", "?")
    proj = info.get("project_name", "?")
    work = info.get("work_dir", "?")

    style = {"done": "green", "partial": "yellow", "failed": "red"}.get(status, "cyan")
    console.print(Panel(
        f"[bold]{proj}[/]  [{style}]{status.upper()}[/]\n[dim]{work}[/]",
        title="[bold]MATRIOSKA[/]", border_style="blue",
    ))

    files = info.get("files", [])
    if files:
        tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", expand=False)
        tbl.add_column("#", style="dim", justify="right")
        tbl.add_column("File")
        tbl.add_column("Status", justify="center")
        tbl.add_column("Size", justify="right", style="dim")
        for f in files:
            st = f.get("status", "?")
            icon = {"done": "[green]✓[/]", "failed": "[red]✗[/]"}.get(st, "[dim]○[/]")
            tbl.add_row(
                str(f.get("order", f.get("name", ""))),
                f"{f['name']}.{f['extension']}",
                f"{icon} {st}",
                f"{f['chars']:,}c" if "chars" in f else "—",
            )
        console.print(tbl)

    tok = info.get("tokens", {})
    if tok:
        console.print(
            f"  [dim]Tokens[/]  prompt={tok.get('prompt_tokens',0):,}  "
            f"completion={tok.get('completion_tokens',0):,}  "
            f"cost=~${tok.get('estimated_cost_usd',0):.4f}"
        )

    cps = info.get("checkpoints", [])
    if cps:
        console.print(f"  [dim]Checkpoints[/]  {len(cps)} saved")


def _pretty_print(data: Any, indent: int = 0) -> None:
    prefix = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}:")
                _pretty_print(v, indent + 1)
            else:
                print(f"{prefix}{k}: {v}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _pretty_print(item, indent + 1)
                print()
            else:
                print(f"{prefix}- {item}")


def _cmd_compile(ns: argparse.Namespace) -> int:
    """Run the DSPy compilation loop over the golden suite."""
    from matrioska.core.config import load_config
    from matrioska.eval.dspy_compiler import compile_target

    overrides = _build_cli_overrides(ns)
    cfg = load_config(overrides)

    summary = compile_target(
        target=getattr(ns, "target", "architect"),
        category=getattr(ns, "category", None),
        max_tasks=getattr(ns, "max_tasks", 10),
        val_fraction=getattr(ns, "val_fraction", 0.3),
        out_dir=getattr(ns, "out_dir", None),
        cfg=cfg,
    )

    print(f"\n── DSPy compile: {summary.target} ──")
    print(f"  train: {summary.n_train}  val: {summary.n_val}")
    print(f"  baseline pass : {summary.baseline_pass:.1%}  (first_pass: {summary.baseline_first_pass_rate:.1%})")
    print(f"  compiled pass : {summary.compiled_pass:.1%}  (first_pass: {summary.compiled_first_pass_rate:.1%})")
    if summary.demos_path:
        print(f"  demos       : {summary.demos_path}")
    if summary.skipped_reason:
        print(f"  note        : skipped ({summary.skipped_reason})")
    print(f"  elapsed     : {summary.elapsed_s}s")
    return 0


def _cmd_init(ns: argparse.Namespace) -> int:
    from matrioska.cli.init_wizard import run_init
    target = getattr(ns, "target_dir", None) or Path.cwd()
    return run_init(cwd=target)


def _cmd_btw(ns: argparse.Namespace) -> int:
    """One-shot LLM query — no pipeline, no episodic note, no vault writes."""
    from matrioska.core.config import load_config, validate_config
    from matrioska.llm.client import LLMClient

    overrides = _build_cli_overrides(ns)
    cfg = load_config(overrides)
    validate_config(cfg)

    question = " ".join(getattr(ns, "question", []) or []).strip()
    if not question:
        print("ERROR: provide a question, e.g. `matrioska btw what is RRF?`",
              file=sys.stderr)
        return 2

    llm = LLMClient(cfg)
    try:
        resp = llm.chat(
            messages=[{"role": "user", "content": question}],
            model_spec=cfg.effective_generator,
            system="You are a concise technical assistant. Answer in 1-5 short sentences.",
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    text = (resp.text or "").strip()
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        Console().print(Markdown(text))
    except ImportError:
        print(text)
    return 0


def _cmd_vault(ns: argparse.Namespace) -> int:
    """Dispatch vault subcommands (search, doctor, graph, list)."""
    from matrioska.memory.vault import GlobalVault, default_vault_dir

    vault_dir = getattr(ns, "vault_dir", None) or default_vault_dir()
    vault = GlobalVault(Path(vault_dir).expanduser())

    sub_cmd = getattr(ns, "vault_cmd", None)
    if sub_cmd == "search":
        return _vault_search(ns, vault)
    if sub_cmd == "doctor":
        return _vault_doctor(vault)
    if sub_cmd == "graph":
        return _vault_graph(ns, vault)
    if sub_cmd == "list":
        return _vault_list(vault)
    print("Unknown vault subcommand.", file=sys.stderr)
    return 2


def _vault_search(ns: argparse.Namespace, vault: Any) -> int:
    query = " ".join(getattr(ns, "query", []) or []).strip()
    if not query:
        print("ERROR: provide a search query", file=sys.stderr)
        return 2

    work_dir = getattr(ns, "work_dir", None) or Path("./matrioska_work")
    results = vault.search(
        query,
        scope=getattr(ns, "scope", "all"),
        project=getattr(ns, "project", None),
        k=getattr(ns, "k", 8),
        local_root=Path(work_dir) / "knowledge" if Path(work_dir).exists() else None,
    )

    if not results:
        print(f"No matches in vault for '{query}' (scope={ns.scope})")
        return 0

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()
        tbl = Table(box=box.SIMPLE, title=f"Vault search: {query!r} (scope={ns.scope})")
        tbl.add_column("#", style="dim", justify="right")
        tbl.add_column("Score", style="green", justify="right")
        tbl.add_column("Kind", style="cyan")
        tbl.add_column("Path")
        tbl.add_column("Snippet", style="dim")
        for i, r in enumerate(results, 1):
            tbl.add_row(str(i), f"{r['score']:.2f}", r["kind"],
                        r["path"], r["snippet"][:80])
        console.print(tbl)
    except ImportError:
        for i, r in enumerate(results, 1):
            print(f"{i:2}. [{r['score']:.2f}] {r['kind']}: {r['path']}")
            print(f"    {r['snippet'][:160]}")
    return 0


def _vault_doctor(vault: Any) -> int:
    report = vault.doctor()
    print(f"Vault: {vault.root}")
    print(f"  Projects:  {report['projects']}")
    print(f"  Concepts:  {report['concepts']}")
    print(f"  Bugs:      {report['bugs']}")
    print(f"  Total notes: {report['total_notes']}")
    if report["orphans"]:
        print(f"\n  Orphan notes (no wikilinks in/out): {len(report['orphans'])}")
        for o in report["orphans"][:10]:
            print(f"    - {o}")
    if report["stale"]:
        print(f"\n  Stale notes (>30d, no recent updates): {len(report['stale'])}")
        for s in report["stale"][:10]:
            print(f"    - {s}")
    if report["broken_links"]:
        print(f"\n  Broken wikilinks: {len(report['broken_links'])}")
        for b in report["broken_links"][:10]:
            print(f"    - {b['from']} → [[{b['target']}]]")
    print(f"\n  Status: {report['status']}")
    return 0 if report["status"] == "healthy" else 1


def _vault_graph(ns: argparse.Namespace, vault: Any) -> int:
    mermaid = vault.export_graph_mermaid()
    out = getattr(ns, "output", None)
    if out:
        Path(out).write_text(mermaid, encoding="utf-8")
        print(f"Wrote Mermaid graph to {out}")
    else:
        print(mermaid)
    return 0


def _vault_list(vault: Any) -> int:
    projects = vault.list_projects()
    if not projects:
        print(f"Vault is empty: {vault.root}")
        return 0
    for p in projects:
        print(f"  {p['name']:30s}  notes={p['notes']:3d}  last_run={p['last_run']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    sub = p.add_subparsers(dest="cmd", required=True)

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

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())

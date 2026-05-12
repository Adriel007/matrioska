"""
Hook system — execute user-defined scripts in response to pipeline events.

Scripts live in `.matrioska/hooks/` (project) or `~/.matrioska/hooks/` (global).
Project hooks take precedence; both dirs are searched.

Supported hook names (executable scripts, any language):
  session_start   — REPL starts
  session_end     — REPL exits
  pre_generate    — before a file is generated (agent_call/generator)
  post_generate   — after a file is generated (file_generated)
  pre_repair      — before a repair attempt (agent_call/repairer)
  phase1_done     — architecture decided
  phase2_done     — all files generated
  run_end         — pipeline finished

Each script receives a JSON object via stdin with the event context.
stdout/stderr are captured and logged at DEBUG level.
Hooks that take > 10 s are killed (timeout).

Example hook (.matrioska/hooks/post_generate.sh):
  #!/usr/bin/env bash
  EVENT=$(cat)
  FILE=$(echo "$EVENT" | jq -r '.file')
  echo "generated: $FILE" >> .matrioska/hook_log.txt
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("matrioska.hooks")

# EventBus event name → hook script name
_EVENT_MAP: Dict[str, str] = {
    "agent_call":     "",          # special: depends on agent field
    "file_generated": "post_generate",
    "phase1_done":    "phase1_done",
    "phase2_done":    "phase2_done",
    "run_end":        "run_end",
}

_AGENT_HOOK_MAP: Dict[str, str] = {
    "generator":      "pre_generate",
    "generator_json": "pre_generate",
    "repairer":       "pre_repair",
}

_HOOK_TIMEOUT = 10   # seconds


def _find_hook(name: str, search_dirs: List[Path]) -> Optional[Path]:
    """Return the first executable hook script found, or None."""
    for d in search_dirs:
        for ext in ("", ".sh", ".py", ".bash"):
            candidate = d / f"{name}{ext}"
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
    return None


class HookRunner:
    """Discovers and runs hook scripts in response to events."""

    def __init__(self, project_dir: Optional[Path] = None) -> None:
        dirs: List[Path] = []
        if project_dir:
            dirs.append(project_dir / ".matrioska" / "hooks")
        dirs.append(Path.home() / ".matrioska" / "hooks")
        self._dirs = [d for d in dirs if d.is_dir()]
        if self._dirs:
            logger.debug("HookRunner active, searching: %s", self._dirs)

    @property
    def active(self) -> bool:
        return bool(self._dirs)

    def run(self, hook_name: str, context: Dict[str, Any]) -> None:
        """Find and execute a named hook, passing context as JSON via stdin."""
        script = _find_hook(hook_name, self._dirs)
        if script is None:
            return
        try:
            result = subprocess.run(
                [str(script)],
                input=json.dumps(context, default=str),
                capture_output=True,
                text=True,
                timeout=_HOOK_TIMEOUT,
            )
            if result.returncode != 0:
                logger.warning(
                    "Hook %s exited %d: %s",
                    hook_name, result.returncode, result.stderr[:200],
                )
            elif result.stdout:
                logger.debug("Hook %s: %s", hook_name, result.stdout[:200])
        except subprocess.TimeoutExpired:
            logger.warning("Hook %s timed out after %ds", hook_name, _HOOK_TIMEOUT)
        except Exception as e:
            logger.debug("Hook %s failed: %s", hook_name, e)

    def subscribe(self, bus: Any) -> None:
        """Subscribe to all relevant EventBus events."""
        if not self.active:
            return

        runner = self   # capture for closures

        def _handler(event: Any) -> None:
            name = getattr(event, "name", "") or ""
            data: Dict[str, Any] = dict(getattr(event, "data", {}) or {})

            if name == "agent_call":
                agent = data.get("agent", "")
                hook = _AGENT_HOOK_MAP.get(agent)
                if hook:
                    runner.run(hook, data)
            else:
                hook = _EVENT_MAP.get(name)
                if hook:
                    runner.run(hook, data)

        bus.on("*", _handler)


def make_hook_runner(work_dir: Path) -> HookRunner:
    """Construct a HookRunner searching project + global hook dirs."""
    return HookRunner(project_dir=work_dir)

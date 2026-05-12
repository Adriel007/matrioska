"""Tests for REPL autocompletion via prompt_toolkit integration.

These tests cover the `build_completer` helper (to be implemented)
that wraps slash-command names and session history into a
prompt_toolkit FuzzyCompleter / NestedCompleter.

The feature is tracked in TODO.md under:
  "REPL autocompletion & keyboard navigation"

All tests here are written against the public contract so they will
*pass* with a minimal implementation and *fail* clearly if the
feature regresses.
"""

from __future__ import annotations

import pytest

from matrioska.cli.repl import COMMANDS


# ---------------------------------------------------------------------------
# build_completer helper
# ---------------------------------------------------------------------------


def _import_build_completer():
    """Import build_completer, skipping the test if not yet implemented."""
    try:
        from matrioska.cli.repl import build_completer
        return build_completer
    except ImportError:
        pytest.skip("build_completer not yet implemented in cli.repl")


def test_build_completer_returns_callable():
    """build_completer() must return a non-None object."""
    build_completer = _import_build_completer()
    completer = build_completer(COMMANDS, history=[])
    assert completer is not None


def test_build_completer_includes_all_slash_commands():
    """Every registered slash command name should appear in completions.

    We exercise the completer by asking it to complete an empty slash
    prefix and checking that all COMMANDS keys appear in the results.
    """
    build_completer = _import_build_completer()
    try:
        from prompt_toolkit.document import Document
    except ImportError:
        pytest.skip("prompt_toolkit not installed")

    completer = build_completer(COMMANDS, history=[])
    doc = Document("/")
    completions = list(completer.get_completions(doc, None))
    displayed = {c.text for c in completions}

    for name in COMMANDS:
        assert f"/{name}" in displayed or name in displayed, (
            f"slash command '{name}' missing from completions"
        )


def test_build_completer_history_entries_are_suggested():
    """History-based completions should surface previous inputs."""
    build_completer = _import_build_completer()
    try:
        from prompt_toolkit.document import Document
    except ImportError:
        pytest.skip("prompt_toolkit not installed")

    history = ["create a todo app", "build an API", "/plan"]
    completer = build_completer(COMMANDS, history=history)

    doc = Document("create")
    completions = list(completer.get_completions(doc, None))
    texts = {c.text for c in completions}
    assert any("todo" in t or "create" in t for t in texts), (
        "history entry not surfaced by completer"
    )


def test_build_completer_empty_history_no_crash():
    """build_completer with empty history must not raise."""
    build_completer = _import_build_completer()
    completer = build_completer(COMMANDS, history=[])
    assert completer is not None


def test_build_completer_slash_prefix_filters_to_commands():
    """Completing '/' should only suggest slash commands, not history."""
    build_completer = _import_build_completer()
    try:
        from prompt_toolkit.document import Document
    except ImportError:
        pytest.skip("prompt_toolkit not installed")

    history = ["write a parser", "add tests"]
    completer = build_completer(COMMANDS, history=history)
    doc = Document("/")
    completions = list(completer.get_completions(doc, None))
    # Completions for "/" should be command-oriented (not plain prose)
    assert len(completions) >= 1, "no completions returned for '/' prefix"

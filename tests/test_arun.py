"""Tests for the MatrioskaStreaming.arun() generator API."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from matrioska.api import MatrioskaStreaming, arun
from matrioska.core.config import Config


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_cfg(tmp_path: Path) -> Config:
    return Config(
        work_dir=tmp_path,
        provider="openai",
        api_key="sk-test",
        dry_run=True,  # dry_run avoids real LLM calls
    )


def _fake_matrioska_run(task: str) -> Dict[str, Any]:
    """Minimal result that Matrioska.run() would return."""
    return {
        "status": "dry_run",
        "project_name": "test_project",
        "artifacts": [],
        "shared_state": {},
        "tokens": {"prompt_tokens": 0, "completion_tokens": 0},
        "work_dir": "/tmp/test",
    }


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestMatrioskaStreamingInit:
    def test_instantiates_with_config(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        client = MatrioskaStreaming(cfg)
        assert client.cfg is cfg
        assert client._thread is None
        assert not client._closed

    def test_instantiates_with_defaults(self, tmp_path: Path):
        """Without explicit cfg it loads defaults — no exception."""
        with patch("matrioska.api.load_config") as mock_load:
            mock_load.return_value = _make_cfg(tmp_path)
            client = MatrioskaStreaming()
        assert client.cfg is not None


class TestMatrioskaStreamingArun:
    def test_dry_run_yields_run_end(self, tmp_path: Path):
        """dry_run=True means Matrioska.run() returns immediately with dry_run status."""
        cfg = _make_cfg(tmp_path)
        client = MatrioskaStreaming(cfg)
        events = list(client.arun("Build a todo CLI"))
        client.close()
        # Must yield at least the run_end event
        assert events, "expected at least one event"
        event_names = [e["event"] for e in events]
        assert "run_end" in event_names

    def test_run_end_has_required_fields(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        client = MatrioskaStreaming(cfg)
        events = list(client.arun("x"))
        client.close()
        run_end = next(e for e in events if e["event"] == "run_end")
        assert "data" in run_end
        assert "timestamp" in run_end
        assert isinstance(run_end["timestamp"], float)
        data = run_end["data"]
        assert "status" in data
        assert "project_name" in data
        assert "files" in data
        assert "tokens" in data

    def test_each_event_has_event_data_timestamp(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        client = MatrioskaStreaming(cfg)
        for event in client.arun("x"):
            assert "event" in event, f"missing 'event' key: {event}"
            assert "data" in event, f"missing 'data' key: {event}"
            assert "timestamp" in event, f"missing 'timestamp' key: {event}"
            assert isinstance(event["data"], dict)
        client.close()

    def test_timestamps_are_monotonic(self, tmp_path: Path):
        """Timestamps should be monotonically non-decreasing."""
        cfg = _make_cfg(tmp_path)
        client = MatrioskaStreaming(cfg)
        events = list(client.arun("x"))
        client.close()
        timestamps = [e["timestamp"] for e in events]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1] - 0.001  # small tolerance


class TestMatrioskaStreamingClose:
    def test_close_is_idempotent(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        client = MatrioskaStreaming(cfg)
        client.close()
        client.close()  # second close must not raise
        assert client._closed

    def test_close_after_run(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        client = MatrioskaStreaming(cfg)
        list(client.arun("x"))  # exhaust the generator
        client.close()
        assert client._closed
        assert client._thread is None


class TestMatrioskaStreamingErrorHandling:
    def test_error_in_run_yields_run_error_event(self, tmp_path: Path):
        """If Matrioska.run() raises, arun() yields a run_error event."""
        cfg = _make_cfg(tmp_path)

        with patch("matrioska.api.Matrioska") as MockM:
            from matrioska.core.events import EventBus

            instance = MockM.return_value
            instance.bus = EventBus()
            instance.run.side_effect = RuntimeError("simulated failure")

            client = MatrioskaStreaming(cfg)
            with patch("matrioska.api.validate_config"):
                events = list(client.arun("will fail"))
            client.close()

        event_names = [e["event"] for e in events]
        assert "run_error" in event_names
        err_event = next(e for e in events if e["event"] == "run_error")
        assert "simulated failure" in err_event["data"]["error"]


class TestArunConvenienceFunction:
    def test_arun_yields_events(self, tmp_path: Path):
        cfg = _make_cfg(tmp_path)
        events = list(arun("Build something", cfg=cfg))
        assert events
        assert any(e["event"] == "run_end" for e in events)

    def test_arun_with_config_overrides(self, tmp_path: Path):
        """arun() calls load_config when keyword config overrides are provided."""
        fake_cfg = _make_cfg(tmp_path)
        with patch("matrioska.api.load_config", return_value=fake_cfg) as mock_load, \
             patch("matrioska.api.validate_config"):
            # exhaust the generator so the finally block runs
            events = list(arun("x", provider="openai"))
        mock_load.assert_called_once_with({"provider": "openai"})


class TestEventInterception:
    def test_pipeline_events_are_captured(self, tmp_path: Path):
        """Events emitted by the bus should appear in the generator output."""
        cfg = _make_cfg(tmp_path)
        from matrioska.core.events import EventBus, Event

        real_bus = EventBus()

        with patch("matrioska.api.Matrioska") as MockM, \
             patch("matrioska.api.validate_config"):
            instance = MockM.return_value
            instance.bus = real_bus

            def _emit_and_return(task):
                # After arun() patches bus.emit, events go through the interceptor
                real_bus.emit(Event(name="phase1_done", data={"ok": True}))
                real_bus.emit(Event(name="phase2_done", data={"files": 2}))
                return {
                    "status": "success",
                    "project_name": "test",
                    "artifacts": [],
                    "shared_state": {},
                    "tokens": {},
                    "work_dir": str(tmp_path),
                }

            instance.run.side_effect = _emit_and_return

            client = MatrioskaStreaming(cfg)
            events = list(client.arun("task"))
            client.close()

        event_names = [e["event"] for e in events]
        assert "phase1_done" in event_names
        assert "phase2_done" in event_names
        assert "run_end" in event_names

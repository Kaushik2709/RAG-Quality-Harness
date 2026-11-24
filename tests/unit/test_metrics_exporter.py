"""Metrics registry and bounded trace exporter."""
from __future__ import annotations

import pytest

from src.observability.metrics import MetricsRegistry
from src.tracing.cost import compute_cost
from src.tracing.exporter import TraceExporter, TraceRecord
from src.tracing.tracer import trace_stage


def test_metrics_counter_gauge_and_histogram_render():
    registry = MetricsRegistry()
    registry.inc("requests_total", {"route": "/query", "status": "200"})
    registry.inc("requests_total", {"route": "/query", "status": "200"}, 2)
    registry.set_gauge("index_chunks", 7)
    for value in (1, 20, 3000):
        registry.observe("latency_ms", value, {"stage": "retrieve"})
    rendered = registry.render()
    assert 'requests_total{route="/query",status="200"} 3' in rendered
    assert "index_chunks 7" in rendered
    assert 'latency_ms_bucket{stage="retrieve",le="5"} 1' in rendered
    assert 'latency_ms_bucket{stage="retrieve",le="+Inf"} 3' in rendered
    assert 'latency_ms_count{stage="retrieve"} 3' in rendered


def test_metrics_labels_are_escaped_and_reset():
    registry = MetricsRegistry()
    registry.inc("events", {"value": 'a"b\nc'})
    assert 'value="a\\"b\\nc"' in registry.render()
    registry.reset()
    assert "events" not in registry.render()


def test_cost_is_static_estimate_and_unknown_model_is_zero():
    assert compute_cost("gemini-2.5-flash", 1_000_000, 1_000_000) == pytest.approx(0.375)
    assert compute_cost("unknown-model", 1000, 1000) == 0.0


def _record(stage="retrieve", query_id="q1"):
    return TraceRecord(query_id=query_id, stage=stage, latency_ms=1.5, tokens_in=2, tokens_out=3)


def test_sqlite_exporter_writes_and_reads(tmp_path):
    exporter = TraceExporter(sink="sqlite", db_path=str(tmp_path / "traces.db"), flush_interval=0.01)
    try:
        assert exporter.export(_record())
        assert exporter.flush(timeout=3)
        rows = exporter.get_recent_traces(limit=10)
        assert len(rows) == 1
        assert rows[0]["query_id"] == "q1"
        assert rows[0]["cost_source"] == "estimated_static_pricing"
        status = exporter.status()
        assert status["written"] == 1
        assert status["persistent"] is True
        assert status["worker_alive"] is True
    finally:
        exporter.close()


def test_full_queue_drops_without_blocking():
    exporter = TraceExporter(sink="sqlite", db_path="unused.db", queue_size=1, batch_size=1)
    exporter._ensure_worker = lambda: None  # deterministic: no worker consumes the queue
    assert exporter.export(_record("a")) is True
    assert exporter.export(_record("b")) is False
    status = exporter.status()
    assert status["dropped"] == 1
    assert status["written"] == 0
    assert status["queue_depth"] == 1


def test_sink_none_never_claims_persistence(tmp_path):
    exporter = TraceExporter(sink="none")
    assert exporter.export(_record()) is False
    assert exporter.status()["persistent"] is False
    assert exporter.get_recent_traces() == []


def test_trace_stage_records_provider_usage_and_estimate_source(tmp_path):
    exporter = TraceExporter(sink="sqlite", db_path=str(tmp_path / "traces.db"), flush_interval=0.01)
    try:
        with trace_stage("generate", query_id="q-usage", model_name="gemini-2.5-flash", exporter=exporter) as span:
            span.set_usage(100, 50, source="provider")
            span.set_attribute("retrieval.index_version", 3)
        assert exporter.flush(timeout=3)
        row = exporter.get_recent_traces(1)[0]
        assert row["usage_source"] == "provider"
        assert row["tokens_in"] == 100
        assert row["index_version"] == 3
        assert row["cost_usd"] > 0
    finally:
        exporter.close()


def test_trace_stage_marks_errors_without_swallowing(tmp_path):
    exporter = TraceExporter(sink="sqlite", db_path=str(tmp_path / "traces.db"), flush_interval=0.01)
    try:
        with pytest.raises(RuntimeError):
            with trace_stage("retrieve", query_id="q-error", exporter=exporter) as span:
                span.set_attribute("retrieval.mode", "hybrid")
                raise RuntimeError("boom")
        assert exporter.flush(timeout=3)
        row = exporter.get_recent_traces(1)[0]
        assert row["status"] == "error"
        assert row["error_category"] == "RuntimeError"
    finally:
        exporter.close()

"""OpenTelemetry span helper plus bounded local trace export.

Token counts are never invented as provider facts: callers must state whether
usage came from provider metadata or from an estimate. Cost is always computed
from a static price table and labelled as an estimate, not a bill.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Generator, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.observability.metrics import metrics
from src.tracing.cost import compute_cost
from src.tracing.exporter import TraceExporter, TraceRecord, get_exporter

tracer = trace.get_tracer("rag.pipeline", "2.0.0")


class PipelineStageSpan:
    def __init__(self, stage: str, query_id: str, model_name: str = "n/a"):
        self.stage = stage
        self.query_id = query_id
        self.model_name = model_name
        self.tokens_in = 0
        self.tokens_out = 0
        self.usage_source = "unavailable"
        self.cost_usd = 0.0
        self.latency_ms = 0.0
        self.status = "ok"
        self.error_category: Optional[str] = None
        self.attributes: Dict[str, Any] = {}
        self.span = None
        self.start_time = 0.0

    def set_tokens(self, tokens_in: int = 0, tokens_out: int = 0, *, source: str = "estimate") -> None:
        """Record token counts. Default source is an estimate by design."""
        self.set_usage(tokens_in, tokens_out, source=source)

    def set_usage(self, tokens_in: int = 0, tokens_out: int = 0, *, source: str = "estimate") -> None:
        self.tokens_in = int(tokens_in or 0)
        self.tokens_out = int(tokens_out or 0)
        self.usage_source = source if self.tokens_in or self.tokens_out else "unavailable"
        self.cost_usd = compute_cost(self.model_name, self.tokens_in, self.tokens_out)
        if self.span:
            self.span.set_attribute("gen_ai.usage.input_tokens", self.tokens_in)
            self.span.set_attribute("gen_ai.usage.output_tokens", self.tokens_out)
            self.span.set_attribute("gen_ai.usage.source", self.usage_source)
            self.span.set_attribute("gen_ai.cost_usd_estimated", self.cost_usd)

    def set_attribute(self, key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            value = ",".join(str(item) for item in value)
        self.attributes[key] = value
        if self.span:
            try:
                self.span.set_attribute(key, value)
            except Exception:  # noqa: BLE001 - telemetry must not break the request
                pass

    def mark_error(self, category: str) -> None:
        self.status = "error"
        self.error_category = category


def _export_record(
    stage: str,
    query_id: str,
    span: PipelineStageSpan,
    exporter: Optional[TraceExporter],
) -> None:
    attributes = span.attributes
    record = TraceRecord(
        query_id=query_id,
        stage=stage,
        latency_ms=round(span.latency_ms, 3),
        tokens_in=span.tokens_in,
        tokens_out=span.tokens_out,
        cost_usd=span.cost_usd,
        model_name=span.model_name,
        usage_source=span.usage_source,
        index_version=attributes.get("retrieval.index_version"),
        candidate_count=attributes.get("retrieval.candidate_count"),
        rerank_status=attributes.get("rerank.status"),
        error_category=span.error_category,
        status=span.status,
    )
    try:
        (exporter or get_exporter()).export(record)
    except Exception:  # noqa: BLE001 - export is best effort and bounded
        pass
    metrics.observe_stage(stage, span.latency_ms, strategy=str(attributes.get("retrieval.mode", "")))
    metrics.inc("rag_stage_total", {"stage": stage, "status": span.status})


@contextlib.contextmanager
def trace_stage(
    stage: str,
    query_id: str,
    model_name: str = "n/a",
    exporter: Optional[TraceExporter] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[PipelineStageSpan, None, None]:
    stage_span = PipelineStageSpan(stage=stage, query_id=query_id, model_name=model_name)
    stage_span.start_time = time.perf_counter()
    for key, value in (attributes or {}).items():
        stage_span.set_attribute(key, value)

    with tracer.start_as_current_span(f"rag.{stage}") as otel_span:
        stage_span.span = otel_span
        otel_span.set_attribute("gen_ai.system", "rag_eval_platform")
        otel_span.set_attribute("gen_ai.pipeline.stage", stage)
        otel_span.set_attribute("gen_ai.query_id", query_id)
        otel_span.set_attribute("gen_ai.request.model", model_name)

        try:
            yield stage_span
            otel_span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            stage_span.mark_error(type(exc).__name__)
            otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
            otel_span.record_exception(exc)
            raise
        finally:
            stage_span.latency_ms = (time.perf_counter() - stage_span.start_time) * 1000.0
            otel_span.set_attribute("gen_ai.latency_ms", stage_span.latency_ms)
            if stage_span.error_category:
                otel_span.set_attribute("error.type", stage_span.error_category)
            _export_record(stage, query_id, stage_span, exporter)

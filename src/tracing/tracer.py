import time
import contextlib
from typing import Optional, Generator
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from src.tracing.cost import compute_cost
from src.tracing.exporter import TraceExporter, TraceRecord, get_exporter

# Setup OpenTelemetry tracer provider
tracer_provider = trace.get_tracer_provider()
tracer = trace.get_tracer("rag.pipeline", "1.0.0")

class PipelineStageSpan:
    def __init__(self, stage: str, query_id: str, model_name: str = "n/a"):
        self.stage = stage
        self.query_id = query_id
        self.model_name = model_name
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self.latency_ms = 0.0
        self.span = None
        self.start_time = 0.0

    def set_tokens(self, tokens_in: int = 0, tokens_out: int = 0):
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = compute_cost(self.model_name, tokens_in, tokens_out)
        if self.span:
            self.span.set_attribute("gen_ai.usage.input_tokens", tokens_in)
            self.span.set_attribute("gen_ai.usage.output_tokens", tokens_out)
            self.span.set_attribute("gen_ai.cost_usd", self.cost_usd)

@contextlib.contextmanager
def trace_stage(stage: str, query_id: str, model_name: str = "n/a", exporter: Optional[TraceExporter] = None) -> Generator[PipelineStageSpan, None, None]:
    stage_span = PipelineStageSpan(stage=stage, query_id=query_id, model_name=model_name)
    stage_span.start_time = time.perf_counter()

    with tracer.start_as_current_span(f"rag.{stage}") as otel_span:
        stage_span.span = otel_span
        otel_span.set_attribute("gen_ai.system", "rag_eval_platform")
        otel_span.set_attribute("gen_ai.pipeline.stage", stage)
        otel_span.set_attribute("gen_ai.query_id", query_id)
        otel_span.set_attribute("gen_ai.request.model", model_name)

        try:
            yield stage_span
            otel_span.set_status(Status(StatusCode.OK))
        except Exception as e:
            otel_span.set_status(Status(StatusCode.ERROR, str(e)))
            otel_span.record_exception(e)
            raise
        finally:
            stage_span.latency_ms = (time.perf_counter() - stage_span.start_time) * 1000.0
            otel_span.set_attribute("gen_ai.latency_ms", stage_span.latency_ms)

            # Export span data
            exp = exporter or get_exporter()
            exp.export(TraceRecord(
                query_id=query_id,
                stage=stage,
                latency_ms=round(stage_span.latency_ms, 2),
                tokens_in=stage_span.tokens_in,
                tokens_out=stage_span.tokens_out,
                cost_usd=stage_span.cost_usd,
                model_name=stage_span.model_name
            ))

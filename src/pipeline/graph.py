"""LangGraph pipeline with explicit retrieval, reranking, generation and citation
validation stages.

Each stage records its own trace/metrics and can fail independently. A skipped
rerank is reported as `unavailable`/`failed`, never as `applied`. A generation
that violates the answer contract raises; it is never converted into a plain
text answer.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.config import settings
from src.generation.citations import abstained_answer, validate_citations
from src.generation.context_builder import build_context
from src.generation.generator import generate_structured
from src.guardrails.detector import GuardrailAction, get_input_guardrail, get_output_guardrail
from src.observability.metrics import metrics
from src.retrieval.retrieval_runtime import get_retrieval_runtime
from src.retrieval.retriever import build_retriever
from src.retrieval.reranker import build_reranked_retriever, rerank_documents
from src.retrieval.vector_store import get_vector_store
from src.runtime.errors import ModelUnavailable
from src.runtime.lifecycle import get_model_registry
from src.tracing.tracer import trace_stage


class RAGState(TypedDict, total=False):
    query: str
    query_id: str
    tenant_id: str
    documents: List[Document]
    reranked_documents: List[Document]
    retrieval: Dict[str, Any]
    rerank: Dict[str, Any]
    context: Dict[str, Any]
    structured_answer: Dict[str, Any]
    citation_validation: Dict[str, Any]
    generation: str
    generation_status: str
    usage: Dict[str, Any]
    guardrail_status: Optional[Dict[str, Any]]
    guardrail_blocked: Optional[bool]
    top_k: Optional[int]
    top_k_rerank: Optional[int]
    use_hybrid: Optional[bool]
    retrieval_mode: Optional[str]


def _query_id(state: RAGState) -> str:
    return state.get("query_id") or "q-default"


def _tenant(state: RAGState) -> str:
    return state.get("tenant_id") or "default"


def input_guardrail_node(state: RAGState) -> Dict[str, Any]:
    query_id = _query_id(state)
    query = state.get("query", "")
    if not settings.ENABLE_INPUT_GUARDRAILS:
        return {"guardrail_blocked": False, "guardrail_status": None}

    with trace_stage("input_guardrail", query_id=query_id) as span:
        result = get_input_guardrail().validate(query)
        span.set_attribute("guardrail.action", result.action.value)
        span.set_attribute("guardrail.rules", result.triggered_rules)
        if result.action == GuardrailAction.BLOCKED:
            metrics.inc("rag_guardrail_total", {"stage": "input", "action": "blocked"})
            metrics.inc("rag_prompt_injection_detected_total", {"tenant_id": _tenant(state), "source": "query"})
            return {
                "guardrail_blocked": True,
                "guardrail_status": result.to_dict(),
                "generation": f"Query blocked by Security Input Guardrail: {result.reason}",
                "generation_status": "blocked",
            }
        metrics.inc("rag_guardrail_total", {"stage": "input", "action": result.action.value})
        updates: Dict[str, Any] = {"guardrail_blocked": False, "guardrail_status": result.to_dict()}
        if result.action == GuardrailAction.SANITIZED:
            updates["query"] = result.sanitized_text
        return updates


def retrieve_node(
    state: RAGState,
    vector_store=None,
    corpus_documents=None,
    retrieval_runtime=None,
) -> Dict[str, Any]:
    if state.get("guardrail_blocked"):
        return {}
    query_id = _query_id(state)
    query = state["query"]
    req_k = state.get("top_k")
    top_k = req_k if (req_k is not None and req_k > 0) else settings.TOP_K_RETRIEVAL
    req_hybrid = state.get("use_hybrid")
    use_hybrid = req_hybrid if req_hybrid is not None else settings.USE_HYBRID_SEARCH
    mode = state.get("retrieval_mode") or ("hybrid" if use_hybrid else "dense")

    with trace_stage("retrieve", query_id=query_id, model_name=settings.EMBEDDING_MODEL) as span:
        span.set_attribute("retrieval.mode", mode)
        span.set_attribute("retrieval.tenant_id", _tenant(state))
        if corpus_documents is not None:
            # Injected/explicit corpus path (evaluation, tests). Building a
            # retriever here is intentionally scoped to the supplied corpus.
            store = vector_store if vector_store is not None else get_vector_store()
            retriever = build_retriever(
                vectorstore=store,
                documents=corpus_documents,
                top_k=top_k,
                mode=mode,
            )
            docs = retriever.invoke(query)
            stage_latency = getattr(retriever, "last_latency_ms", {})
            retrieval_meta = {
                "mode": mode,
                "index_version": None,
                "corpus_fingerprint": None,
                "candidate_count": len(docs),
                "stage_latency_ms": stage_latency,
                "degraded_reasons": [],
            }
        else:
            store = vector_store if vector_store is not None else get_vector_store()
            runtime = retrieval_runtime or get_retrieval_runtime()
            result = runtime.search(
                store,
                query,
                top_k=top_k,
                mode=mode,
                tenant_id=_tenant(state),
            )
            docs = result.documents
            retrieval_meta = result.metadata()
            span.set_attribute("retrieval.index_version", result.index_version)
            span.set_attribute("retrieval.corpus_fingerprint", result.corpus_fingerprint)
            span.set_attribute("retrieval.candidate_count", result.candidate_count)
            for stage, elapsed in result.stage_latency_ms.items():
                metrics.observe_stage(stage, elapsed, strategy=mode)
        span.set_attribute("retrieval.candidate_count", len(docs))
        span.set_attribute("retrieval.chunk_ids", [d.metadata.get("chunk_id", "") for d in docs[:10]])
        span.set_attribute("retrieval.degraded_reasons", retrieval_meta.get("degraded_reasons"))
        metrics.inc("rag_retrieval_requests_total", {"strategy": mode, "status": "success"})
    return {"documents": docs, "retrieval": retrieval_meta}


def rerank_node(state: RAGState) -> Dict[str, Any]:
    if state.get("guardrail_blocked"):
        return {}
    query_id = _query_id(state)
    query = state["query"]
    input_docs = state.get("documents", [])
    req_rerank = state.get("top_k_rerank")
    top_k_rerank = req_rerank if (req_rerank is not None and req_rerank > 0) else settings.TOP_K_RERANK

    if settings.RERANK_POLICY == "disabled":
        metrics.inc("rag_reranker_total", {"status": "disabled"})
        return {
            "reranked_documents": input_docs,
            "rerank": {"status": "disabled", "degraded": False, "scores": {}},
        }

    with trace_stage("rerank", query_id=query_id, model_name=settings.RERANKER_MODEL) as span:
        span.set_attribute("rerank.policy", settings.RERANK_POLICY)
        if not input_docs:
            span.set_attribute("rerank.status", "skipped_empty")
            metrics.inc("rag_reranker_total", {"status": "skipped_empty"})
            return {"reranked_documents": [], "rerank": {"status": "skipped_empty", "degraded": False, "scores": {}}}

        try:
            reranker = get_model_registry().get_reranker()
        except ModelUnavailable:
            span.set_attribute("rerank.status", "unavailable")
            metrics.inc("rag_reranker_total", {"status": "unavailable"})
            raise
        if reranker is None:
            # Degraded policy: explicitly skip and expose it.
            span.set_attribute("rerank.status", "unavailable")
            span.set_attribute("rerank.degraded", True)
            metrics.inc("rag_reranker_total", {"status": "unavailable"})
            return {
                "reranked_documents": input_docs,
                "rerank": {
                    "status": "unavailable",
                    "degraded": True,
                    "scores": {},
                    "reason": "cross-encoder unavailable; policy=degraded",
                },
            }

        try:
            ranked, scores = rerank_documents(reranker, query, input_docs, top_k_rerank)
        except Exception as exc:  # noqa: BLE001
            span.mark_error(type(exc).__name__)
            metrics.inc("rag_reranker_total", {"status": "failed"})
            if settings.RERANK_POLICY == "strict":
                raise
            return {
                "reranked_documents": input_docs,
                "rerank": {
                    "status": "failed",
                    "degraded": True,
                    "scores": {},
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            }
        top_scores = dict(list(scores.items())[:10])
        span.set_attribute("rerank.status", "applied")
        span.set_attribute("rerank.top_scores", top_scores)
        metrics.inc("rag_reranker_total", {"status": "applied"})
    return {
        "reranked_documents": ranked,
        "rerank": {"status": "applied", "degraded": False, "scores": top_scores, "top_n": top_k_rerank},
    }


def generate_node(state: RAGState) -> Dict[str, Any]:
    if state.get("guardrail_blocked"):
        return {}
    query_id = _query_id(state)
    query = state["query"]
    reranked = state.get("reranked_documents", [])
    bundle = build_context(reranked)
    metrics.set_gauge("rag_context_utilization", bundle.utilization)

    if not bundle.chunks:
        answer = abstained_answer("no_relevant_evidence")
        return {
            "generation": answer.answer,
            "generation_status": "abstained",
            "structured_answer": answer.model_dump(),
            "context": bundle.metadata(),
            "usage": {},
        }

    with trace_stage("generate", query_id=query_id, model_name=get_model_registry().status()["models"]["llm"]["model_id"]) as span:
        span.set_attribute("context.chunks", len(bundle.chunks))
        span.set_attribute("context.estimated_tokens", bundle.estimated_tokens)
        span.set_attribute("context.token_counts_estimated", bundle.token_counts_estimated)
        span.set_attribute("context.utilization", bundle.utilization)
        span.set_attribute("context.untrusted_evidence", bundle.untrusted_scan.flagged)
        if bundle.untrusted_scan.flagged:
            metrics.inc("rag_untrusted_evidence_total", {"stage": "context"})
        llm = get_model_registry().get_llm()
        result = generate_structured(query, bundle.text, chat_model=llm)
        span.set_usage(
            result.usage.get("input_tokens", 0),
            result.usage.get("output_tokens", 0),
            source=result.usage_source,
        )
        span.set_attribute("generation.model_id", result.model_id)
        span.set_attribute("generation.attempts", result.attempts)
        span.set_attribute("generation.abstained", result.structured.abstained)
        metrics.inc("rag_generation_requests_total", {"status": "success", "abstained": str(result.structured.abstained).lower()})

    return {
        "generation": result.structured.answer,
        "generation_status": "abstained" if result.structured.abstained else "generated",
        "structured_answer": result.structured.model_dump(),
        "context": bundle.metadata(),
        "usage": {"input_tokens": result.usage.get("input_tokens", 0),
                  "output_tokens": result.usage.get("output_tokens", 0),
                  "total_tokens": result.usage.get("total_tokens", 0),
                  "source": result.usage_source},
    }


def citation_validation_node(state: RAGState) -> Dict[str, Any]:
    if state.get("guardrail_blocked"):
        return {}
    payload = state.get("structured_answer")
    if not payload:
        return {"citation_validation": {"status": "skipped", "reason": "no structured answer"}}
    from src.generation.citations import StructuredAnswer

    answer = StructuredAnswer(**payload)
    documents = state.get("reranked_documents", [])
    query_id = _query_id(state)
    policy = getattr(settings, "CITATION_VALIDATION_POLICY", "strict")

    with trace_stage("citation_validation", query_id=query_id) as span:
        validation = validate_citations(answer, documents, tenant_id=_tenant(state), policy=policy)
        span.set_attribute("citations.status", validation.status)
        span.set_attribute("citations.precision", validation.citation_precision)
        span.set_attribute("citations.recall", validation.retrieved_evidence_utilization)
        span.set_attribute("citations.claim_coverage", validation.claim_coverage)
        span.set_attribute("citations.rejected", len(validation.rejected))
        span.set_attribute("citations.is_fully_supported", validation.is_fully_supported)
        metrics.inc("rag_citation_validation_total", {"status": validation.status})
        outcome = "abstained" if answer.abstained else (validation.status.value if hasattr(validation.status, "value") else str(validation.status).lower())
        metrics.inc("rag_citation_validation_outcomes_total", {"tenant_id": _tenant(state), "outcome": outcome})

        if not answer.abstained:
            if policy == "strict" and not validation.is_fully_supported:
                # In strict mode: reject complete answer if any required evidence or claim fails
                metrics.inc("rag_citation_rejections_total", {"reason": "strict_policy_rejection"})
                fallback = abstained_answer("citations_failed_validation")
                return {
                    "generation": fallback.answer,
                    "generation_status": "abstained",
                    "structured_answer": fallback.model_dump(),
                    "citation_validation": {**validation.to_dict(), "original_answer_rejected": True},
                }
            if policy == "review" and not validation.is_fully_supported:
                return {
                    "generation_status": "review_required",
                    "citation_validation": {
                        **validation.to_dict(),
                        "original_answer_rejected": False,
                        "review_required": True,
                    },
                }
            if not validation.accepted:
                fallback = abstained_answer("citations_failed_validation")
                return {
                    "generation": fallback.answer,
                    "generation_status": "abstained",
                    "structured_answer": fallback.model_dump(),
                    "citation_validation": {**validation.to_dict(), "original_answer_rejected": True},
                }
    return {"citation_validation": validation.to_dict()}


def output_guardrail_node(state: RAGState) -> Dict[str, Any]:
    if state.get("guardrail_blocked"):
        return {}
    query_id = _query_id(state)
    raw_generation = state.get("generation", "")
    if not settings.ENABLE_OUTPUT_GUARDRAILS:
        return {}

    with trace_stage("output_guardrail", query_id=query_id) as span:
        result = get_output_guardrail().validate(raw_generation)
        span.set_attribute("guardrail.action", result.action.value)
        span.set_attribute("guardrail.rules", result.triggered_rules)
        existing = state.get("guardrail_status") or {}
        combined = {"input": existing, "output": result.to_dict()}
        metrics.inc("rag_guardrail_total", {"stage": "output", "action": result.action.value})
        if result.action == GuardrailAction.SANITIZED:
            return {"generation": result.sanitized_text, "guardrail_status": combined}
        return {"guardrail_status": combined}


def route_input_guardrail(state: RAGState) -> str:
    return END if state.get("guardrail_blocked") else "retrieve"


def compile_rag_graph(vector_store=None, corpus_documents=None, retrieval_runtime=None):
    builder = StateGraph(RAGState)
    builder.add_node("input_guardrail", input_guardrail_node)
    builder.add_node(
        "retrieve",
        partial(
            retrieve_node,
            vector_store=vector_store,
            corpus_documents=corpus_documents,
            retrieval_runtime=retrieval_runtime,
        ),
    )
    builder.add_node("rerank", rerank_node)
    builder.add_node("generate", generate_node)
    builder.add_node("citation_validation", citation_validation_node)
    builder.add_node("output_guardrail", output_guardrail_node)

    builder.set_entry_point("input_guardrail")
    builder.add_conditional_edges("input_guardrail", route_input_guardrail)
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "citation_validation")
    builder.add_edge("citation_validation", "output_guardrail")
    builder.add_edge("output_guardrail", END)
    return builder.compile()

"""Isolated retrieval regression CLI. Paid generation/judging is explicitly opt-in."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from time import perf_counter

from src.config import settings
from src.evaluation.golden_dataset import load_golden_dataset
from src.evaluation.retrieval_metrics import compute_all_retrieval_metrics
from src.ingestion.loaders import load_document
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import get_vector_store
from src.retrieval.retriever import build_retriever, FixedRetriever

MODES = ("dense", "bm25", "hybrid", "hybrid-rerank")
ROOT = Path(__file__).resolve().parents[2]


def load_corpus(corpus_dir):
    files = sorted(p for p in Path(corpus_dir).rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".html", ".pdf"})
    if not files:
        raise ValueError("Corpus is empty")
    docs = []
    for path in files:
        loaded = load_document(str(path))
        if any(d.metadata.get("error") for d in loaded):
            raise ValueError(f"Failed loading corpus document: {path.name}")
        for d in loaded:
            d.metadata["doc_id"] = path.relative_to(corpus_dir).as_posix()
        docs.extend(loaded)
    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError("Corpus has no readable chunks")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(corpus_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return chunks, len(files), digest.hexdigest()


def ingest_sample_corpus(vector_store):
    chunks, _, _ = load_corpus(ROOT / "data/sample_docs")
    vector_store.add_documents(chunks, ids=[d.metadata["chunk_id"] for d in chunks])
    return chunks


def percentiles(values):
    def percentile(p):
        ordered = sorted(values)
        if not ordered:
            return None
        pos = (len(ordered)-1)*p
        low, high = math.floor(pos), math.ceil(pos)
        return ordered[low] + (ordered[high]-ordered[low])*(pos-low)
    return {"p50_ms": percentile(.5), "p95_ms": percentile(.95), "samples": len(values)}


def versions():
    result = {"python": platform.python_version()}
    result.update(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions() if dist.metadata.get("Name")))
    return result


def run_evaluation(dataset_path=None, *, corpus_dir=None, mode="hybrid", backend="offline",
                   judge=False, output_dir=None, candidate_k=10, k=5):
    if mode not in MODES or backend not in {"offline", "google", "local"}:
        raise ValueError("Unknown evaluation configuration")
    if k <= 0 or candidate_k < k:
        raise ValueError("Require candidate_k >= k > 0")
    dataset_path = Path(dataset_path or ROOT / "eval/golden_dataset.jsonl")
    corpus_dir = Path(corpus_dir or ROOT / "data/sample_docs")
    triples = load_golden_dataset(str(dataset_path))
    start = perf_counter()
    chunks, doc_count, corpus_hash = load_corpus(corpus_dir)
    known_ids = {d.metadata["doc_id"] for d in chunks}
    unknown = {i for t in triples for i in t.expected_relevant_doc_ids} - known_ids
    if unknown:
        raise ValueError(f"Unknown relevance IDs: {sorted(unknown)}")
    report = {
        "schema_version": 2, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "completed", "total_queries": len(triples),
        "dataset": {"path": str(dataset_path), "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                    "corpus_sha256": corpus_hash, "documents": doc_count, "chunks": len(chunks),
                    "label_provenance": "See eval/METHODOLOGY.md; labels are not asserted manually verified"},
        "config": {"mode": mode, "backend": backend, "candidate_k": candidate_k, "k": k,
                   "chunk_size": settings.CHUNK_SIZE, "chunk_overlap": settings.CHUNK_OVERLAP,
                   "chunk_strategy": settings.CHUNKER_STRATEGY, "fusion_weights": [.6, .4], "rrf_constant": 60,
                   "embedding_model": {"offline": "sha256-lexical-512-v1", "google": settings.EMBEDDING_MODEL,
                                       "local": "BAAI/bge-small-en-v1.5"}[backend],
                   "reranker_model": settings.RERANKER_MODEL if mode == "hybrid-rerank" else None},
        "versions": versions(), "judge_status": "skipped" if not judge else "pending",
        "usage": {"tokens": None, "cost_usd": None, "status": "not measured"},
        "metrics": {}, "details": [], "latency_ms": {},
    }
    latencies = {"retrieve": [], "rerank": [], "generate": [], "judge": [], "query_total": []}
    try:
        if backend == "offline":
            from src.evaluation.offline import HashEmbeddings
            embeddings = HashEmbeddings()
        elif backend == "google":
            import os
            key = settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Google embeddings unavailable: missing credentials")
            from src.ingestion.embedder import get_embeddings_model
            embeddings = get_embeddings_model(api_key=key)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        # Explicit fresh in-memory instance: no Qdrant or development collection access.
        store = get_vector_store(embeddings=embeddings, use_in_memory=True)
        store.add_documents(chunks, ids=[d.metadata["chunk_id"] for d in chunks])
        retriever = build_retriever(store, chunks, top_k=candidate_k, mode="hybrid" if mode == "hybrid-rerank" else mode)
        reranker = None
        if mode == "hybrid-rerank":
            if backend == "offline":
                raise RuntimeError("Real cross-encoder unavailable in deterministic offline mode; use --backend local or google")
            from src.retrieval.reranker import build_reranked_retriever
            reranker = build_reranked_retriever(FixedRetriever(documents=[]), top_n=candidate_k, strict=True)
        judge_engine, chat = None, None
        if judge:
            import os
            key = (settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")) if settings.LLM_PROVIDER == "gemini" else (settings.CLAUDE_API_KEY or os.getenv("ANTHROPIC_API_KEY"))
            if not key:
                report["judge_status"] = "unavailable"
                report["judge_error"] = "Missing generation/judge credentials"
            else:
                from src.generation.generator import get_chat_model
                from src.evaluation.judge import LLMJudge
                chat = get_chat_model()
                judge_engine = LLMJudge(chat_model=chat)
                report["judge_status"] = "completed"
                report["config"]["judge_model"] = settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else settings.CLAUDE_MODEL
        report["latency_ms"]["setup"] = (perf_counter()-start)*1000
        for triple in triples:
            row = {"query_id": triple.query_id, "query": triple.query, "expected_docs": triple.expected_relevant_doc_ids,
                   "retrieved_docs": [], "retrieved_chunks": [], "status": "completed", "latency_ms": {}}
            query_start = perf_counter()
            try:
                stage_start = perf_counter()
                docs = retriever.invoke(triple.query)
                row["latency_ms"]["retrieve"] = (perf_counter()-stage_start)*1000
                row["latency_ms"].update(getattr(retriever, "last_latency_ms", {}))
                if reranker:
                    stage_start = perf_counter()
                    docs = list(reranker.base_compressor.compress_documents(docs, triple.query))
                    row["latency_ms"]["rerank"] = (perf_counter()-stage_start)*1000
                row["retrieved_chunks"] = [d.metadata["chunk_id"] for d in docs]
                # Document-level judgments: collapse repeated chunks before assigning ranks.
                row["retrieved_docs"] = list(dict.fromkeys(d.metadata["doc_id"] for d in docs))
                row["metrics"] = compute_all_retrieval_metrics(row["retrieved_docs"][:k], triple.expected_relevant_doc_ids, k)
                row["missed_docs"] = sorted(set(triple.expected_relevant_doc_ids)-set(row["retrieved_docs"][:k]))
                if judge_engine:
                    from src.generation.prompt_builder import build_chat_prompt_template
                    context = "\n\n".join(d.page_content for d in docs[:k])
                    stage_start = perf_counter()
                    answer = (build_chat_prompt_template() | chat).invoke({"query": triple.query, "context": context})
                    row["latency_ms"]["generate"] = (perf_counter()-stage_start)*1000
                    stage_start = perf_counter()
                    result = judge_engine.evaluate(triple.query, context, str(answer.content))
                    row["latency_ms"]["judge"] = (perf_counter()-stage_start)*1000
                    row["judge_score"] = result.model_dump()
                    row["provider_usage"] = {"generation": getattr(answer, "usage_metadata", None),
                                             "judge": judge_engine.last_usage_metadata}
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                report["status"] = "failed"
                # Failures stay in the denominator; no survivorship-biased averages.
                row.setdefault("metrics", compute_all_retrieval_metrics([], triple.expected_relevant_doc_ids, k))
                if judge_engine:
                    report["judge_status"] = "failed"
            row["latency_ms"]["query_total"] = (perf_counter()-query_start)*1000
            for stage, elapsed in row["latency_ms"].items():
                latencies.setdefault(stage, []).append(elapsed)
            report["details"].append(row)
        keys = compute_all_retrieval_metrics([], [], k)
        report["metrics"] = {name: sum(row["metrics"][name] for row in report["details"])/len(triples) for name in keys}
        if report["judge_status"] == "completed":
            for name in ("faithfulness", "answer_relevancy", "context_precision"):
                report["metrics"]["mean_"+name] = sum(row["judge_score"][name] for row in report["details"])/len(triples)
    except Exception as exc:
        report.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        if report["judge_status"] == "pending":
            report["judge_status"] = "unavailable"
    usages = [usage for row in report["details"] for usage in row.get("provider_usage", {}).values() if usage]
    if usages:
        report["usage"] = {"tokens": {name: sum(usage.get(name, 0) for usage in usages)
                                       for name in ("input_tokens", "output_tokens", "total_tokens")},
                           "responses_with_usage": len(usages), "cost_usd": None,
                           "status": "provider-reported generation/judge tokens; embedding tokens and cost not measured"}
    report["latency_ms"].update({stage: percentiles(values) for stage, values in latencies.items()})
    output = Path(output_dir or ROOT / "eval/reports")
    output.mkdir(parents=True, exist_ok=True)
    filename = output / ("eval_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")+"_"+mode+".json")
    filename.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(filename), "status": report["status"], "judge_status": report["judge_status"], "metrics": report["metrics"]}))
    return report


def comparison_identity(report):
    return {"schema": report.get("schema_version"), "dataset": report.get("dataset", {}).get("sha256"),
            "corpus": report.get("dataset", {}).get("corpus_sha256"), "config": report.get("config"),
            "versions": report.get("versions")}


def compare_with_baseline(current_metrics, baseline_path="eval/baseline_metrics.json"):
    path = Path(baseline_path)
    if not path.exists():
        return False
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if current_metrics.get("status") != "completed" or baseline.get("status") != "completed":
        return False
    if comparison_identity(current_metrics) != comparison_identity(baseline):
        print("Incompatible baseline: corpus, dataset, dependencies or configuration changed")
        return False
    required = ("recall_at_5", "mrr", "ndcg_at_5")
    return all(name in current_metrics["metrics"] and name in baseline["metrics"]
               and current_metrics["metrics"][name] >= baseline["metrics"][name]-settings.MAX_RECALL_DROP_TOLERANCE
               for name in required) and current_metrics["metrics"]["recall_at_5"] >= settings.MIN_RECALL_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--corpus")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--backend", choices=("offline", "google", "local"), default="offline")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--baseline")
    parser.add_argument("--accept-baseline", action="store_true")
    parser.add_argument("--min-recall", type=float, default=settings.MIN_RECALL_THRESHOLD)
    args = parser.parse_args()
    result = run_evaluation(args.dataset, corpus_dir=args.corpus, mode=args.mode, backend=args.backend,
                            judge=args.judge, output_dir=args.output_dir)
    if result["status"] != "completed" or (args.judge and result["judge_status"] != "completed"):
        raise SystemExit(2)
    if result["metrics"]["recall_at_5"] < args.min_recall:
        raise SystemExit(1)
    if args.judge and result["metrics"].get("mean_faithfulness", 0) < settings.MIN_FAITHFULNESS_THRESHOLD:
        raise SystemExit(1)
    if args.accept_baseline:
        Path(args.baseline or "eval/baseline_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.baseline and not compare_with_baseline(result, args.baseline):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Deterministic local retrieval benchmark.

This measures this process on this machine. It is not a production capacity
claim: the vector store is in-memory, embeddings are lexical hashes, and the
client and server share a CPU. Use the JSON output to detect regressions in the
same environment, not to promise internet-scale throughput.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from src.config import settings
from src.evaluation.offline import HashEmbeddings
from src.retrieval.index_manager import IndexManager
from src.retrieval.retrieval_runtime import RetrievalRuntime

WORDS = (
    "orion lyra nova atlas rigel vega retention export backup delete timeout auth policy "
    "tenant audit log archive recovery grace certificate rotation database network recipe telescope"
).split()


def make_chunks(count: int) -> List[Document]:
    chunks = []
    for index in range(count):
        product = WORDS[index % 12]
        tail = " ".join(WORDS[(index + offset) % len(WORDS)] for offset in range(3, 23))
        text = f"{product} policy section {index}. {product} guidance: {tail} requirement {index}."
        chunks.append(
            Document(
                page_content=text,
                metadata={
                    "chunk_id": f"bench::{index}",
                    "doc_id": f"doc-{index // 10}",
                    "tenant_id": "default",
                    "version_state": "active",
                },
            )
        )
    return chunks


def percentile(values: List[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    pos = (len(ordered) - 1) * p
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "samples": len(values),
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "p99_ms": round(percentile(values, 0.99), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def versions() -> Dict[str, str]:
    result = {"python": platform.python_version()}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name and name.lower() in {"langchain-core", "rank-bm25", "numpy", "pytest"}:
            result[name] = dist.version
    return dict(sorted(result.items()))


def run(mode: str, chunks_count: int, query_count: int, concurrencies: List[int]) -> Dict:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tracemalloc.start()
    total_start = time.perf_counter()
    chunks = make_chunks(chunks_count)
    embeddings = HashEmbeddings()
    store = InMemoryVectorStore(embedding=embeddings)
    embed_start = time.perf_counter()
    store.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])
    embed_ms = (time.perf_counter() - embed_start) * 1000

    runtime = RetrievalRuntime(IndexManager(max_chunks=max(chunks_count + 1, settings.MAX_INDEX_CHUNKS)))
    index_start = time.perf_counter()
    snapshot = runtime.refresh_from_store(store)
    index_ms = (time.perf_counter() - index_start) * 1000

    queries = [f"policy requirement {index % 97} retention export" for index in range(query_count)]
    for query in queries[:3]:
        runtime.search(store, query, top_k=10, mode=mode)

    scenarios = []
    for concurrency in concurrencies:
        latencies: List[float] = []
        errors = 0

        def worker(query: str) -> float:
            started = time.perf_counter()
            result = runtime.search(store, query, top_k=10, mode=mode)
            assert result.documents
            return (time.perf_counter() - started) * 1000

        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(worker, query) for query in queries]
            for future in futures:
                try:
                    latencies.append(future.result())
                except Exception:  # noqa: BLE001 - counted, never hidden
                    errors += 1
        wall_seconds = time.perf_counter() - wall_start
        scenarios.append(
            {
                "concurrency": concurrency,
                "queries": len(queries),
                "errors": errors,
                "throughput_qps": round(len(queries) / wall_seconds, 3) if wall_seconds else None,
                "latency": summarize(latencies) if latencies else None,
            }
        )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": mode,
        "machine": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()},
        "config": {
            "chunks": chunks_count,
            "documents": len({chunk.metadata["doc_id"] for chunk in chunks}),
            "queries_per_scenario": query_count,
            "top_k": 10,
            "concurrencies": concurrencies,
            "embedding": "sha256-lexical-512-v1",
            "vector_store": "langchain InMemoryVectorStore",
            "index_version": snapshot.version,
            "corpus_fingerprint": snapshot.corpus_fingerprint,
            "corpus_fingerprint_sha256": hashlib.sha256(snapshot.corpus_fingerprint.encode()).hexdigest(),
            "fusion_weights": [settings.FUSION_DENSE_WEIGHT, settings.FUSION_SPARSE_WEIGHT],
            "rrf_constant": settings.FUSION_RRF_K,
        },
        "setup": {
            "embed_and_write_ms": round(embed_ms, 3),
            "index_build_ms": round(index_ms, 3),
            "total_setup_ms": round((time.perf_counter() - total_start) * 1000, 3),
        },
        "memory": {
            "tracemalloc_current_mb": round(current / 1024 / 1024, 3),
            "tracemalloc_peak_mb": round(peak / 1024 / 1024, 3),
        },
        "scenarios": scenarios,
        "versions": versions(),
        "limitations": [
            "Single-process, in-memory store; no network, Qdrant, embedding provider or LLM.",
            "Hash embeddings are lexical, not semantic; do not read this as model-quality evidence.",
            "Latency is machine-dependent and includes Python thread scheduling overhead.",
            "No sustained load, memory leak profile, multi-tenant isolation or distributed test.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--concurrency", default="1,5,10")
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    concurrencies = [int(value) for value in args.concurrency.split(",") if value.strip()]
    result = run(args.mode, args.chunks, args.queries, concurrencies)
    output = Path(args.output) if args.output else ROOT / "benchmarks/results" / f"retrieval_{args.mode}_{args.chunks}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "mode": args.mode, "scenarios": result["scenarios"]}, indent=2))


if __name__ == "__main__":
    main()

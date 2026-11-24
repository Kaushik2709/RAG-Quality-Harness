"""Deterministic end-to-end performance benchmarking and concurrency validation.

Measures:
- Cold start vs warm query latency.
- Concurrency scaling across 1, 5, 10, and 25 workers.
- Corpus sizes: Small (1,000), Medium (10,000), Large (50,000 chunks).
- Peak memory during sparse index construction.
- Snapshot memory and throughput curves (QPS).
- Versioned, structured JSON reports with environment provenance.
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
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    "tenant audit log archive recovery grace certificate rotation database network recipe telescope "
    "encryption compliance jurisdiction latency failover replication snapshot parameter"
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
    if not values:
        return {}
    return {
        "samples": len(values),
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "p99_ms": round(percentile(values, 0.99), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "std_dev_ms": round(statistics.stdev(values) if len(values) > 1 else 0.0, 3),
    }


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def get_environment_info() -> Dict[str, Any]:
    deps = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name and name.lower() in {
            "langchain-core", "rank-bm25", "numpy", "pytest", "pydantic", "fastapi"
        }:
            deps[name] = dist.version

    return {
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
        "dependencies": dict(sorted(deps.items())),
    }


def run_benchmark(
    mode: str = "hybrid",
    chunks_count: int = 10_000,
    query_count: int = 40,
    concurrencies: Optional[List[int]] = None,
) -> Dict[str, Any]:
    concurrencies = concurrencies or [1, 5, 10, 25]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    tracemalloc.start()
    total_start = time.perf_counter()

    # 1. Corpus generation & embedding
    chunks = make_chunks(chunks_count)
    embeddings = HashEmbeddings()
    store = InMemoryVectorStore(embedding=embeddings)

    embed_start = time.perf_counter()
    store.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])
    embed_ms = (time.perf_counter() - embed_start) * 1000

    # 2. Sparse Index Construction & Memory Profiling
    tracemalloc.reset_peak()
    runtime = RetrievalRuntime(IndexManager(max_chunks=max(chunks_count + 1, settings.MAX_INDEX_CHUNKS)))
    index_start = time.perf_counter()
    snapshot = runtime.refresh_from_store(store)
    index_ms = (time.perf_counter() - index_start) * 1000
    _, index_peak_bytes = tracemalloc.get_traced_memory()
    index_build_peak_mb = round(index_peak_bytes / 1024 / 1024, 3)

    queries = [f"policy requirement {index % 97} retention export security timeout" for index in range(query_count)]

    # 3. Cold Start vs Warm Measurement
    cold_start_t = time.perf_counter()
    cold_res = runtime.search(store, queries[0], top_k=10, mode=mode)
    cold_latency_ms = round((time.perf_counter() - cold_start_t) * 1000, 3)

    warm_latencies = []
    for q in queries[1:11]:
        t0 = time.perf_counter()
        runtime.search(store, q, top_k=10, mode=mode)
        warm_latencies.append((time.perf_counter() - t0) * 1000)
    warm_mean_ms = round(statistics.mean(warm_latencies), 3) if warm_latencies else cold_latency_ms

    # 4. Concurrency Scenarios
    scenarios = []
    baseline_qps = 1.0

    for idx, concurrency in enumerate(concurrencies):
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
                except Exception:
                    errors += 1
        wall_seconds = time.perf_counter() - wall_start
        throughput_qps = round(len(queries) / wall_seconds, 2) if wall_seconds > 0 else 0.0

        if concurrency == 1:
            baseline_qps = throughput_qps

        speedup_factor = round(throughput_qps / baseline_qps, 2) if baseline_qps > 0 else 1.0

        scenarios.append(
            {
                "concurrency": concurrency,
                "queries": len(queries),
                "errors": errors,
                "throughput_qps": throughput_qps,
                "speedup_vs_single_worker": speedup_factor,
                "latency": summarize(latencies) if latencies else None,
                "raw_latencies_ms": [round(x, 2) for x in latencies],
            }
        )

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time_ms = round((time.perf_counter() - total_start) * 1000, 3)

    return {
        "benchmark_id": f"retrieval_bench_{mode}_{chunks_count}chunks",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": mode,
        "environment": get_environment_info(),
        "config": {
            "chunks": chunks_count,
            "chunks_count": chunks_count,
            "documents_count": len({chunk.metadata["doc_id"] for chunk in chunks}),
            "queries_per_scenario": query_count,
            "top_k": 10,
            "concurrencies": concurrencies,
            "embedding": "sha256-lexical-512-v1",
            "vector_store": "langchain InMemoryVectorStore",
            "index_version": snapshot.version,
            "corpus_fingerprint": snapshot.corpus_fingerprint,
            "fusion_weights": [settings.FUSION_DENSE_WEIGHT, settings.FUSION_SPARSE_WEIGHT],
            "rrf_k": settings.FUSION_RRF_K,
        },
        "performance_summary": {
            "cold_start_latency_ms": cold_latency_ms,
            "warm_mean_latency_ms": warm_mean_ms,
            "cold_to_warm_ratio": round(cold_latency_ms / warm_mean_ms, 2) if warm_mean_ms > 0 else 1.0,
            "embed_and_stage_ms": round(embed_ms, 3),
            "index_build_ms": round(index_ms, 3),
            "total_execution_ms": total_time_ms,
        },
        "memory_profile": {
            "sparse_index_build_peak_mb": index_build_peak_mb,
            "total_benchmark_peak_mb": round(peak_mem / 1024 / 1024, 3),
            "resident_final_mb": round(current_mem / 1024 / 1024, 3),
        },
        "concurrency_scenarios": scenarios,
        "scenarios": scenarios,
        "limitations": [
            "Deterministic offline benchmark using in-memory vector store and lexical embeddings.",
            "Execution is single-node, CPU-bound; thread speedup is bounded by Python GIL.",
            "Measures algorithmic retrieval complexity and concurrency scaling, not network/Qdrant I/O.",
        ],
    }


run = run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Performance & Concurrency Benchmark")
    parser.add_argument("--chunks", type=int, default=10_000, help="Number of chunks (e.g. 1000, 10000, 50000)")
    parser.add_argument("--queries", type=int, default=40, help="Queries per scenario")
    parser.add_argument("--concurrency", default="1,5,10,25", help="Comma-separated concurrency workers")
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    concurrencies = [int(v.strip()) for v in args.concurrency.split(",") if v.strip()]
    result = run_benchmark(
        mode=args.mode,
        chunks_count=args.chunks,
        query_count=args.queries,
        concurrencies=concurrencies,
    )

    out_dir = ROOT / "benchmarks/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = result["environment"]["git_commit"][:7]
    filename = f"benchmark_{args.mode}_{args.chunks}chunks_{commit}.json"
    output_path = Path(args.output) if args.output else (out_dir / filename)

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[Benchmark Complete] Saved artifact to {output_path}")
    print("\n--- PERFORMANCE & CONCURRENCY SUMMARY ---")
    print(f"Mode: {args.mode} | Chunks: {args.chunks} | Cold Start: {result['performance_summary']['cold_start_latency_ms']} ms | Warm Mean: {result['performance_summary']['warm_mean_latency_ms']} ms")
    print(f"Index Build Peak Memory: {result['memory_profile']['sparse_index_build_peak_mb']} MB")
    print(f"{'Concurrency':<12} {'QPS':<10} {'Speedup':<10} {'p50 (ms)':<10} {'p95 (ms)':<10}")
    print("-" * 55)
    for sc in result["concurrency_scenarios"]:
        lat = sc.get("latency", {})
        print(f"{sc['concurrency']:<12} {sc['throughput_qps']:<10.1f} {sc['speedup_vs_single_worker']:<10.2f} {lat.get('p50_ms', 0):<10.2f} {lat.get('p95_ms', 0):<10.2f}")


if __name__ == "__main__":
    main()

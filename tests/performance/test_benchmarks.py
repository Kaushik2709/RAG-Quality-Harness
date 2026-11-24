"""Smoke test for benchmark suite: verifies metrics, percentiles, and timing harness."""
import pytest
from benchmarks.retrieval_benchmark import make_chunks, percentile, run, summarize


def test_percentile_and_summarize():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    p50 = percentile(values, 0.50)
    assert p50 == pytest.approx(30.0)

    summary = summarize(values)
    assert summary["samples"] == 5
    assert summary["mean_ms"] == 30.0
    assert summary["p50_ms"] == 30.0


def test_benchmark_run_smoke():
    # Miniature benchmark with 20 chunks, 4 queries, concurrency 1 and 2
    res = run(mode="bm25", chunks_count=20, query_count=4, concurrencies=[1, 2])
    assert res["mode"] == "bm25"
    assert res["config"]["chunks"] == 20
    assert len(res["scenarios"]) == 2

    c1 = res["scenarios"][0]
    assert c1["concurrency"] == 1
    assert c1["errors"] == 0
    assert c1["throughput_qps"] > 0.0
    assert "latency" in c1
    assert c1["latency"]["p50_ms"] >= 0.0

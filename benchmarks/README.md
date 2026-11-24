# Performance benchmarks

`retrieval_benchmark.py` measures the request-path retrieval runtime on a
deterministic 10,000-chunk in-memory corpus at concurrency 1, 5 and 10.

```bash
python benchmarks/retrieval_benchmark.py --chunks 10000 --queries 30 --mode hybrid
```

What it actually measures:

- SHA-256 lexical hash embeddings (`sha256-lexical-512-v1`), not a semantic model.
- A LangChain `InMemoryVectorStore` in the same process as the client.
- Index construction, per-query p50/p95/p99, throughput, errors and
  `tracemalloc` peak memory.

What it does **not** measure or promise:

- Qdrant/network latency, provider embedding calls, LLM generation, reranking.
- Semantic retrieval quality.
- Sustained load, multi-process contention, leak-free operation, or
  production capacity. The workload is deliberately short.

Committed results live in `benchmarks/results/`. Re-run on the same machine and
configuration before comparing numbers; a different CPU, Python version or
dependency version invalidates a direct comparison. Results are never merged
into evaluation quality metrics.

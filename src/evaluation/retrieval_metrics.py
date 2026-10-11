"""Exact-ID binary metrics; duplicates/missing IDs consume rank without gain.
Precision divides by K even for short lists. Empty labels/nonpositive K score zero.
"""
import math


def _relevant(ids):
    return {x for x in ids if isinstance(x, str) and x.strip()}


def _gains(retrieved_ids, expected_ids):
    expected, seen = _relevant(expected_ids), set()
    for item in retrieved_ids:
        hit = isinstance(item, str) and bool(item.strip()) and item in expected and item not in seen
        if isinstance(item, str):
            seen.add(item)
        yield int(hit)


def recall_at_k(retrieved_ids, expected_ids, k=5):
    expected = _relevant(expected_ids)
    return sum(_gains(retrieved_ids[:max(k, 0)], expected)) / len(expected) if expected and k > 0 else 0.0


def precision_at_k(retrieved_ids, expected_ids, k=5):
    return sum(_gains(retrieved_ids[:k], expected_ids)) / k if k > 0 else 0.0


def mrr(retrieved_ids, expected_ids):
    return next((1 / rank for rank, gain in enumerate(_gains(retrieved_ids, expected_ids), 1) if gain), 0.0)


def ndcg_at_k(retrieved_ids, expected_ids, k=5):
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(_relevant(expected_ids)), max(k, 0)) + 1))
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(_gains(retrieved_ids[:max(k, 0)], expected_ids), 1))
    return dcg / ideal if ideal else 0.0


def compute_all_retrieval_metrics(retrieved_ids, expected_ids, k=5):
    return {f"recall_at_{k}": recall_at_k(retrieved_ids, expected_ids, k),
            f"precision_at_{k}": precision_at_k(retrieved_ids, expected_ids, k),
            "mrr": mrr(retrieved_ids, expected_ids),
            f"ndcg_at_{k}": ndcg_at_k(retrieved_ids, expected_ids, k)}

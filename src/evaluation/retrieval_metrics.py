"""Exact-ID binary and graded retrieval evaluation metrics.

Formulas & Mathematical Definitions:
-------------------------------------
1. Precision@K:
   Precision@K = (1 / K) * sum_{i=1}^K rel_i
   where rel_i in {0, 1} is 1 if retrieved candidate at rank i is relevant, and 0 otherwise.

2. Recall@K:
   Recall@K = (1 / |Relevant|) * sum_{i=1}^K rel_i
   where |Relevant| is the count of ground truth relevant items for the query.

3. MRR@K (Mean Reciprocal Rank):
   RR = 1 / rank_first
   where rank_first is the 1-based rank of the first relevant item retrieved, or 0 if none in top K.

4. nDCG@K (Normalized Discounted Cumulative Gain):
   DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
   IDCG@K = sum_{i=1}^{min(|Relevant|, K)} 1 / log2(i + 1)
   nDCG@K = DCG@K / IDCG@K

5. Average Precision@K (AP@K) and MAP@K:
   AP@K = (1 / min(|Relevant|, K)) * sum_{i=1}^K [Precision@i * rel_i]
   MAP@K is the mean of AP@K across all evaluated queries.

6. Document-Level Recall@K:
   Maps retrieved chunk IDs to document IDs, deduplicating within top K:
   DocRecall@K = |{doc(c) : c in Retrieved[:K]} cap RelevantDocs| / |RelevantDocs|

7. Chunk-Level Recall@K:
   Direct exact match between retrieved chunk IDs and expected chunk IDs:
   ChunkRecall@K = |Retrieved[:K] cap RelevantChunks| / |RelevantChunks|
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set


def _relevant(ids: Sequence[Any]) -> Set[str]:
    return {str(x).strip() for x in ids if isinstance(x, (str, int)) and str(x).strip()}


def _gains(retrieved_ids: Sequence[Any], expected_ids: Sequence[Any]):
    expected = _relevant(expected_ids)
    seen = set()
    for item in retrieved_ids:
        s_item = str(item).strip() if item is not None else ""
        hit = bool(s_item) and s_item in expected and s_item not in seen
        if s_item:
            seen.add(s_item)
        yield int(hit)


def recall_at_k(retrieved_ids: Sequence[Any], expected_ids: Sequence[Any], k: int = 5) -> float:
    """Computes Recall@K: fraction of relevant items retrieved in top K."""
    expected = _relevant(expected_ids)
    if not expected or k <= 0:
        return 0.0
    return sum(_gains(retrieved_ids[:max(k, 0)], expected)) / len(expected)


def precision_at_k(retrieved_ids: Sequence[Any], expected_ids: Sequence[Any], k: int = 5) -> float:
    """Computes Precision@K: fraction of top K retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    return sum(_gains(retrieved_ids[:k], expected_ids)) / k


def mrr(retrieved_ids: Sequence[Any], expected_ids: Sequence[Any], k: Optional[int] = None) -> float:
    """Computes Reciprocal Rank (RR): reciprocal of the rank of the first relevant item."""
    cands = retrieved_ids[:k] if k is not None and k > 0 else retrieved_ids
    return next((1.0 / rank for rank, gain in enumerate(_gains(cands, expected_ids), 1) if gain), 0.0)


def ndcg_at_k(retrieved_ids: Sequence[Any], expected_ids: Sequence[Any], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    expected = _relevant(expected_ids)
    if not expected or k <= 0:
        return 0.0
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(expected), max(k, 0)) + 1))
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(_gains(retrieved_ids[:max(k, 0)], expected), 1))
    return dcg / ideal if ideal else 0.0


def average_precision_at_k(retrieved_ids: Sequence[Any], expected_ids: Sequence[Any], k: int = 5) -> float:
    """AP@K averaged over queries by the caller to obtain MAP@K."""
    expected = _relevant(expected_ids)
    if not expected or k <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, gain in enumerate(_gains(retrieved_ids[:k], expected), 1):
        if gain:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(expected), k)


def hit_rate_at_k(retrieved_ids: Sequence[Any], expected_ids: Sequence[Any], k: int = 5) -> float:
    """Computes HitRate@K (1.0 if at least one relevant item in top K, else 0.0)."""
    if k <= 0 or not _relevant(expected_ids):
        return 0.0
    return 1.0 if any(_gains(retrieved_ids[:k], expected_ids)) else 0.0


def retrieval_empty(retrieved_ids: Sequence[Any]) -> float:
    return 1.0 if not [item for item in (retrieved_ids or []) if item is not None and str(item).strip()] else 0.0


def document_recall_at_k(
    retrieved_chunk_ids: Sequence[Any],
    expected_doc_ids: Sequence[Any],
    k: int = 5,
) -> float:
    """Computes Document-level Recall@K by extracting document IDs from chunk IDs.

    Chunks formatted as '{doc_id}::c{idx}::{digest}' map to '{doc_id}'.
    """
    retrieved_doc_ids = []
    seen = set()
    for cid in retrieved_chunk_ids[:k]:
        if cid is None:
            continue
        s_cid = str(cid).strip()
        doc_id = s_cid.split("::")[0] if "::" in s_cid else s_cid
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            retrieved_doc_ids.append(doc_id)
    return recall_at_k(retrieved_doc_ids, expected_doc_ids, k=k)


def chunk_recall_at_k(
    retrieved_chunk_ids: Sequence[Any],
    expected_chunk_ids: Sequence[Any],
    k: int = 5,
) -> float:
    """Computes Chunk-level Recall@K matching exact chunk identifiers."""
    return recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k=k)


def compute_all_retrieval_metrics(retrieved_ids: Sequence[Any], expected_ids: Sequence[Any], k: int = 5) -> Dict[str, float]:
    return {
        f"recall_at_{k}": recall_at_k(retrieved_ids, expected_ids, k),
        f"precision_at_{k}": precision_at_k(retrieved_ids, expected_ids, k),
        "mrr": mrr(retrieved_ids, expected_ids, k=k),
        f"ndcg_at_{k}": ndcg_at_k(retrieved_ids, expected_ids, k),
        f"map_at_{k}": average_precision_at_k(retrieved_ids, expected_ids, k),
        f"hit_rate_at_{k}": hit_rate_at_k(retrieved_ids, expected_ids, k),
        f"doc_recall_at_{k}": document_recall_at_k(retrieved_ids, expected_ids, k),
        "retrieval_empty": retrieval_empty(retrieved_ids),
    }

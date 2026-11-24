"""Pure fusion logic, separated from storage and request handling.

Reciprocal Rank Fusion (RRF) is deliberately simple and score-scale free. Its
limitation is equally important: it only uses rank, so it cannot express
confidence differences between retrievers, and its constant controls how
quickly rank influence decays. Both are recorded in evaluation reports.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Tuple[float, Sequence[Any]]],
    *,
    id_getter=lambda item: item.metadata.get("chunk_id"),
    rrf_k: int = 60,
    limit: int | None = None,
) -> List[Any]:
    """Fuse ranked lists by weighted RRF with deterministic tie-breaking.

    Duplicate IDs within one ranked list count once, matching the previous
    implementation. Missing/empty IDs are skipped rather than being treated as
    a shared identity.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    fused: Dict[str, float] = {}
    items: Dict[str, Any] = {}
    for weight, ranked in ranked_lists:
        if weight < 0:
            raise ValueError("fusion weights must be non-negative")
        seen = set()
        for rank, item in enumerate(ranked, 1):
            key = id_getter(item)
            if not key or key in seen:
                continue
            seen.add(key)
            items[key] = item
            fused[key] = fused.get(key, 0.0) + weight / (rrf_k + rank)
    ordered = [items[key] for key in sorted(fused, key=lambda key: (-fused[key], key))]
    return ordered[:limit] if limit is not None else ordered


def fused_scores(ranked_lists, *, id_getter=lambda item: item.metadata.get("chunk_id"), rrf_k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for weight, ranked in ranked_lists:
        seen = set()
        for rank, item in enumerate(ranked, 1):
            key = id_getter(item)
            if not key or key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
    return scores

import math
from typing import List, Set, Dict

def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Recall@K: Proportion of expected relevant documents present in the top-K retrieved.
    """
    if not expected_ids:
        return 0.0
    k_retrieved = set(retrieved_ids[:k])
    expected_set = set(expected_ids)
    
    # Check both exact match and source substring match
    hits = 0
    for exp_id in expected_set:
        if any(exp_id in ret_id or ret_id in exp_id for ret_id in k_retrieved):
            hits += 1

    return float(hits) / float(len(expected_set))

def precision_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Precision@K: Proportion of top-K retrieved documents that are relevant.
    """
    if not retrieved_ids or k <= 0:
        return 0.0
    k_retrieved = retrieved_ids[:k]
    expected_set = set(expected_ids)

    hits = 0
    for ret_id in k_retrieved:
        if any(exp_id in ret_id or ret_id in exp_id for exp_id in expected_set):
            hits += 1

    return float(hits) / float(min(k, len(k_retrieved)))

def mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """
    Mean Reciprocal Rank (MRR): Reciprocal of the rank position of the first relevant document.
    """
    expected_set = set(expected_ids)
    for rank_idx, ret_id in enumerate(retrieved_ids, start=1):
        if any(exp_id in ret_id or ret_id in exp_id for exp_id in expected_set):
            return 1.0 / float(rank_idx)
    return 0.0

def ndcg_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG@K).
    Binary relevance: 1 if document in expected_ids, 0 otherwise.
    """
    if not expected_ids or not retrieved_ids or k <= 0:
        return 0.0

    k_retrieved = retrieved_ids[:k]
    expected_set = set(expected_ids)

    # Compute DCG@K
    dcg = 0.0
    for rank_idx, ret_id in enumerate(k_retrieved, start=1):
        is_rel = 1.0 if any(exp_id in ret_id or ret_id in exp_id for exp_id in expected_set) else 0.0
        dcg += is_rel / math.log2(rank_idx + 1)

    # Compute Ideal DCG@K (IDCG)
    idcg = 0.0
    ideal_hits = min(len(expected_set), k)
    for rank_idx in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank_idx + 1)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)

def compute_all_retrieval_metrics(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> Dict[str, float]:
    return {
        f"recall_at_{k}": round(recall_at_k(retrieved_ids, expected_ids, k=k), 4),
        f"precision_at_{k}": round(precision_at_k(retrieved_ids, expected_ids, k=k), 4),
        "mrr": round(mrr(retrieved_ids, expected_ids), 4),
        f"ndcg_at_{k}": round(ndcg_at_k(retrieved_ids, expected_ids, k=k), 4),
    }

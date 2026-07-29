import pytest
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    compute_all_retrieval_metrics
)

def test_recall_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    expected = ["doc_b", "doc_x"]
    # At K=2: retrieved are doc_a, doc_b. Expected are doc_b, doc_x. Hits = 1 / 2 = 0.5
    assert recall_at_k(retrieved, expected, k=2) == 0.5
    # At K=4: retrieved contains doc_b. Hits = 1 / 2 = 0.5
    assert recall_at_k(retrieved, expected, k=4) == 0.5

def test_precision_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    expected = ["doc_b", "doc_x"]
    # At K=2: retrieved are doc_a, doc_b. 1 out of 2 is relevant => 0.5
    assert precision_at_k(retrieved, expected, k=2) == 0.5
    # At K=4: 1 out of 4 is relevant => 0.25
    assert precision_at_k(retrieved, expected, k=4) == 0.25

def test_mrr():
    # First relevant item is at rank 2 ("doc_b") => 1/2 = 0.5
    retrieved = ["doc_a", "doc_b", "doc_c"]
    expected = ["doc_b"]
    assert mrr(retrieved, expected) == 0.5

    # First item is relevant => 1.0
    assert mrr(["doc_b", "doc_a"], expected) == 1.0

    # None relevant => 0.0
    assert mrr(["doc_x", "doc_y"], expected) == 0.0

def test_ndcg_at_k():
    retrieved = ["doc_b", "doc_a", "doc_c"]
    expected = ["doc_b"]
    # Document b is at rank 1. DCG@1 = 1 / log2(2) = 1.0, IDCG@1 = 1.0 => nDCG = 1.0
    assert ndcg_at_k(retrieved, expected, k=1) == 1.0

    # Empty expected => 0.0
    assert ndcg_at_k(retrieved, [], k=5) == 0.0

def test_compute_all():
    retrieved = ["doc_1", "doc_2", "doc_3"]
    expected = ["doc_1", "doc_3"]
    res = compute_all_retrieval_metrics(retrieved, expected, k=3)
    assert "recall_at_3" in res
    assert "precision_at_3" in res
    assert "mrr" in res
    assert "ndcg_at_3" in res
    assert res["mrr"] == 1.0

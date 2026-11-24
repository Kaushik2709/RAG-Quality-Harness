"""Unit tests for scientific evaluation, dataset integrity, and judge calibration."""
from __future__ import annotations

import pytest
from src.evaluation.answer_metrics import (
    claim_coverage,
    citation_precision,
    citation_recall,
    citation_quote_validity_rate,
    grounded_generation_metrics,
    numerical_consistency,
)
from src.evaluation.datasets import annotate_split, dataset_manifest
from src.evaluation.golden_dataset import GoldenTriple, load_golden_dataset
from src.evaluation.judge import (
    bootstrap_ci,
    calibration_report,
    cohens_kappa,
    detect_position_bias,
    detect_verbosity_bias,
    percent_agreement,
)
from src.evaluation.retrieval_metrics import (
    average_precision_at_k,
    chunk_recall_at_k,
    compute_all_retrieval_metrics,
    document_recall_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_human_annotated_queries_integrity_and_no_leakage():
    """Verify eval/human_annotated/queries.jsonl has zero leakage and consistent annotations."""
    triples = load_golden_dataset("eval/human_annotated/queries.jsonl")
    assert len(triples) == 10

    # human-001 and human-002 are paraphrases and must share the same split
    h1 = next(t for t in triples if t.query_id == "human-001")
    h2 = next(t for t in triples if t.query_id == "human-002")
    assert h1.metadata.get("split") == h2.metadata.get("split") == "dev"
    assert h2.metadata.get("paraphrase_of") == "human-001"

    # human-006 is unanswerable and must not have relevant doc IDs
    h6 = next(t for t in triples if t.query_id == "human-006")
    assert h6.answerable is False
    assert h6.expected_relevant_doc_ids == []
    assert h6.expected_abstention_reason == "ambiguous_query_unspecified_service"

    # All items are adjudicated
    assert all(t.metadata.get("annotation_status") == "adjudicated" for t in triples)


def test_leakage_resistant_cluster_splitting():
    """Queries sharing a cluster_id or paraphrase_of must be assigned to the same split."""
    triples = [
        GoldenTriple(query_id="q1", query="Query 1", expected_relevant_doc_ids=["d1"], reference_answer="a1", metadata={"cluster_id": "c1"}),
        GoldenTriple(query_id="q1_para", query="Query 1 paraphrase", expected_relevant_doc_ids=["d1"], reference_answer="a1", metadata={"paraphrase_of": "q1"}),
        GoldenTriple(query_id="q2", query="Query 2", expected_relevant_doc_ids=["d2"], reference_answer="a2", metadata={"cluster_id": "c2"}),
        GoldenTriple(query_id="q3", query="Query 3", expected_relevant_doc_ids=["d3"], reference_answer="a3", metadata={"cluster_id": "c3"}),
    ]
    split_triples = annotate_split(triples, dev_ratio=0.5, seed=42)
    s1 = next(t.metadata["split"] for t in split_triples if t.query_id == "q1")
    s1_para = next(t.metadata["split"] for t in split_triples if t.query_id == "q1_para")
    assert s1 == s1_para, "Paraphrases must share the same split to avoid data leakage"


def test_document_and_chunk_level_recall():
    """Verify document-level vs chunk-level recall calculations."""
    retrieved_chunks = ["docA::c0::123", "docA::c1::456", "docB::c0::789", "docC::c0::000"]
    expected_docs = ["docA", "docB", "docD"]
    expected_chunks = ["docA::c0::123", "docD::c0::999"]

    # In top 3: docA and docB retrieved -> 2 / 3
    doc_rec = document_recall_at_k(retrieved_chunks, expected_docs, k=3)
    assert doc_rec == pytest.approx(2 / 3)

    # In top 3: only docA::c0::123 is retrieved -> 1 / 2
    chunk_rec = chunk_recall_at_k(retrieved_chunks, expected_chunks, k=3)
    assert chunk_rec == pytest.approx(1 / 2)


def test_grounded_generation_metrics():
    """Verify claim coverage, citation precision/recall, and numerical consistency."""
    validation_dict = {
        "accepted": [{"chunk_id": "c1", "quote": "30 days retention"}],
        "rejected": [{"chunk_id": "c2", "quote": "invalid quote"}],
        "claims": [
            {"claim_id": "cl1", "supported": True},
            {"claim_id": "cl2", "supported": False},
        ],
        "citation_recall": 0.5,
    }
    metrics = grounded_generation_metrics(
        validation_dict,
        reference_answer="Logs are kept for 30 days and 24 hours.",
        generated_answer="Logs are retained for 30 days.",
    )
    assert metrics["claim_coverage"] == 0.5  # 1 / 2 claims supported
    assert metrics["citation_precision"] == 0.5  # 1 / 2 citations accepted
    assert metrics["quote_validity_rate"] == 0.5
    # Numerical consistency: reference has 30 and 24, answer has 30 -> 1 / 2
    assert metrics["numerical_consistency"] == 0.5


def test_judge_calibration_cohens_kappa_and_bias():
    """Verify Cohen's Kappa, percent agreement, position bias, and verbosity bias."""
    rater1 = [1, 1, 0, 0, 1, 1, 0, 1]
    rater2 = [1, 1, 0, 0, 1, 0, 0, 1]  # 7 / 8 match
    assert percent_agreement(rater1, rater2) == 0.875
    kappa = cohens_kappa(rater1, rater2)
    assert 0.70 <= kappa <= 1.0

    # Position bias
    pos_a = [0.9, 0.95, 0.85]
    pos_b = [0.7, 0.75, 0.65]
    bias = detect_position_bias(pos_a, pos_b)
    assert bias["position_bias_delta"] == pytest.approx(0.20)

    # Verbosity bias
    scores = [0.5, 0.7, 0.9, 0.95]
    lengths = [10, 50, 150, 300]
    v_bias = detect_verbosity_bias(scores, lengths)
    assert v_bias["status"] == "computed"
    assert v_bias["correlation"] > 0.8

    # Bootstrap CI
    ci = bootstrap_ci([0.8, 0.85, 0.9, 0.95, 0.88], samples=200, seed=123)
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

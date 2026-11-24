"""Evaluation statistics, answer metrics, manifests and diagnostics."""
from __future__ import annotations

import json

import pytest

from src.evaluation.answer_metrics import answer_metrics, exact_match, token_f1
from src.evaluation.datasets import (
    CATEGORIES,
    annotate_split,
    dataset_manifest,
    filter_split,
    validate_dataset,
)
from src.evaluation.golden_dataset import GoldenTriple
from src.evaluation.regression_runner import run_evaluation
from src.evaluation.retrieval_metrics import (
    average_precision_at_k,
    compute_all_retrieval_metrics,
    hit_rate_at_k,
    retrieval_empty,
)
from src.evaluation.statistical_tests import bootstrap_ci, cohens_d, paired_bootstrap_difference


def test_map_hit_rate_and_empty_detection():
    retrieved = ["x", "a", "b"]
    expected = ["a", "b"]
    assert average_precision_at_k(retrieved, expected, 3) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert hit_rate_at_k(retrieved, expected, 3) == 1.0
    assert hit_rate_at_k(["x"], expected, 3) == 0.0
    assert retrieval_empty([]) == 1.0
    assert retrieval_empty([None, ""]) == 1.0
    assert retrieval_empty(["a"]) == 0.0
    metrics = compute_all_retrieval_metrics(retrieved, expected, 3)
    assert metrics["map_at_3"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert metrics["retrieval_empty"] == 0.0


def test_answer_lexical_metrics_are_explicit_about_their_limits():
    assert exact_match("30 days", " 30 days. ") == 1.0
    assert token_f1("Orion retains logs for 30 days", "Orion keeps logs for 30 days") > 0.5
    assert token_f1("", "anything") == 0.0
    result = answer_metrics("Orion retains logs for 30 days", "Orion logs: 30 days")
    assert result["contains_reference"] == 0.0
    assert result["token_f1"] > 0


def test_bootstrap_ci_and_paired_difference_are_deterministic():
    values = [1.0, 0.0, 1.0, 0.5, 1.0]
    first = bootstrap_ci(values, samples=200, seed=7)
    second = bootstrap_ci(values, samples=200, seed=7)
    assert first == second
    assert first["lower"] <= first["mean"] <= first["upper"]
    paired = paired_bootstrap_difference([0, 1, 1, 0], [1, 1, 1, 1], samples=200, seed=7)
    assert paired["mean_difference"] == 0.5
    assert 0 <= paired["p_value_two_sided"] <= 1
    with pytest.raises(ValueError):
        paired_bootstrap_difference([1], [1, 2])
    with pytest.raises(ValueError):
        bootstrap_ci([])
    assert cohens_d([0, 1, 0, 1], [2, 3, 3, 2]) > 1


def _triples():
    return [
        GoldenTriple(
            query_id="direct",
            query="q1",
            expected_relevant_doc_ids=["d1"],
            reference_answer="a1",
            metadata={"category": "direct", "annotation_status": "unverified", "split": "dev"},
        ),
        GoldenTriple(
            query_id="compare",
            query="q2",
            expected_relevant_doc_ids=["d1", "d2"],
            reference_answer="a2",
            metadata={"category": "near-neighbor", "annotation_status": "unverified", "split": "test"},
        ),
        GoldenTriple(
            query_id="no-answer",
            query="q3",
            expected_relevant_doc_ids=[],
            reference_answer="no answer",
            answerable=False,
            metadata={"category": "no_answer", "annotation_status": "unverified", "split": "test"},
        ),
    ]


def test_dataset_manifest_reports_coverage_and_unverified_labels():
    manifest = dataset_manifest(_triples(), corpus_sha256="abc")
    assert manifest["total_queries"] == 3
    assert manifest["answerable_queries"] == 2
    assert manifest["unanswerable_queries"] == 1
    assert manifest["categories"]["near_neighbor"] == 1
    assert manifest["human_verified"] is False
    assert manifest["corpus_sha256"] == "abc"


def test_dataset_split_and_validation():
    triples = annotate_split(_triples(), dev_ratio=0.34, seed=1)
    assert {triple.metadata["split"] for triple in triples} == {"dev", "test"}
    assert len(filter_split(triples, "dev")) == 1
    assert len(filter_split(triples, "all")) == 3
    validate_dataset(triples, ["d1", "d2"])
    with pytest.raises(ValueError, match="Unknown relevance IDs"):
        validate_dataset(triples, ["d1"])
    with pytest.raises(ValueError):
        filter_split(triples, "missing")


def test_golden_dataset_rejects_inconsistent_answerability(tmp_path):
    path = tmp_path / "d.jsonl"
    row = {
        "query_id": "q",
        "query": "query",
        "expected_relevant_doc_ids": [],
        "reference_answer": "answer",
        "answerable": True,
    }
    path.write_text(json.dumps(row) + "\n")
    from src.evaluation.golden_dataset import load_golden_dataset

    with pytest.raises(ValueError):
        load_golden_dataset(str(path))
    row["answerable"] = False
    row["expected_relevant_doc_ids"] = ["d1"]
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        load_golden_dataset(str(path))


def test_diagnostic_suite_covers_ten_categories_and_no_answer(tmp_path):
    result = run_evaluation(
        "eval/fixtures/diagnostic_queries.jsonl",
        corpus_dir="eval/fixtures/corpus",
        mode="hybrid",
        output_dir=tmp_path,
        bootstrap_samples=100,
    )
    assert result["status"] == "completed"
    assert result["total_queries"] == 10
    assert result["manifest"]["category_coverage_complete"] is True
    assert set(result["manifest"]["category_coverage"]) == set(CATEGORIES)
    assert result["manifest"]["human_verified"] is False
    assert result["unanswerable"]["queries"] == 1
    assert "recall_at_5" in result["metrics"]
    assert "recall_at_5" in result["confidence_intervals"]
    # Unanswerable queries are excluded from retrieval averages and expose their
    # own observed count rather than a fabricated accuracy score.
    assert result["details"][4]["answerable"] is False
    assert result["details"][4]["metrics"] == {}


def test_split_filter_selects_subset(tmp_path):
    result = run_evaluation(
        "eval/fixtures/diagnostic_queries.jsonl",
        corpus_dir="eval/fixtures/corpus",
        mode="bm25",
        output_dir=tmp_path,
        split="dev",
        bootstrap=False,
    )
    assert result["status"] == "completed"
    assert result["total_queries"] == 5
    assert all(row["split"] == "dev" for row in result["details"])
    assert result["confidence_intervals"] == {}

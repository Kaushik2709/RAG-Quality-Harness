"""Tests for Retrieval Optimization Laboratory (E1-E7) and advanced experiments."""
import pytest
from langchain_core.documents import Document

from src.evaluation.experiments import (
    EXPERIMENT_SUITE,
    AdaptiveRetriever,
    ContextCompressor,
    ExperimentConfig,
    ExperimentRunner,
    LostInTheMiddleEvaluator,
    ParentDocumentRetriever,
    QueryRewriter,
    linear_score_fusion,
)
from src.evaluation.golden_dataset import GoldenTriple


def test_experiment_config_immutability():
    cfg = EXPERIMENT_SUITE["E1"]
    assert cfg.dense is True
    assert cfg.sparse is False
    with pytest.raises(Exception):
        cfg.dense = False  # frozen dataclass


def test_linear_score_fusion():
    dense = {"c1": 0.9, "c2": 0.5, "c3": 0.1}
    sparse = {"c1": 10.0, "c2": 5.0, "c3": 1.0}
    combined = linear_score_fusion(dense, sparse, dense_weight=1.0, sparse_weight=1.0)
    assert combined["c1"] > combined["c2"] > combined["c3"]
    assert combined["c1"] == pytest.approx(2.0)
    assert combined["c3"] == pytest.approx(0.0)

    # Empty inputs
    assert linear_score_fusion({}, {}) == {}


def test_query_rewriter_deterministic():
    rewriter = QueryRewriter(mode="rule_based")
    res = rewriter.rewrite("atlas auth and timeout")
    assert "authentication" in res["rewritten_query"]
    assert "latency" in res["rewritten_query"]
    assert res["strategy"] == "deterministic_keyword_expansion"
    assert res["latency_ms"] >= 0.0

    # Query with no synonyms remains unchanged
    res_plain = rewriter.rewrite("unmatchedxyz")
    assert res_plain["rewritten_query"] == "unmatchedxyz"


def test_parent_document_retriever():
    retriever = ParentDocumentRetriever()
    retriever.register("chunk-1", "doc-A")
    retriever.register("chunk-2", "doc-A")
    retriever.register("chunk-3", "doc-B")

    docs = [
        Document(page_content="c1", metadata={"chunk_id": "chunk-1"}),
        Document(page_content="c2", metadata={"chunk_id": "chunk-2"}),
        Document(page_content="c3", metadata={"chunk_id": "chunk-3"}),
    ]
    parents = retriever.resolve_parents(docs)
    assert parents == ["doc-A", "doc-B"]


def test_context_compressor():
    doc1 = Document(
        page_content=(
            "The audit logs retention policy is 30 days. "
            "Sentence one about something else entirely. "
            "Sentence two describing weather patterns. "
            "Sentence three detailing astronomy facts. "
            "Sentence four on baking sourdough bread."
        ),
        metadata={"chunk_id": "c1"},
    )
    doc2 = Document(
        page_content=(
            "Database cluster connects via port 5432. "
            "Sentence five on ancient Greek history. "
            "Sentence six about quantum mechanics. "
            "Sentence seven on gardening tips."
        ),
        metadata={"chunk_id": "c2"},
    )
    res = ContextCompressor.compress([doc1, doc2], query="audit logs retention", max_sentences_per_doc=1)
    assert res["original_tokens"] > res["compressed_tokens"]
    assert "audit logs retention" in res["compressed_text"]
    assert res["savings_pct"] > 0.0
    assert res["chunk_count"] == 2


def test_lost_in_the_middle_evaluator():
    gold = Document(page_content="Target gold chunk with key secret answer.", metadata={"chunk_id": "gold"})
    d1 = Document(page_content="Distractor one astronomy galaxy.", metadata={"chunk_id": "d1"})
    d2 = Document(page_content="Distractor two cooking recipes.", metadata={"chunk_id": "d2"})

    head = LostInTheMiddleEvaluator.position_context(gold, [d1, d2], placement="head")
    assert head[0].metadata["chunk_id"] == "gold"

    mid = LostInTheMiddleEvaluator.position_context(gold, [d1, d2], placement="middle")
    assert mid[1].metadata["chunk_id"] == "gold"

    tail = LostInTheMiddleEvaluator.position_context(gold, [d1, d2], placement="tail")
    assert tail[-1].metadata["chunk_id"] == "gold"

    # Evaluation simulation
    def fake_eval(docs, q):
        # Head gets highest score, middle slightly discounted
        pos = next(i for i, d in enumerate(docs) if d.metadata["chunk_id"] == "gold")
        return 1.0 / (pos + 1)

    study = LostInTheMiddleEvaluator.run_position_study(gold, [d1, d2], "key secret", fake_eval)
    assert "head" in study["scores_by_position"]
    assert "middle" in study["scores_by_position"]
    assert "tail" in study["scores_by_position"]
    assert study["middle_penalty"] >= 0


def test_adaptive_retriever():
    d1 = Document(page_content="audit retention policy 30 days", metadata={"chunk_id": "c1"})
    d2 = Document(page_content="unrelated text", metadata={"chunk_id": "c2"})

    retriever = AdaptiveRetriever(sufficiency_threshold=0.5, initial_k=1, expanded_k=2)

    def fake_retriever_fn(q, k):
        return [d2] if k == 1 else [d2, d1]

    res = retriever.execute("audit retention", fake_retriever_fn)
    # Since initial docs [d2] have 0 overlap with 'audit retention', expand should trigger!
    assert res["expanded_triggered"] is True
    assert res["decision"] == "generate"
    assert len(res["documents"]) == 1 or len(res["documents"]) == 2


def test_experiment_runner_basic(monkeypatch):
    class FakeVectorStore:
        def similarity_search(self, q, k):
            return [Document(page_content="audit logs 30 days", metadata={"chunk_id": "c1", "doc_id": "d1"})]

    chunks = [Document(page_content="audit logs 30 days", metadata={"chunk_id": "c1", "doc_id": "d1"})]
    runner = ExperimentRunner(
        vector_store=FakeVectorStore(),
        chunks=chunks,
        corpus_hash="fake-corpus-hash",
        dataset_hash="fake-dataset-hash",
        bm25=None,
    )

    triple = GoldenTriple(
        query_id="q1",
        query="What is the audit log retention?",
        reference_answer="30 days",
        expected_relevant_doc_ids=["d1"],
        answerable=True,
    )

    result = runner.run_experiment(EXPERIMENT_SUITE["E1"], [triple])
    assert result["experiment_id"] == "E1"
    assert result["total_queries"] == 1
    assert result["aggregate_metrics"]["recall_at_5"] == 1.0
    assert len(result["per_query_results"]) == 1
    assert result["per_query_results"][0]["ranking_after_fusion"] == ["c1"]

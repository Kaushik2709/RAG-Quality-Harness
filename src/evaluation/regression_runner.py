import os
import sys
import json
import argparse
import datetime
import glob
from typing import Dict, Any, List

from src.config import settings
from src.evaluation.golden_dataset import load_golden_dataset, GoldenTriple
from src.evaluation.retrieval_metrics import compute_all_retrieval_metrics
from src.evaluation.judge import LLMJudge
from src.ingestion.loaders import load_document
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import get_vector_store
from src.pipeline.graph import compile_rag_graph

def ingest_sample_corpus(vector_store):
    """Ingests all sample files using LangChain loaders & text splitters."""
    sample_files = glob.glob("data/sample_docs/*")
    if not sample_files:
        print("[RegressionRunner] Warning: No sample docs found in data/sample_docs/")
        return

    all_chunks = []
    for file_path in sample_files:
        docs = load_document(file_path)
        chunks = chunk_documents(
            docs,
            strategy=settings.CHUNKER_STRATEGY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        all_chunks.extend(chunks)

    if all_chunks:
        vector_store.add_documents(all_chunks)

def run_evaluation(dataset_path: str = "eval/golden_dataset.jsonl") -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"[RUNNER] Starting LangChain & LangGraph RAG Regression Evaluation")
    print(f"Dataset: {dataset_path}")
    print(f"Chunk Size: {settings.CHUNK_SIZE} | Overlap: {settings.CHUNK_OVERLAP} | Strategy: {settings.CHUNKER_STRATEGY}")
    print(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"=======================================================\n")

    triples = load_golden_dataset(dataset_path)

    # Inmemory vector store for evaluation
    vector_store = get_vector_store(use_in_memory=True)
    ingest_sample_corpus(vector_store)

    rag_graph = compile_rag_graph()
    judge = LLMJudge()

    per_query_results = []
    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    total_faithfulness = 0.0
    total_relevancy = 0.0
    total_context_precision = 0.0

    k = 5
    for idx, triple in enumerate(triples, start=1):
        query_id = f"eval-q{idx}"
        state = rag_graph.invoke({
            "query": triple.query,
            "query_id": query_id,
            "documents": [],
            "reranked_documents": [],
            "generation": ""
        })

        reranked_docs = state.get("reranked_documents", [])
        retrieved_ids = [d.metadata.get("doc_id", d.metadata.get("source", "")) for d in reranked_docs]
        retrieved_sources = [f"{d.metadata.get('source', '')}_{d.metadata.get('doc_id', '')}" for d in reranked_docs] + retrieved_ids

        # 3. Compute Retrieval Metrics
        metrics = compute_all_retrieval_metrics(retrieved_sources, triple.expected_relevant_doc_ids, k=k)

        # 4. Generated Answer
        gen_answer = state.get("generation", "")

        # 5. LLM-as-Judge Evaluation
        context_str = "\n".join([d.page_content for d in reranked_docs])
        judge_res = judge.evaluate(
            query=triple.query,
            retrieved_context=context_str,
            generated_answer=gen_answer
        )

        total_recall += metrics[f"recall_at_{k}"]
        total_precision += metrics[f"precision_at_{k}"]
        total_mrr += metrics["mrr"]
        total_ndcg += metrics[f"ndcg_at_{k}"]
        total_faithfulness += judge_res.faithfulness
        total_relevancy += judge_res.answer_relevancy
        total_context_precision += judge_res.context_precision

        per_query_results.append({
            "query_id": triple.query_id,
            "query": triple.query,
            "expected_docs": triple.expected_relevant_doc_ids,
            "retrieved_docs": retrieved_ids,
            "metrics": metrics,
            "judge_score": judge_res.model_dump(),
            "generated_answer": gen_answer
        })

        import time
        time.sleep(2.0)

        print(f"[EVAL] Query [{idx}/{len(triples)}] ({triple.query_id}): Recall@{k}={metrics[f'recall_at_{k}']} | MRR={metrics['mrr']} | Faithfulness={judge_res.faithfulness}")

    num_samples = len(triples)
    summary_metrics = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_queries": num_samples,
        "framework": "LangChain + LangGraph",
        "config": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "llm_provider": settings.LLM_PROVIDER
        },
        "metrics": {
            f"recall_at_{k}": round(total_recall / num_samples, 4),
            f"precision_at_{k}": round(total_precision / num_samples, 4),
            "mrr": round(total_mrr / num_samples, 4),
            f"ndcg_at_{k}": round(total_ndcg / num_samples, 4),
            "mean_faithfulness": round(total_faithfulness / num_samples, 4),
            "mean_relevancy": round(total_relevancy / num_samples, 4),
            "mean_context_precision": round(total_context_precision / num_samples, 4)
        },
        "details": per_query_results
    }

    # Save report
    os.makedirs("eval/reports", exist_ok=True)
    report_filename = f"eval/reports/eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"\n[SUMMARY] Aggregate Evaluation Metrics:")
    for m, v in summary_metrics["metrics"].items():
        print(f"  - {m}: {v}")
    print(f"\n[REPORT] Saved evaluation report to: {report_filename}")

    return summary_metrics

def compare_with_baseline(current_metrics: Dict[str, Any], baseline_path: str = "eval/baseline_metrics.json") -> bool:
    if not os.path.exists(baseline_path):
        print(f"\n[BASELINE] Warning: Baseline file '{baseline_path}' not found. Cannot perform regression diff.")
        return True

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    base = baseline_data.get("metrics", {})
    curr = current_metrics.get("metrics", {})

    print(f"\n=======================================================")
    print(f"[BASELINE COMPARISON] Regression Test Results vs Baseline")
    print(f"=======================================================")

    passed = True
    for metric_name, curr_val in curr.items():
        base_val = base.get(metric_name)
        if base_val is None:
            continue

        diff = curr_val - base_val
        status = "PASS"

        # Threshold rules
        if metric_name.startswith("recall_at_"):
            if curr_val < (base_val - settings.MAX_RECALL_DROP_TOLERANCE):
                status = "FAIL (RECALL REGRESSION)"
                passed = False
            elif curr_val < settings.MIN_RECALL_THRESHOLD:
                status = "FAIL (BELOW MIN THRESHOLD)"
                passed = False
        elif metric_name == "mean_faithfulness":
            if curr_val < settings.MIN_FAITHFULNESS_THRESHOLD:
                status = "FAIL (FAITHFULNESS BREACH)"
                passed = False

        tag = "[PASS]" if "PASS" in status else "[FAIL]"
        print(f" {tag:7s} {metric_name:25s} | Current: {curr_val:.4f} | Baseline: {base_val:.4f} | Diff: {diff:+.4f} | Status: {status}")

    return passed

def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG Evaluation & Regression Runner")
    parser.add_argument("--accept-baseline", action="store_true", help="Overwrite baseline_metrics.json with current results")
    parser.add_argument("--dataset", default="eval/golden_dataset.jsonl", help="Path to golden dataset jsonl")
    args = parser.parse_args()

    results = run_evaluation(dataset_path=args.dataset)

    if args.accept_baseline or not os.path.exists("eval/baseline_metrics.json"):
        with open("eval/baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("\n[SUCCESS] Baseline metrics snapshot created/updated at 'eval/baseline_metrics.json'.")
        sys.exit(0)

    is_success = compare_with_baseline(results)
    if not is_success:
        print("\n[FAILURE] REGRESSION FAILURE DETECTED: Quality gates breached!")
        sys.exit(1)

    print("\n[SUCCESS] All quality gates PASSED successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()

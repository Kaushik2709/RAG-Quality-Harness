import os
import pytest
from src.evaluation.regression_runner import run_evaluation, compare_with_baseline

def test_regression_evaluation():
    results = run_evaluation(dataset_path="eval/golden_dataset.jsonl")
    assert "metrics" in results
    metrics = results["metrics"]
    
    assert "recall_at_5" in metrics
    assert "precision_at_5" in metrics
    assert "mrr" in metrics
    assert "mean_faithfulness" in metrics
    assert metrics["recall_at_5"] >= 0.5
    assert metrics["mean_faithfulness"] >= 0.7

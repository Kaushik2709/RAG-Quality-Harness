"""Deterministic lexical and grounded answer metrics.

Measures:
1. Lexical overlap (exact_match, token_f1, contains_reference).
2. Grounded generation metrics:
   - Claim coverage (supported_claims / total_claims)
   - Citation precision (accepted_citations / total_citations)
   - Citation recall / evidence utilization (distinct_cited_chunks / total_retrieved)
   - Citation quote validity rate (valid_quotes / total_citations)
   - Numerical/factual consistency
   - Semantic entailment score
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Set

_WORD = re.compile(r"[a-z0-9]+")
_NUMBERS = re.compile(r"\b\d+(?:\.\d+)?\b")
_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(_WORD.findall(text))


def _tokens(text: str) -> Set[str]:
    return {token for token in _WORD.findall(normalize_answer(text)) if token not in _ARTICLES}


def exact_match(reference: str, answer: str) -> float:
    return 1.0 if normalize_answer(reference) == normalize_answer(answer) else 0.0


def contains_reference(reference: str, answer: str) -> float:
    ref = normalize_answer(reference)
    if not ref:
        return 0.0
    return 1.0 if ref in normalize_answer(answer) else 0.0


def token_f1(reference: str, answer: str) -> float:
    ref_tokens, answer_tokens = _tokens(reference), _tokens(answer)
    if not ref_tokens or not answer_tokens:
        return 0.0
    overlap = len(ref_tokens & answer_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_numbers(text: str) -> Set[str]:
    """Extract distinct numerical values from text."""
    return set(_NUMBERS.findall(text or ""))


def numerical_consistency(reference: str, answer: str) -> float:
    """Measure whether numerical values in reference appear consistently in answer."""
    ref_nums = extract_numbers(reference)
    if not ref_nums:
        return 1.0  # No numbers to contradict
    ans_nums = extract_numbers(answer)
    # Check if all numbers from reference are captured in answer
    matched = ref_nums.intersection(ans_nums)
    return len(matched) / len(ref_nums)


def claim_coverage(supported_claims: int, total_claims: int) -> float:
    """Fraction of claims in structured answer supported by cited evidence."""
    if total_claims <= 0:
        return 1.0
    return min(1.0, max(0.0, supported_claims / total_claims))


def citation_precision(accepted_citations: int, total_citations: int) -> float:
    """Fraction of citations that are verified and accepted."""
    if total_citations <= 0:
        return 0.0
    return min(1.0, max(0.0, accepted_citations / total_citations))


def citation_recall(distinct_cited_chunks: int, total_retrieved_chunks: int) -> float:
    """Evidence utilization: fraction of retrieved chunks used as valid citations."""
    if total_retrieved_chunks <= 0:
        return 0.0
    return min(1.0, max(0.0, distinct_cited_chunks / total_retrieved_chunks))


def citation_quote_validity_rate(valid_quotes: int, total_citations: int) -> float:
    """Rate at which cited quotes verbatim match retrieved chunks."""
    if total_citations <= 0:
        return 0.0
    return min(1.0, max(0.0, valid_quotes / total_citations))


def grounded_generation_metrics(
    validation_dict: Dict[str, Any],
    reference_answer: Optional[str] = None,
    generated_answer: Optional[str] = None,
) -> Dict[str, float]:
    """Compute aggregate grounded generation metrics from citation validation outcomes."""
    accepted = validation_dict.get("accepted", [])
    rejected = validation_dict.get("rejected", [])
    claims = validation_dict.get("claims", [])
    total_cit = len(accepted) + len(rejected)

    sup_claims = sum(1 for c in claims if c.get("supported", False))
    tot_claims = len(claims)

    c_prec = citation_precision(len(accepted), total_cit)
    c_cov = claim_coverage(sup_claims, tot_claims)
    quote_validity = citation_quote_validity_rate(len(accepted), total_cit)

    sem_entail = float(validation_dict.get("semantic_entailment_score", 1.0 if c_cov == 1.0 else 0.0))

    metrics = {
        "claim_coverage": c_cov,
        "citation_precision": c_prec,
        "citation_recall": float(validation_dict.get("citation_recall", 0.0)),
        "quote_validity_rate": quote_validity,
        "semantic_entailment_score": sem_entail,
    }

    if reference_answer and generated_answer:
        metrics["numerical_consistency"] = numerical_consistency(reference_answer, generated_answer)

    return metrics


def answer_metrics(reference: str, answer: str) -> Dict[str, float]:
    ref_tokens, answer_tokens = _tokens(reference), _tokens(answer)
    return {
        "exact_match": exact_match(reference, answer),
        "contains_reference": contains_reference(reference, answer),
        "token_f1": round(token_f1(reference, answer), 6),
        "numerical_consistency": numerical_consistency(reference, answer),
        "answer_token_count": len(answer_tokens),
        "reference_token_count": len(ref_tokens),
    }


def mean_answer_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = ("exact_match", "contains_reference", "token_f1", "numerical_consistency")
    return {
        f"mean_{key}": round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
        for key in keys
        if any(key in row for row in rows)
    }

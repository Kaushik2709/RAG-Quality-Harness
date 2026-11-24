"""Deterministic lexical answer metrics.

These metrics are cheap regression signals, not semantic correctness. A
lexical mismatch is not proof that an answer is wrong, and a lexical match is
not proof that it is grounded. Semantic grounding remains the job of citation
validation, the optional LLM judge and human review.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Sequence, Set

_WORD = re.compile(r"[a-z0-9]+")
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


def answer_metrics(reference: str, answer: str) -> Dict[str, float]:
    ref_tokens, answer_tokens = _tokens(reference), _tokens(answer)
    return {
        "exact_match": exact_match(reference, answer),
        "contains_reference": contains_reference(reference, answer),
        "token_f1": round(token_f1(reference, answer), 6),
        "answer_token_count": len(answer_tokens),
        "reference_token_count": len(ref_tokens),
    }


def mean_answer_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = ("exact_match", "contains_reference", "token_f1")
    return {
        f"mean_{key}": round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
        for key in keys
    }

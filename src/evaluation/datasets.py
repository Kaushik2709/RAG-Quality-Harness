"""Dataset taxonomy, manifest and deterministic dev/test splitting.

The manifest states what the labels are (and are not). An unverified synthetic
dataset remains unverified no matter how many categories it covers; category
coverage is not annotation quality.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from src.evaluation.golden_dataset import GoldenTriple

CATEGORIES = (
    "direct",
    "near_neighbor",
    "comparison",
    "multi_hop",
    "no_answer",
    "ambiguous",
    "temporal",
    "numeric",
    "policy_exception",
    "distractor",
)

CATEGORY_ALIASES = {
    "near-neighbor": "near_neighbor",
    "multi-hop": "multi_hop",
    "policy-exception": "policy_exception",
    "no-answer": "no_answer",
}

ANNOTATION_STATUSES = ("unverified", "proposed", "human_verified", "adjudicated")


def normalize_category(value: str | None) -> str:
    value = (value or "uncategorized").strip().lower()
    return CATEGORY_ALIASES.get(value, value)


def query_hash(triples: Sequence[GoldenTriple]) -> str:
    digest = hashlib.sha256()
    for triple in sorted(triples, key=lambda item: item.query_id):
        digest.update(triple.query_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(triple.query.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_manifest(
    triples: Sequence[GoldenTriple],
    *,
    dataset_path: str | None = None,
    corpus_sha256: str | None = None,
) -> Dict[str, Any]:
    categories = Counter(normalize_category(triple.metadata.get("category")) for triple in triples)
    splits = Counter(triple.metadata.get("split", "unspecified") for triple in triples)
    statuses = Counter(triple.metadata.get("annotation_status", "unverified") for triple in triples)
    answerable = sum(1 for triple in triples if triple.answerable)
    return {
        "dataset_path": dataset_path,
        "query_hash": query_hash(triples),
        "corpus_sha256": corpus_sha256,
        "total_queries": len(triples),
        "answerable_queries": answerable,
        "unanswerable_queries": len(triples) - answerable,
        "categories": dict(sorted(categories.items())),
        "category_coverage": sorted(categories),
        "category_coverage_complete": set(CATEGORIES).issubset(categories),
        "splits": dict(sorted(splits.items())),
        "annotation_status": dict(sorted(statuses.items())),
        "human_verified": bool(triples) and all(
            triple.metadata.get("annotation_status") == "human_verified" for triple in triples
        ),
        "label_provenance": sorted(
            {str(triple.metadata.get("provenance", "unrecorded")) for triple in triples}
        ),
    }


def validate_dataset(triples: Sequence[GoldenTriple], known_ids: Iterable[str]) -> None:
    known = set(known_ids)
    for triple in triples:
        unknown = set(triple.expected_relevant_doc_ids) - known
        if unknown:
            raise ValueError(f"Unknown relevance IDs: {sorted(unknown)}")
        # Categories and annotation statuses are descriptive metadata. Unknown
        # values are surfaced by the manifest instead of rejecting a dataset
        # whose taxonomy is richer than this repository's built-in list.
        if not triple.answerable and triple.expected_relevant_doc_ids:
            raise ValueError(f"Unanswerable query {triple.query_id} must not declare relevant documents")


def annotate_split(
    triples: Sequence[GoldenTriple],
    *,
    dev_ratio: float = 0.5,
    seed: int = 12345,
) -> List[GoldenTriple]:
    """Deterministically assign a dev/test split.

    Callers must not repeatedly re-split and report the best test score. The
    split is recorded in each query's metadata so a report can be audited.
    """
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be between 0 and 1")
    indexed = list(enumerate(triples))
    random.Random(seed).shuffle(indexed)
    dev_count = max(1, min(len(indexed) - 1, int(round(len(indexed) * dev_ratio)))) if len(indexed) > 1 else 1
    dev_indices = {index for index, _ in indexed[:dev_count]}
    result: List[GoldenTriple] = []
    for index, triple in enumerate(triples):
        metadata = dict(triple.metadata)
        metadata["split"] = "dev" if index in dev_indices else "test"
        metadata["split_seed"] = seed
        result.append(triple.model_copy(update={"metadata": metadata}))
    return result


def filter_split(triples: Sequence[GoldenTriple], split: str | None) -> List[GoldenTriple]:
    if split in (None, "", "all"):
        return list(triples)
    selected = [triple for triple in triples if triple.metadata.get("split", "unspecified") == split]
    if not selected:
        raise ValueError(f"dataset has no queries in split '{split}'")
    return selected

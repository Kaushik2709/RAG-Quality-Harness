"""Dataset taxonomy, manifest and deterministic leakage-resistant splitting.

Guarantees that semantic duplicates, paraphrases, and family clusters are
strictly assigned to the same split (train, dev, or test), preventing train/test
leakage across evaluation partitions.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
    "long_document": "direct",
    "conflicting": "policy_exception",
    "keyword": "direct",
    "paraphrase": "near_neighbor",
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
            triple.metadata.get("annotation_status") in {"human_verified", "adjudicated"} for triple in triples
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
        if not triple.answerable and triple.expected_relevant_doc_ids:
            raise ValueError(f"Unanswerable query {triple.query_id} must not declare relevant documents")


def _resolve_cluster_id(triple: GoldenTriple) -> str:
    meta = triple.metadata or {}
    return str(
        meta.get("cluster_id")
        or meta.get("paraphrase_of")
        or meta.get("family_id")
        or triple.query_id
    )


def annotate_split(
    triples: Sequence[GoldenTriple],
    *,
    dev_ratio: float = 0.5,
    seed: int = 12345,
    split_ratios: Optional[Dict[str, float]] = None,
) -> List[GoldenTriple]:
    """Deterministically assign dev/test or train/dev/test splits with cluster grouping.

    Queries in the same cluster/family (e.g. paraphrases) are assigned as an
    atomic group to the same split, eliminating data leakage.
    """
    if not (0.0 < dev_ratio < 1.0) and not split_ratios:
        raise ValueError("dev_ratio must be between 0 and 1")

    # Group queries into clusters
    clusters: Dict[str, List[int]] = defaultdict(list)
    for index, triple in enumerate(triples):
        cid = _resolve_cluster_id(triple)
        clusters[cid].append(index)

    sorted_cluster_ids = sorted(clusters.keys())
    rng = random.Random(seed)
    rng.shuffle(sorted_cluster_ids)

    assigned_splits: Dict[int, str] = {}

    if split_ratios:
        # 3-way split: e.g. {"train": 0.5, "dev": 0.25, "test": 0.25}
        total_ratio = sum(split_ratios.values())
        norm_ratios = {k: v / total_ratio for k, v in split_ratios.items()}
        split_names = list(norm_ratios.keys())
        counts = {name: 0 for name in split_names}
        total_queries = len(triples)

        for cid in sorted_cluster_ids:
            indices = clusters[cid]
            # Pick split that is furthest below its target ratio
            best_split = split_names[0]
            best_deficit = -1e9
            for s_name in split_names:
                target = norm_ratios[s_name] * total_queries
                deficit = target - counts[s_name]
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_split = s_name
            for idx in indices:
                assigned_splits[idx] = best_split
            counts[best_split] += len(indices)
    else:
        # Standard 2-way dev/test split
        dev_target = int(round(len(triples) * dev_ratio))
        dev_count = 0
        for cid in sorted_cluster_ids:
            indices = clusters[cid]
            if dev_count + len(indices) <= dev_target or dev_count == 0:
                s_name = "dev"
                dev_count += len(indices)
            else:
                s_name = "test"
            for idx in indices:
                assigned_splits[idx] = s_name

    result: List[GoldenTriple] = []
    for index, triple in enumerate(triples):
        metadata = dict(triple.metadata)
        metadata["split"] = assigned_splits.get(index, "dev")
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

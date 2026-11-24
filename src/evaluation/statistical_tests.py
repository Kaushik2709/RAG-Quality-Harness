"""Deterministic paired bootstrap utilities for evaluation reports.

These are resampling estimates over the observed query set, not guarantees
about future traffic. With small fixtures the intervals are correspondingly
wide and must not be presented as production confidence.
"""
from __future__ import annotations

import math
import random
from statistics import mean
from typing import Callable, Dict, List, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval plus the observed statistic."""
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    observed = float(statistic(list(values)))
    rng = random.Random(seed)
    draws = [
        float(statistic([values[rng.randrange(len(values))] for _ in range(len(values))]))
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": round(observed, 6),
        "lower": round(percentile(draws, alpha), 6),
        "upper": round(percentile(draws, 1.0 - alpha), 6),
        "std_error": round(_stddev(draws), 6),
        "samples": samples,
        "confidence": confidence,
        "n": len(values),
    }


def paired_bootstrap_difference(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> Dict[str, float]:
    """Bootstrap the mean paired difference (candidate - baseline).

    A two-sided p-value is the fraction of resampled mean differences that
    cross zero. It is descriptive and should not be read as proof of
    superiority; use a held-out, sufficiently large set for decisions.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired bootstrap requires equal-length vectors")
    if not baseline:
        raise ValueError("paired bootstrap requires at least one pair")
    diffs = [float(c) - float(b) for b, c in zip(baseline, candidate)]
    observed = mean(diffs)
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(samples):
        draw = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        means.append(mean(draw))
    alpha = (1.0 - confidence) / 2.0
    nonpositive = sum(1 for value in means if value <= 0)
    nonnegative = sum(1 for value in means if value >= 0)
    p_value = min(1.0, 2.0 * min(nonpositive, nonnegative) / samples)
    return {
        "mean_difference": round(observed, 6),
        "lower": round(percentile(means, alpha), 6),
        "upper": round(percentile(means, 1.0 - alpha), 6),
        "p_value_two_sided": round(p_value, 6),
        "samples": samples,
        "confidence": confidence,
        "n": len(diffs),
    }


def cohens_d(baseline: Sequence[float], candidate: Sequence[float]) -> float:
    """Standardised mean difference using pooled sample variance."""
    if len(baseline) < 2 or len(candidate) < 2:
        raise ValueError("cohens_d requires at least two values per group")
    n1, n2 = len(baseline), len(candidate)
    var1 = _variance(baseline)
    var2 = _variance(candidate)
    pooled = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (mean(candidate) - mean(baseline)) / pooled


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return sum((value - mu) ** 2 for value in values) / (len(values) - 1)


def _stddev(values: Sequence[float]) -> float:
    return math.sqrt(_variance(values))

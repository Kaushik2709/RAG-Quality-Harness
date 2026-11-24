"""Prometheus-compatible, process-local metrics.

These are *system* metrics only. Offline evaluation metrics live in eval/reports
and are never mixed into this registry, so a dashboards cannot confuse a
benchmark score with live telemetry.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

_LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], List[int]] = {}
        self._hist_sums: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = defaultdict(float)
        self._hist_counts: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], int] = defaultdict(int)
        self._started = time.time()

    @staticmethod
    def _key(labels: Dict[str, str] | None) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))

    def inc(self, name: str, labels: Dict[str, str] | None = None, value: float = 1.0) -> None:
        key = (name, self._key(labels))
        with self._lock:
            self._counters[key] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, value: float, labels: Dict[str, str] | None = None) -> None:
        key = (name, self._key(labels))
        with self._lock:
            buckets = self._histograms.setdefault(key, [0] * (len(_LATENCY_BUCKETS_MS) + 1))
            for idx, edge in enumerate(_LATENCY_BUCKETS_MS):
                if value <= edge:
                    buckets[idx] += 1
                    break
            else:
                buckets[-1] += 1
            self._hist_sums[key] += value
            self._hist_counts[key] += 1

    def observe_stage(self, stage: str, latency_ms: float, strategy: str = "") -> None:
        labels = {"stage": stage}
        if strategy:
            labels["strategy"] = strategy
        self.observe("rag_stage_latency_ms", latency_ms, labels)

    def render(self) -> str:
        lines: List[str] = []
        with self._lock:
            lines.append("# HELP rag_uptime_seconds Process uptime in seconds.")
            lines.append("# TYPE rag_uptime_seconds gauge")
            lines.append(f"rag_uptime_seconds {time.time() - self._started:.3f}")

            for (name, label_key), value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{_render_labels(label_key)} {value:g}")

            for name, value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value:g}")

            for (name, label_key), buckets in sorted(self._histograms.items()):
                lines.append(f"# TYPE {name} histogram")
                cumulative = 0
                for edge, count in zip(_LATENCY_BUCKETS_MS, buckets[:-1]):
                    cumulative += count
                    lines.append(f"{name}_bucket{_render_labels(label_key + (('le', str(edge)),))} {cumulative}")
                cumulative += buckets[-1]
                lines.append(f"{name}_bucket{_render_labels(label_key + (('le', '+Inf'),))} {cumulative}")
                lines.append(f"{name}_sum{_render_labels(label_key)} {self._hist_sums[(name, label_key)]:.6f}")
                lines.append(f"{name}_count{_render_labels(label_key)} {self._hist_counts[(name, label_key)]}")

            lines.append("# HELP rag_build_info Build information as labels.")
            lines.append("# TYPE rag_build_info gauge")
            lines.append("rag_build_info{service=\"rag-quality-harness\"} 1")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._hist_sums.clear()
            self._hist_counts.clear()
            self._started = time.time()


def _render_labels(label_key: Tuple[Tuple[str, str], ...]) -> str:
    if not label_key:
        return ""
    escaped = ",".join(f'{key}="{_escape(value)}"' for key, value in label_key)
    return "{" + escaped + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


metrics = MetricsRegistry()

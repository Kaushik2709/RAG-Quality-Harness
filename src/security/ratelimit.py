"""Bounded in-process rate limiting.

This is a single-node control, not a distributed quota. It exists to bound
accidental or abusive bursts in the reference deployment; the limitation is
documented rather than presented as a multi-instance guarantee.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    def __init__(self, per_minute: int, burst: int):
        self.per_minute = per_minute
        self.burst = max(burst, 1)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._last_gc = time.monotonic()

    def allow(self, key: str) -> tuple[bool, float]:
        if self.per_minute <= 0:
            return True, 0.0
        now = time.monotonic()
        rate = self.per_minute / 60.0
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.burst), updated_at=now)
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(float(self.burst), bucket.tokens + elapsed * rate)
                bucket.updated_at = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            retry_after = (1.0 - bucket.tokens) / rate if rate > 0 else 60.0
            self._gc(now)
            return False, retry_after

    def _gc(self, now: float) -> None:
        if now - self._last_gc < 300:
            return
        self._last_gc = now
        stale = [key for key, bucket in self._buckets.items() if now - bucket.updated_at > 900]
        for key in stale:
            self._buckets.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

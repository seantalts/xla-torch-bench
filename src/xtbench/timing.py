"""Per-call timing for torch and jax, and stats summary."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


def time_torch(fn: Callable[[], object]) -> int:
    """Time one call of a torch-returning callable. Returns nanoseconds."""
    t0 = time.perf_counter_ns()
    out = fn()
    # CPU torch is synchronous; no explicit sync needed.
    _ = out
    return time.perf_counter_ns() - t0


def time_jax(fn: Callable[[], object]) -> int:
    """Time one call of a jax-returning callable; defeat async dispatch."""
    t0 = time.perf_counter_ns()
    out = fn()
    out.block_until_ready()
    return time.perf_counter_ns() - t0


@dataclass
class Stats:
    median_ms: float
    p10_ms: float
    p90_ms: float


def summarize(samples_ns: list[int]) -> Stats:
    a = np.asarray(samples_ns, dtype=np.float64) / 1e6  # → ms
    return Stats(
        median_ms=float(np.median(a)),
        p10_ms=float(np.percentile(a, 10, method="nearest")),
        p90_ms=float(np.percentile(a, 90, method="nearest")),
    )

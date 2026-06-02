"""Compare a torch output and a jax output as numpy arrays."""
from __future__ import annotations

import numpy as np


class EquivalenceError(AssertionError):
    pass


def assert_close(a: np.ndarray, b: np.ndarray, atol: float) -> None:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise EquivalenceError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b).max()
    if not np.isfinite(diff) or diff > atol:
        raise EquivalenceError(
            f"max abs diff {diff:.3g} exceeds atol {atol:.3g}"
        )

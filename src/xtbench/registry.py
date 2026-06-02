"""Spec dataclass and @register decorator. No framework imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


def _default_inputs(shape: tuple, dtype: str, seed: int) -> list[np.ndarray]:
    """Default: one random standard-normal input of the given shape."""
    rng = np.random.default_rng(seed)
    np_dtype = {"f32": np.float32, "bf16": np.float32}[dtype]
    # bf16 generated as f32; harness converts per-framework downstream.
    return [rng.standard_normal(shape).astype(np_dtype)]


@dataclass
class Spec:
    """One benchmark: a PyTorch module + a JAX function over the same compute."""
    torch_module: Callable[..., Any]
    jax_fn: Callable[..., Any]
    shapes: list[tuple[int, ...]]
    dtypes: list[str]
    params: dict[str, Callable[[tuple[int, ...]], Any]] = field(default_factory=dict)
    inputs: Callable[[tuple, str, int], list[np.ndarray]] = _default_inputs


REGISTRY: list[tuple[str, Spec]] = []


def register(fn: Callable[[], Spec]) -> Callable[[], Spec]:
    """Decorator: a zero-arg function returning a Spec becomes a registered benchmark."""
    name = fn.__name__
    if any(n == name for n, _ in REGISTRY):
        raise ValueError(f"benchmark {name!r} already registered")
    REGISTRY.append((name, fn()))
    return fn

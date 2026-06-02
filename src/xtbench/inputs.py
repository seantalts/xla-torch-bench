"""Convert numpy inputs into framework tensors at the right dtype."""
from __future__ import annotations

import numpy as np


def _check_dtype(dtype: str) -> None:
    if dtype not in ("f32", "bf16"):
        raise ValueError(f"unknown dtype: {dtype!r}")


def to_torch(arr: np.ndarray, dtype: str):
    _check_dtype(dtype)
    import torch
    t = torch.from_numpy(arr.astype(np.float32, copy=False))
    if dtype == "bf16":
        t = t.to(torch.bfloat16)
    return t


def to_jax(arr: np.ndarray, dtype: str):
    _check_dtype(dtype)
    import jax.numpy as jnp
    target = {"f32": jnp.float32, "bf16": jnp.bfloat16}[dtype]
    return jnp.asarray(arr).astype(target)


def dtype_atol(dtype: str) -> float:
    _check_dtype(dtype)
    return {"f32": 1e-4, "bf16": 1e-2}[dtype]

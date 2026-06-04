"""Convert numpy inputs into framework tensors at the right dtype."""
from __future__ import annotations

import numpy as np


def _check_dtype(dtype: str) -> None:
    if dtype not in ("f32", "bf16"):
        raise ValueError(f"unknown dtype: {dtype!r}")


def to_torch(arr: np.ndarray, dtype: str):
    _check_dtype(dtype)
    import torch
    # Integer arrays (used as indices in gather/scatter benchmarks) bypass the
    # dtype cast — converting them through f32 or bf16 loses precision (bf16
    # has only 7 mantissa bits, so values above 256 round). Keep their native
    # integer dtype; the benchmark side casts to int64 as needed.
    if np.issubdtype(arr.dtype, np.integer):
        return torch.from_numpy(arr)
    t = torch.from_numpy(arr.astype(np.float32, copy=False))
    if dtype == "bf16":
        t = t.to(torch.bfloat16)
    return t


def to_jax(arr: np.ndarray, dtype: str):
    _check_dtype(dtype)
    import jax.numpy as jnp
    if np.issubdtype(arr.dtype, np.integer):
        # Preserve integer dtype for index tensors; see to_torch above.
        return jnp.asarray(arr)
    target = {"f32": jnp.float32, "bf16": jnp.bfloat16}[dtype]
    return jnp.asarray(arr).astype(target)


def dtype_atol(dtype: str) -> float:
    _check_dtype(dtype)
    # f32: compiled reductions (Inductor's vs XLA's tree shapes) drift by ~2e-4
    # on long axes (~4096 elements). 5e-4 covers that without masking real bugs.
    # bf16: 1 bf16 ULP at magnitude ~1 is 0.0625; fused-vs-decomposed kernel
    # pairs routinely produce a 1-ULP diff. 7e-2 gives headroom over that floor.
    return {"f32": 5e-4, "bf16": 7e-2}[dtype]

"""Per-row gather of TOPK columns from an (R, C) tensor. Common in top-k / MoE.

Input ``x`` is ``(ROWS, COLS)``; indices ``idx`` are ``(ROWS, TOPK)`` with each
value in ``[0, COLS)``. The result is ``(ROWS, TOPK)`` where
``out[r, k] = x[r, idx[r, k]]``.

- torch: ``torch.gather(x, 1, idx)``.
- jax:   ``jnp.take_along_axis(x, idx, axis=1)``.

This is a row-strided gather, which on XLA typically lowers to ``lax.gather``
with an indices-vector-dim — a well-known emitter weakness for non-contiguous
inner-dim reads.
"""
import numpy as np

from xtbench import Spec, register

ROWS = 4096
COLS = 768
TOPK = 32


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((ROWS, COLS)).astype(np.float32)
    idx = rng.integers(low=0, high=COLS, size=(ROWS, TOPK), dtype=np.int64)
    return [x, idx]


@register
def take_along_axis() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, idx):
                return torch.gather(x, 1, idx.to(torch.int64))
        return M()

    def jax_fn():
        import jax.numpy as jnp

        def f(x, idx):
            return jnp.take_along_axis(x, idx.astype(jnp.int32), axis=1)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(ROWS, COLS)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )

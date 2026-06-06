"""Embedding-table lookup: gather rows from a (vocab, embed) table by integer indices.

Idiomatic patterns:
- torch: ``nn.functional.embedding`` (a gather of rows by `int64` indices).
- jax:   ``embed[indices]`` (which lowers to ``lax.gather``).

The shape passed in is the index shape ``(batch, seq)``; the embedding table is
held internally as ``(VOCAB, EMBED)`` and contributes the bf16/f32 dtype of the
benchmark. Indices are int64 numpy arrays which the harness passes through
unchanged (see ``to_torch``/``to_jax`` integer fast path).
"""
import numpy as np

from xtbench import Spec, register

VOCAB = 50_000
EMBED = 768
WEIGHT_CONST = 0.02  # deterministic table so equivalence holds modulo dtype


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(low=0, high=VOCAB, size=shape, dtype=np.int64)
    # `dtype_carrier` is a 1-element float tensor whose only purpose is to let
    # the module/fn read off the bench dtype (`carrier.dtype`) at call time —
    # the harness doesn't pass `dtype` through to the closures so we route it
    # through the input list. It does not participate in the gather output.
    dtype_carrier = np.zeros((1,), dtype=np.float32)
    return [idx, dtype_carrier]


@register
def embedding_lookup() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.full((VOCAB, EMBED), WEIGHT_CONST)
                )

            def forward(self, idx, dtype_carrier):
                w = self.weight.to(dtype_carrier.dtype)
                return torch.nn.functional.embedding(idx.to(torch.int64), w)
        return M()

    def jax_fn():
        import jax.numpy as jnp
        weight = jnp.full((VOCAB, EMBED), WEIGHT_CONST)

        def f(idx, dtype_carrier):
            w = weight.astype(dtype_carrier.dtype)
            return w[idx.astype(jnp.int32)]
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 512), (1, 2048)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )

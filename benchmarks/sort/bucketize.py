"""bucketize: assign each value in a (4096, 768) tensor to a sorted bucket.

Two inputs (values + boundaries). Output is integer bucket indices — the
harness compares as f32, which is exact for small ints. Torch's default
bucketize semantics (right=False) match jax.numpy.searchsorted (side='left').
The values tensor is the only one carrying a meaningful dtype; the
boundaries we always feed as f32 (in jax we cast to match for searchsorted,
torch keeps them as the meaningful dtype for the comparison).
"""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    # Values: random normal, shape (4096, 768).
    x = rng.standard_normal(shape).astype(np.float32)
    # Sorted boundaries: 256 evenly-spaced across the value range, perturbed
    # slightly so the test isn't degenerate. linspace already sorted.
    boundaries = np.linspace(-3.0, 3.0, 256).astype(np.float32)
    return [x, boundaries]


@register
def bucketize() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, boundaries):
                # bucketize requires boundaries to share dtype with values.
                return torch.bucketize(x, boundaries.to(x.dtype))
        return M()

    def jax_fn():
        import jax.numpy as jnp

        def f(x, boundaries):
            return jnp.searchsorted(boundaries.astype(x.dtype), x)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(4096, 768)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )

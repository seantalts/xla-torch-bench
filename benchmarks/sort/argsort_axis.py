"""argsort along the last axis.

We compare *sorted values* rather than the index tensor: with bf16 ties
torch and jax can pick different indices for equal elements while still
producing a correctly-sorted output. Gathering via take_along_axis with
each side's own indices reduces to the standard 'sort returns the sorted
sequence' contract, which is unambiguous.
"""
from xtbench import Spec, register


@register
def argsort_axis() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                idx = torch.argsort(x, dim=-1)
                return torch.gather(x, -1, idx)
        return M()

    def jax_fn():
        import jax.numpy as jnp

        def f(x):
            idx = jnp.argsort(x, axis=-1)
            return jnp.take_along_axis(x, idx, axis=-1)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1024, 512)],
        dtypes=["f32", "bf16"],
    )

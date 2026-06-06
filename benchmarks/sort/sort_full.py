"""Full sort of a 2D tensor along the last axis (no top-k slicing).

Returns the sorted values directly. Single tensor output, unambiguous
across torch and jax even with bf16 ties.
"""
from xtbench import Spec, register


@register
def sort_full() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                v, _ = torch.sort(x, dim=-1)
                return v
        return M()

    def jax_fn():
        import jax.numpy as jnp

        def f(x):
            return jnp.sort(x, axis=-1)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1024, 4096)],
        dtypes=["f32", "bf16"],
    )

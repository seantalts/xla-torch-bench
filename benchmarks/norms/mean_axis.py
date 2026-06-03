"""mean along last dim."""
from xtbench import Spec, register


@register
def mean_axis() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                return torch.mean(x, dim=-1)
        return M()

    def jax_fn():
        import jax.numpy as jnp
        def f(x):
            return jnp.mean(x, axis=-1)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 4096, 1024), (1, 1024, 4096)],
        dtypes=["f32", "bf16"],
    )

"""silu(a) * b — pure pointwise."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32),
            rng.standard_normal(shape).astype(np.float32)]


@register
def silu_mul() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b):
                return torch.nn.functional.silu(a) * b
        return M()

    def jax_fn():
        import jax
        def f(a, b):
            return jax.nn.silu(a) * b
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768), (1, 4096, 768)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )

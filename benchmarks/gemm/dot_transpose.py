"""y = a @ b.T. XLA and Inductor may handle the transpose differently."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    m, k, n = shape
    return [
        rng.standard_normal((m, k)).astype(np.float32),
        rng.standard_normal((n, k)).astype(np.float32),
    ]


@register
def dot_transpose() -> Spec:
    def torch_module():
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b):
                return a @ b.transpose(-1, -2)
        return M()

    def jax_fn():
        import jax.numpy as jnp
        def f(a, b):
            return a @ jnp.swapaxes(b, -1, -2)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(512, 768, 512)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )

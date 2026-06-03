"""Small matmul: (M, K) @ (K, N) with M=K=N=256."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    m, k, n = shape
    return [
        rng.standard_normal((m, k)).astype(np.float32),
        rng.standard_normal((k, n)).astype(np.float32),
    ]


@register
def dot_small() -> Spec:
    def torch_module():
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b):
                return a @ b
        return M()

    def jax_fn():
        def f(a, b):
            return a @ b
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(256, 256, 256)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )

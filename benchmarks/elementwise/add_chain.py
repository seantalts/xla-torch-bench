"""a + b + c + d over a 3D tensor — pure pointwise fusion target."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32) for _ in range(4)]


@register
def add_chain() -> Spec:
    def torch_module():
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b, c, d):
                return a + b + c + d
        return M()

    def jax_fn():
        def f(a, b, c, d):
            return a + b + c + d
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768), (1, 4096, 768)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )

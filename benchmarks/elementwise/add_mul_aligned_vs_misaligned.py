"""(a + b) * c at shapes that hit vs miss vector boundaries.

The 'aligned' shape (d=1024) is a multiple of the 16-lane f32 vector size on
AVX-512 and the 8-lane vector size on NEON. The 'misaligned' shape (d=1023)
forces a scalar tail loop — this is the case xtile-aligned-ops targets.
"""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32) for _ in range(3)]


@register
def add_mul_aligned_vs_misaligned() -> Spec:
    def torch_module():
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b, c):
                return (a + b) * c
        return M()

    def jax_fn():
        def f(a, b, c):
            return (a + b) * c
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 4096, 1024), (1, 4096, 1023), (1, 4096, 1025)],
        dtypes=["f32"],
        inputs=_inputs,
    )

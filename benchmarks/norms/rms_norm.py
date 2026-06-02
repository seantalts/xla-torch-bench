"""RMSNorm — rsqrt(mean(x^2) + eps) * x * w."""
from xtbench import Spec, register


@register
def rms_norm() -> Spec:
    def torch_module(d):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(d))

            def forward(self, x):
                rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                return x * rms * self.w
        return M()

    def jax_fn(d):
        import jax
        import jax.numpy as jnp
        w = jnp.ones((d,))

        def f(x):
            rms = jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
            return x * rms * w
        return f

    return Spec(
        torch_module=torch_module,
        jax_fn=jax_fn,
        shapes=[(1, 1024, 768), (1, 4096, 768)],
        dtypes=["f32", "bf16"],
        params={"d": lambda s: s[-1]},
    )

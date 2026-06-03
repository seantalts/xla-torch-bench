"""LayerNorm on last dim, with affine weights."""
from xtbench import Spec, register


@register
def layer_norm() -> Spec:
    def torch_module(d):
        # Reimplement manually rather than using nn.LayerNorm; nn.LayerNorm
        # upcasts bf16 inputs to f32 internally, so the bf16 row would partly
        # measure that upcast rather than bf16 arithmetic.
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(d))
                self.b = nn.Parameter(torch.zeros(d))

            def forward(self, x):
                w = self.w.to(x.dtype)
                b = self.b.to(x.dtype)
                mean = x.mean(-1, keepdim=True)
                var = ((x - mean) ** 2).mean(-1, keepdim=True)
                return (x - mean) / torch.sqrt(var + 1e-5) * w + b
        return M()

    def jax_fn(d):
        import jax.numpy as jnp
        w = jnp.ones((d,))
        b = jnp.zeros((d,))

        def f(x):
            wc = w.astype(x.dtype)
            bc = b.astype(x.dtype)
            mean = jnp.mean(x, -1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, -1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5) * wc + bc
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768), (1, 4096, 768)],
        dtypes=["f32", "bf16"],
        params={"d": lambda s: s[-1]},
    )

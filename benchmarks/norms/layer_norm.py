"""LayerNorm on last dim, with affine weights."""
from xtbench import Spec, register


@register
def layer_norm() -> Spec:
    def torch_module(d):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(d, eps=1e-5)
                with torch.no_grad():
                    self.ln.weight.fill_(1.0)
                    self.ln.bias.fill_(0.0)

            def forward(self, x):
                return self.ln(x)
        return M()

    def jax_fn(d):
        import jax.numpy as jnp
        w = jnp.ones((d,))
        b = jnp.zeros((d,))

        def f(x):
            mean = jnp.mean(x, -1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, -1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5) * w + b
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768), (1, 4096, 768)],
        dtypes=["f32", "bf16"],
        params={"d": lambda s: s[-1]},
    )

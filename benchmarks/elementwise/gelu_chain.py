"""Linear → GELU → Linear. Weights set to identity for rosetta-stone equivalence."""
from xtbench import Spec, register


@register
def gelu_chain() -> Spec:
    def torch_module(d):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                w = torch.eye(d)
                self.w1 = nn.Parameter(w.clone())
                self.w2 = nn.Parameter(w.clone())

            def forward(self, x):
                w1 = self.w1.to(x.dtype)
                w2 = self.w2.to(x.dtype)
                y = x @ w1
                y = torch.nn.functional.gelu(y, approximate="tanh")
                return y @ w2
        return M()

    def jax_fn(d):
        import jax
        import jax.numpy as jnp
        w1 = jnp.eye(d)
        w2 = jnp.eye(d)

        def f(x):
            w1c = w1.astype(x.dtype)
            w2c = w2.astype(x.dtype)
            y = x @ w1c
            y = jax.nn.gelu(y, approximate=True)
            return y @ w2c
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768)],
        dtypes=["f32", "bf16"],
        params={"d": lambda s: s[-1]},
    )

"""y = gelu(x @ w + b). Tests whether the bias add + activation fuse with the dot."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    b, m, k = shape
    return [rng.standard_normal((b, m, k)).astype(np.float32)]


@register
def dot_bias_gelu() -> Spec:
    def torch_module(k, n):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(k, n)
                with torch.no_grad():
                    self.lin.weight.fill_(0.01)
                    self.lin.bias.fill_(0.1)

            def forward(self, x):
                w = self.lin.weight.to(x.dtype)
                b = self.lin.bias.to(x.dtype)
                return torch.nn.functional.gelu(
                    torch.nn.functional.linear(x, w, b), approximate="tanh"
                )
        return M()

    def jax_fn(k, n):
        import jax
        import jax.numpy as jnp
        w = jnp.full((k, n), 0.01)
        b = jnp.full((n,), 0.1)

        def f(x):
            return jax.nn.gelu(x @ w + b, approximate=True)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768)],
        dtypes=["f32", "bf16"],
        params={"k": lambda s: s[-1], "n": lambda s: s[-1]},
        inputs=_inputs,
    )

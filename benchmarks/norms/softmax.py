"""softmax on last dim."""
from xtbench import Spec, register


@register
def softmax() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=-1)
        return M()

    def jax_fn():
        import jax
        def f(x):
            return jax.nn.softmax(x, axis=-1)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 12, 512, 512), (1, 16, 1024, 1024)],
        dtypes=["f32", "bf16"],
    )

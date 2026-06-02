"""Scaled dot-product attention block (no masking).

Shape convention: (batch, num_heads, seq, head_dim).
"""
import math
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
    ]


@register
def mha_block() -> Spec:
    def torch_module(scale):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = scale

            def forward(self, q, k, v):
                attn = (q @ k.transpose(-1, -2)) * self.scale
                attn = torch.softmax(attn, dim=-1)
                return attn @ v
        return M()

    def jax_fn(scale):
        import jax
        import jax.numpy as jnp
        def f(q, k, v):
            attn = (q @ jnp.swapaxes(k, -1, -2)) * scale
            attn = jax.nn.softmax(attn, axis=-1)
            return attn @ v
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[
            (1, 12, 128, 64),
            (1, 12, 512, 64),
            (1, 12, 2048, 64),
        ],
        dtypes=["f32", "bf16"],
        params={"scale": lambda s: 1.0 / math.sqrt(s[-1])},
        inputs=_inputs,
    )

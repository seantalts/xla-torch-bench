"""Grouped-query attention: kv heads are repeated across query heads.

Shape: q is (b, h_q, s, d); k, v are (b, h_kv, s, d). We expand k/v to match.
"""
import math
import numpy as np

from xtbench import Spec, register

H_Q = 32
H_KV = 8
HEAD_D = 128


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    b, seq = shape
    q = rng.standard_normal((b, H_Q, seq, HEAD_D)).astype(np.float32)
    k = rng.standard_normal((b, H_KV, seq, HEAD_D)).astype(np.float32)
    v = rng.standard_normal((b, H_KV, seq, HEAD_D)).astype(np.float32)
    return [q, k, v]


@register
def gqa_block() -> Spec:
    def torch_module(scale):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = scale

            def forward(self, q, k, v):
                repeat = H_Q // H_KV
                k_full = k.repeat_interleave(repeat, dim=1)
                v_full = v.repeat_interleave(repeat, dim=1)
                attn = (q @ k_full.transpose(-1, -2)) * self.scale
                attn = torch.softmax(attn, dim=-1)
                return attn @ v_full
        return M()

    def jax_fn(scale):
        import jax
        import jax.numpy as jnp
        def f(q, k, v):
            repeat = H_Q // H_KV
            k_full = jnp.repeat(k, repeat, axis=1)
            v_full = jnp.repeat(v, repeat, axis=1)
            attn = (q @ jnp.swapaxes(k_full, -1, -2)) * scale
            attn = jax.nn.softmax(attn, axis=-1)
            return attn @ v_full
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 512), (1, 2048)],
        dtypes=["f32", "bf16"],
        params={"scale": lambda s: 1.0 / math.sqrt(HEAD_D)},
        inputs=_inputs,
    )

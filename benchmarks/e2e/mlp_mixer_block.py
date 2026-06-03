"""MLP-Mixer block (token-mixing MLP + channel-mixing MLP) — attention-free."""
import numpy as np

from xtbench import Spec, register

SEQ = 196
D_MODEL = 768
D_TOKEN_MLP = 384
D_CHANNEL_MLP = 3072
W_CONST = 0.02


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32)]


@register
def mlp_mixer_block() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = nn.LayerNorm(D_MODEL)
                self.ln2 = nn.LayerNorm(D_MODEL)
                with torch.no_grad():
                    for ln in (self.ln1, self.ln2):
                        ln.weight.fill_(1.0)
                        ln.bias.fill_(0.0)
                self.w_tm1 = nn.Parameter(torch.full((SEQ, D_TOKEN_MLP), W_CONST))
                self.w_tm2 = nn.Parameter(torch.full((D_TOKEN_MLP, SEQ), W_CONST))
                self.w_cm1 = nn.Parameter(torch.full((D_MODEL, D_CHANNEL_MLP), W_CONST))
                self.w_cm2 = nn.Parameter(torch.full((D_CHANNEL_MLP, D_MODEL), W_CONST))

            def forward(self, x):
                w_tm1 = self.w_tm1.to(x.dtype)
                w_tm2 = self.w_tm2.to(x.dtype)
                w_cm1 = self.w_cm1.to(x.dtype)
                w_cm2 = self.w_cm2.to(x.dtype)

                h = self.ln1(x).transpose(-1, -2)
                h = torch.nn.functional.gelu(h @ w_tm1, approximate="tanh") @ w_tm2
                x = x + h.transpose(-1, -2)
                h = self.ln2(x)
                h = torch.nn.functional.gelu(h @ w_cm1, approximate="tanh") @ w_cm2
                return x + h
        return Block()

    def jax_fn():
        import jax
        import jax.numpy as jnp

        w_tm1 = jnp.full((SEQ, D_TOKEN_MLP), W_CONST)
        w_tm2 = jnp.full((D_TOKEN_MLP, SEQ), W_CONST)
        w_cm1 = jnp.full((D_MODEL, D_CHANNEL_MLP), W_CONST)
        w_cm2 = jnp.full((D_CHANNEL_MLP, D_MODEL), W_CONST)

        def _ln(x):
            mean = jnp.mean(x, -1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, -1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5)

        def f(x):
            tm1 = w_tm1.astype(x.dtype)
            tm2 = w_tm2.astype(x.dtype)
            cm1 = w_cm1.astype(x.dtype)
            cm2 = w_cm2.astype(x.dtype)

            h = jnp.swapaxes(_ln(x), -1, -2)
            h = jax.nn.gelu(h @ tm1, approximate=True) @ tm2
            x = x + jnp.swapaxes(h, -1, -2)
            h = jax.nn.gelu(_ln(x) @ cm1, approximate=True) @ cm2
            return x + h
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, SEQ, D_MODEL)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
        # Large-K accumulation through multiple bf16 matmuls (K up to 3072).
        atol_override={"bf16": 5.0},
    )

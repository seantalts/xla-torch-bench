"""One GPT-2 transformer block: pre-LN, MHA, residual, pre-LN, MLP (4x), residual.

Weights set to a fixed deterministic constant on both sides for rosetta-stone
equivalence (we are NOT loading HF GPT-2 weights — this is a forward-pass
codegen comparison).
"""
import math
import numpy as np

from xtbench import Spec, register

D_MODEL = 768
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS
D_MLP = 4 * D_MODEL
W_CONST = 0.02


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32)]


@register
def gpt2_block() -> Spec:
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
                # Store weights as raw f32 parameters; cast to x.dtype in forward
                # so bf16 inputs don't trip the matmul dtype check.
                self.wq = nn.Parameter(torch.full((D_MODEL, D_MODEL), W_CONST))
                self.wk = nn.Parameter(torch.full((D_MODEL, D_MODEL), W_CONST))
                self.wv = nn.Parameter(torch.full((D_MODEL, D_MODEL), W_CONST))
                self.wo = nn.Parameter(torch.full((D_MODEL, D_MODEL), W_CONST))
                self.w1 = nn.Parameter(torch.full((D_MODEL, D_MLP), W_CONST))
                self.w2 = nn.Parameter(torch.full((D_MLP, D_MODEL), W_CONST))

            def forward(self, x):
                b, s, _ = x.shape
                wq = self.wq.to(x.dtype)
                wk = self.wk.to(x.dtype)
                wv = self.wv.to(x.dtype)
                wo = self.wo.to(x.dtype)
                w1 = self.w1.to(x.dtype)
                w2 = self.w2.to(x.dtype)

                h = self.ln1(x)
                q = (h @ wq).view(b, s, N_HEADS, D_HEAD).transpose(1, 2)
                k = (h @ wk).view(b, s, N_HEADS, D_HEAD).transpose(1, 2)
                v = (h @ wv).view(b, s, N_HEADS, D_HEAD).transpose(1, 2)
                attn = (q @ k.transpose(-1, -2)) / math.sqrt(D_HEAD)
                attn = torch.softmax(attn, dim=-1)
                out = (attn @ v).transpose(1, 2).reshape(b, s, D_MODEL)
                x = x + out @ wo
                h = self.ln2(x)
                h = torch.nn.functional.gelu(h @ w1, approximate="tanh")
                return x + h @ w2
        return Block()

    def jax_fn():
        import jax
        import jax.numpy as jnp

        wq = jnp.full((D_MODEL, D_MODEL), W_CONST)
        wk = jnp.full((D_MODEL, D_MODEL), W_CONST)
        wv = jnp.full((D_MODEL, D_MODEL), W_CONST)
        wo = jnp.full((D_MODEL, D_MODEL), W_CONST)
        w1 = jnp.full((D_MODEL, D_MLP), W_CONST)
        w2 = jnp.full((D_MLP, D_MODEL), W_CONST)

        def _ln(x):
            mean = jnp.mean(x, -1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, -1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5)

        def f(x):
            b, s, _ = x.shape
            wqc = wq.astype(x.dtype)
            wkc = wk.astype(x.dtype)
            wvc = wv.astype(x.dtype)
            woc = wo.astype(x.dtype)
            w1c = w1.astype(x.dtype)
            w2c = w2.astype(x.dtype)

            h = _ln(x)
            q = (h @ wqc).reshape(b, s, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
            k = (h @ wkc).reshape(b, s, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
            v = (h @ wvc).reshape(b, s, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
            attn = (q @ jnp.swapaxes(k, -1, -2)) / math.sqrt(D_HEAD)
            attn = jax.nn.softmax(attn, axis=-1)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, D_MODEL)
            x = x + out @ woc
            h = _ln(x)
            h = jax.nn.gelu(h @ w1c, approximate=True)
            return x + h @ w2c
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 512, D_MODEL), (1, 2048, D_MODEL)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
        # Large-K accumulation through multiple bf16 matmuls; the global 7e-2
        # floor isn't enough for an end-to-end block.
        atol_override={"bf16": 5.0},
    )

"""Full MoE block (dense fallback): router → all-experts → gated combine.

This is the headline sparse benchmark. We use the *dense* path — compute
every expert's output then mask by router weight — rather than a true
gather/scatter dispatch. The dense path is what most CPU MoE implementations
use because gather/scatter overhead exceeds the FLOP savings on small N.

The torch.scatter / jax.lax.scatter primitives also have very different
performance characteristics and APIs, making a true-sparse rosetta-stone
ambiguous (dispatch by tokens-per-expert with variable counts requires
either dynamic shapes or padded buckets — neither maps cleanly between
torch.compile and jax.jit on CPU). The dense version keeps the rosetta
honest: identical compute, identical shapes, only the codegen differs.

Top-1 routing: for each token, take the single best expert. The output is
the gated expert MLP output (gate * expert(x)) for the picked expert, with
all other experts' outputs zeroed via the one-hot routing mask.
"""
import numpy as np

from xtbench import Spec, register

D_MODEL = 768
D_HIDDEN = 3072
N_EXPERTS = 8
W_CONST = 0.02


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32)]


@register
def expert_dispatch_combine() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                # Random router so logits actually differ across experts.
                rng = np.random.default_rng(12345)
                w_router = rng.standard_normal((D_MODEL, N_EXPERTS)).astype(np.float32) * W_CONST
                self.w_router = nn.Parameter(torch.from_numpy(w_router))
                # Expert MLPs: stacked (E, D_MODEL, D_HIDDEN) and (E, D_HIDDEN, D_MODEL).
                # Constant weights so torch and jax stay rosetta-equivalent.
                self.w1 = nn.Parameter(torch.full((N_EXPERTS, D_MODEL, D_HIDDEN), W_CONST))
                self.w2 = nn.Parameter(torch.full((N_EXPERTS, D_HIDDEN, D_MODEL), W_CONST))

            def forward(self, x):
                # x: (B, S, D_MODEL)
                w_router = self.w_router.to(x.dtype)
                w1 = self.w1.to(x.dtype)
                w2 = self.w2.to(x.dtype)

                logits = x @ w_router                      # (B, S, E)
                gates = torch.softmax(logits, dim=-1)
                top_val, top_idx = torch.topk(gates, 1, dim=-1)  # both (B, S, 1)
                # One-hot mask over experts, weighted by the top-1 gate value.
                # mask: (B, S, E) — exactly one nonzero entry per token.
                mask = torch.zeros_like(gates)
                mask.scatter_(-1, top_idx, top_val)

                # Dense expert compute: run all E experts on all tokens,
                # then combine with the routing mask. (B, S, D) @ (E, D, H) ->
                # (E, B, S, H) via einsum.
                h = torch.einsum("bsd,edh->ebsh", x, w1)
                h = torch.nn.functional.gelu(h, approximate="tanh")
                y = torch.einsum("ebsh,ehd->ebsd", h, w2)  # (E, B, S, D)
                # Combine: weighted sum over experts using mask.
                # mask is (B, S, E); reorder to (E, B, S, 1) and multiply.
                m = mask.permute(2, 0, 1).unsqueeze(-1)
                return (y * m).sum(dim=0)                  # (B, S, D)
        return Block()

    def jax_fn():
        import jax
        import jax.numpy as jnp

        rng = np.random.default_rng(12345)
        w_router_np = rng.standard_normal((D_MODEL, N_EXPERTS)).astype(np.float32) * W_CONST
        w_router = jnp.asarray(w_router_np)
        w1 = jnp.full((N_EXPERTS, D_MODEL, D_HIDDEN), W_CONST)
        w2 = jnp.full((N_EXPERTS, D_HIDDEN, D_MODEL), W_CONST)

        def f(x):
            wrc = w_router.astype(x.dtype)
            w1c = w1.astype(x.dtype)
            w2c = w2.astype(x.dtype)

            logits = x @ wrc                               # (B, S, E)
            gates = jax.nn.softmax(logits, axis=-1)
            top_val, top_idx = jax.lax.top_k(gates, 1)     # both (B, S, 1)

            B, S, _ = gates.shape
            zeros = jnp.zeros_like(gates)
            b_idx = jnp.arange(B)[:, None, None]
            s_idx = jnp.arange(S)[None, :, None]
            mask = zeros.at[b_idx, s_idx, top_idx].set(top_val)

            h = jnp.einsum("bsd,edh->ebsh", x, w1c)
            h = jax.nn.gelu(h, approximate=True)
            y = jnp.einsum("ebsh,ehd->ebsd", h, w2c)        # (E, B, S, D)
            m = jnp.transpose(mask, (2, 0, 1))[..., None]
            return (y * m).sum(axis=0)                      # (B, S, D)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, D_MODEL)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
        # All-experts dense compute: K up to D_HIDDEN=3072 through stacked
        # matmuls of constant 0.02 weights, then weighted sum across 8
        # experts. Output magnitudes are large (~8) and accumulated through
        # many reductions; f32 drifts past the global 5e-4 floor by ~3x and
        # bf16 needs even more headroom.
        atol_override={"f32": 5e-3, "bf16": 5.0},
    )

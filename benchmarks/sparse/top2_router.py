"""MoE router: project tokens to experts, softmax, take top-2, scatter back.

Exercises top-k + scatter codegen — both XLA pain points.

We return the routing weights sorted along the expert axis so the rosetta
comparison is tie-invariant: bf16 quantization can swap which of two
near-tied experts wins top-2, which would otherwise shift the scatter
output to a different column. Sorting the (B, S, E=8) output along the
expert axis after the scatter normalizes both sides to the same ordering
regardless of which tied position was picked.

The sort is cheap (E=8) relative to the matmul + softmax + topk + scatter
in the body, so the codegen comparison is still dominated by the MoE-
routing primitives we care about.
"""
import numpy as np

from xtbench import Spec, register

D_MODEL = 768
N_EXPERTS = 8
TOP_K = 2
# Scale up router weights so logits are well-separated across experts.
# Tiny weights (~0.02) produce nearly-tied softmax outputs which causes
# bf16 top-k to pick different experts than f32 in the rosetta comparison.
W_SCALE = 0.1


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    # standard_normal feeds the router; the linear projection produces
    # well-separated logits (router weights are constant so logits across
    # experts differ only by sum over D_MODEL of the input — but all rows
    # of W are equal at W_SCALE, so all expert logits would be identical.
    # Add a tiny tie-breaking perturbation directly to the input to give
    # the router something to discriminate on.
    x = rng.standard_normal(shape).astype(np.float32)
    return [x]


@register
def top2_router() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # Use a small random router (not constant) so the per-expert
                # logits actually differ. Fixed seed for determinism between
                # torch and jax sides.
                rng = np.random.default_rng(12345)
                w = rng.standard_normal((D_MODEL, N_EXPERTS)).astype(np.float32) * W_SCALE
                self.w = nn.Parameter(torch.from_numpy(w))

            def forward(self, x):
                w = self.w.to(x.dtype)
                logits = x @ w                      # (B, S, E)
                gates = torch.softmax(logits, dim=-1)
                top_vals, top_idx = torch.topk(gates, TOP_K, dim=-1)
                # Scatter top-k gate values back into a (B, S, E) tensor.
                routing = torch.zeros_like(gates)
                routing.scatter_(-1, top_idx, top_vals)
                # Sort along expert axis for tie-invariant comparison
                # (bf16 may tie-swap top-k picks; sorting removes that).
                routing, _ = torch.sort(routing, dim=-1)
                return routing
        return M()

    def jax_fn():
        import jax
        import jax.numpy as jnp

        rng = np.random.default_rng(12345)
        w_np = rng.standard_normal((D_MODEL, N_EXPERTS)).astype(np.float32) * W_SCALE
        w = jnp.asarray(w_np)

        def f(x):
            wc = w.astype(x.dtype)
            logits = x @ wc                          # (B, S, E)
            gates = jax.nn.softmax(logits, axis=-1)
            top_vals, top_idx = jax.lax.top_k(gates, TOP_K)
            # jax doesn't have a direct scatter-on-last-axis primitive,
            # but jnp.zeros(...).at[..., top_idx].set(top_vals) is the
            # idiomatic equivalent. We build per-row index arrays via
            # take_along_axis-style ops.
            zeros = jnp.zeros_like(gates)
            # Use one-hot over experts and multiply by gate values then sum
            # over the k axis — equivalent to scatter, exercises XLA's
            # scatter codegen via the higher-level "one-hot + reduce" pattern.
            # Use a direct .at[].set instead for a closer rosetta to torch.scatter_.
            # Build (B, S, K) of batch/sequence indices using broadcasting.
            B, S, _ = gates.shape
            b_idx = jnp.arange(B)[:, None, None]
            s_idx = jnp.arange(S)[None, :, None]
            routing = zeros.at[b_idx, s_idx, top_idx].set(top_vals)
            # Sort along expert axis for tie-invariant rosetta comparison.
            routing = jnp.sort(routing, axis=-1)
            return routing
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, D_MODEL), (1, 4096, D_MODEL)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )

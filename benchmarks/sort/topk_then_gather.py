"""MoE-like pattern: top-k over a per-token expert score, then gather
the corresponding expert-weight rows for downstream multiplication.

Shapes:
    scores:  (T, E)   — T tokens, E experts.
    weights: (E, D)   — per-expert hidden dim.
Output:    (T, k, D) — selected expert rows per token.

We return only the gathered weights (single tensor). f32 only — under bf16
two scores that round to the same value cause torch and jax to pick
different experts (different tie-break), which changes a whole row of the
gathered output by O(1). Setting atol high enough to cover that defeats
the equivalence check, so we drop bf16 for this bench rather than
construct an artificially tie-free score distribution (which would
destroy the "realistic MoE score" character).
"""
import numpy as np

from xtbench import Spec, register

_T = 512  # tokens
_E = 64   # experts
_D = 128  # expert hidden dim
_K = 4    # selected experts per token


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((_T, _E)).astype(np.float32)
    weights = rng.standard_normal((_E, _D)).astype(np.float32)
    return [scores, weights]


@register
def topk_then_gather() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, scores, weights):
                # topk over experts: (T, k)
                _, idx = torch.topk(scores, k=_K, dim=-1)
                # gather rows of weights: weights[idx] -> (T, k, D)
                return weights[idx]
        return M()

    def jax_fn():
        import jax.lax as lax
        import jax.numpy as jnp

        def f(scores, weights):
            _, idx = lax.top_k(scores, k=_K)
            return weights[idx]
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        # Single shape; the actual sizes are pinned via the inputs callback.
        # The shape tuple is used only for naming + the params lambdas (none
        # here). We declare it as (T, E) for clarity in the report row.
        shapes=[(_T, _E)],
        dtypes=["f32"],
        inputs=_inputs,
    )

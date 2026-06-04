"""Row-scatter-add into a zeroed accumulator. Forces gather-on-write codegen.

Input rows ``(N, EMBED)`` are scattered into a ``(NUM_SEGMENTS, EMBED)``
accumulator at random row-indices in ``[0, NUM_SEGMENTS)`` — collisions are
expected and the op is order-independent (sum semantics).

- torch: ``torch.zeros(...).index_add_(0, idx, src)``.
- jax:   ``jax.ops.segment_sum(src, idx, num_segments=NUM_SEGMENTS)``.

bf16 with summation collisions does have order-dependent rounding, so the bf16
atol gets bumped — the expected count of collisions per segment is
``N/NUM_SEGMENTS = 4`` here and N(0,1) sums of 4 terms have stdev 2, so 1-ULP
error at magnitude ~2 is ~0.125; we give some headroom.
"""
import numpy as np

from xtbench import Spec, register

N = 4096
EMBED = 768
NUM_SEGMENTS = 1024


def _inputs(shape, dtype, seed):
    # shape is purely a tag for the run table; the dims used internally are
    # fixed (N, EMBED, NUM_SEGMENTS).
    rng = np.random.default_rng(seed)
    src = rng.standard_normal((N, EMBED)).astype(np.float32)
    idx = rng.integers(low=0, high=NUM_SEGMENTS, size=(N,), dtype=np.int64)
    return [src, idx]


@register
def scatter_add() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, src, idx):
                out = torch.zeros(
                    (NUM_SEGMENTS, EMBED), dtype=src.dtype, device=src.device
                )
                # index_add is the natural torch scatter-sum primitive.
                return out.index_add(0, idx.to(torch.int64), src)
        return M()

    def jax_fn():
        import jax
        import jax.numpy as jnp

        def f(src, idx):
            return jax.ops.segment_sum(src, idx.astype(jnp.int32),
                                       num_segments=NUM_SEGMENTS)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        # Shape tag is informational; the bench uses fixed dims internally.
        shapes=[(N, EMBED)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
        # bf16 sum-with-collisions accumulates ~4 terms per segment in different
        # orders across the two frameworks; magnitudes ~sqrt(4)=2 give 1-ULP
        # error ~0.125. f32 with the same 4-term sum is well under 5e-4.
        atol_override={"bf16": 0.4},
    )

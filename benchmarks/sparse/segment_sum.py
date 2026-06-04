"""Segment-sum: scatter-add (N, C) rows into (num_segments, C) by row index.

This is the core primitive behind MoE-style dispatch and any grouped-reduce
op. Torch uses `index_add_`, JAX uses `jax.ops.segment_sum`. XLA's scatter
codegen on CPU is known to serialize per-segment, so this is expected to
be a soft XLA loss.

Indices are integer but our input pipeline coerces to float, so we pass
indices as float32 in [0, 256), then cast to int inside both forward
functions. bf16 represents integers up to 256 exactly, so the f32 -> bf16
coercion preserves the index values losslessly.

bf16 segment-sum with many collisions per segment has summation-order
sensitivity (torch's per-element scatter loop vs XLA's scatter codegen
may sum in different orders), so we bump the bf16 atol.
"""
import numpy as np

from xtbench import Spec, register

N_ROWS = 4096
N_CHANNELS = 64
NUM_SEGMENTS = 256


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_ROWS, N_CHANNELS)).astype(np.float32)
    # Random segment indices in [0, NUM_SEGMENTS).
    idx = rng.integers(0, NUM_SEGMENTS, size=(N_ROWS,)).astype(np.float32)
    return [x, idx]


@register
def segment_sum() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, idx_f):
                # idx arrives as float (possibly bf16) — cast to int64 for
                # index_add_. Values 0..255 are exact in both f32 and bf16.
                idx = idx_f.to(torch.int64)
                out = torch.zeros(NUM_SEGMENTS, N_CHANNELS, dtype=x.dtype, device=x.device)
                out.index_add_(0, idx, x)
                return out
        return M()

    def jax_fn():
        import jax
        import jax.numpy as jnp

        def f(x, idx_f):
            idx = idx_f.astype(jnp.int32)
            return jax.ops.segment_sum(x, idx, num_segments=NUM_SEGMENTS)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        # Shape is informational — the inputs() callable hardcodes N_ROWS,
        # N_CHANNELS, NUM_SEGMENTS. We list one shape so the harness runs.
        shapes=[(N_ROWS, N_CHANNELS)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
        # ~16 rows per segment on average. Each row is sum of N(0,1)s
        # across 64 channels — magnitudes ~sqrt(16)=4. bf16 sum of 16
        # values has ~16 * ULP(4) = 16 * 0.03125 = 0.5 worst-case order-of-
        # operations drift, plus torch's per-element accumulation order
        # differs from XLA's scatter tree. Use 1.0 to be safe.
        atol_override={"bf16": 1.0},
    )

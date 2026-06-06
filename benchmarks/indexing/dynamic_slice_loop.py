"""KV-cache-lookup pattern: many dynamic slices of an inner axis, stacked.

Take ``N_SLICES`` length-``SLICE_LEN`` contiguous slices along the seq axis of a
``(1, SEQ, EMBED)`` tensor at offsets supplied at runtime, and stack into a
``(N_SLICES, SLICE_LEN, EMBED)`` output. This is the structural shape of a
sliding-window / per-head KV cache read.

- torch: a python loop of ``torch.narrow`` calls then ``torch.stack``.
- jax:   a python loop of ``jax.lax.dynamic_slice`` calls then ``jnp.stack``.

Both are static-trip-count python loops so ``torch.compile`` and ``jax.jit``
both unroll. The offsets are constrained so that ``offset + SLICE_LEN <= SEQ``,
giving us non-trivially varying slice starts that ``jax.lax.dynamic_slice``
clamps if needed (we keep them in range so clamp is a no-op).
"""
import numpy as np

from xtbench import Spec, register

SEQ = 2048
EMBED = 768
N_SLICES = 64
SLICE_LEN = 32


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, SEQ, EMBED)).astype(np.float32)
    offsets = rng.integers(
        low=0, high=SEQ - SLICE_LEN + 1, size=(N_SLICES,), dtype=np.int64
    )
    return [x, offsets]


@register
def dynamic_slice_loop() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, offsets):
                offsets = offsets.to(torch.int64)
                outs = []
                # index_select with `start + arange` keeps the offset symbolic
                # under torch.compile (it doesn't require a python-int start
                # the way `torch.narrow` does).
                base = torch.arange(SLICE_LEN, device=x.device, dtype=torch.int64)
                for i in range(N_SLICES):
                    sl = torch.index_select(x, 1, offsets[i] + base)
                    outs.append(sl)
                return torch.stack(outs, dim=0)
        return M()

    def jax_fn():
        import jax
        import jax.numpy as jnp

        def f(x, offsets):
            offsets = offsets.astype(jnp.int32)
            outs = []
            for i in range(N_SLICES):
                sl = jax.lax.dynamic_slice(x, (0, offsets[i], 0),
                                            (1, SLICE_LEN, EMBED))
                outs.append(sl)
            return jnp.stack(outs, axis=0)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, SEQ, EMBED)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )

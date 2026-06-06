"""Dilated 3x3 conv: (1, 64, 56, 56), dilation=2, stride 1, SAME-equivalent padding.

Segmentation / audio model pattern (atrous conv). Dilated convs often fall to
generic codegen because oneDNN's optimized kernels target dilation=1.

Padding choice: with k=3 and dilation=2, the effective kernel size is 5, so
padding=2 preserves spatial dims (analogous to padding=1 for k=3, d=1).
"""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32)]


@register
def dilated_conv_3x3() -> Spec:
    in_ch, out_ch, k = 64, 64, 3
    dilation = 2
    padding = 2  # (k-1)*dilation/2 = 2 keeps H,W unchanged

    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(
                    in_ch, out_ch, k, stride=1, padding=padding,
                    dilation=dilation, bias=False,
                )
                with torch.no_grad():
                    self.conv.weight.fill_(0.01)

            def forward(self, x):
                w = self.conv.weight.to(x.dtype)
                return torch.nn.functional.conv2d(
                    x, w, stride=1, padding=padding, dilation=dilation
                )
        return M()

    def jax_fn():
        import jax
        import jax.numpy as jnp
        w = jnp.full((out_ch, in_ch, k, k), 0.01)

        def f(x):
            wc = w.astype(x.dtype)
            return jax.lax.conv_general_dilated(
                x, wc,
                window_strides=(1, 1),
                padding=((padding, padding), (padding, padding)),
                rhs_dilation=(dilation, dilation),
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, in_ch, 56, 56)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
        atol_override={"bf16": 1.0},
    )

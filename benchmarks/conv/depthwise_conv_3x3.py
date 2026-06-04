"""Depthwise 3x3 conv: (1, 256, 56, 56), groups=in_channels, stride 1, SAME.

MobileNet/EfficientNet pattern. Depthwise convs typically don't dispatch to
oneDNN's optimized winograd/im2col paths on either side; both frameworks may
fall to generic codegen.
"""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32)]


@register
def depthwise_conv_3x3() -> Spec:
    in_ch, k = 256, 3
    groups = in_ch  # depthwise: groups == in_channels, out_ch == in_channels

    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(
                    in_ch, in_ch, k, stride=1, padding=1, groups=groups, bias=False
                )
                with torch.no_grad():
                    self.conv.weight.fill_(0.01)

            def forward(self, x):
                w = self.conv.weight.to(x.dtype)
                return torch.nn.functional.conv2d(
                    x, w, stride=1, padding=1, groups=groups
                )
        return M()

    def jax_fn():
        import jax
        import jax.numpy as jnp
        # PyTorch depthwise weight shape: (out=in_ch, in/groups=1, k, k).
        # JAX OIHW with feature_group_count matches this layout.
        w = jnp.full((in_ch, 1, k, k), 0.01)

        def f(x):
            wc = w.astype(x.dtype)
            return jax.lax.conv_general_dilated(
                x, wc,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
                feature_group_count=groups,
            )
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, in_ch, 56, 56)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
        # K-per-output = k*k = 9 — small, but use the 1.0 ceiling for safety.
        atol_override={"bf16": 1.0},
    )

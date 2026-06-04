"""Grouped 3x3 conv: (1, 128, 28, 28), groups=4, stride 1, SAME. ResNeXt-style.

Groups between 1 (dense conv -> oneDNN GEMM-im2col) and in_channels (depthwise)
often miss the most optimized dispatch paths in both frameworks.
"""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32)]


@register
def grouped_conv_3x3() -> Spec:
    in_ch, out_ch, k = 128, 128, 3
    groups = 4

    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(
                    in_ch, out_ch, k, stride=1, padding=1, groups=groups, bias=False
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
        # PyTorch grouped weight shape: (out, in/groups, k, k).
        w = jnp.full((out_ch, in_ch // groups, k, k), 0.01)

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
        shapes=[(1, in_ch, 28, 28)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
        # K per output = (in_ch/groups) * k * k = 32 * 9 = 288. Use 1.0 ceiling.
        atol_override={"bf16": 1.0},
    )

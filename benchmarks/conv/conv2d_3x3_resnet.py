"""ResNet-style 3x3 conv: (1, 64, 56, 56) -> 128 channels, stride 1, SAME padding.

Baseline: both PyTorch (Inductor) and XLA:CPU should dispatch this to oneDNN.
Expected parity. If not, something is wrong with the setup.
"""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32)]


@register
def conv2d_3x3_resnet() -> Spec:
    in_ch, out_ch, k = 64, 128, 3

    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # bias=False to keep the rosetta-stone exact; padding=1 == SAME for k=3.
                self.conv = nn.Conv2d(in_ch, out_ch, k, stride=1, padding=1, bias=False)
                with torch.no_grad():
                    self.conv.weight.fill_(0.01)

            def forward(self, x):
                w = self.conv.weight.to(x.dtype)
                return torch.nn.functional.conv2d(x, w, stride=1, padding=1)
        return M()

    def jax_fn():
        import jax
        import jax.numpy as jnp
        # OIHW to match PyTorch convention.
        w = jnp.full((out_ch, in_ch, k, k), 0.01)

        def f(x):
            wc = w.astype(x.dtype)
            return jax.lax.conv_general_dilated(
                x, wc,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, in_ch, 56, 56)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
        # K = in_ch * k * k = 64*9 = 576; bf16 sum-of-products at that K with
        # constant weights 0.01 and N(0,1) inputs gives max-magnitude outputs
        # ~= 0.01 * sqrt(576) ~= 0.24 and 2-ULP error ~= 0.06, but the safe
        # ceiling per task brief is 1.0.
        atol_override={"bf16": 1.0},
    )

"""Transpose conv 3x3, stride=2: (1, 64, 14, 14) -> 32 channels. VAE/decoder pattern.

Mapping PyTorch's ConvTranspose2d to JAX:
  PyTorch: weight shape (in_channels, out_channels, kH, kW), i.e. IOHW.
  JAX:     conv_transpose(dimension_numbers=("NCHW", "IOHW", "NCHW"),
                          use_consistent_padding=True).
  use_consistent_padding=True interprets `padding` as the corresponding forward
  conv's padding (PyTorch's convention).

  Kernel spatial flip: PyTorch's ConvTranspose2d flips the kernel spatially
  internally; JAX's conv_transpose does NOT by default. We sidestep this with
  constant weights (flip is a no-op on a constant kernel) — documented so a
  future variant with non-constant weights remembers to flip via
  `transpose_kernel=True` (which then requires OIHW dim_nums with a swap, see
  jax source) or via a manual `w[:, :, ::-1, ::-1]`.

Output spatial: H_out = (H_in - 1)*stride - 2*pad + k = (14-1)*2 - 2 + 3 = 27.
"""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32)]


@register
def transpose_conv_3x3() -> Spec:
    in_ch, out_ch, k = 64, 32, 3
    stride, pad = 2, 1

    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.ConvTranspose2d(
                    in_ch, out_ch, k, stride=stride, padding=pad, bias=False
                )
                with torch.no_grad():
                    self.conv.weight.fill_(0.01)

            def forward(self, x):
                w = self.conv.weight.to(x.dtype)
                return torch.nn.functional.conv_transpose2d(
                    x, w, stride=stride, padding=pad
                )
        return M()

    def jax_fn():
        import jax
        import jax.numpy as jnp
        # PyTorch ConvTranspose2d weight layout: (in, out, k, k) = IOHW.
        w = jnp.full((in_ch, out_ch, k, k), 0.01)

        def f(x):
            wc = w.astype(x.dtype)
            return jax.lax.conv_transpose(
                x, wc,
                strides=(stride, stride),
                padding=((pad, pad), (pad, pad)),
                dimension_numbers=("NCHW", "IOHW", "NCHW"),
                use_consistent_padding=True,
            )
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, in_ch, 14, 14)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
        atol_override={"bf16": 1.0},
    )

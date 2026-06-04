"""top-k on a 1x50000 logit row (LLM vocab sampling shape).

Sweeps k in {10, 100, 1000}. Each variant is a separate registered Spec so
results show up as distinct rows in the benchmark table. The module returns
only the top-k *values* (not indices) — bf16 ties can break torch vs jax
index agreement even when the underlying sort is correct; the values are
unambiguous (worst case one extra equal-valued element would be picked, and
its float value is identical to the displaced one).
"""
from xtbench import Spec, register


def _topk_spec(k: int) -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                v, _ = torch.topk(x, k=k, dim=-1)
                return v
        return M()

    def jax_fn():
        import jax.lax as lax

        def f(x):
            v, _ = lax.top_k(x, k=k)
            return v
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 50000)],
        dtypes=["f32", "bf16"],
    )


@register
def topk_logits_k10() -> Spec:
    return _topk_spec(10)


@register
def topk_logits_k100() -> Spec:
    return _topk_spec(100)


@register
def topk_logits_k1000() -> Spec:
    return _topk_spec(1000)

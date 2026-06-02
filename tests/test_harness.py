from xtbench.harness import run_benchmark, Result
from xtbench.registry import Spec


def test_run_benchmark_on_identity_spec():
    """End-to-end: both sides compute x * 2 (constant matches, no weight drift)."""
    def torch_module():
        import torch.nn as nn
        class M(nn.Module):
            def forward(self, x):
                return x * 2.0
        return M()

    def jax_fn():
        def f(x):
            return x * 2.0
        return f

    spec = Spec(torch_module=torch_module, jax_fn=jax_fn,
                shapes=[(8,)], dtypes=["f32"], params={})
    r = run_benchmark(spec, shape=(8,), dtype="f32", warmup=1, iters=3)
    assert isinstance(r, Result)
    assert r.equiv_ok
    assert r.torch.median_ms > 0
    assert r.jax.median_ms > 0
    assert r.compile_ms_torch > 0
    assert r.compile_ms_jax > 0


def test_run_benchmark_marks_equiv_failure():
    """Two sides deliberately disagree; equiv_ok must be False."""
    def torch_module():
        import torch.nn as nn
        class M(nn.Module):
            def forward(self, x):
                return x * 2.0
        return M()

    def jax_fn():
        def f(x):
            return x * 3.0  # deliberate mismatch
        return f

    spec = Spec(torch_module=torch_module, jax_fn=jax_fn,
                shapes=[(8,)], dtypes=["f32"], params={})
    r = run_benchmark(spec, shape=(8,), dtype="f32", warmup=1, iters=3)
    assert not r.equiv_ok
    assert "max abs diff" in r.equiv_error

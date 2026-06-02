"""Run one (spec, shape, dtype) end-to-end: compile, equiv-check, warmup, time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xtbench.equivalence import assert_close, EquivalenceError
from xtbench.inputs import dtype_atol, to_jax, to_torch
from xtbench.registry import Spec
from xtbench.timing import Stats, summarize, time_jax, time_torch


@dataclass
class Result:
    torch: Stats
    jax: Stats
    compile_ms_torch: float
    compile_ms_jax: float
    equiv_ok: bool
    equiv_error: str = ""


def _resolve_params(spec: Spec, shape: tuple) -> dict:
    return {k: fn(shape) for k, fn in spec.params.items()}


def run_benchmark(
    spec: Spec, shape: tuple, dtype: str, *, warmup: int = 5, iters: int = 50,
    seed: int = 0,
) -> Result:
    import torch  # noqa: F401  (ensures torch is initialized; thread pin done by caller)

    kwargs = _resolve_params(spec, shape)
    np_inputs = spec.inputs(shape, dtype, seed)

    # ---- PyTorch side ----
    torch_inputs = [to_torch(a, dtype) for a in np_inputs]
    m = torch.compile(spec.torch_module(**kwargs), backend="inductor")
    compile_ms_torch = time_torch(lambda: m(*torch_inputs)) / 1e6

    for _ in range(warmup):
        m(*torch_inputs)
    torch_samples = [time_torch(lambda: m(*torch_inputs)) for _ in range(iters)]

    # ---- JAX side ----
    import jax
    jax_inputs = [to_jax(a, dtype) for a in np_inputs]
    f_jit = jax.jit(spec.jax_fn(**kwargs))
    compile_ms_jax = time_jax(lambda: f_jit(*jax_inputs)) / 1e6

    for _ in range(warmup):
        f_jit(*jax_inputs).block_until_ready()
    jax_samples = [time_jax(lambda: f_jit(*jax_inputs)) for _ in range(iters)]

    # ---- Equivalence check ----
    torch_out = m(*torch_inputs).detach().to(torch.float32).numpy()
    jax_out = np.asarray(f_jit(*jax_inputs), dtype=np.float32)
    try:
        assert_close(torch_out, jax_out, atol=dtype_atol(dtype))
        equiv_ok, equiv_error = True, ""
    except EquivalenceError as e:
        equiv_ok, equiv_error = False, str(e)

    return Result(
        torch=summarize(torch_samples),
        jax=summarize(jax_samples),
        compile_ms_torch=compile_ms_torch,
        compile_ms_jax=compile_ms_jax,
        equiv_ok=equiv_ok,
        equiv_error=equiv_error,
    )

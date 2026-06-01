# xla-torch-bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python benchmark suite that compares XLA:CPU (via `jax.jit`) against PyTorch Inductor (via `torch.compile`) on a shared set of "rosetta-stone" workloads — each benchmark written twice (PyTorch `nn.Module` + JAX function) expressing the same compute.

**Architecture:** A small harness (`src/xtbench/`) provides the `Spec` dataclass + `@register` decorator, thread pinning, input generation, equivalence checking, timing, and stdout reporting. Benchmarks live under `benchmarks/` and are pure data — each file defines a single `Spec` via `@register`. A pytest parametrized test (`tests/test_equivalence.py`) iterates the registry and asserts numerical equivalence between the two implementations.

**Tech Stack:** Python 3.11+, PyTorch (with `torch.compile`/Inductor), JAX (`jax[cpu]` pulls XLA wheels), NumPy, pytest. Argparse (stdlib) for the CLI — chosen over click to keep imports minimal so we can set env vars before any framework import.

---

## File Structure

```
xla-torch-bench/
  pyproject.toml                          # Task 1
  src/xtbench/
    __init__.py                           # Task 1 (re-exports register, Spec)
    registry.py                           # Task 2
    inputs.py                             # Task 4
    equivalence.py                        # Task 5
    timing.py                             # Task 6
    harness.py                            # Task 7
    report.py                             # Task 8
    cli.py                                # Task 9
    __main__.py                           # Task 9
  benchmarks/
    __init__.py                           # Task 10 (imports all categories)
    norms/__init__.py                     # Task 10 / Task 13
    norms/rms_norm.py                     # Task 10
    norms/layer_norm.py                   # Task 13
    norms/softmax.py                      # Task 13
    norms/sum_axis.py                     # Task 13
    norms/mean_axis.py                    # Task 13
    elementwise/__init__.py               # Task 12
    elementwise/add_chain.py              # Task 12
    elementwise/gelu_chain.py             # Task 12
    elementwise/silu_mul.py               # Task 12
    elementwise/add_mul_aligned_vs_misaligned.py  # Task 12
    gemm/__init__.py                      # Task 14
    gemm/dot_small.py                     # Task 14
    gemm/dot_medium.py                    # Task 14
    gemm/dot_bias_gelu.py                 # Task 14
    gemm/dot_transpose.py                 # Task 14
    attention/__init__.py                 # Task 15
    attention/mha_block.py                # Task 15
    attention/gqa_block.py                # Task 15
    e2e/__init__.py                       # Task 16
    e2e/gpt2_block.py                     # Task 16
    e2e/mlp_mixer_block.py                # Task 16
  tests/
    __init__.py                           # Task 2
    test_registry.py                      # Task 2
    test_inputs.py                        # Task 4
    test_equivalence_check.py             # Task 5
    test_timing.py                        # Task 6
    test_harness.py                       # Task 7
    test_report.py                        # Task 8
    test_equivalence.py                   # Task 11 (parametrized over registry)
```

Boundary calls:

- `registry.py` owns `Spec` + `REGISTRY` + `@register`. No framework imports.
- `inputs.py`, `equivalence.py`, `timing.py` each have one job; import torch/jax inside functions (lazy) so the CLI can set env vars before they load.
- `harness.py` orchestrates a single `(spec, shape, dtype)` run, calling the four single-job modules.
- `cli.py` parses args and sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`XLA_FLAGS` *before* `from xtbench.harness import …`.
- `benchmarks/__init__.py` imports every category package; importing `benchmarks` is what populates the registry.

---

## Phase 0: One-time environment setup

### Task 0: Create a virtual env and install bench deps

This is a manual prep step, not committed. The plan's commands assume the venv is active.

- [ ] **Step 1: Create venv and activate**

```bash
cd /Users/xitrium/claud/xla-torch-bench
python3.11 -m venv .venv
source .venv/bin/activate
```

- [ ] **Step 2: Install torch, jax, numpy, pytest**

```bash
pip install --upgrade pip
pip install torch numpy pytest
pip install -U "jax[cpu]"
```

Expected: all four install cleanly. `python -c "import torch, jax, numpy; print(torch.__version__, jax.__version__)"` prints versions.

- [ ] **Step 3: Confirm `torch.compile` works on this machine**

```bash
python -c "import torch; m = torch.compile(torch.nn.Linear(4,4)); print(m(torch.zeros(1,4)).shape)"
```

Expected: `torch.Size([1, 4])` with no errors. (On macOS, Inductor needs a working `clang`.)

---

## Phase 1: Scaffolding

### Task 1: pyproject.toml + package init

**Files:**
- Create: `pyproject.toml`
- Create: `src/xtbench/__init__.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "xtbench"
version = "0.0.1"
description = "XLA:CPU vs PyTorch Inductor CPU benchmarks (rosetta-stone)."
requires-python = ">=3.11"
dependencies = [
  "numpy",
  "torch",
  "jax[cpu]",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
pythonpath = ["."]   # so `import benchmarks` works in tests
```

- [ ] **Step 2: Write src/xtbench/__init__.py (placeholder)**

```python
"""xtbench: XLA:CPU vs PyTorch Inductor CPU benchmarks."""
```

- [ ] **Step 3: Install editable**

```bash
pip install -e ".[dev]"
```

Expected: `Successfully installed xtbench-0.0.1` (or similar).

- [ ] **Step 4: Confirm pytest sees the empty test dir**

```bash
mkdir -p tests && touch tests/__init__.py
pytest -q
```

Expected: `no tests ran in <time>s`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/xtbench/__init__.py tests/__init__.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add pyproject and empty package skeleton.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Spec dataclass + register decorator + registry

**Files:**
- Create: `src/xtbench/registry.py`
- Modify: `src/xtbench/__init__.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

`tests/test_registry.py`:

```python
import pytest
from xtbench.registry import Spec, register, REGISTRY


def setup_function():
    REGISTRY.clear()


def test_register_adds_spec_to_registry():
    @register
    def my_bench():
        return Spec(
            torch_module=lambda d: object(),
            jax_fn=lambda d: (lambda x: x),
            shapes=[(4,)],
            dtypes=["f32"],
            params={"d": lambda s: s[-1]},
        )

    assert len(REGISTRY) == 1
    name, spec = REGISTRY[0]
    assert name == "my_bench"
    assert spec.shapes == [(4,)]
    assert spec.dtypes == ["f32"]


def test_register_rejects_duplicate_names():
    @register
    def dup():
        return Spec(torch_module=None, jax_fn=None, shapes=[(1,)], dtypes=["f32"])

    with pytest.raises(ValueError, match="already registered"):
        @register
        def dup():  # noqa: F811
            return Spec(torch_module=None, jax_fn=None, shapes=[(1,)], dtypes=["f32"])


def test_spec_default_params_and_inputs():
    spec = Spec(
        torch_module=lambda: None, jax_fn=lambda: None,
        shapes=[(8,)], dtypes=["f32"],
    )
    assert spec.params == {}
    assert callable(spec.inputs)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError: xtbench.registry`.

- [ ] **Step 3: Implement registry.py**

`src/xtbench/registry.py`:

```python
"""Spec dataclass and @register decorator. No framework imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


def _default_inputs(shape: tuple, dtype: str, seed: int) -> list[np.ndarray]:
    """Default: one random standard-normal input of the given shape."""
    rng = np.random.default_rng(seed)
    np_dtype = {"f32": np.float32, "bf16": np.float32}[dtype]
    # bf16 generated as f32; harness converts per-framework downstream.
    return [rng.standard_normal(shape).astype(np_dtype)]


@dataclass
class Spec:
    """One benchmark: a PyTorch module + a JAX function over the same compute."""
    torch_module: Callable[..., Any]
    jax_fn: Callable[..., Any]
    shapes: list[tuple[int, ...]]
    dtypes: list[str]
    params: dict[str, Callable[[tuple[int, ...]], Any]] = field(default_factory=dict)
    inputs: Callable[[tuple, str, int], list[np.ndarray]] = _default_inputs


REGISTRY: list[tuple[str, Spec]] = []


def register(fn: Callable[[], Spec]) -> Callable[[], Spec]:
    """Decorator: a zero-arg function returning a Spec becomes a registered benchmark."""
    name = fn.__name__
    if any(n == name for n, _ in REGISTRY):
        raise ValueError(f"benchmark {name!r} already registered")
    REGISTRY.append((name, fn()))
    return fn
```

- [ ] **Step 4: Re-export from __init__.py**

`src/xtbench/__init__.py`:

```python
"""xtbench: XLA:CPU vs PyTorch Inductor CPU benchmarks."""
from xtbench.registry import REGISTRY, Spec, register

__all__ = ["REGISTRY", "Spec", "register"]
```

- [ ] **Step 5: Run test to verify pass**

```bash
pytest tests/test_registry.py -v
```

Expected: all three tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/xtbench/__init__.py src/xtbench/registry.py tests/test_registry.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add Spec dataclass and @register decorator.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Environment / thread pinning

No tests here — this module exists purely for its side effects on `os.environ` and is verified by Task 9's CLI test. Keep it small.

**Files:**
- Create: `src/xtbench/env.py`

- [ ] **Step 1: Write env.py**

`src/xtbench/env.py`:

```python
"""Apply thread-pinning env vars. MUST be called before importing torch or jax."""
from __future__ import annotations

import os


def pin_threads(n: int) -> None:
    """Pin all CPU thread pools to n threads.

    Must run before torch/jax imports; OMP and MKL respect these vars at
    first allocation, and XLA reads XLA_FLAGS at jaxlib init.
    """
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    xla_flags = os.environ.get("XLA_FLAGS", "")
    extra = (
        f"--xla_cpu_multi_thread_eigen=true "
        f"--xla_cpu_eigen_num_threads={n}"
    )
    os.environ["XLA_FLAGS"] = (xla_flags + " " + extra).strip()


def belt_and_suspenders_torch(n: int) -> None:
    """Call AFTER `import torch` to clamp torch's intra-op pool."""
    import torch
    torch.set_num_threads(n)
```

- [ ] **Step 2: Smoke-test by hand**

```bash
python -c "
import xtbench.env as e
e.pin_threads(8)
import os
assert os.environ['OMP_NUM_THREADS'] == '8'
assert '--xla_cpu_eigen_num_threads=8' in os.environ['XLA_FLAGS']
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/xtbench/env.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add env.pin_threads — set OMP/MKL/XLA thread flags before framework imports.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Input generation utilities

**Files:**
- Create: `src/xtbench/inputs.py`
- Create: `tests/test_inputs.py`

- [ ] **Step 1: Write the failing test**

`tests/test_inputs.py`:

```python
import numpy as np
import pytest

from xtbench.inputs import to_torch, to_jax, dtype_atol


def test_to_torch_f32_roundtrip():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = to_torch(arr, "f32")
    assert tuple(t.shape) == (2, 2)
    assert str(t.dtype) == "torch.float32"
    np.testing.assert_array_equal(t.numpy(), arr)


def test_to_torch_bf16_downcasts():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    t = to_torch(arr, "bf16")
    assert str(t.dtype) == "torch.bfloat16"


def test_to_jax_f32_roundtrip():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    j = to_jax(arr, "f32")
    assert str(j.dtype) == "float32"
    np.testing.assert_array_equal(np.asarray(j), arr)


def test_to_jax_bf16_downcasts():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    j = to_jax(arr, "bf16")
    assert str(j.dtype) == "bfloat16"


def test_dtype_atol():
    assert dtype_atol("f32") == 1e-4
    assert dtype_atol("bf16") == 1e-2


def test_unknown_dtype_raises():
    with pytest.raises(ValueError, match="unknown dtype"):
        to_torch(np.zeros(1, dtype=np.float32), "f16")
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_inputs.py -v
```

Expected: FAIL — `ModuleNotFoundError: xtbench.inputs`.

- [ ] **Step 3: Implement inputs.py**

`src/xtbench/inputs.py`:

```python
"""Convert numpy inputs into framework tensors at the right dtype."""
from __future__ import annotations

import numpy as np


def _check_dtype(dtype: str) -> None:
    if dtype not in ("f32", "bf16"):
        raise ValueError(f"unknown dtype: {dtype!r}")


def to_torch(arr: np.ndarray, dtype: str):
    _check_dtype(dtype)
    import torch
    t = torch.from_numpy(arr.astype(np.float32, copy=False))
    if dtype == "bf16":
        t = t.to(torch.bfloat16)
    return t


def to_jax(arr: np.ndarray, dtype: str):
    _check_dtype(dtype)
    import jax.numpy as jnp
    target = {"f32": jnp.float32, "bf16": jnp.bfloat16}[dtype]
    return jnp.asarray(arr).astype(target)


def dtype_atol(dtype: str) -> float:
    _check_dtype(dtype)
    return {"f32": 1e-4, "bf16": 1e-2}[dtype]
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_inputs.py -v
```

Expected: all six tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xtbench/inputs.py tests/test_inputs.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add inputs.to_torch / to_jax / dtype_atol helpers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Equivalence check

**Files:**
- Create: `src/xtbench/equivalence.py`
- Create: `tests/test_equivalence_check.py`

- [ ] **Step 1: Write the failing test**

`tests/test_equivalence_check.py`:

```python
import numpy as np
import pytest

from xtbench.equivalence import assert_close, EquivalenceError


def test_close_values_pass():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = a + 1e-6
    assert_close(a, b, atol=1e-4)


def test_far_values_raise():
    a = np.array([1.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    with pytest.raises(EquivalenceError, match="max abs diff"):
        assert_close(a, b, atol=1e-4)


def test_shape_mismatch_raises():
    a = np.zeros((2, 2), dtype=np.float32)
    b = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(EquivalenceError, match="shape mismatch"):
        assert_close(a, b, atol=1e-4)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_equivalence_check.py -v
```

Expected: FAIL — `ModuleNotFoundError: xtbench.equivalence`.

- [ ] **Step 3: Implement equivalence.py**

`src/xtbench/equivalence.py`:

```python
"""Compare a torch output and a jax output as numpy arrays."""
from __future__ import annotations

import numpy as np


class EquivalenceError(AssertionError):
    pass


def assert_close(a: np.ndarray, b: np.ndarray, atol: float) -> None:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise EquivalenceError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b).max()
    if not np.isfinite(diff) or diff > atol:
        raise EquivalenceError(
            f"max abs diff {diff:.3g} exceeds atol {atol:.3g}"
        )
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_equivalence_check.py -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xtbench/equivalence.py tests/test_equivalence_check.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add equivalence.assert_close with shape and atol checks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Timing primitives

**Files:**
- Create: `src/xtbench/timing.py`
- Create: `tests/test_timing.py`

The two timing helpers (`time_torch`, `time_jax`) wrap one call of a compiled function and return elapsed nanoseconds. The JAX variant calls `.block_until_ready()` on the result to defeat async dispatch.

- [ ] **Step 1: Write the failing test**

`tests/test_timing.py`:

```python
import time
import numpy as np
import pytest

from xtbench.timing import time_torch, time_jax, Stats, summarize


def test_summarize_basic():
    s = summarize([1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000])
    assert isinstance(s, Stats)
    assert s.median_ms == pytest.approx(3.0, rel=1e-6)
    assert s.p10_ms == pytest.approx(1.0, rel=1e-6)
    assert s.p90_ms == pytest.approx(5.0, rel=1e-6)


def test_time_torch_runs_callable():
    import torch
    x = torch.zeros(4)
    elapsed = time_torch(lambda: x + 1)
    assert elapsed > 0


def test_time_jax_runs_callable_and_blocks():
    import jax.numpy as jnp
    x = jnp.zeros(4)
    elapsed = time_jax(lambda: x + 1)
    assert elapsed > 0
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_timing.py -v
```

Expected: FAIL — `ModuleNotFoundError: xtbench.timing`.

- [ ] **Step 3: Implement timing.py**

`src/xtbench/timing.py`:

```python
"""Per-call timing for torch and jax, and stats summary."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


def time_torch(fn: Callable[[], object]) -> int:
    """Time one call of a torch-returning callable. Returns nanoseconds."""
    t0 = time.perf_counter_ns()
    out = fn()
    # CPU torch is synchronous; no explicit sync needed.
    _ = out
    return time.perf_counter_ns() - t0


def time_jax(fn: Callable[[], object]) -> int:
    """Time one call of a jax-returning callable; defeat async dispatch."""
    t0 = time.perf_counter_ns()
    out = fn()
    out.block_until_ready()
    return time.perf_counter_ns() - t0


@dataclass
class Stats:
    median_ms: float
    p10_ms: float
    p90_ms: float


def summarize(samples_ns: list[int]) -> Stats:
    a = np.asarray(samples_ns, dtype=np.float64) / 1e6  # → ms
    return Stats(
        median_ms=float(np.median(a)),
        p10_ms=float(np.percentile(a, 10)),
        p90_ms=float(np.percentile(a, 90)),
    )
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_timing.py -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xtbench/timing.py tests/test_timing.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add timing helpers: time_torch, time_jax, summarize.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Harness — run_benchmark orchestration

This is the integration point. It takes a `Spec`, one `shape`, one `dtype`, runs both frameworks end-to-end (compile, equivalence check, warmup, time), and returns a `Result`.

**Files:**
- Create: `src/xtbench/harness.py`
- Create: `tests/test_harness.py`

- [ ] **Step 1: Write the failing test**

`tests/test_harness.py`:

```python
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
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_harness.py -v
```

Expected: FAIL — `ModuleNotFoundError: xtbench.harness`.

- [ ] **Step 3: Implement harness.py**

`src/xtbench/harness.py`:

```python
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
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_harness.py -v
```

Expected: both tests PASS. (First run will be slow because `torch.compile` warms up Inductor on the trivial module.)

- [ ] **Step 5: Commit**

```bash
git add src/xtbench/harness.py tests/test_harness.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add run_benchmark: orchestrates compile, equivalence, warmup, timing per (spec, shape, dtype).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Reporting

**Files:**
- Create: `src/xtbench/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:

```python
from xtbench.harness import Result
from xtbench.report import format_row, format_table, RowKey
from xtbench.timing import Stats


def _result(t_med, j_med):
    return Result(
        torch=Stats(median_ms=t_med, p10_ms=t_med * 0.95, p90_ms=t_med * 1.05),
        jax=Stats(median_ms=j_med, p10_ms=j_med * 0.95, p90_ms=j_med * 1.05),
        compile_ms_torch=100.0, compile_ms_jax=80.0,
        equiv_ok=True,
    )


def test_format_row_speedup_gt_1():
    key = RowKey(name="rms_norm", shape=(1, 1024, 768), dtype="f32")
    row = format_row(key, _result(t_med=1.0, j_med=0.5))
    assert "rms_norm" in row
    assert "2.00x" in row


def test_format_row_equiv_fail():
    key = RowKey(name="bad", shape=(8,), dtype="f32")
    r = Result(
        torch=Stats(0, 0, 0), jax=Stats(0, 0, 0),
        compile_ms_torch=0, compile_ms_jax=0,
        equiv_ok=False, equiv_error="max abs diff 1.0",
    )
    row = format_row(key, r)
    assert "EQUIV FAIL" in row


def test_format_table_has_header_and_compile_summary():
    rows = [
        (RowKey("rms_norm", (1, 1024, 768), "f32"), _result(1.0, 0.5)),
        (RowKey("rms_norm", (1, 1024, 768), "bf16"), _result(0.8, 0.4)),
    ]
    out = format_table(rows)
    assert "benchmark" in out and "speedup" in out
    assert "rms_norm" in out
    assert "compile time" in out
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_report.py -v
```

Expected: FAIL — `ModuleNotFoundError: xtbench.report`.

- [ ] **Step 3: Implement report.py**

`src/xtbench/report.py`:

```python
"""Format a list of (RowKey, Result) into a human-readable stdout table."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from xtbench.harness import Result


@dataclass(frozen=True)
class RowKey:
    name: str
    shape: tuple
    dtype: str


def _fmt_stat(s) -> str:
    return f"{s.median_ms:.2f} [{s.p10_ms:.2f},{s.p90_ms:.2f}]"


_HEADER = f"{'benchmark':<24} {'shape':<20} {'dtype':<6} {'torch (ms)':<22} {'xla (ms)':<22} {'speedup':<8}"


def format_row(key: RowKey, r: Result) -> str:
    name = key.name[:24]
    shape = str(tuple(key.shape))[:20]
    if not r.equiv_ok:
        return f"{name:<24} {shape:<20} {key.dtype:<6} EQUIV FAIL ({r.equiv_error})"
    speedup = r.torch.median_ms / r.jax.median_ms if r.jax.median_ms > 0 else float("nan")
    return (
        f"{name:<24} {shape:<20} {key.dtype:<6} "
        f"{_fmt_stat(r.torch):<22} {_fmt_stat(r.jax):<22} {speedup:.2f}x"
    )


def format_table(rows: list[tuple[RowKey, Result]]) -> str:
    lines = [_HEADER, "-" * len(_HEADER)]
    for key, r in rows:
        lines.append(format_row(key, r))
    if rows:
        torch_c = median([r.compile_ms_torch for _, r in rows if r.equiv_ok])
        jax_c = median([r.compile_ms_jax for _, r in rows if r.equiv_ok])
        lines.append("")
        lines.append(
            f"compile time (median, first-call): "
            f"torch {torch_c:.0f} ms / xla {jax_c:.0f} ms across {len(rows)} benchmarks"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_report.py -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xtbench/report.py tests/test_report.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add stdout table formatter (report.format_row / format_table).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: CLI

**Files:**
- Create: `src/xtbench/cli.py`
- Create: `src/xtbench/__main__.py`

The CLI parses args, pins threads, then imports the rest of the package (which imports torch/jax). Order matters.

- [ ] **Step 1: Implement cli.py**

`src/xtbench/cli.py`:

```python
"""xtbench CLI. Sets env vars before importing torch/jax."""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Sequence


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="xtbench")
    p.add_argument("--filter", default=".*",
                   help="regex on benchmark name (default: match all)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                   help="thread pool size for torch/jax/OpenMP")
    p.add_argument("--dtypes", default="",
                   help="comma list of dtypes (default: all in spec)")
    p.add_argument("--shapes", default="",
                   help="comma list of '(d0,d1,...)' tuples; overrides spec shapes")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    return p.parse_args(argv)


def _parse_shapes(s: str) -> list[tuple] | None:
    if not s.strip():
        return None
    out = []
    for chunk in s.split("),"):
        chunk = chunk.strip().strip("()")
        if not chunk:
            continue
        out.append(tuple(int(x) for x in chunk.split(",") if x.strip()))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Pin threads BEFORE importing torch/jax.
    from xtbench.env import pin_threads, belt_and_suspenders_torch
    pin_threads(args.threads)

    # Now safe to import everything else.
    import benchmarks  # noqa: F401  (populates registry as a side effect)
    from xtbench.harness import run_benchmark
    from xtbench.registry import REGISTRY
    from xtbench.report import RowKey, format_table

    belt_and_suspenders_torch(args.threads)

    pat = re.compile(args.filter)
    dtype_filter = set(d.strip() for d in args.dtypes.split(",") if d.strip())
    shape_override = _parse_shapes(args.shapes)

    rows = []
    for name, spec in REGISTRY:
        if not pat.search(name):
            continue
        shapes = shape_override if shape_override else spec.shapes
        for shape in shapes:
            for dtype in spec.dtypes:
                if dtype_filter and dtype not in dtype_filter:
                    continue
                result = run_benchmark(
                    spec, shape=shape, dtype=dtype,
                    warmup=args.warmup, iters=args.iters,
                )
                rows.append((RowKey(name, shape, dtype), result))

    print(format_table(rows))
    return 0
```

- [ ] **Step 2: Implement __main__.py**

`src/xtbench/__main__.py`:

```python
from xtbench.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Smoke test (no benchmarks yet — should print empty table)**

```bash
python -m xtbench --threads 4 2>&1 | head -5
```

Expected: prints the table header and a `compile time …` line absent, OR a clean empty output (no crash). If it crashes on `import benchmarks`, that's fine — Task 10 creates the package.

- [ ] **Step 4: Commit**

```bash
git add src/xtbench/cli.py src/xtbench/__main__.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add CLI entry point (python -m xtbench). Pins threads before framework imports.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: First benchmark + auto-equivalence test

### Task 10: First benchmark — rms_norm

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/norms/__init__.py`
- Create: `benchmarks/norms/rms_norm.py`

- [ ] **Step 1: Create the benchmarks package**

`benchmarks/__init__.py`:

```python
"""Importing this package populates xtbench.registry.REGISTRY."""
from benchmarks import norms  # noqa: F401
```

`benchmarks/norms/__init__.py`:

```python
from benchmarks.norms import rms_norm  # noqa: F401
```

- [ ] **Step 2: Write the rms_norm benchmark**

`benchmarks/norms/rms_norm.py`:

```python
"""RMSNorm — rsqrt(mean(x^2) + eps) * x * w."""
from xtbench import Spec, register


@register
def rms_norm() -> Spec:
    def torch_module(d):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(d))

            def forward(self, x):
                rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                return x * rms * self.w
        return M()

    def jax_fn(d):
        import jax
        import jax.numpy as jnp
        w = jnp.ones((d,))

        def f(x):
            rms = jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
            return x * rms * w
        return f

    return Spec(
        torch_module=torch_module,
        jax_fn=jax_fn,
        shapes=[(1, 1024, 768), (1, 4096, 768)],
        dtypes=["f32", "bf16"],
        params={"d": lambda s: s[-1]},
    )
```

- [ ] **Step 3: Run the CLI end-to-end on this one benchmark**

```bash
python -m xtbench --filter rms_norm --threads 4 --warmup 2 --iters 5
```

Expected: a table with 4 rows (2 shapes × 2 dtypes), all marked equivalent, with positive ms timings for both columns and a speedup figure per row. No crashes.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/__init__.py benchmarks/norms/__init__.py benchmarks/norms/rms_norm.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add first benchmark: rms_norm (norms category).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Parametrized equivalence test over the registry

**Files:**
- Create: `tests/test_equivalence.py`

- [ ] **Step 1: Write the test**

`tests/test_equivalence.py`:

```python
"""Rosetta-stone invariant: for every (spec, shape, dtype), torch ≈ jax."""
import numpy as np
import pytest

import benchmarks  # noqa: F401  (populates registry)
from xtbench.equivalence import assert_close
from xtbench.inputs import dtype_atol, to_jax, to_torch
from xtbench.registry import REGISTRY


def _cases():
    for name, spec in REGISTRY:
        for shape in spec.shapes:
            for dtype in spec.dtypes:
                yield pytest.param(name, spec, shape, dtype,
                                   id=f"{name}-{shape}-{dtype}")


@pytest.mark.parametrize("name,spec,shape,dtype", list(_cases()))
def test_equivalence(name, spec, shape, dtype):
    import torch
    import jax

    kwargs = {k: fn(shape) for k, fn in spec.params.items()}
    np_inputs = spec.inputs(shape, dtype, seed=0)

    m = spec.torch_module(**kwargs)
    torch_inputs = [to_torch(a, dtype) for a in np_inputs]
    with torch.no_grad():
        torch_out = m(*torch_inputs).detach().to(torch.float32).numpy()

    f = spec.jax_fn(**kwargs)
    jax_inputs = [to_jax(a, dtype) for a in np_inputs]
    jax_out = np.asarray(f(*jax_inputs), dtype=np.float32)

    assert_close(torch_out, jax_out, atol=dtype_atol(dtype))
```

- [ ] **Step 2: Run it**

```bash
pytest tests/test_equivalence.py -v
```

Expected: 4 PASS (2 shapes × 2 dtypes for `rms_norm`). bf16 may be close to the 1e-2 tolerance; if it consistently fails, raise atol in `inputs.dtype_atol` to 5e-2 for `bf16` (note the change in commit message).

- [ ] **Step 3: Commit**

```bash
git add tests/test_equivalence.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add parametrized equivalence test over the full registry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Microbenchmark seed

Each benchmark in this phase is a small additive file. Pattern for every one:

1. Write the file under `benchmarks/<category>/<name>.py`.
2. Add `from benchmarks.<category> import <name>  # noqa: F401` to that category's `__init__.py`.
3. Run `pytest tests/test_equivalence.py -k <name>` — must pass.
4. Run `python -m xtbench --filter <name> --warmup 2 --iters 5` — must produce a row.
5. Commit.

If a benchmark fails equivalence, the rosetta-stone is broken; fix the math before continuing.

### Task 12: Elementwise benchmarks

**Files:**
- Create: `benchmarks/elementwise/__init__.py`
- Create: `benchmarks/elementwise/add_chain.py`
- Create: `benchmarks/elementwise/gelu_chain.py`
- Create: `benchmarks/elementwise/silu_mul.py`
- Create: `benchmarks/elementwise/add_mul_aligned_vs_misaligned.py`
- Modify: `benchmarks/__init__.py` (add `elementwise` import)

- [ ] **Step 1: Wire up the category**

`benchmarks/elementwise/__init__.py`:

```python
from benchmarks.elementwise import (  # noqa: F401
    add_chain, gelu_chain, silu_mul, add_mul_aligned_vs_misaligned,
)
```

Update `benchmarks/__init__.py`:

```python
"""Importing this package populates xtbench.registry.REGISTRY."""
from benchmarks import norms, elementwise  # noqa: F401
```

- [ ] **Step 2: `add_chain` — 4-arg pointwise add**

`benchmarks/elementwise/add_chain.py`:

```python
"""a + b + c + d over a 3D tensor — pure pointwise fusion target."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32) for _ in range(4)]


@register
def add_chain() -> Spec:
    def torch_module():
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b, c, d):
                return a + b + c + d
        return M()

    def jax_fn():
        def f(a, b, c, d):
            return a + b + c + d
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768), (1, 4096, 768)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )
```

- [ ] **Step 3: `gelu_chain` — linear → gelu → linear**

`benchmarks/elementwise/gelu_chain.py`:

```python
"""Linear → GELU → Linear. Weights set to identity for rosetta-stone equivalence."""
from xtbench import Spec, register


@register
def gelu_chain() -> Spec:
    def torch_module(d):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                w = torch.eye(d)
                self.w1 = nn.Parameter(w.clone())
                self.w2 = nn.Parameter(w.clone())

            def forward(self, x):
                y = x @ self.w1
                y = torch.nn.functional.gelu(y, approximate="tanh")
                return y @ self.w2
        return M()

    def jax_fn(d):
        import jax
        import jax.numpy as jnp
        w1 = jnp.eye(d)
        w2 = jnp.eye(d)

        def f(x):
            y = x @ w1
            y = jax.nn.gelu(y, approximate=True)
            return y @ w2
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768)],
        dtypes=["f32", "bf16"],
        params={"d": lambda s: s[-1]},
    )
```

- [ ] **Step 4: `silu_mul` — silu(a) * b (the SwiGLU half)**

`benchmarks/elementwise/silu_mul.py`:

```python
"""silu(a) * b — pure pointwise."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32),
            rng.standard_normal(shape).astype(np.float32)]


@register
def silu_mul() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b):
                return torch.nn.functional.silu(a) * b
        return M()

    def jax_fn():
        import jax
        def f(a, b):
            return jax.nn.silu(a) * b
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768), (1, 4096, 768)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )
```

- [ ] **Step 5: `add_mul_aligned_vs_misaligned` — vector boundary stress**

`benchmarks/elementwise/add_mul_aligned_vs_misaligned.py`:

```python
"""(a + b) * c at shapes that hit vs miss vector boundaries.

The 'aligned' shape (d=1024) is a multiple of the 16-lane f32 vector size on
AVX-512 and the 8-lane vector size on NEON. The 'misaligned' shape (d=1023)
forces a scalar tail loop — this is the case xtile-aligned-ops targets.
"""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32) for _ in range(3)]


@register
def add_mul_aligned_vs_misaligned() -> Spec:
    def torch_module():
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b, c):
                return (a + b) * c
        return M()

    def jax_fn():
        def f(a, b, c):
            return (a + b) * c
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 4096, 1024), (1, 4096, 1023), (1, 4096, 1025)],
        dtypes=["f32"],
        inputs=_inputs,
    )
```

- [ ] **Step 6: Run equivalence tests for elementwise**

```bash
pytest tests/test_equivalence.py -v -k "add_chain or gelu_chain or silu_mul or add_mul"
```

Expected: all PASS.

- [ ] **Step 7: Run the CLI on all elementwise**

```bash
python -m xtbench --filter "add_chain|gelu_chain|silu_mul|add_mul" --warmup 2 --iters 5
```

Expected: one row per (benchmark, shape, dtype). Manually sanity-check the misaligned row shows a different speedup than the aligned one.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/elementwise/ benchmarks/__init__.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add elementwise benchmarks: add_chain, gelu_chain, silu_mul, add_mul_aligned_vs_misaligned.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Remaining norms (layer_norm, softmax, sum_axis, mean_axis)

**Files:**
- Create: `benchmarks/norms/layer_norm.py`
- Create: `benchmarks/norms/softmax.py`
- Create: `benchmarks/norms/sum_axis.py`
- Create: `benchmarks/norms/mean_axis.py`
- Modify: `benchmarks/norms/__init__.py`

- [ ] **Step 1: Wire up the imports**

Update `benchmarks/norms/__init__.py`:

```python
from benchmarks.norms import (  # noqa: F401
    rms_norm, layer_norm, softmax, sum_axis, mean_axis,
)
```

- [ ] **Step 2: layer_norm**

`benchmarks/norms/layer_norm.py`:

```python
"""LayerNorm on last dim, with affine weights."""
from xtbench import Spec, register


@register
def layer_norm() -> Spec:
    def torch_module(d):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(d, eps=1e-5)
                with torch.no_grad():
                    self.ln.weight.fill_(1.0)
                    self.ln.bias.fill_(0.0)

            def forward(self, x):
                return self.ln(x)
        return M()

    def jax_fn(d):
        import jax.numpy as jnp
        w = jnp.ones((d,))
        b = jnp.zeros((d,))

        def f(x):
            mean = jnp.mean(x, -1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, -1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5) * w + b
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768), (1, 4096, 768)],
        dtypes=["f32", "bf16"],
        params={"d": lambda s: s[-1]},
    )
```

- [ ] **Step 3: softmax**

`benchmarks/norms/softmax.py`:

```python
"""softmax on last dim."""
from xtbench import Spec, register


@register
def softmax() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=-1)
        return M()

    def jax_fn():
        import jax
        def f(x):
            return jax.nn.softmax(x, axis=-1)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 12, 512, 512), (1, 16, 1024, 1024)],
        dtypes=["f32", "bf16"],
    )
```

- [ ] **Step 4: sum_axis**

`benchmarks/norms/sum_axis.py`:

```python
"""sum along last dim (single reduction, no normalization tail)."""
from xtbench import Spec, register


@register
def sum_axis() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                return torch.sum(x, dim=-1)
        return M()

    def jax_fn():
        import jax.numpy as jnp
        def f(x):
            return jnp.sum(x, axis=-1)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 4096, 1024), (1, 1024, 4096)],
        dtypes=["f32", "bf16"],
    )
```

- [ ] **Step 5: mean_axis**

`benchmarks/norms/mean_axis.py`:

```python
"""mean along last dim."""
from xtbench import Spec, register


@register
def mean_axis() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                return torch.mean(x, dim=-1)
        return M()

    def jax_fn():
        import jax.numpy as jnp
        def f(x):
            return jnp.mean(x, axis=-1)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 4096, 1024), (1, 1024, 4096)],
        dtypes=["f32", "bf16"],
    )
```

- [ ] **Step 6: Run equivalence and a quick smoke run**

```bash
pytest tests/test_equivalence.py -v -k "layer_norm or softmax or sum_axis or mean_axis"
python -m xtbench --filter "layer_norm|softmax|sum_axis|mean_axis" --warmup 2 --iters 5
```

Expected: equivalence PASS for all; CLI prints one row per case.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/norms/
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add norm benchmarks: layer_norm, softmax, sum_axis, mean_axis.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: GEMM benchmarks

**Files:**
- Create: `benchmarks/gemm/__init__.py`
- Create: `benchmarks/gemm/dot_small.py`
- Create: `benchmarks/gemm/dot_medium.py`
- Create: `benchmarks/gemm/dot_bias_gelu.py`
- Create: `benchmarks/gemm/dot_transpose.py`
- Modify: `benchmarks/__init__.py`

- [ ] **Step 1: Wire up**

`benchmarks/gemm/__init__.py`:

```python
from benchmarks.gemm import (  # noqa: F401
    dot_small, dot_medium, dot_bias_gelu, dot_transpose,
)
```

Update `benchmarks/__init__.py`:

```python
"""Importing this package populates xtbench.registry.REGISTRY."""
from benchmarks import norms, elementwise, gemm  # noqa: F401
```

- [ ] **Step 2: dot_small — 256×256×256, two-input**

`benchmarks/gemm/dot_small.py`:

```python
"""Small matmul: (M, K) @ (K, N) with M=K=N=256."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    m, k, n = shape
    return [
        rng.standard_normal((m, k)).astype(np.float32),
        rng.standard_normal((k, n)).astype(np.float32),
    ]


@register
def dot_small() -> Spec:
    def torch_module():
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b):
                return a @ b
        return M()

    def jax_fn():
        def f(a, b):
            return a @ b
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(256, 256, 256)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )
```

- [ ] **Step 3: dot_medium — 1024³**

`benchmarks/gemm/dot_medium.py`:

```python
"""Medium matmul: 1024 × 1024 × 1024."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    m, k, n = shape
    return [
        rng.standard_normal((m, k)).astype(np.float32),
        rng.standard_normal((k, n)).astype(np.float32),
    ]


@register
def dot_medium() -> Spec:
    def torch_module():
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b):
                return a @ b
        return M()

    def jax_fn():
        def f(a, b):
            return a @ b
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1024, 1024, 1024)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )
```

- [ ] **Step 4: dot_bias_gelu — fused trio**

`benchmarks/gemm/dot_bias_gelu.py`:

```python
"""y = gelu(x @ w + b). Tests whether the bias add + activation fuse with the dot."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    b, m, k = shape
    return [rng.standard_normal((b, m, k)).astype(np.float32)]


@register
def dot_bias_gelu() -> Spec:
    def torch_module(k, n):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(k, n)
                with torch.no_grad():
                    self.lin.weight.fill_(0.01)
                    self.lin.bias.fill_(0.1)

            def forward(self, x):
                return torch.nn.functional.gelu(self.lin(x), approximate="tanh")
        return M()

    def jax_fn(k, n):
        import jax
        import jax.numpy as jnp
        w = jnp.full((k, n), 0.01)
        b = jnp.full((n,), 0.1)

        def f(x):
            return jax.nn.gelu(x @ w + b, approximate=True)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768)],
        dtypes=["f32", "bf16"],
        params={"k": lambda s: s[-1], "n": lambda s: s[-1]},
        inputs=_inputs,
    )
```

NB: torch's `nn.Linear(k, n)` computes `x @ w.T + b`. The JAX side here computes `x @ w + b` with `w` of shape `(k, n)`. To make them equivalent, set torch's weight as the *transpose* of the jax weight. Since both are constant 0.01, `w == w.T` and they match.

- [ ] **Step 5: dot_transpose — exercises transpose+gemm fusion**

`benchmarks/gemm/dot_transpose.py`:

```python
"""y = a @ b.T. XLA and Inductor may handle the transpose differently."""
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    m, k, n = shape
    return [
        rng.standard_normal((m, k)).astype(np.float32),
        rng.standard_normal((n, k)).astype(np.float32),
    ]


@register
def dot_transpose() -> Spec:
    def torch_module():
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, a, b):
                return a @ b.transpose(-1, -2)
        return M()

    def jax_fn():
        import jax.numpy as jnp
        def f(a, b):
            return a @ jnp.swapaxes(b, -1, -2)
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(512, 768, 512)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )
```

- [ ] **Step 6: Equivalence + smoke run**

```bash
pytest tests/test_equivalence.py -v -k "dot_"
python -m xtbench --filter "dot_" --warmup 2 --iters 5
```

Expected: all equivalence checks PASS. (Tolerance on bf16 matmul-of-1024³ is close to the 1e-2 floor; if it fails, bump to 5e-2 — bf16 GEMM accumulates error at K terms.)

- [ ] **Step 7: Commit**

```bash
git add benchmarks/gemm/ benchmarks/__init__.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add GEMM benchmarks: dot_small, dot_medium, dot_bias_gelu, dot_transpose.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Attention benchmarks

**Files:**
- Create: `benchmarks/attention/__init__.py`
- Create: `benchmarks/attention/mha_block.py`
- Create: `benchmarks/attention/gqa_block.py`
- Modify: `benchmarks/__init__.py`

- [ ] **Step 1: Wire up**

`benchmarks/attention/__init__.py`:

```python
from benchmarks.attention import mha_block, gqa_block  # noqa: F401
```

Update `benchmarks/__init__.py`:

```python
"""Importing this package populates xtbench.registry.REGISTRY."""
from benchmarks import norms, elementwise, gemm, attention  # noqa: F401
```

- [ ] **Step 2: mha_block — Q@K → softmax → @V**

`benchmarks/attention/mha_block.py`:

```python
"""Scaled dot-product attention block (no masking).

Shape convention: (batch, num_heads, seq, head_dim).
"""
import math
import numpy as np

from xtbench import Spec, register


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
    ]


@register
def mha_block() -> Spec:
    def torch_module(scale):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = scale

            def forward(self, q, k, v):
                attn = (q @ k.transpose(-1, -2)) * self.scale
                attn = torch.softmax(attn, dim=-1)
                return attn @ v
        return M()

    def jax_fn(scale):
        import jax
        import jax.numpy as jnp
        def f(q, k, v):
            attn = (q @ jnp.swapaxes(k, -1, -2)) * scale
            attn = jax.nn.softmax(attn, axis=-1)
            return attn @ v
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[
            (1, 12, 128, 64),
            (1, 12, 512, 64),
            (1, 12, 2048, 64),
        ],
        dtypes=["f32", "bf16"],
        params={"scale": lambda s: 1.0 / math.sqrt(s[-1])},
        inputs=_inputs,
    )
```

- [ ] **Step 3: gqa_block — grouped query attention (LLM shapes)**

`benchmarks/attention/gqa_block.py`:

```python
"""Grouped-query attention: kv heads are repeated across query heads.

Shape: q is (b, h_q, s, d); k, v are (b, h_kv, s, d). We expand k/v to match.
"""
import math
import numpy as np

from xtbench import Spec, register

H_Q = 32
H_KV = 8
HEAD_D = 128


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    b, seq = shape
    q = rng.standard_normal((b, H_Q, seq, HEAD_D)).astype(np.float32)
    k = rng.standard_normal((b, H_KV, seq, HEAD_D)).astype(np.float32)
    v = rng.standard_normal((b, H_KV, seq, HEAD_D)).astype(np.float32)
    return [q, k, v]


@register
def gqa_block() -> Spec:
    def torch_module(scale):
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = scale

            def forward(self, q, k, v):
                # Expand kv heads to match query heads: (b, H_KV, s, d) -> (b, H_Q, s, d)
                repeat = H_Q // H_KV
                k_full = k.repeat_interleave(repeat, dim=1)
                v_full = v.repeat_interleave(repeat, dim=1)
                attn = (q @ k_full.transpose(-1, -2)) * self.scale
                attn = torch.softmax(attn, dim=-1)
                return attn @ v_full
        return M()

    def jax_fn(scale):
        import jax
        import jax.numpy as jnp
        def f(q, k, v):
            repeat = H_Q // H_KV
            k_full = jnp.repeat(k, repeat, axis=1)
            v_full = jnp.repeat(v, repeat, axis=1)
            attn = (q @ jnp.swapaxes(k_full, -1, -2)) * scale
            attn = jax.nn.softmax(attn, axis=-1)
            return attn @ v_full
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 512), (1, 2048)],
        dtypes=["f32", "bf16"],
        params={"scale": lambda s: 1.0 / math.sqrt(HEAD_D)},
        inputs=_inputs,
    )
```

- [ ] **Step 4: Equivalence + smoke run**

```bash
pytest tests/test_equivalence.py -v -k "mha_block or gqa_block"
python -m xtbench --filter "mha_block|gqa_block" --warmup 2 --iters 5
```

Expected: PASS. bf16 attention at seq=2048 is the tightest case; if equiv fails, bump bf16 atol.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/attention/ benchmarks/__init__.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add attention benchmarks: mha_block, gqa_block.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: E2E benchmarks

### Task 16: E2E block benchmarks

Pure-forward, randomly initialized, no tokenizers, no weight downloads. Two blocks to give one transformer-style and one attention-free profile.

**Files:**
- Create: `benchmarks/e2e/__init__.py`
- Create: `benchmarks/e2e/gpt2_block.py`
- Create: `benchmarks/e2e/mlp_mixer_block.py`
- Modify: `benchmarks/__init__.py`

- [ ] **Step 1: Wire up**

`benchmarks/e2e/__init__.py`:

```python
from benchmarks.e2e import gpt2_block, mlp_mixer_block  # noqa: F401
```

Update `benchmarks/__init__.py`:

```python
"""Importing this package populates xtbench.registry.REGISTRY."""
from benchmarks import norms, elementwise, gemm, attention, e2e  # noqa: F401
```

- [ ] **Step 2: gpt2_block**

`benchmarks/e2e/gpt2_block.py`:

```python
"""One GPT-2 transformer block: pre-LN, MHA, residual, pre-LN, MLP (4x), residual.

Weights set to a fixed deterministic constant on both sides for rosetta-stone
equivalence (we are NOT loading HF GPT-2 weights — this is a forward-pass
codegen comparison).
"""
import math
import numpy as np

from xtbench import Spec, register

D_MODEL = 768
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS
D_MLP = 4 * D_MODEL
W_CONST = 0.02  # deterministic constant for all weights


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32)]


@register
def gpt2_block() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        def _fill(m, v=W_CONST):
            with torch.no_grad():
                for p in m.parameters():
                    p.fill_(v)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = nn.LayerNorm(D_MODEL); _fill(self.ln1, 1.0)
                with torch.no_grad():
                    self.ln1.bias.fill_(0.0)
                self.q = nn.Linear(D_MODEL, D_MODEL, bias=False); _fill(self.q)
                self.k = nn.Linear(D_MODEL, D_MODEL, bias=False); _fill(self.k)
                self.v = nn.Linear(D_MODEL, D_MODEL, bias=False); _fill(self.v)
                self.o = nn.Linear(D_MODEL, D_MODEL, bias=False); _fill(self.o)
                self.ln2 = nn.LayerNorm(D_MODEL); _fill(self.ln2, 1.0)
                with torch.no_grad():
                    self.ln2.bias.fill_(0.0)
                self.fc1 = nn.Linear(D_MODEL, D_MLP, bias=False); _fill(self.fc1)
                self.fc2 = nn.Linear(D_MLP, D_MODEL, bias=False); _fill(self.fc2)

            def forward(self, x):
                b, s, _ = x.shape
                h = self.ln1(x)
                q = self.q(h).view(b, s, N_HEADS, D_HEAD).transpose(1, 2)
                k = self.k(h).view(b, s, N_HEADS, D_HEAD).transpose(1, 2)
                v = self.v(h).view(b, s, N_HEADS, D_HEAD).transpose(1, 2)
                attn = (q @ k.transpose(-1, -2)) / math.sqrt(D_HEAD)
                attn = torch.softmax(attn, dim=-1)
                out = (attn @ v).transpose(1, 2).reshape(b, s, D_MODEL)
                x = x + self.o(out)
                h = self.ln2(x)
                h = torch.nn.functional.gelu(self.fc1(h), approximate="tanh")
                return x + self.fc2(h)
        return Block()

    def jax_fn():
        import jax
        import jax.numpy as jnp

        wq = jnp.full((D_MODEL, D_MODEL), W_CONST)
        wk = jnp.full((D_MODEL, D_MODEL), W_CONST)
        wv = jnp.full((D_MODEL, D_MODEL), W_CONST)
        wo = jnp.full((D_MODEL, D_MODEL), W_CONST)
        w1 = jnp.full((D_MODEL, D_MLP), W_CONST)
        w2 = jnp.full((D_MLP, D_MODEL), W_CONST)

        def _ln(x):
            mean = jnp.mean(x, -1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, -1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5)

        def f(x):
            b, s, _ = x.shape
            h = _ln(x)
            q = (h @ wq).reshape(b, s, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
            k = (h @ wk).reshape(b, s, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
            v = (h @ wv).reshape(b, s, N_HEADS, D_HEAD).transpose(0, 2, 1, 3)
            attn = (q @ jnp.swapaxes(k, -1, -2)) / math.sqrt(D_HEAD)
            attn = jax.nn.softmax(attn, axis=-1)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, D_MODEL)
            x = x + out @ wo
            h = _ln(x)
            h = jax.nn.gelu(h @ w1, approximate=True)
            return x + h @ w2
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 512, D_MODEL), (1, 2048, D_MODEL)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )
```

NB on the torch side: `nn.Linear(in, out)` computes `x @ w.T + b` and stores `w` as `(out, in)`. We set `w` to the constant `W_CONST` everywhere, so `w == w.T` and the JAX formulation `x @ wq` with `wq = (D_MODEL, D_MODEL)` constant matches.

- [ ] **Step 3: mlp_mixer_block**

`benchmarks/e2e/mlp_mixer_block.py`:

```python
"""MLP-Mixer block (token-mixing MLP + channel-mixing MLP) — attention-free."""
import numpy as np

from xtbench import Spec, register

SEQ = 196          # 14x14 patch grid (ImageNet-like)
D_MODEL = 768
D_TOKEN_MLP = 384
D_CHANNEL_MLP = 3072
W_CONST = 0.02


def _inputs(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32)]


@register
def mlp_mixer_block() -> Spec:
    def torch_module():
        import torch
        import torch.nn as nn

        def _fill(m, v=W_CONST):
            with torch.no_grad():
                for p in m.parameters():
                    p.fill_(v)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = nn.LayerNorm(D_MODEL); _fill(self.ln1, 1.0)
                with torch.no_grad():
                    self.ln1.bias.fill_(0.0)
                self.tm1 = nn.Linear(SEQ, D_TOKEN_MLP, bias=False); _fill(self.tm1)
                self.tm2 = nn.Linear(D_TOKEN_MLP, SEQ, bias=False); _fill(self.tm2)
                self.ln2 = nn.LayerNorm(D_MODEL); _fill(self.ln2, 1.0)
                with torch.no_grad():
                    self.ln2.bias.fill_(0.0)
                self.cm1 = nn.Linear(D_MODEL, D_CHANNEL_MLP, bias=False); _fill(self.cm1)
                self.cm2 = nn.Linear(D_CHANNEL_MLP, D_MODEL, bias=False); _fill(self.cm2)

            def forward(self, x):
                h = self.ln1(x).transpose(-1, -2)
                h = self.tm2(torch.nn.functional.gelu(self.tm1(h), approximate="tanh"))
                x = x + h.transpose(-1, -2)
                h = self.ln2(x)
                h = self.cm2(torch.nn.functional.gelu(self.cm1(h), approximate="tanh"))
                return x + h
        return Block()

    def jax_fn():
        import jax
        import jax.numpy as jnp

        w_tm1 = jnp.full((SEQ, D_TOKEN_MLP), W_CONST)
        w_tm2 = jnp.full((D_TOKEN_MLP, SEQ), W_CONST)
        w_cm1 = jnp.full((D_MODEL, D_CHANNEL_MLP), W_CONST)
        w_cm2 = jnp.full((D_CHANNEL_MLP, D_MODEL), W_CONST)

        def _ln(x):
            mean = jnp.mean(x, -1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, -1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5)

        def f(x):
            h = jnp.swapaxes(_ln(x), -1, -2)
            h = jax.nn.gelu(h @ w_tm1, approximate=True) @ w_tm2
            x = x + jnp.swapaxes(h, -1, -2)
            h = jax.nn.gelu(_ln(x) @ w_cm1, approximate=True) @ w_cm2
            return x + h
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, SEQ, D_MODEL)],
        dtypes=["f32", "bf16"],
        inputs=_inputs,
    )
```

- [ ] **Step 4: Equivalence + smoke run**

```bash
pytest tests/test_equivalence.py -v -k "gpt2_block or mlp_mixer_block"
python -m xtbench --filter "gpt2_block|mlp_mixer_block" --warmup 2 --iters 3
```

Expected: equivalence PASS for both (f32 should be tight; bf16 across a full block may be ~5e-2 — bump if needed). Smoke run prints both rows.

- [ ] **Step 5: Final full sweep**

```bash
pytest -v
python -m xtbench --threads 8 --warmup 5 --iters 50
```

Expected: all tests PASS, full benchmark table prints with all rows non-EQUIV-FAIL.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/e2e/ benchmarks/__init__.py
git -c user.name=seantalts -c user.email=talts@google.com commit -m "$(cat <<'EOF'
Add E2E benchmarks: gpt2_block, mlp_mixer_block.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done criteria

- `pytest` from the repo root: all tests PASS (registry, inputs, equivalence, timing, harness, report, all parametrized equivalence cases).
- `python -m xtbench --threads 8 --warmup 5 --iters 50` runs end-to-end without crashing and prints one row per `(benchmark, shape, dtype)` triple, with non-zero speedup figures and no `EQUIV FAIL` rows.
- Repo has one commit per task; `git log --oneline` reads as a clean chronological build.

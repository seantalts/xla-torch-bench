# xla-torch-bench

CPU benchmarks comparing XLA:CPU (via `jax.jit`) against PyTorch Inductor (via `torch.compile`) on a shared set of "rosetta-stone" workloads — each benchmark is written twice, once as a `torch.nn.Module` and once as a JAX function, expressing the same compute idiomatically in each framework.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# run the full suite
python -m xtbench --threads 8 --warmup 5 --iters 50

# focused subset
python -m xtbench --filter "rms_norm|attention" --threads 4

# unit tests + rosetta-stone equivalence
pytest
```

Output is a stdout table with median + [p10, p90] latency in ms for each side and a `speedup` column (torch_median / xla_median). Compile time is reported as a separate trailing line.

## CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--filter REGEX` | `.*` | regex on benchmark name |
| `--threads N` | `os.cpu_count()` | sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `XLA_FLAGS`, and `torch.set_num_threads` — must run before any framework import (the CLI handles this) |
| `--dtypes f32,bf16` | all in spec | comma list |
| `--shapes "(d0,d1),(...)"` | per-spec | overrides spec shapes |
| `--warmup N` / `--iters N` | 5 / 50 | first call is timed separately as compile time, never folded into steady-state |

## Adding a benchmark

A benchmark is a Python file under `benchmarks/<category>/<name>.py` that exports a `Spec` via the `@register` decorator. Simplest possible form:

```python
# benchmarks/elementwise/my_bench.py
from xtbench import Spec, register


@register
def my_bench() -> Spec:
    def torch_module():
        import torch.nn as nn
        class M(nn.Module):
            def forward(self, x):
                return x * 2.0 + 1.0
        return M()

    def jax_fn():
        def f(x):
            return x * 2.0 + 1.0
        return f

    return Spec(
        torch_module=torch_module, jax_fn=jax_fn,
        shapes=[(1, 1024, 768)],
        dtypes=["f32", "bf16"],
    )
```

Then add the import to `benchmarks/<category>/__init__.py`:

```python
from benchmarks.elementwise import my_bench  # noqa: F401
```

That's it. The parametrized equivalence test (`tests/test_equivalence.py`) and the CLI pick up new specs automatically.

### Patterns to follow

- **Lazy imports.** `import torch` and `import jax` go *inside* the `torch_module` / `jax_fn` closures. Top-level `import torch` in a benchmark module body breaks thread pinning (the CLI sets env vars before importing `benchmarks`, so torch must not load until after that).
- **Weight dtype.** If your benchmark holds weights that participate in bf16 math, cast them inside `forward`: `w = self.w.to(x.dtype)` on the torch side and `wc = w.astype(x.dtype)` on the JAX side. See `benchmarks/elementwise/gelu_chain.py` for the canonical pattern. Skipping this raises `RuntimeError: mat1 and mat2 must have the same dtype` on bf16.
- **Multi-input benchmarks.** Provide a custom `inputs(shape, dtype, seed)` callable on the Spec returning a list of `np.ndarray`. See `benchmarks/elementwise/add_chain.py`. The harness converts each to torch and jax tensors before calling your modules positionally.
- **Shape-derived constants.** Use the `params` dict on Spec: `params={"d": lambda s: s[-1]}` resolves at run time and is passed as kwargs to `torch_module(**resolved)` and `jax_fn(**resolved)`.

### Numerical tolerance

The default per-dtype atol lives in `src/xtbench/inputs.py`:

- `f32 = 5e-4` — covers Inductor-vs-XLA tree-reduction drift on long reduction axes (~2e-4 observed at K=4096).
- `bf16 = 7e-2` — covers fused-vs-decomposed kernel pairs (1 bf16 ULP at magnitude 1 is 0.0625).

These are global floors. If your benchmark's bf16 output magnitudes are large (e.g. unscaled GEMM with K=1024 produces outputs of magnitude `sqrt(K) ≈ 32`, with 2-ULP error ≈ 1.0), use the per-spec escape hatch:

```python
return Spec(
    ...,
    atol_override={"bf16": 1.0},
)
```

See `benchmarks/gemm/dot_medium.py` for the rationale.

## Design and plan

- Spec: `docs/superpowers/specs/2026-06-01-xla-pytorch-cpu-benchmarks-design.md`
- Plan: `docs/superpowers/plans/2026-06-01-xla-torch-bench.md`

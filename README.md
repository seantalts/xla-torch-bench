# xla-torch-bench

CPU benchmarks comparing XLA:CPU (via `jax.jit`) against PyTorch Inductor (via `torch.compile`) on a shared set of "rosetta-stone" workloads — each benchmark is written twice, once as a `torch.nn.Module` and once as a JAX function, expressing the same compute idiomatically in each framework.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Linux only: install CPU-only torch first so pip doesn't pull the ~2GB CUDA wheel.
# Skip this line on macOS — the default wheel is already CPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"

# run the full suite
python -m xtbench --threads 8 --warmup 5 --iters 50

# focused subset
python -m xtbench --filter "rms_norm|attention" --threads 4

# unit tests + rosetta-stone equivalence
pytest
```

System requirements:
- Python 3.11+
- A working C++ compiler on PATH (`torch.compile` invokes it at runtime). macOS: Xcode CLT. Ubuntu: `sudo apt install build-essential`.

CI runs the test suite on Ubuntu 22.04 (`.github/workflows/test.yml`). Local development on macOS Apple Silicon also works.

Output is a stdout table with median + [p10, p90] latency in ms for each side and a `speedup` column (torch_median / xla_median). Compile time is reported as a separate trailing line.

## CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--filter REGEX` | `.*` | regex on benchmark name |
| `--threads N` | `os.cpu_count()` | sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `PJRT_NPROC`, `NPROC`, `XLA_FLAGS`, and `torch.set_num_threads` — must run before any framework import (the CLI handles this). XLA's PJRT CPU client reads `PJRT_NPROC`/`NPROC` to size `eigen_intraop_pool_`; without them XLA uses every host core regardless of `OMP_NUM_THREADS`. |
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

These constants were empirically calibrated against Apple Silicon's bf16 math. Linux x86 with AVX-512-BF16 / AMX may need them re-tuned — `test_equivalence.py` will surface any per-spec failure, and the fix is to bump that benchmark's `atol_override`, not the global floor.

If your benchmark's bf16 output magnitudes are large (e.g. unscaled GEMM with K=1024 produces outputs of magnitude `sqrt(K) ≈ 32`, with 2-ULP error ≈ 1.0), use the per-spec escape hatch:

```python
return Spec(
    ...,
    atol_override={"bf16": 1.0},
)
```

See `benchmarks/gemm/dot_medium.py` for the rationale.

## Sample results

Snapshot from one machine — these numbers will move with library versions, CPU, and thread count. Reproduce locally rather than treating these as a contract.

- **Machine**: Apple M3 Ultra (32 cores), macOS 26.4
- **Versions**: torch 2.12.0, jax 0.10.1, numpy 2.4.6, Python 3.11
- **Command**: `python -m xtbench --threads 8 --warmup 5 --iters 50`
- **Date**: 2026-06-02

`speedup = torch_median / xla_median` — values above 1.0 mean XLA:CPU is faster.

```
benchmark                shape                dtype  torch (ms)             xla (ms)               speedup
-----------------------------------------------------------------------------------------------------------
rms_norm                 (1, 1024, 768)       f32    0.16 [0.15,0.17]       0.12 [0.12,0.14]       1.30x
rms_norm                 (1, 1024, 768)       bf16   0.12 [0.12,0.14]       0.12 [0.11,0.13]       1.08x
rms_norm                 (1, 4096, 768)       f32    0.35 [0.33,0.37]       0.41 [0.40,0.43]       0.84x
rms_norm                 (1, 4096, 768)       bf16   0.26 [0.26,0.27]       0.31 [0.29,0.34]       0.85x
layer_norm               (1, 1024, 768)       f32    0.25 [0.25,0.27]       0.30 [0.28,0.31]       0.84x
layer_norm               (1, 1024, 768)       bf16   0.28 [0.26,0.32]       0.40 [0.38,0.42]       0.69x
layer_norm               (1, 4096, 768)       f32    0.43 [0.41,0.50]       1.52 [1.48,1.56]       0.28x
layer_norm               (1, 4096, 768)       bf16   0.50 [0.48,0.53]       1.71 [1.67,1.73]       0.29x
softmax                  (1, 12, 512, 512)    f32    0.52 [0.51,0.53]       0.69 [0.68,0.82]       0.75x
softmax                  (1, 12, 512, 512)    bf16   0.63 [0.63,0.65]       0.91 [0.91,0.93]       0.69x
softmax                  (1, 16, 1024, 1024)  f32    2.80 [2.79,2.83]       3.81 [3.77,3.84]       0.74x
softmax                  (1, 16, 1024, 1024)  bf16   3.26 [3.25,3.29]       4.90 [4.88,4.94]       0.66x
sum_axis                 (1, 4096, 1024)      f32    0.15 [0.14,0.16]       0.21 [0.20,0.21]       0.71x
sum_axis                 (1, 4096, 1024)      bf16   0.11 [0.10,0.12]       0.39 [0.35,0.43]       0.28x
sum_axis                 (1, 1024, 4096)      f32    0.17 [0.16,0.18]       0.20 [0.20,0.21]       0.82x
sum_axis                 (1, 1024, 4096)      bf16   0.12 [0.12,0.13]       0.39 [0.36,0.41]       0.31x
mean_axis                (1, 4096, 1024)      f32    0.17 [0.16,0.19]       0.20 [0.20,0.22]       0.84x
mean_axis                (1, 4096, 1024)      bf16   0.13 [0.13,0.14]       0.38 [0.36,0.42]       0.34x
mean_axis                (1, 1024, 4096)      f32    0.19 [0.19,0.23]       0.20 [0.20,0.21]       0.95x
mean_axis                (1, 1024, 4096)      bf16   0.15 [0.14,0.15]       0.38 [0.36,0.42]       0.38x
add_chain                (1, 1024, 768)       f32    0.11 [0.10,0.11]       0.10 [0.09,0.11]       1.07x
add_chain                (1, 1024, 768)       bf16   0.12 [0.12,0.14]       0.13 [0.12,0.15]       0.93x
add_chain                (1, 4096, 768)       f32    0.39 [0.37,0.46]       0.55 [0.55,0.59]       0.70x
add_chain                (1, 4096, 768)       bf16   0.29 [0.29,0.30]       0.23 [0.21,0.25]       1.27x
gelu_chain               (1, 1024, 768)       f32    0.90 [0.88,0.97]       1.37 [1.35,1.40]       0.65x
gelu_chain               (1, 1024, 768)       bf16   555.30 [552.78,557.47] 1.63 [1.61,1.66]       339.99x
silu_mul                 (1, 1024, 768)       f32    0.17 [0.16,0.18]       0.14 [0.13,0.14]       1.23x
silu_mul                 (1, 1024, 768)       bf16   0.21 [0.21,0.22]       0.21 [0.20,0.23]       1.00x
silu_mul                 (1, 4096, 768)       f32    0.46 [0.46,0.49]       0.40 [0.38,0.40]       1.17x
silu_mul                 (1, 4096, 768)       bf16   0.64 [0.64,0.65]       0.55 [0.53,0.57]       1.16x
add_mul_aligned_vs_misal (1, 4096, 1024)      f32    0.48 [0.43,0.53]       0.57 [0.56,0.63]       0.85x
add_mul_aligned_vs_misal (1, 4096, 1023)      f32    0.47 [0.41,0.50]       0.56 [0.55,0.58]       0.83x
add_mul_aligned_vs_misal (1, 4096, 1025)      f32    0.40 [0.39,0.44]       0.57 [0.56,0.59]       0.71x
dot_small                (256, 256, 256)      f32    0.04 [0.04,0.05]       0.14 [0.13,0.14]       0.31x
dot_small                (256, 256, 256)      bf16   4.49 [4.49,4.57]       0.16 [0.16,0.17]       27.32x
dot_medium               (1024, 1024, 1024)   f32    0.42 [0.42,0.45]       0.99 [0.97,1.00]       0.43x
dot_medium               (1024, 1024, 1024)   bf16   565.84 [565.68,566.00] 1.21 [1.18,1.22]       469.21x
dot_bias_gelu            (1, 1024, 768)       f32    0.79 [0.73,0.98]       0.78 [0.76,0.80]       1.01x
dot_bias_gelu            (1, 1024, 768)       bf16   5.02 [4.96,7.45]       1.00 [0.98,1.03]       5.02x
dot_transpose            (512, 768, 512)      f32    0.14 [0.14,0.20]       0.32 [0.30,0.33]       0.44x
dot_transpose            (512, 768, 512)      bf16   1.43 [1.43,1.44]       0.43 [0.40,0.45]       3.33x
mha_block                (1, 12, 128, 64)     f32    0.12 [0.12,0.13]       0.30 [0.30,0.31]       0.41x
mha_block                (1, 12, 128, 64)     bf16   0.75 [0.74,0.76]       0.41 [0.40,0.42]       1.84x
mha_block                (1, 12, 512, 64)     f32    0.74 [0.71,0.77]       1.85 [1.82,1.89]       0.40x
mha_block                (1, 12, 512, 64)     bf16   10.88 [10.74,11.32]    2.48 [2.46,2.52]       4.38x
mha_block                (1, 12, 2048, 64)    f32    8.32 [8.14,8.62]       17.11 [16.67,17.49]    0.49x
mha_block                (1, 12, 2048, 64)    bf16   178.96 [177.80,179.20] 26.25 [25.97,27.08]    6.82x
gqa_block                (1, 512)             f32    3.01 [2.90,3.12]       4.80 [4.73,4.91]       0.63x
gqa_block                (1, 512)             bf16   71.63 [71.54,71.74]    6.42 [6.34,6.51]       11.16x
gqa_block                (1, 2048)            f32    40.99 [35.90,41.74]    57.21 [56.34,58.07]    0.72x
gqa_block                (1, 2048)            bf16   1142.41 [1140.47,1144.94] 88.55 [87.93,89.31]    12.90x
gpt2_block               (1, 512, 768)        f32    4.68 [4.34,4.91]       6.30 [6.19,6.38]       0.74x
gpt2_block               (1, 512, 768)        bf16   1791.79 [1780.20,1802.02] 9.78 [9.68,9.94]       183.24x
gpt2_block               (1, 2048, 768)       f32    18.03 [17.40,19.31]    31.04 [30.65,31.66]    0.58x
gpt2_block               (1, 2048, 768)       bf16   7267.29 [7246.58,7290.07] 79.07 [78.64,79.38]    91.91x
mlp_mixer_block          (1, 196, 768)        f32    1.95 [1.91,2.04]       2.25 [2.16,2.34]       0.86x
mlp_mixer_block          (1, 196, 768)        bf16   477.91 [475.96,479.57] 2.72 [2.63,2.81]       175.80x
```

### What stands out

- **bf16 disparity is huge.** On Apple Silicon, PyTorch Inductor's bf16 path appears to fall off a cliff for matmul-heavy ops (`dot_medium` 469x, `gpt2_block` seq=512 183x, `gqa_block` seq=2048 12.9x). This likely reflects an Inductor bf16 fallback rather than XLA being magically faster — worth confirming under x86 AVX-512 / AMX before drawing conclusions.
- **f32 XLA generally loses on this machine** (most rows 0.3x–0.95x). XLA:CPU f32 codegen has room to close vs PyTorch's f32 path here.
- **Alignment signal** in `add_mul_aligned_vs_misaligned`: shape `(4096, 1024)` = 0.85x vs `(4096, 1023)` = 0.83x vs `(4096, 1025)` = 0.71x. The misaligned tails show real, measurable speedup differentiation — this is the headline xtile-aligned-ops signal in repo form.
- **Small-shape pointwise** (`rms_norm`, `silu_mul`, `add_chain` at 1024) consistently shows XLA at parity or better; **medium shapes** (4096) flip the other way.

## Design and plan

- Spec: `docs/superpowers/specs/2026-06-01-xla-pytorch-cpu-benchmarks-design.md`
- Plan: `docs/superpowers/plans/2026-06-01-xla-torch-bench.md`

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
- **Date**: 2026-06-04 (re-run after the `PJRT_NPROC` threading fix; earlier numbers used XLA at full 32 cores)

`speedup = torch_median / xla_median` — values above 1.0 mean XLA:CPU is faster.

```
benchmark                shape                dtype  torch (ms)             xla (ms)               speedup
-----------------------------------------------------------------------------------------------------------
rms_norm                 (1, 1024, 768)       f32    0.18 [0.17,0.20]       0.12 [0.12,0.13]       1.43x
rms_norm                 (1, 1024, 768)       bf16   0.20 [0.19,0.23]       0.15 [0.14,0.16]       1.32x
rms_norm                 (1, 4096, 768)       f32    0.29 [0.28,0.33]       0.41 [0.39,0.42]       0.71x
rms_norm                 (1, 4096, 768)       bf16   0.34 [0.33,0.49]       0.36 [0.34,0.39]       0.96x
layer_norm               (1, 1024, 768)       f32    0.25 [0.24,0.28]       0.33 [0.30,0.35]       0.76x
layer_norm               (1, 1024, 768)       bf16   0.26 [0.25,0.29]       0.43 [0.40,0.44]       0.62x
layer_norm               (1, 4096, 768)       f32    0.42 [0.39,0.50]       1.52 [1.47,1.54]       0.27x
layer_norm               (1, 4096, 768)       bf16   0.48 [0.46,0.64]       1.70 [1.66,1.72]       0.28x
softmax                  (1, 12, 512, 512)    f32    0.52 [0.51,0.55]       0.69 [0.68,0.73]       0.75x
softmax                  (1, 12, 512, 512)    bf16   0.63 [0.63,0.66]       0.91 [0.91,0.93]       0.69x
softmax                  (1, 16, 1024, 1024)  f32    2.81 [2.79,2.82]       3.80 [3.76,3.85]       0.74x
softmax                  (1, 16, 1024, 1024)  bf16   3.28 [3.27,3.30]       4.88 [4.86,4.94]       0.67x
sum_axis                 (1, 4096, 1024)      f32    0.15 [0.14,0.17]       0.20 [0.19,0.20]       0.74x
sum_axis                 (1, 4096, 1024)      bf16   0.11 [0.10,0.12]       0.33 [0.32,0.36]       0.33x
sum_axis                 (1, 1024, 4096)      f32    0.17 [0.16,0.18]       0.20 [0.20,0.22]       0.81x
sum_axis                 (1, 1024, 4096)      bf16   0.12 [0.12,0.13]       0.33 [0.32,0.36]       0.37x
mean_axis                (1, 4096, 1024)      f32    0.18 [0.17,0.20]       0.20 [0.20,0.20]       0.90x
mean_axis                (1, 4096, 1024)      bf16   0.13 [0.13,0.15]       0.33 [0.32,0.37]       0.40x
mean_axis                (1, 1024, 4096)      f32    0.20 [0.19,0.22]       0.20 [0.20,0.22]       0.98x
mean_axis                (1, 1024, 4096)      bf16   0.15 [0.14,0.19]       0.33 [0.32,0.36]       0.46x
add_chain                (1, 1024, 768)       f32    0.11 [0.10,0.12]       0.09 [0.08,0.10]       1.19x
add_chain                (1, 1024, 768)       bf16   0.12 [0.12,0.14]       0.12 [0.11,0.14]       1.05x
add_chain                (1, 4096, 768)       f32    0.38 [0.37,0.41]       0.54 [0.53,0.58]       0.71x
add_chain                (1, 4096, 768)       bf16   0.29 [0.28,0.32]       0.32 [0.32,0.33]       0.90x
gelu_chain               (1, 1024, 768)       f32    0.91 [0.87,1.19]       3.09 [3.07,3.13]       0.30x
gelu_chain               (1, 1024, 768)       bf16   554.15 [551.14,557.61] 3.50 [3.48,3.90]       158.32x
silu_mul                 (1, 1024, 768)       f32    0.17 [0.17,0.19]       0.12 [0.12,0.14]       1.43x
silu_mul                 (1, 1024, 768)       bf16   0.21 [0.21,0.22]       0.30 [0.29,0.33]       0.71x
silu_mul                 (1, 4096, 768)       f32    0.46 [0.46,0.48]       0.39 [0.38,0.40]       1.18x
silu_mul                 (1, 4096, 768)       bf16   0.64 [0.64,0.65]       1.07 [1.07,1.10]       0.60x
add_mul_aligned_vs_misal (1, 4096, 1024)      f32    0.48 [0.41,0.60]       0.57 [0.56,0.58]       0.84x
add_mul_aligned_vs_misal (1, 4096, 1023)      f32    0.44 [0.41,0.49]       0.58 [0.57,0.58]       0.77x
add_mul_aligned_vs_misal (1, 4096, 1025)      f32    0.38 [0.37,0.46]       0.57 [0.56,0.60]       0.68x
dot_small                (256, 256, 256)      f32    0.04 [0.04,0.05]       0.13 [0.13,0.14]       0.32x
dot_small                (256, 256, 256)      bf16   4.53 [4.51,4.61]       0.16 [0.16,0.17]       28.40x
dot_medium               (1024, 1024, 1024)   f32    0.42 [0.42,0.49]       2.66 [2.65,2.67]       0.16x
dot_medium               (1024, 1024, 1024)   bf16   565.73 [565.36,565.85] 2.86 [2.84,2.89]       197.72x
dot_bias_gelu            (1, 1024, 768)       f32    0.75 [0.73,0.91]       1.60 [1.59,2.02]       0.47x
dot_bias_gelu            (1, 1024, 768)       bf16   4.92 [4.88,6.86]       1.99 [1.97,2.30]       2.46x
dot_transpose            (512, 768, 512)      f32    0.14 [0.14,0.20]       0.53 [0.52,0.54]       0.27x
dot_transpose            (512, 768, 512)      bf16   1.43 [1.43,1.44]       0.62 [0.61,0.65]       2.30x
mha_block                (1, 12, 128, 64)     f32    0.12 [0.11,0.14]       0.29 [0.29,0.30]       0.39x
mha_block                (1, 12, 128, 64)     bf16   0.77 [0.77,0.78]       0.39 [0.38,0.41]       1.97x
mha_block                (1, 12, 512, 64)     f32    0.75 [0.71,0.77]       2.25 [2.24,2.27]       0.33x
mha_block                (1, 12, 512, 64)     bf16   11.22 [11.18,11.34]    2.88 [2.86,2.90]       3.90x
mha_block                (1, 12, 2048, 64)    f32    8.40 [8.06,8.71]       25.97 [25.56,26.75]    0.32x
mha_block                (1, 12, 2048, 64)    bf16   177.63 [174.08,179.23] 35.45 [35.11,35.96]    5.01x
gqa_block                (1, 512)             f32    2.92 [2.79,3.11]       7.49 [7.34,7.77]       0.39x
gqa_block                (1, 512)             bf16   71.68 [71.60,71.79]    9.15 [9.01,9.34]       7.83x
gqa_block                (1, 2048)            f32    40.75 [37.75,42.11]    105.57 [104.19,107.65] 0.39x
gqa_block                (1, 2048)            bf16   1141.64 [1140.88,1142.97] 136.13 [135.56,139.50] 8.39x
gpt2_block               (1, 512, 768)        f32    4.65 [4.31,4.82]       10.10 [10.00,10.27]    0.46x
gpt2_block               (1, 512, 768)        bf16   1794.64 [1781.27,1803.44] 13.89 [13.77,14.17]    129.20x
gpt2_block               (1, 2048, 768)       f32    17.75 [17.45,19.49]    56.72 [56.07,57.37]    0.31x
gpt2_block               (1, 2048, 768)       bf16   7233.25 [7219.29,7253.37] 105.86 [105.39,106.79] 68.33x
mlp_mixer_block          (1, 196, 768)        f32    1.94 [1.90,2.10]       3.31 [3.29,3.42]       0.58x
mlp_mixer_block          (1, 196, 768)        bf16   478.07 [476.72,479.22] 3.75 [3.70,3.85]       127.51x
embedding_lookup         (1, 512)             f32    0.13 [0.12,0.17]       0.05 [0.04,0.06]       2.48x
embedding_lookup         (1, 512)             bf16   0.14 [0.14,0.17]       0.04 [0.03,0.05]       3.74x
embedding_lookup         (1, 2048)             f32    0.17 [0.15,0.19]       0.07 [0.06,0.09]       2.43x
embedding_lookup         (1, 2048)             bf16   0.21 [0.20,0.24]       0.06 [0.05,0.07]       3.61x
scatter_add              (4096, 768)          f32    0.89 [0.85,0.93]       0.49 [0.40,0.66]       1.83x
scatter_add              (4096, 768)          bf16   1.48 [1.44,1.51]       1.19 [1.07,1.26]       1.25x
take_along_axis          (4096, 768)          f32    0.12 [0.11,0.14]       0.06 [0.06,0.08]       2.05x
take_along_axis          (4096, 768)          bf16   0.12 [0.11,0.14]       0.06 [0.05,0.07]       1.97x
dynamic_slice_loop       (1, 2048, 768)       f32    0.43 [0.40,0.46]       0.14 [0.14,0.15]       2.99x
dynamic_slice_loop       (1, 2048, 768)       bf16   0.36 [0.34,0.38]       0.28 [0.26,0.30]       1.29x
conv2d_3x3_resnet        (1, 64, 56, 56)      f32    0.56 [0.52,0.61]       0.97 [0.96,0.98]       0.58x
conv2d_3x3_resnet        (1, 64, 56, 56)      bf16   2.02 [2.00,2.04]       0.80 [0.78,0.98]       2.53x
depthwise_conv_3x3       (1, 256, 56, 56)     f32    19.08 [18.45,19.26]    2.16 [2.15,2.18]       8.82x
depthwise_conv_3x3       (1, 256, 56, 56)     bf16   17.50 [17.27,17.84]    2.17 [2.16,2.19]       8.06x
dilated_conv_3x3         (1, 64, 56, 56)      f32    0.38 [0.37,0.40]       0.52 [0.51,0.53]       0.74x
dilated_conv_3x3         (1, 64, 56, 56)      bf16   28.38 [28.28,28.51]    0.54 [0.52,0.56]       52.86x
transpose_conv_3x3       (1, 64, 14, 14)      f32    0.11 [0.11,0.14]       0.10 [0.10,0.11]       1.09x
transpose_conv_3x3       (1, 64, 14, 14)      bf16   0.44 [0.43,0.47]       0.10 [0.10,0.11]       4.31x
grouped_conv_3x3         (1, 128, 28, 28)     f32    0.30 [0.29,0.32]       0.47 [0.46,0.48]       0.65x
grouped_conv_3x3         (1, 128, 28, 28)     bf16   4.77 [4.70,4.82]       0.47 [0.46,0.48]       10.09x
topk_logits_k10          (1, 50000)           f32    0.07 [0.07,0.07]       0.05 [0.05,0.05]       1.49x
topk_logits_k10          (1, 50000)           bf16   0.07 [0.07,0.07]       5.68 [5.65,5.82]       0.01x
topk_logits_k100         (1, 50000)           f32    0.08 [0.08,0.08]       0.06 [0.06,0.06]       1.36x
topk_logits_k100         (1, 50000)           bf16   0.08 [0.08,0.09]       5.66 [5.64,5.75]       0.02x
topk_logits_k1000        (1, 50000)           f32    0.18 [0.18,0.19]       0.21 [0.21,0.22]       0.85x
topk_logits_k1000        (1, 50000)           bf16   0.23 [0.23,0.26]       5.68 [5.64,6.40]       0.04x
argsort_axis             (1024, 512)          f32    2.01 [1.96,2.06]       62.22 [62.05,63.05]    0.03x
argsort_axis             (1024, 512)          bf16   2.36 [2.30,2.41]       92.16 [92.00,92.47]    0.03x
bucketize                (4096, 768)          f32    2.45 [2.44,2.46]       15.91 [15.81,16.13]    0.15x
bucketize                (4096, 768)          bf16   3.51 [3.50,3.52]       14.57 [14.52,14.68]    0.24x
sort_full                (1024, 4096)         f32    20.24 [20.19,20.33]    392.33 [389.34,393.32] 0.05x
sort_full                (1024, 4096)         bf16   23.70 [23.62,23.75]    621.97 [621.64,622.88] 0.04x
topk_then_gather         (512, 64)            f32    0.20 [0.16,0.30]       0.12 [0.10,0.15]       1.63x
top2_router              (1, 1024, 768)       f32    0.34 [0.33,0.42]       0.18 [0.17,0.20]       1.90x
top2_router              (1, 1024, 768)       bf16   1.28 [1.23,1.38]       0.40 [0.38,0.43]       3.21x
top2_router              (1, 4096, 768)       f32    0.78 [0.74,0.86]       0.56 [0.54,0.64]       1.39x
top2_router              (1, 4096, 768)       bf16   4.26 [4.19,4.55]       1.33 [1.31,1.36]       3.21x
segment_sum              (4096, 64)           f32    0.20 [0.19,0.23]       0.04 [0.04,0.04]       5.35x
segment_sum              (4096, 64)           bf16   0.23 [0.23,0.27]       0.11 [0.10,0.11]       2.23x
expert_dispatch_combine  (1, 1024, 768)       f32    32.86 [30.51,37.31]    107.54 [99.58,127.72]  0.31x
expert_dispatch_combine  (1, 1024, 768)       bf16   25079.75 [24782.81,25257.92] 118.45 [117.84,119.39] 211.72x
```

### What stands out

- **Sort family is the worst category**. `argsort_axis` 0.03x, `sort_full` 0.05x, `bucketize` 0.15x f32. The bf16 `topk_logits` is 0.01x at constant ~5.7ms regardless of `k` ∈ {10, 100, 1000} — strong signal of algorithmic fallback (likely full sort or scalar) for bf16 topk on CPU.
- **f32 GEMM is a big and broad loss**. `dot_medium 0.16x`, `dot_transpose 0.27x`, `dot_bias_gelu 0.47x`, `dot_small 0.32x`. This cascades into every transformer benchmark (`gpt2_block 2048 f32 → 0.31x`, `mha_block at all seqs → 0.32–0.39x`, `gqa_block → 0.39x`).
- **`layer_norm (1,4096,768)` two-pass mean/variance is still the headline regression** (0.27x f32, 0.28x bf16), localized to the YNN REDUCE library-fusion path (see `docs/perf-wins-ynn-followup-2026-06-02.md`).
- **bf16 reductions** (`sum_axis 0.33x`, `mean_axis 0.40x` at 4096) and **softmax** (0.66–0.69x) consistent with the reduction-emitter gaps documented in `docs/perf-wins-2026-06-02.md`.
- **bf16 disparity** on PyTorch's matmul-heavy paths is still huge on Apple Silicon (`dot_medium bf16 197x`, `gpt2_block bf16 seq=512 129x`); these are likely Inductor bf16 fallback cliffs, not real XLA wins. Worth confirming on Linux x86 with AVX-512-BF16 / AMX before drawing conclusions.
- **Where XLA dominates**: indexing/gather (`embedding_lookup 3.7x bf16`, `dynamic_slice_loop 3.0x f32`), MoE primitives (`segment_sum 5.4x f32`, `top2_router 3.2x bf16`), depthwise conv (8.8x — Inductor weak on this path), small pointwise (`silu_mul 1.4x`, `add_chain 1.2x` at 1024).
- **Alignment signal** in `add_mul_aligned_vs_misaligned`: 1024 = 0.84x vs 1023 = 0.77x vs 1025 = 0.68x. The misaligned-tail differentiation is still there post-threading-fix.

## Design and plan

- Spec: `docs/superpowers/specs/2026-06-01-xla-pytorch-cpu-benchmarks-design.md`
- Plan: `docs/superpowers/plans/2026-06-01-xla-torch-bench.md`

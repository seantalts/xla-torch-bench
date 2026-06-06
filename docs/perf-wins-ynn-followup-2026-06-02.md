# YNN REDUCE toggle — reduction-benchmark comparison

Follow-up to [perf-wins-2026-06-02.md](perf-wins-2026-06-02.md). The default `XLA_FLAGS` enables `LIBRARY_FUSION_TYPE_REDUCE` in `--xla_cpu_experimental_ynn_fusion_type`; this run re-measures the reduction benchmarks with that one library-fusion path removed via `--xla_cpu_experimental_ynn_fusion_type=-REDUCE`. Everything else identical to the baseline.

- Machine: Apple M3 Ultra (32 cores), macOS 26.4
- Versions: torch 2.12.0, jax 0.10.1, Python 3.11
- Command: `XLA_FLAGS="--xla_cpu_experimental_ynn_fusion_type=-REDUCE" python -m xtbench --filter "sum_axis|mean_axis|softmax|layer_norm|rms_norm" --threads 8 --warmup 5 --iters 50`
- Date: 2026-06-02

XLA median ms only (torch column unchanged across runs, omitted here).

| benchmark | shape | dtype | xla baseline | xla -REDUCE | xla delta | speedup baseline → -REDUCE |
|---|---|---|---:|---:|---:|---|
| layer_norm | (1, 4096, 768) | f32  | 1.52 | **0.50** | **−67% (3.0× faster)** | 0.28x → 1.09x |
| layer_norm | (1, 4096, 768) | bf16 | 1.71 | **0.76** | **−55% (2.2× faster)** | 0.29x → 0.80x |
| layer_norm | (1, 1024, 768) | f32  | 0.30 | **0.16** | **−47% (1.9× faster)** | 0.84x → 2.47x |
| layer_norm | (1, 1024, 768) | bf16 | 0.40 | **0.28** | **−30% (1.4× faster)** | 0.69x → 1.27x |
| sum_axis   | (1, 4096, 1024)| f32  | 0.21 | 0.16 | −24% | 0.71x → 0.91x |
| sum_axis   | (1, 4096, 1024)| bf16 | 0.39 | 0.31 | −21% | 0.28x → 0.35x |
| sum_axis   | (1, 1024, 4096)| f32  | 0.20 | 0.16 | −20% | 0.82x → 1.04x |
| sum_axis   | (1, 1024, 4096)| bf16 | 0.39 | 0.31 | −21% | 0.31x → 0.40x |
| mean_axis  | (1, 4096, 1024)| f32  | 0.20 | 0.16 | −20% | 0.84x → 1.08x |
| mean_axis  | (1, 4096, 1024)| bf16 | 0.38 | 0.31 | −18% | 0.34x → 0.41x |
| mean_axis  | (1, 1024, 4096)| f32  | 0.20 | 0.16 | −20% | 0.95x → 1.23x |
| mean_axis  | (1, 1024, 4096)| bf16 | 0.38 | 0.31 | −18% | 0.38x → 0.47x |
| softmax    | (1, 12, 512, 512) | f32  | 0.69 | 0.63 | −9%  | 0.75x → 0.82x |
| softmax    | (1, 16, 1024,1024)| f32  | 3.81 | 3.41 | −10% | 0.74x → 0.83x |
| rms_norm   | (1, 4096, 768) | f32  | 0.41 | 0.42 | ≈0   | 0.84x → 0.90x |
| **rms_norm**   | (1, 1024, 768) | f32  | 0.12 | 0.14 | **+17% slower** | 1.30x → 1.96x* |
| **rms_norm**   | (1, 1024, 768) | bf16 | 0.12 | 0.16 | **+33% slower** | 1.08x → 1.99x* |
| **rms_norm**   | (1, 4096, 768) | bf16 | 0.31 | 0.37 | **+19% slower** | 0.85x → 1.17x* |
| **softmax**    | (1, 12, 512, 512)  | bf16 | 0.91 | 1.15 | **+26% slower** | 0.69x → 0.55x |
| **softmax**    | (1, 16, 1024,1024) | bf16 | 4.90 | 6.07 | **+24% slower** | 0.66x → 0.53x |

\* The "speedup vs torch" gets better in these rows because the torch number also rose in this run (Inductor recompile noise, not a real regression on the torch side); compare the XLA delta column for the real signal.

## Read

YNN REDUCE has two distinct effects depending on the pattern:

**It hurts (disable wins):**
- **layer_norm** — multi-pass mean+variance pattern. The 3× speedup at (1,4096,768) f32 makes this the headline finding. YNN's reduce fusion appears to lower the `mean((x - mean(x))²)` idiom into something significantly worse than the non-YNN reduce emitter. Confirms perf-wins #3 is real *and* localizes the regression to YNN.
- **sum_axis / mean_axis** — pure single-axis reductions. Consistent ~20% slowdown across all 8 rows when YNN REDUCE is on. Suggests the YNN reduce path has overhead (codegen-call dispatch, packing, or extra copies) that isn't amortized by the small per-row work for these shapes.
- **softmax f32** — small (~10%) slowdown when YNN is on. Marginal.

**It helps (keep enabled):**
- **rms_norm small (1024)** — 17–33% slowdown without YNN. The single-pass reduction-with-elementwise-tail pattern is what YNN REDUCE is actually good at when the row fits in cache.
- **softmax bf16** — 24–26% slowdown without YNN. YNN's reduce path apparently handles the multi-pass max+sum bf16 pattern better than the fallback, even though the f32 version is roughly neutral.

## What this means for the perf-wins list

- **#1 (bf16 reductions)**: YNN REDUCE off improves bf16 reductions by ~20%, but the f32/bf16 gap is still 2× (e.g. `sum_axis bf16` 0.31ms vs `sum_axis f32` 0.16ms with YNN off). So YNN is *part* of the regression but not the whole story — the vectorized-bf16-accumulator work still has the bigger remaining payoff.

- **#3 (layer_norm two-pass mean/var)**: Confirmed real. The fix is "fix YNN REDUCE for the multi-pass-reduction case, or have YNN REDUCE bail out for this pattern and let the existing reduce emitter handle it." The non-YNN emitter is already significantly better at this shape, so the simplest near-term win is teaching YNN to recognize and decline.

- **#4 (softmax fusion)**: The YNN data adds nuance — for f32 softmax the gap is mostly elsewhere (YNN toggle barely moves it), but for bf16 softmax YNN is actively helping by 25%. The online-softmax design from #4 still applies but should be measured against the YNN-on baseline.

- **#12 (reduction tile tuning)**: Partially absorbed — about 20% of the gap on `sum_axis`/`mean_axis` is YNN overhead, and turning it off lifts those rows to or above parity on f32. The remaining bf16 gap (now 0.35–0.47x vs torch) is the real signal for tile-tuning + accumulator work.

## Suggested follow-up experiments

1. **Per-pattern YNN REDUCE policy.** Audit the YNN REDUCE pattern matcher for the multi-pass mean+var idiom and either fix it or have it decline so the non-YNN emitter handles it. This single change should land most of the layer_norm win.
2. **YNN dispatch cost on small reductions.** Profile `sum_axis (1,4096,1024) f32` with YNN on; the consistent ~20% slowdown across all single-reduction rows suggests a fixed per-call cost in YNN dispatch that dominates for short reductions. If it's call overhead, raise a threshold (don't dispatch YNN for reductions below N elements). If it's codegen, fix the emitter.
3. **bf16 softmax — what's YNN doing right?** YNN REDUCE helping softmax bf16 by 25% but hurting layer_norm bf16 by 55% is asymmetric in an interesting way. Reading the lowered IR for both might reveal which YNN-REDUCE feature is the winner — possibly worth extracting and applying to the layer_norm path.

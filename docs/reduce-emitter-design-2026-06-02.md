# Vanilla XLA:CPU reduce emitter — design

Date: 2026-06-02
Author: seantalts
Status: prototype landed on `feat/reduce-emitter-multi-accumulator` (XLA monorepo)

## Problem

The YNN follow-up data ([`perf-wins-ynn-followup-2026-06-02.md`](perf-wins-ynn-followup-2026-06-02.md))
showed that disabling `LIBRARY_FUSION_TYPE_REDUCE` (the library-fusion path) wins
big on multi-pass reductions (layer_norm 3.0×) and on pure single-axis reductions
(~20% across sum/mean rows), but **loses** on cache-resident single-pass patterns
(rms_norm small, bf16 softmax). The current non-YNN emitter is therefore close to
the "right answer" for some shapes and far from it for others. The goal of this
work is to close the bf16-and-small-rows gap so the vanilla emitter can win
consistently without needing YNN REDUCE at all. Out of scope: the YNN path
itself, and the structural multi-output recognizer that would automate the
layer_norm 3× win (sketched here, implementation deferred).

## Where the vanilla emitter lives today

- `xla/backends/cpu/codegen/tiled/transforms/vectorized_reduce_emitter.{h,cc}`
  is the entry point — `EmitVectorizedReduction(...)`.
- Called from `shlo_to_vector.cc` via the `xtile-cpu-shlo-to-vector` pass on a
  `stablehlo.reduce` op.
- Generates an outer `scf.for` over non-reduced non-minor dimensions, a middle
  `scf.for` over non-minor reduced dimensions carrying a single vector
  accumulator, and (when the minor dim is reduced) a final `vector.reduction`.

Two structural observations from reading the existing code:

1. The middle loop carries **one** vector accumulator across all reduced rows
   (`scf.for ... iter_args = %acc`). For `tensor<4096x768>` reduced over dim 0
   that is 4096 sequentially-dependent FMAs into the same SIMD register — fully
   pipelined hardware FMA latency (4–5 cycles on Apple M-series) gates
   throughput. This is the canonical "1-accumulator chain" anti-pattern.
2. The minor-dim reduction (`EmitMinorReduction`) emits a single
   `vector.reduction <add>` on the row vector with `FastMathFlags::reassoc`. The
   reassoc flag lets LLVM tree-reduce, so this leg is OK once it's reached — the
   problem is **getting there**: for bf16, every row load is a bf16 vector that
   would need to be widened to f32 *before* it enters the accumulator chain,
   which currently isn't done — we accumulate in bf16, losing both precision
   and (more importantly) ILP because bf16 fma is narrower-throughput than f32
   fma on most CPUs.

## The three pieces

### (1) Multi-accumulator loop body

Where: inner `scf.for` in `EmitReductionLoop` — the non-minor reduced
dimension loop. (For the `reduce_outer` shape `tensor<R x C>` reducing over
dim 0, this is the only loop carrying the accumulator.)

What changes: instead of `scf.for %i = 0..N step 1 iter_args(%acc = %init)`,
emit `scf.for %i = 0..N-tail step K iter_args(%acc0 = %init, %acc1 = %init, ...,
%acc_{K-1} = %init)` where K is the number of independent accumulators (default
K=4 — matches typical reduction-kernel hand-rolls and the M3's 4-cycle FMA
latency / 1-per-cycle throughput).

Body: K independent vector loads at `(i, i+1, ..., i+K-1)` shifted, K
independent applications of the reducer body, K yields.

Combine: after the loop, fold the K accumulators by repeating the body
(`acc01 = body(acc0, acc1); acc23 = body(acc2, acc3); result = body(acc01,
acc23)`) — log2(K) latency, one accumulator on exit.

Tail: if `N % K != 0`, emit a short single-accumulator loop over the remainder
before the combine.

Disable when: K > N (degenerate, fall back to original path), or the reducer
body is non-associative-floating-point and fast-math is off (we already require
reassoc via `add_reduction_fast_math_flags.cc`, so this should always be safe
for the reductions XLA cares about).

### (2) bf16 widen-on-load

(Designed only; not in this prototype if budget runs out.)

Where: every `ExtractVector` call site in the emitter that produces a bf16
vector that immediately feeds a reducer body.

What changes:
- After `vector.transfer_read` produces `vector<W x bf16>`, immediately
  `arith.extf` it to `vector<W x f32>`.
- Promote `init_value` from bf16 to f32 once at the top.
- Run the entire accumulator chain in f32 (vector and scalar).
- At the end, `arith.truncf` the scalar/vector result back to bf16 before the
  store.

Why this matters: the perf-wins data shows bf16 reductions running 2× slower
than f32 — most of that is "bf16 accumulator chain has more rounding noise and
in practice the LLVM backend can't pack as efficiently." Widening once at the
load and accumulating in f32 is what every hand-tuned bf16 reduction kernel
does and what cuBLAS/oneDNN do for bf16 sgemm.

Risk: changes numerics. The XLA bf16 reduction semantics already permit f32
accumulation in many CPU code paths (it's how YNN does it). Add a flag to gate
this on the same fast-math envelope the reassoc rewrite already lives under
(`add_reduction_fast_math_flags.cc`).

### (3) `vector.multi_reduction` + `VectorUnrollPass`

Where: replace the single big `vector.reduction` on the natural-width
accumulator with `vector.multi_reduction` on a `vector<K x W>` of K accumulators
(post-(1)), letting MLIR's `VectorUnrollPass` shred the operation into hardware
vector width per target.

What this buys us: ISA agnosticism. We emit at, say, `vector<32xf32>` regardless
of whether the target is AVX2 (8-wide), AVX-512 (16-wide), or NEON (4-wide), and
MLIR's `populateVectorUnrollPatterns` lowering with a target-width callback
produces the right number of register-width ops. Right now the emitter picks the
vector width = the tensor's static minor dim, which couples the IR to the
problem shape rather than to the hardware.

Status check during prototyping: `mlir::vector::populateVectorUnrollPatterns`
and `UnrollVectorOptions` are available in the imported MLIR
(`@llvm-project//mlir:VectorTransforms` is already on the `passes` and
`vectorized_reduce_emitter` deps lines in `BUILD`). However, there is no
existing pass in the XLA CPU pipeline that runs it on the reduce emitter
output — `grep -rn VectorUnroll xla/` returned only dep references, no
invocations. So adding an emit-time call to `vector.multi_reduction` will work,
but a follow-on pass needs to be added that runs `populateVectorUnrollPatterns`
post-emitter (could live in the same `passes` library, e.g.
`xtile-cpu-unroll-vectors-to-native-width`). The width callback should consult
the target triple — for this prototype, hard-code 8 (AVX2 / 256-bit) since the
benchmark target is Apple Silicon NEON-128 ⇒ 4 × f32, but the M3 SVE has
256-bit, and we want to leave room.

## Integration points and pass order

Files modified by this prototype:

- `xla/backends/cpu/codegen/tiled/transforms/vectorized_reduce_emitter.cc` —
  multi-accumulator loop emission. Add a helper
  `EmitMultiAccumulatorReductionLoop` invoked from `EmitReductionLoop` when the
  preconditions hold.
- `xla/backends/cpu/codegen/tiled/transforms/vectorized_reduce_emitter.h` —
  expose the multi-accumulator factor as a tunable constant.
- `xla/backends/cpu/codegen/tiled/transforms/tests/shlo_to_vector.mlir` — extend
  with checks for the multi-accumulator pattern on a representative shape.
- (Stretch) `xla/backends/cpu/codegen/tiled/transforms/passes.{h,td,cc}` — new
  `xtile-cpu-unroll-vectors-to-native-width` pass running
  `populateVectorUnrollPatterns` after the shlo-to-vector pass. *Not in this
  prototype; designed only.*

Pass order in the existing pipeline (verified by reading
`tiled/transforms/passes.td` and `emitters/transforms/passes.td`):

1. `xtile-cpu-shlo-to-vector` — emits the multi-accumulator loops via this
   file's changes.
2. (Proposed) `xtile-cpu-unroll-vectors-to-native-width` — splits emitted
   wide-vector ops to hardware width.
3. `xtile-cpu-tensor-ops-to-bufferizable` — unchanged.
4. Bufferization passes — unchanged.
5. `xtile-cpu-linalg-elementwise-to-vector` — unchanged.
6. `cpu-add-reduction-fast-math-flags` (already exists) — attaches `reassoc`
   to our fma chain, which is what allows the multi-acc combine to be folded
   without changing IEEE semantics expectations.

## Structural multi-output reduction recognizer (future work)

This is the lever that unlocks the layer_norm 3× win without ever naming
LayerNorm. It is **deferred**; designed here so a follow-up task can pick it up.

Algorithm sketch:

1. Walk fusion DAGs (post-fusion, pre-lowering — the `xla_cpu_fusion` op or
   equivalent in the tiled pipeline).
2. For each fusion, collect all `stablehlo.reduce` ops it contains.
3. Group reductions by `(reduction_dims, axis-order-equivalent input load)` —
   two reductions are mergeable if they reduce over the same dims and their
   input tensors are "load-equivalent" (defined as: traversing both DAGs from
   the reduce input back toward fusion parameters yields identical
   `(elementwise-op-chain, parameter)` paths). This is a structural match — it
   never inspects op identities like "is this mean of x?"
4. For each group with ≥2 reductions, rewrite into a single multi-output
   reduce: emit N accumulators in the *same* `scf.for` loop, sharing the load
   of x, applying both reducer bodies in lock-step.

Payoff: `layer_norm` lowers to two reductions over the same axis of the same
tensor (mean and mean-of-squares). With shared loads and shared loop overhead,
this matches what a hand-rolled `welford_pass` kernel does in one pass. Combined
with multi-accumulator (1), this is the entire 3× headline win in the YNN
follow-up.

Where to put it: a new pre-lowering pass, probably in
`xla/backends/cpu/codegen/tiled/transforms/`, named
`xtile-cpu-fuse-coreduction` or similar. Run before `xtile-cpu-shlo-to-vector`
so that by the time the emitter sees the reduce, it's already been merged.

Estimated complexity: medium. The load-equivalence check is the only subtle
piece; everything else is a straightforward `OpRewritePattern`. Allow
1–2 days to land + measure.

## IREE-style ukernels — escape hatch

If pure codegen plateaus on bf16 softmax / very-small-row rms_norm patterns
(where YNN is currently winning), the next step is to ship target-specialized
microkernels (IREE-style) for a small set of structural patterns. The mechanism
would be: detect the structural pattern in the same recognizer above, and
instead of lowering through `EmitVectorizedReduction`, emit a call to a
pre-compiled `__xla_cpu_ukernel_reduce_<dtype>_<axis>` symbol. We don't go
this direction in this pass because (a) it's a large investment for a narrow
win, (b) codegen-first solutions cover more shapes for cheaper, and (c) the
ukernel set is a maintenance liability — every new dtype/axis combo needs a
hand-written kernel. Reach for them only after the multi-acc + widen-on-load +
multi-reduction set has been measured and a residual gap clearly localizes to a
small enumerable set of patterns.

## Plan-level surprise: the tiled emitter doesn't see reduces today

Found during prototype benchmarking, important for any follow-up:

- `vectorized_reduce_emitter.cc` lives inside the **tiled** CPU emitter
  pipeline (`AddTiledOptimizationPasses` in
  `xla/backends/cpu/codegen/fusion_compiler.cc`, invoked from
  `EmitTiledFusionKernel`).
- Dispatch into the tiled emitter is gated by
  `IsSupportedTiledFusion` →
  `IsSupportedInstruction` in
  `xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.cc`. That function's
  switch statement does **not** list `HloOpcode::kReduce`, and the `default:`
  arm returns `inst.IsElementwise()` which is false for reduce. So any fusion
  containing a reduce is rejected by the tiled emitter and falls back to the
  legacy emitter.
- Standalone (non-fused) reduces go through an entirely separate path:
  `ThunkEmitter::EmitReductionKernelThunk` in
  `xla/service/cpu/thunk_emitter.cc`. The tiled MLIR pipeline isn't on this
  path at all.

What this means: **the multi-accumulator change here is correct in the file it
lives in, but as of 2026-06-02 no benchmark actually exercises that path.** The
benchmark numbers in this prototype are accordingly flat (within noise),
because the binary keeps using the legacy emitter regardless. To get end-to-end
perf signal from this change, a separate follow-up is needed to either:

1. Add `HloOpcode::kReduce` to `IsSupportedInstruction` (and audit the rest of
   the tiled-emitter ops for reduce-compatibility — symbolic tile analysis,
   linalg lowering, etc.), routing fused-reduce shapes through the tiled
   emitter. This is the cleaner win because it also picks up multi-accumulator
   for layer_norm-style reduce-inside-fusion patterns.
2. Port the multi-accumulator logic into the legacy `ir_emitter` reduce path
   in `xla/service/cpu/ir_emitter*.cc`. That's a duplication of effort and
   doesn't help the layer_norm structural recognizer story, so option (1) is
   preferred.

The prototype here is best understood as "the tiled emitter side of the work
done first." The work is unblocked once the dispatch question is settled.

## Risks / open questions

- **Tail handling**: the prototype handles `N % K != 0` with a scalar-step
  cleanup loop. For pathological N (like N = K+1) this dominates. In practice
  reduction lengths are powers of 2 or multiples of 16/32, so this shouldn't
  matter, but it's a sharp edge to know about.
- **Default K**: 4 is right for Apple M-series and most x86. AVX-512 with FMA
  pipelining might prefer K=8. Left as a constant in this prototype; a real
  follow-up would target-tune it.
- **Vector unroll pass placement**: I haven't verified that the existing
  bufferization → LLVM lowering doesn't already unroll wide vectors at some
  later stage. If it does, piece (3) might be a no-op for us. Worth measuring
  before adding the new pass.
- **bf16 widen interaction with reassoc**: widening to f32 then accumulating
  with reassoc is standard, but verifying numeric envelope against the existing
  reduction tests is required before flipping on by default.
- **Numerics-of-multi-acc tree combine**: combining 4 partials via
  `body(body(a,b), body(c,d))` introduces a different reduction order than the
  single-accumulator left-fold. Already covered by `reassoc` in the existing
  fast-math pass; flagged here so reviewers know the semantics envelope this
  lives in.
- **The unroll pass isn't actually wired up in this prototype** — the prototype
  emits at `vector<32xf32>` (the row width for the test shape), trusting that
  LLVM's backend will unroll to native registers. That works for small shapes
  but won't scale to 1024-wide rows without an explicit unroll. Tracked as the
  first follow-up after this lands.

## Implemented vs deferred

Implemented in `feat/reduce-emitter-multi-accumulator`:

- Piece (1): multi-accumulator loop body (K=4, configurable constant) in the
  non-minor reduced inner loop of `EmitReductionLoop`. Builds cleanly, lit
  tests pass.
- A lit test demonstrating the multi-accumulator IR is emitted for the
  representative `reduce_outer` shape, plus a `reduce_outer_tiny` case
  documenting the single-acc fallback.
- Four new benchmark cases pulled from the YNN follow-up (4096×768 f32 and
  bf16, 4096×1024 f32 and bf16 — corresponding to layer_norm-batch and
  sum_axis-last-axis patterns).

Deferred to this doc:

- Tiled-emitter reduce dispatch (the new "plan-level surprise" finding above) —
  blocks end-to-end perf signal from the multi-acc change.
- Piece (2): bf16 widen-on-load (designed; not coded).
- Piece (3) end-to-end: `vector.multi_reduction` emission is straightforward to
  add, but wiring `populateVectorUnrollPatterns` into a new
  `xtile-cpu-unroll-vectors-to-native-width` pass requires touching the pass
  pipeline registration and was scoped out. Prototype emits at the source row
  width and trusts the backend.
- Structural multi-output recognizer (the layer_norm 3× lever).
- IREE-style ukernels.

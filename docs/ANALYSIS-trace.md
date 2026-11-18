# Trace time of the reduced-space kernels (m410_CLay0203HRS, m34_convex2), 2026-11-17

**Question.** Cold, m410 spent 10.4 s tracing and 13 s compiling its bound kernel before the root bound (28.8 s at load 3.3). OBBT (12.7 s) and OA (18 s) traces and compiles competed with the main thread. What in the trace can go, and is any trace output shared between kernels?

## Diagnosis

Sources: py-spy (raw stacks per thread) of a cold m410 solve; `jax.monitoring` trace and compile spans; a standalone harness that builds `_Kernels` with the options `minimize` uses and traces and compiles each kernel alone.

1. **The same work in every kernel.**
   - Each relaxer instance rebuilt its tape: the bound kernel's, OBBT's, the cut pool's, the dive's FBBT, alphaBB, the convexity certificate and the memory estimate. Each build is a `make_jaxpr` of the `.nl` functions plus flattening.
   - On top of the tape, each also redid the dependency masks, the scalarized program, the certificate and the FBBT propagator (scalarize, schedule, layout).
   - This was about 20% of the tracer thread. The wrappers (`lambda x: atleast_1d(fn(x))`) were new objects per instance, so nothing could be shared.
2. **Eager dependency masks.** `univariate._dep_masks` ran each tape node's rule as eager `jnp` operations. About 70% of its time was XLA compiling one tiny executable per operation and shape: 1.6 s of the tracer thread at load 0, and the 2–9 s `jit(abs)` / `jit(add)` spans seen under load.
3. **Functions prepared but never used.**
   - With `SPLIT_AFFINE` and exact affine cuts, a function with no nonlinear part is not relaxed at the cut points. On m410 that is `f` and the 24 equalities; on m34, the equalities.
   - Their box preparation was still traced in every kernel and then dropped by XLA: alphaBB's interval Hessian (`jacfwd(jacfwd)`, an eigen-decomposition of (24, 90, 90)) and the univariate hull setup.
   - This was 43% of m410's OBBT trace and 35% of the bound trace. alphaBB also evaluated the function at `lo` only to learn its output size.
4. **`vmap` of loops.**
   - JAX's `while`/`scan` batching rule re-traces the body until it knows which carries are batched: once per round in which an unbatched initial carry (counter, flag, constant buffer) becomes batched, then once more to batch all carries when the condition is batched.
   - Nested loops multiply this: the Kelley loop holds the cut-chunk loop and the IPM, each batched 2–3 times per outer pass.
   - This was 18% of the tracer thread, and 46% of the bound trace after 1–3.
5. **Not a cause: the `.nl` expressions.**
   - `make_jaxpr(vmap(jacfwd(ineq)))` on m410 traces in 0.18 s, the same as evaluating a cached jaxpr. The time attributed to `nl.py` frames in ANALYSIS-compile is the transformations running beneath them, repeated per kernel.
   - Vectorizing the linear terms would change the scalar tapes that the analyses read (scalarize, affine forms, convexity, structure).
6. **OA (`MilpBound`).** Its 7 s trace is almost all the generic McCormick relaxation of `g`: standalone, 4.6 s for `g`, 0.4 s for `h`, 0.1 s for `f`. Unchanged here.

## Changes

- **Shared tapes (`fbbt.make_tape`, `joint_tape`).**
  - One tape per (function, input shape, dtype) for the process, as `jax.jit` keys its traces; LRU of 64.
  - A tape whose constants are tracers is not kept.
  - `lp._as_vec` and `convexity._flat_of` return one wrapper per function.
- **Analyses memoized per tape (`fbbt_interp._memo`).** `scalarize`, `make_propagate`, `convexity.certificate_of` and the layout behind `memory.tape_slots`. Keys include the scalar constants and functions of the modules a build reads, so patching one builds anew.
- **numpy dependency masks (`univariate._dep_masks`).** Elementwise rules, slice, squeeze, reshape, broadcast, concatenate and stack in numpy; other data movement and `dot_general` through the primitives as before; memoized per tape.
- **No preparation for left-out functions (`lp._parts_fn`).**
  - `prepare(lo, hi, skip)` skips the functions `_SubParts` leaves out; used by the node bound and OBBT.
  - `make_pool_cuts` does not relax functions with no nonlinear part; their parts get `b = -inf`, and `pool_parts` never pools them.
  - alphaBB reads the output size from the tape.
- **Batched loop carries (`fbbt.batched_init`).**
  - A `custom_vmap` that broadcasts an unbatched carry along the batch axis when the reference is batched — the broadcast `vmap` inserts anyway when the condition is batched. The body is then batched once.
  - Applied to the Kelley loops, the cut-chunk loop, the IPMs (`lp._ipm_box`, `quadratic`) and the FBBT interpreter's loops, where every carry ends up batched.
  - A first version built on `select(p, v, v)` cost 4 s more XLA compile CPU on the m410 bound kernel. The `custom_vmap` version leaves the optimized HLO unchanged.

## Effect

**Kernels, fresh process with `minimize`'s options, load 7–16, two runs each (trace s, base → new):**

| kernel | m410 | m34 |
|---|---|---|
| bound | 6.2–7.1 → 3.0–3.1 | 5.3–5.5 → 2.7–3.0 |
| OBBT | 3.6–3.7 → 1.1–1.5 | 3.0–3.3 → 0.9–1.0 |
| cut pool | 1.8–1.9 → 0.5 | 1.6–1.7 → 0.2–0.3 |
| dive | 1.5–1.9 → 1.1–1.2 | 1.1–1.3 → 1.0–1.2 |
| total | 13.2–14.5 → 5.8–6.4 | 11.2–11.6 → 5.0–5.2 |

- XLA compile CPU is unchanged (m410 bound 18.9–19.6 → 18.6–20.5 s; m34 22.2–22.3 → 22.3–23.1 s).
- Optimized HLO instruction counts of the bound, dive and OBBT kernels are identical.
- The cut-pool kernel is smaller (m410 7257 → 5155 instructions; compile CPU 7.2–8.8 → 4.9–6.7 s).

**Cold solves (`profile_root`, back to back, load 13–14), base → new:**

| | m410 | m34 |
|---|---|---|
| bound trace | 12.0 → 5.7 s | 14.2 → 7.2 s |
| OBBT trace | 21.2 → 3.1 s | 17.4 → 1.5 s |
| eager mask compiles on the tracer thread | 9.8 s → 0 | 11.9 s → 0 |
| root bound ready | 39.0 → 32.6 s | 40.7 → 35.4 s |
| search starts | 55.5 → 48.0 s | 55.6 → 49.4 s |
| nodes in 60 s | 7 → 15 | 3 → 11 |

At this load the bound kernel's compile (19–24 s) is the remaining wait.

**A/B, paired** (base and new on the same instance at the same time, 2 threads each, cold, 60 s, load 12–18):

| set | solved | geo-mean time (both solved) |
|---|---|---|
| int50 | 35 → 36 | 23.07 → 22.62 s |
| start50 | 34 → 34 | 20.81 → 20.31 s |

No losses, no wrong answers; `check_wrong` with seeds 0 and 1 is clean. Stage-2b (paired, load 13–19): no model solved in either arm. New reached the same or better bounds on most, e.g. m410 a root bound where base had none, m394 dual −0.0003 → 3190, m474 gap 6146.6/6475.9 → 6271.6/6272.7.

## Validity

- **Pure caches.** Tapes and analyses are deterministic functions of the function and its input shape; a tape holding tracers is never shared.
- **Skipped preparation was unused.** The data skipped is that of functions whose relaxation the kernel never evaluates; their interval bounds come from their affine forms as before.
- **Skipped pool parts.** Pool-cut parts of affine functions become "no cut" (`b = -inf`); `pool_parts` never selects them.
- **`batched_init`** changes only when a carry becomes batched, on loops where all carries end up batched anyway.
- **Bit-identical results** on m410 and m34 (8 boxes each, root box and boxes shrinking around the incumbent): node bounds (batched, single box, `exact`), OBBT boxes and live pool cuts.
- **Certificates.** Weak-duality certificates and `safe` mode are untouched.

## What is left

- The bound kernel's XLA compile (8 s at load 0, about 20 s at load 13) is now the m410 critical path.
- OA: `g` is certified convex on m410. Its tangents (`convexity`) instead of McCormick would make the OA program cheaper and tighter.
- The rows-free pool kernel is traced and compiled again during the search.
- Row-level alphaBB pruning for functions with both affine and nonlinear rows (m47, m6, kall_*) needs restricted evaluation of a function's rows; differentiating and then slicing does not prune them.

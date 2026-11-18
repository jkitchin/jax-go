# Cold compile and trace time in a stage-2b solve, 2026-11-16

**Question.** On a cold cache, how much of the 60 s goes to tracing and XLA compiles, and what can be removed? Round-2 profiles, averaged over the 19 models, showed about 9 s of waiting on background compiles and about 10 s of main-thread trace and compile.

## Diagnosis

Sources: `benchmarks/profile_root.py` (cold), `jax_log_compiles` per thread, py-spy of the bound kernel's trace, and HLO line attribution from `as_text(debug_info=True)`.

1. **Compiles compete.**
   - One XLA:CPU compile of a large kernel is internally parallel: m34_convex2's bound kernel used 30 s of CPU in 11 s of wall time.
   - Cold, the dive, cut-pool and OBBT compiles started next to it. The bound kernel's compile then took 22–25 s instead of about 15 s, and the root bound waited for it.
   - Tiny eager compiles inside the OBBT trace took 2–4.6 s each instead of under 0.3 s.
2. **Compiles that were never needed.**
   - The root OBBT kernel was compiled while the bound kernel was still building (15–21 s on m34 and m410). On m34 the cost rule then skipped the call.
   - The screening functions (`fvals`, `viol`) were compiled once per power-of-two batch size, on the main thread during the root dives: 2.7 s each on m34, and 10.6 s of main-thread compile in total.
3. **Kernel size, m34 bound kernel (35k HLO lines).**
   - In the `SPLIT_AFFINE` layout, the affine parts' centre cuts came from a forward-mode pass over the whole relaxation: 9.4k lines.
   - At each cut point every function was relaxed and only the nonlinear parts kept. XLA does not drop an unused function's interpreted relaxation: the equality rows (all affine on m34) cost 5.3k lines.
   - The OBBT kernel had the same pattern.
4. **GIL convoy.**
   - At Python's default 5 ms switch interval, a background thread that is tracing holds the GIL for up to 5 ms every time the main thread retakes it. The main thread retakes it at every local-solve callback, kernel call and eager op.
   - Microbenchmark, a loop of short jitted calls: 0.06 s alone, 1.49 s next to a busy thread at 5 ms, 0.14 s at 1 ms.
   - On m34, the final float64 check (`_verify`, eager ops) took 22 s next to the OA kernel's trace.
5. **Time limit.**
   - A kernel call dispatched before the deadline ran to the end. m47_bcp8's first bound call takes about 67 s.
   - With the kernel ready early (primed cache), the solve returned at 124 s; in the round-2 cold run, at 122.6 s.
6. **XLA options.**
   - `xla_llvm_disable_expensive_passes` and `xla_cpu_parallel_codegen_split_count` made no difference beyond noise.
   - `xla_backend_optimization_level=0` halves the compile (m34: 12.2 → 5.7 s; 21 → 11 s of CPU) but calls run about 30% slower (1.95 → 2.5 s per 16 boxes).

## Changes

- **`bnb._Gate`.**
  - The dive, cut-pool and tree-OBBT compiles are traced as before but compiled only after the root bound kernel's compile.
  - Root OBBT compiles after the root bound only if its call fits its cost rules with half of the time left (`obbt_may_fit`, `_OBBT_FIT_SHARE`).
  - A held compile starts as soon as a caller waits for it.
  - Nothing is held when the bound kernel was traced in under `_GATE_TRACE_S` (1 s).
  - All gates open when `minimize` returns.
- **`bnb._await`.**
  - A kernel call still running at the deadline raises `_Deadline`. The call finishes in the background and its result is dropped, as if it had not been dispatched.
  - It polls only on a kernel's first call, or when the time left is under `_AWAIT_SAFE` (10) × the kernel's slowest call + `_AWAIT_MIN` (1 s); otherwise it blocks.
- **`bnb._batched`.** Screening uses two fixed batch sizes, 8 and 64, both compiled off the main thread at the start (`_warm_screen`).
- **`bnb.SWITCH_INTERVAL`.** The GIL switch interval is 1 ms while `minimize` runs and is restored on return.
- **`lp._affine_forms` and `AFFINE_EXACT`.**
  - The coefficients of affine parts are propagated through the scalarized tape, and their cuts are taken from those coefficients instead of the centre pass. An affine part's relaxation is the function itself.
  - Only when `safe` is off.
- **`lp._SubParts(..., sizes)`.** At the cut points, a function with no nonlinear part is not relaxed at all; its interval bounds are the range of its affine form over the box. Applies to the node bound and to `make_obbt`, which now uses `SPLIT_AFFINE` as well.
- **`lp._joint`.** The joint tape and its scalarized program are built once per problem; `_nonlinear_vars`, `_affine_forms` and `_tape_eqns` share them.
- **Option, off by default: `bnb._FAST_COMPILE`.**
  - The root kernel is compiled at `-O0` first, then again with XLA's defaults when the search starts; calls switch to the optimized executable once it is ready.
  - Left off because the root bound came earlier but root probing ran on the slower kernel.

## Effect

**Kernels, standalone at load about 9; the same bounds as base.**

| kernel | HLO lines | compile CPU | trace |
|---|---:|---:|---:|
| m34 bound | 35,478 → 14,588 | 33.1 → 24.1 s | 14.8 → 11.9 s |
| m34 OBBT | 27,035 → 10,148 | 23.0 → 19.0 s | — |
| m410 bound | 36,649 → 15,904 | 32.3 → 21.0 s | 14.9 → 8.4 s |
| m410 OBBT | 27,987 → 11,871 | 22.3 → 16.9 s | — |

The lifted quadratic kernel (the CLay models, m474) is unchanged.

**Cold profiles, base vs new back to back, load 7–10.**

| model | root bound ready | bound compile in the solve | main-thread compile | nodes |
|---|---:|---:|---:|---:|
| m34 | 40.7 → 25.7 s | 25.4 → 12.6 s | 10.6 → 2.7 s | 1 → 27 |
| m410 | 47.8 → 31.2 s | 21.8 → 9.7 s | 13.2 → 10.3 s | 1 → 7 |
| m394 | 27.2 → 23.2 s | 10.7 → 7.7 s | 7.8 → 5.8 s | 2665 → 3985 |

On m34 the OBBT kernel (20.7 s of compile) is no longer built.

**run.py, cold, 60 s, `--jobs 1`, back to back at load 3–4.6.**

| model | base | new |
|---|---|---|
| m34 | time limit, 39 nodes | optimal, 48.5 s, 25 nodes |
| m47 | time limit, returned at 118.3 s | time limit, 60.1 s |
| m394 | time limit, bound 7582 | time limit, bound 8050 (optimum 8092.5) |
| m399 | optimal, 57.4 s | optimal, 56.3 s |
| m406 | optimal, 57.7 s | optimal, 57.2 s |
| m410 | time limit, bound 0 | time limit, bound 1373, 55 nodes |
| m474 | optimal, 61.1 s | time limit at 60.4 s, gap 1.4e-4 |

- **m47 with a primed cache:** 123.9 s → 60.2 s.
- **Stage-2b solved in 60 s:** 6 → 6 (m34 gained; m474 lost at the margin).
- **int50 A/B** (60 s, `--jobs 2 --threads-per-job 2`): 42 → 45 solved, no losses. In the half run at equal load: 20 → 21 solved, geometric mean time 19.1 → 19.3 s.
- **start50 A/B:** 39 → 40 solved. Lost ex6_2_6: base solved it in 58.7 s at load 3.7, and it timed out in both arms in a back-to-back rerun.
- **Wrong answers:** none in either A/B; `check_wrong` with seeds 0 and 1 is clean.
- **m422.** The dive kernel is now ready in time, so dives run (4 s at the root, 3.5 s in the search). The incumbent is better (4.755 against 5.026) and fewer nodes are searched.

## Validity

- Gating, `_await` and `_batched` change when work happens, not what it computes. A call dropped at the deadline leaves the state as before the call.
- The exact affine cuts are the same LP rows as the relaxation pass, up to float rounding. The pass computed the same rows in floats without outward rounding when `safe` was off, and the exact cuts are not used when `safe` is on.
- The interval bounds of the functions left out are the exact range of their affine forms over the box: they are not wider than the relaxation's.
- Bounds remain weak-duality certificates.
- Tests compare the new layouts with the old: node bounds to 1e-6, and OBBT boxes to 1e-4 of the box width.

## What is left

- **End-of-solve overrun (up to about 4 s on m410).** `repair_incumbent` and `_verify` (eager ops) wait for the GIL behind the OA kernel's trace, which continues past the deadline.
- **Trace time, still about 9–12 s on m34 and m410.**
  - About 30% of it is evaluating the `.nl` scalar expressions, each linear term as its own op; a vectorized linear part in `nl.py` is the next large cut.
  - About 30% is the alphaBB traced Hessian over every output, mostly affine on these models. It should be restricted to the nonlinear rows, with element-level pruning of dead rows.
- **m47.** Its first bound call (dense LPs with about 51k rows per box) takes 67 s: a kernel-design problem.
- **Tiered `-O0` compile.** Measure it at low load before turning it on.

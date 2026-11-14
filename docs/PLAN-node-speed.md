# Plan: faster B&B nodes (node throughput)

## Diagnosis (summary; full report in the PR / docs)

minlp.org m424 (k-means, convex MIQCP): ~11.3 ms per node vs SCIP ~1 ms.

- 85-88 %: the node LP, a cold dense IPM (`quadratic._ipm`, fixed sigma = 0.2, no predictor-corrector,
  hard-coded start z = 0.5, y = 1) solved 3.5-4.5 times per node (Kelley rounds of
  `quadratic.make_quadratic_bounder.round_`; the lifted path has no stall rule like
  `lp._kelley_go`, so ~88 % of unpruned boxes run all 5 rounds); 15-19 iterations each at ~90 us
  (dense 90x90 Cholesky incl. 19-32 fixed columns, scatter-built normal matrix).
- The vmapped `while_loop` runs the batch maximum of iterations: 36 % of executed iterations are
  for boxes that already stopped. Batching gives no CPU speedup (cost linear in boxes).
- Effectively one core: cpu/wall 1.35-1.4; XLA intra-op threads and OpenBLAS threads are a net
  loss (NPROC=1 is faster per box; OPENBLAS_NUM_THREADS=1 faster on hadamard_5).
- FBBT 9 % (1.05 ms/box), LP build < 1 %, host Python small.
- Measured (uncommitted test edit): dispatching the 16-box batch calls in `bnb.minimize`'s
  `run` from a 4-thread pool: m424 60.7 -> 327 nodes/s (batch 16), 373 (batch 4), 566 (batch
  4, n_kelley=2); bound at 45 s 0.054 -> 0.30-0.37 (opt 0.3737). Kernel replay on hadamard_5,
  tspn05, ex6_2_6: 2.2-4.6x. With NPROC=1 thread dispatch gives nothing (XLA then runs one
  executable at a time).
- A warm-started dual simplex (SoPlex) needs 0.09-0.15 ms per node LP vs 8.4-10 ms of IPM;
  warm-starting the IPM saves only ~15 % of iterations.

Profiling tool: `benchmarks/profile_node.py` (records a run, replays boxes: batch x threads,
LP sizes, iterations per node, iteration cost split, FBBT / cut-generation alone, `--simplex`),
on branch `worktree-agent-ad7c6072d53293ced` commit efcd81d.

## Phases (each a separate commit, measured before the next)

### Phase 1: concurrent dispatch of bound-kernel calls (expected 3-6x on multi-core)
- `jaxglobal/bnb.py`, `minimize`: where `evaluate` / `run` loop over batch chunks one after
  another, submit the chunks to a thread pool and gather in order (`ex.map` keeps order).
- Threads: `min(chunks, workers)` with `workers` from `_ncpu()` (honours RAYON_NUM_THREADS /
  NPROC caps of `benchmarks/run.py --threads-per-job`); a module constant / option to disable
  (`1` = today's behaviour). With an XLA pool of 1 thread there is no gain: document it.
- Shared state written from the chunk calls (`_Precompiled.dispatched`, `bound_call[0]`, kernel
  start/compile bookkeeping, deadline checks, kcache) must be made thread-safe or kept on the
  main thread. Keep results identical to sequential dispatch (same per-box bounds to 1e-12,
  same order): add a test comparing both on a small integer problem.
- Re-measure the default `batch_sizes` (16 vs 4 vs 8) under threads.
- Consider pinning OpenBLAS to 1 thread inside worker processes when threads are used (measure
  first; hadamard_5 was faster with OPENBLAS_NUM_THREADS=1).

### Phase 2: fewer, cheaper LP solves in the lifted quadratic path (expected 1.3-2x more)
- `jaxglobal/quadratic.py`, `make_quadratic_bounder` rounds: add a stall rule like
  `lp._kelley_go` (stop when a round raises the bound by <= KELLEY_STALL of the gap, or the box is
  pruned). Re-tune KELLEY_STALL / rounds for this path (n_kelley=2 was better on m424).
- `quadratic._ipm`: port Mehrotra predictor-corrector from `lp._ipm_box` (`body_pc`); measured
  34 vs 60 own iterations per node, 1.2-1.3x time.
- Optionally warm-start round k+1 from round k's (z, y) (with Mehrotra: 29 iterations).
- Keep every bound going through the existing dual certificate (`_certificate` / weak duality),
  including `safe=True` margins; the bound must stay valid for any iterate.

### Phase 3 (only if Phases 1-2 land and time allows; separate decision): host dual simplex
- Prototype behind an option: solve the node LP rows that `_lp` / `_unit` / tangents already
  build with a warm-started dual simplex (HiGHS via scipy has no basis warm start; SoPlex via
  pyscipopt `LP` or highspy does), carry the parent's basis to children, and feed the duals to
  the same certificate. Expected LP part 8.4-10 ms -> ~0.7 ms per node. Report a go / no-go with
  measurements; do not make it default in this work.

### Not in scope now
Fixed-column elimination / sparse normal matrix in `quadratic._ipm` (<= 1.5-2x, high effort
under static shapes); FBBT round trimming (9 %).

## Validation (every phase)
1. Unit tests for touched modules, files run in separate processes with JAXGLOBAL_NO_CACHE=1
   (the whole suite in one process crashes XLA): at least test_quadratic, test_lifted_poly,
   test_binary_cuts, test_eigen_cuts, test_integer, test_milpbound, plus any bnb tests touched.
2. `python benchmarks/check_wrong.py --sample 12 --time-limit 20 --seed 0` and `--seed 1`: no
   wrong answers.
3. Throughput: `benchmarks/profile_node.py` / nodes per second on m424, m423, m419, m425
   (minlp.org) and hadamard_5, tspn05, ex6_2_6 (MINLPLib), before and after.
4. A/B on MINLPLib int50 and start50 (`benchmarks/sets/`, 60 s): run baseline and candidate on a
   quiet machine back to back or interleaved, never with another heavy process running (a
   loaded baseline earlier made a neutral change look like +8 solved). Report solved counts,
   geometric mean time on commonly solved, any losses.
5. Bounds stay valid: no lower bound above a reference optimum on any run.

## Done when
Phase 1 and 2 committed with measurements in `docs/ANALYSIS-node-speed.md`; m424 nodes/s and
time-to-solve on the k-means models vs before; int50 / start50 A/B shows no losses.

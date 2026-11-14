# Node speed: results

Plan and diagnosis: `docs/PLAN-node-speed.md`. Machine: 4 cores, jaxlib 0.11, one heavy
process at a time (load average at the start of each run is in the result CSVs). Every
configuration compared has its own kernel cache (`JAXGLOBAL_CACHE_DIR`); k-means runs were warmed
with a 30 s run first, the benchmark-set A/B runs are cold vs cold.

Summary (original code -> Phase 1 + 2 defaults):

| | before | after |
|---|---|---|
| m424 nodes/s | 78 | 416 (5.3x) |
| m424, 120 s | not solved, lb 0.237 (opt 0.3737) | optimal in 61-75 s |
| m419 | optimal 62.4 s | optimal 16.0 s |
| m425 lb at 120 s (ub 1.113) | 0.296 | 0.676 |
| int50 solved (60 s) | 46 | 48 |
| start50 solved (60 s) | 40 | 44 |
| losses on the sets | | none |
| wrong answers (check_wrong seeds 0, 1) | | none |

## Phase 1: concurrent dispatch of the bound-kernel calls

`bnb.minimize`'s `run` submits the padded calls of one batch to a thread pool of
`bnb._dispatch_threads()` threads and returns their results in call order
(`bnb.DISPATCH_THREADS`, env `JAXGLOBAL_DISPATCH_THREADS`; default `_ncpu()`, which honours
`RAYON_NUM_THREADS`, so `benchmarks/run.py --threads-per-job K` gives K dispatch threads; `1`
is the old one-call-after-another behaviour). All calls are waited for before the results are
returned or the first failure (in call order) is raised; each call still checks the deadline
before it starts. `_Precompiled.dispatched` became per thread, the failed-compile path uses
`dict.pop` so two threads cannot race on it. Every call computes exactly what it computed
before: `tests/test_dispatch.py` checks bit-identical outputs of every bound-kernel call and the
same result with 1 and 4 threads.

Since the kernels are unchanged, the A/B is the same tree with `JAXGLOBAL_DISPATCH_THREADS=1`
(sequential) and without (4 threads), same warm cache, back to back.

minlp.org k-means, 120 s, `--jobs 1`:

| model | sequential | nodes | nodes/s | 4 dispatch threads | nodes | nodes/s |
|---|---|---:|---:|---|---:|---:|
| m419 | optimal 62.4 s | 6255 | 100 | optimal 19.8 s | 6255 | 316 |
| m423 | optimal 63.7 s | 5459 | 86 | optimal 61.6 s | 33679 | 547 |
| m424 | limit, lb 0.237 (opt 0.3737) | 9403 | 78 | optimal 61.2 s | 21107 | 345 |
| m425 | limit, lb 0.296 (ub 1.113) | 6135 | 51 | limit, lb 0.629 | 28407 | 236 |

m419 explores the identical tree (3.2x faster). m423's search differs between runs (see the
node counts): its time to solve is set by when the incumbent is found, not by the node rate.

MINLPLib `int50` / `start50`, 60 s, `--jobs 1` (uncapped: 4 dispatch threads), cold caches,
sequential first and threaded right after on a quiet machine (load ~1.5, nothing else running).
"Solved" is `report.classify` against the MINLPLib reference; time is the geometric mean (each
time floored at 1 s) over the instances both solved.

| set | run | solved | wrong | geo-mean time, common | total nodes |
|---|---|---:|---:|---:|---:|
| int50 | sequential | 46 | 0 | 12.31 s (46) | 79 408 |
| int50 | 4 threads | 47 | 0 | 12.19 s (46) | 260 908 |
| start50 | sequential | 40 | 0 | 8.49 s (40) | 156 098 |
| start50 | 4 threads | 44 | 0 | 7.91 s (40) | 370 328 |

Gains: ex1252a (int50, unsolved -> 26.8 s), kall_circles_c6a / c6b / c7a and
kall_circlespolygons_c1p12 (start50, unsolved -> 17-47 s), ex8_4_5 38 -> 15 s, kall_diffcircles_8
46 -> 15 s. Losses: none (no instance lost, none solved more than 1.5x + 2 s slower). Most
int50 / start50 instances that solve do so in a few batches, where compile and local solves
dominate, so the geometric mean barely moves; the gain is on the searches that need nodes.

Validation: `tests/test_dispatch.py`, test_batching, test_time_limit, test_kernels,
test_quadratic, test_lifted_poly, test_binary_cuts, test_eigen_cuts, test_integer,
test_milpbound, test_nl, test_kcache, test_incumbent pass (one file per process,
`JAXGLOBAL_NO_CACHE=1`). `check_wrong.py --sample 12 --time-limit 20` seeds 0 and 1: no wrong
answers (5 and 8 solved; it runs 2 jobs, i.e. 2 dispatch threads each).

Notes:
- With `NPROC=1` (XLA's pool capped at one thread) concurrent calls run one at a time and the
  threads give nothing (measured in the diagnosis, not re-measured here); `run.py --jobs N` sets `RAYON_NUM_THREADS = cpu_count // N`, so each job
  gets that many dispatch threads and the machine is not oversubscribed.
- The mid-iteration deadline test (`test_deadline_mid_search_keeps_pre_iteration_bound`) pins
  `DISPATCH_THREADS = 1`: with a pool all calls of an iteration start together, so the check
  before a call only refuses calls queued behind busy threads. A solve still overruns its limit
  by at most one round of concurrent calls.
- The slowest-call estimate (`bound_call`, used for dive and root-OBBT budgets) is now measured
  on calls that run concurrently, so it is somewhat larger than a lone call.

## Phase 2: Mehrotra predictor-corrector and a round stall rule in the lifted quadratic path

`quadratic._ipm` takes Mehrotra predictor-corrector steps (`IPM_PC = True`), ported from
`lp._ipm_box`'s `body_pc`: an affine-scaling step sets `sigma = (mu_aff / mu)^3`, then a
second-order corrected step; both solves share the iteration's one Jacobi-scaled Cholesky
factorization (one more triangular solve and sparse product per iteration). It applies to the
bounder and to the lifted OBBT LPs. `make_quadratic_bounder(stall=...)` / `ROUND_STALL` adds
`lp._kelley_go`'s stall rule to the lifted rounds (stop when a round raised the bound by at most
`stall` of the gap); it is implemented but **off by default** (below).

Validity: only the iterates change. Every bound is still `_certificate(y)` of some dual `y`
(weak duality with the rounding error, `SAFE_K` margins under `safe=True`), the best over the
iterations and rounds. New test `test_rounds_and_steps_keep_bounds_valid` (12 configurations:
predictor-corrector on/off x stall off / 1e-4 / 0.5 x safe on/off): on random boxes of a
nonconvex QCQP no sampled feasible point is below the bound, and the one-LP bound equals
HiGHS' LP value.

### Recorded node boxes (bounder alone, batch 16, `JAXGLOBAL_DISPATCH_THREADS=1`)

Boxes recorded from a 20-25 s run, FBBT-tightened, replayed through `make_quadratic_bounder`
(256 boxes; `mean lb` is the mean of min(lb, cutoff)):

| model | fixed sigma (before) | predictor-corrector | pc + stall 1e-4 | pc, n_kelley 2 |
|---|---|---|---|---|
| m424 | 9.93 ms, lb 0.07984 | 6.82 ms, lb 0.07980 | 5.95 ms, lb 0.07523 | 3.56 ms, lb 0.07334 |
| m425 | 14.18 ms, lb 0.08292 | 10.49 ms, lb 0.08292 | 7.65 ms, lb 0.08192 | |
| m419 | 7.49 ms, lb 1.23705 | 5.07 ms, lb 1.23700 | 5.43 ms, lb 1.23700 | |
| hadamard_5 | 34.24 ms, lb -17.3177 | 20.94 ms, lb -17.3177 | 20.80 ms, lb -17.3177 | |

Predictor-corrector: 1.35-1.65x per box at the same mean bound (single boxes up to 1.4e-3 lower:
the rounds' tangents are taken at slightly different LP points). The stall rule
trades bound for time (m424: 13 % faster, mean bound 6 % lower); fewer Kelley rounds even more
so. (tspn05: the 20 s recording run made no node calls, nothing to replay.)

### minlp.org k-means (120 s, warm caches, 4 dispatch threads)

| model | Phase 1 | nodes/s | + predictor-corrector | nodes/s | + pc + stall 1e-4 | nodes/s |
|---|---|---:|---|---:|---|---:|
| m419 | optimal 19.8 s | 316 | optimal 16.0 s | 354 | optimal 18.3 s | 313 |
| m423 | optimal 61.6 s | 547 | optimal 60.1 s | 562 | optimal 48.0 s | 701 |
| m424 | optimal 61.2 s | 345 | optimal 74.5 s | 416 | optimal 68.1 s | 458 |
| m425 | limit, lb 0.629 | 236 | limit, lb 0.676 | 306 | limit, lb 0.698 | 308 |

m424 needed 31 000 nodes with predictor-corrector against 21 100 before (the trees differ:
bounds per box move in the last digits and change the search order), so its time to solve went
up although its node rate went up 20 %. Against the original sequential code m424 went from
not solved in 120 s (lb 0.237) to solved in 61-75 s; m425's bound at 120 s from 0.296 to
0.676 (upper bound 1.113, not closed).

### int50 / start50 A/B (60 s, cold caches, back to back after the Phase 1 runs)

| set | run | solved | wrong | geo-mean time, common | total nodes |
|---|---|---:|---:|---:|---:|
| int50 | Phase 1 (threads) | 47 | 0 | 12.39 s (47) | 260 908 |
| int50 | + predictor-corrector | 48 | 0 | 11.85 s (47) | 212 352 |
| int50 | + pc + stall 1e-4 | 48 | 0 | 12.27 s (48; pc: 12.08) | 210 304 |
| start50 | Phase 1 (threads) | 44 | 0 | 9.03 s (44) | 370 328 |
| start50 | + predictor-corrector | 44 | 0 | 9.15 s (44) | 396 246 |
| start50 | + pc + stall 1e-4 | 44 | 0 | 8.96 s (44; pc: 9.15) | 395 926 |

Predictor-corrector: tln4 (int50) unsolved -> 30.0 s; no instance lost and none solved more
than 1.5x + 2 s slower. The stall rule: no instance changes class, geometric means within
noise (+-2 %). Since it helped two k-means models, hurt one and changed nothing on the sets,
`ROUND_STALL` stays off (`-1`): it is a knob, not a default.

Overall, original code -> Phase 1 + 2 (defaults): int50 46 -> 48 solved, start50 40 -> 44,
no losses.

Validation: test_quadratic (with the new test), test_lifted_poly, test_binary_cuts,
test_eigen_cuts, test_integer, test_milpbound, test_nl, test_dispatch, test_constrained,
test_tightening, test_cut_pool pass (one file per process, `JAXGLOBAL_NO_CACHE=1`);
`check_wrong.py --sample 12 --time-limit 20` seeds 0 and 1: no wrong answers (5 and 8 solved).

### Tried and dropped / not done

- Stall rule as default: see above (kept as `ROUND_STALL`, off).
- `n_kelley = 2` (better on m424 in the diagnosis): 2.8x cheaper per box than 4 rounds with
  predictor-corrector but an 8 % lower mean bound on m424; it is a `minimize` default for every
  lifted model and was not A/B tested on the sets, so `n_kelley = 4` stays.
- Warm start of round k+1 from round k's (z, y): not implemented. The diagnosis measured 29 vs
  34 iterations with predictor-corrector (15 %); a warm start needs an interior point for rows
  that, by construction, cut the previous point off, i.e. a shift heuristic whose robustness
  would have to be shown, for a gain smaller than the noise of the k-means runs.

### Batch size under threads (`batch_sizes`)

k-means, 120 s, predictor-corrector, warm caches; with Phase 1+2 the trees are identical across
batch sizes (same node counts), so the times compare directly:

| model | 16 (default) | 8 | 4 |
|---|---|---|---|
| m419 | 16.0 s | 15.4 s | 13.1 s |
| m423 | 60.1 s | 57.9 s | 54.3 s |
| m424 | 74.5 s | 68.8 s | 59.7 s |
| m425 (lb at 120 s) | 0.676 | 0.692 | 0.687 |

Replay of the real m424 kernel (64 boxes, `profile_node.py`): batch 16 x 4 threads 1.84 ms per
box, batch 4 x 4 threads 1.40, batch 1 x 4 threads 1.64 (one thread: 8.05 / 7.34 / 9.90).

On the sets `batch_sizes=(4,)` was neutral: int50 48 / 48 solved, geometric mean 12.08 ->
12.31 s; start50 44 / 44, 9.15 -> 8.63 s; one loss, kall_circlespolygons_c1p12 14.5 -> 26.0 s.
It helps the k-means models by 10-25 % and not the sets, so the default stays `(16,)`.

Not measured: pinning OpenBLAS to one thread in worker processes under threaded dispatch.

## Phase 3: host warm-started dual simplex (go / no-go)

Not implemented; measured with `benchmarks/profile_node.py --simplex` on m424 after Phases 1-2
(30 s recording, 64 node boxes, the same LP rows as the lifted bounder, SoPlex through
`pyscipopt.LP`, one core):

| per node | IPM in the bound kernel (Phase 2) | dual simplex (SoPlex) |
|---|---|---|
| first LP | ~8.4 iterations x 116 us, executed as the batch maximum | cold 583 us (29 iterations); from the parent's basis 91 us (12 iterations) |
| each Kelley round | ~8 iterations | 158 us (15 iterations) after adding the round's tangent rows |
| LP part of a node | 7.2 ms (62.5 executed iterations x 116 us) | ~0.6 ms (warm first LP + ~3 re-solves) |

The whole node kernel costs 8.05 ms per box on one thread and 1.84 ms per box of wall time with
4 dispatch threads; FBBT alone is 1.14 ms per box. A warm-started simplex would cut the LP part
~10x on one core, so a node would cost ~2 ms of CPU (FBBT, LP build, cuts, host round trips)
instead of ~8.

**Go, as separate work behind an option, not as part of this change.** What it takes:
- the bound kernel is one jitted, vmapped JAX function; a host LP needs the rounds split into
  JAX pieces around a host loop (5 round trips per batch) or a `pure_callback` inside the
  kernel (sequential under `vmap`; parallel only if the solver releases the GIL);
- per-node basis storage and parent -> child hand-over in `bnb.minimize`'s host loop;
- the duals go through the same `quadratic._certificate` (valid for any `y >= 0`, so the
  bound and the `safe` margins stay as they are); SoPlex's own optimality is never trusted;
- an optional dependency (pyscipopt; scipy's HiGHS has no basis warm start; highspy is not
  installed here).
The expected gain over Phase 1+2 on 4 cores is smaller than the 10x LP ratio, because the
threaded IPM already uses all cores: roughly 2-3x on the lifted k-means nodes if the host
solves run on several threads, less if they serialize on the GIL.

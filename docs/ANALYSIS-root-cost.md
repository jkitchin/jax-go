# Root cost on large minlp.org models (analysis, 2026-11-15)

Question: why does jaxglobal not get past the root within 60 s on the larger stage-2b models
(`m47_bcp8`, `m400_CLay0304HRcone`, `m441_kClus_3_10_2_HRc_1`; `benchmarks/results/minlporg_stage2b_cold.csv`)
and the stage-2 models over 150 variables (`m20_MTG_EX1a`, `m21_MTG_EX1b`), and which general
changes fix it?

Machine: 4-core VM shared with four other agents; the 1-min load average was 3-17 during these
runs and is given per run. Profiles: `benchmarks/profile_root.py` (`bnb._mark` stages; cold = no
cache, prime = empty cache dir, warm = the dir prime filled), 60 s, one solve at a time.

## 1. Where the 60 s went (before)

`profile_root.py` on `origin/claude/stage2b`:

| bucket (s) | m441 cold | m441 prime | m441 warm | m400 cold | m400 prime | m400 warm | m47 cold | m47 prime | m47 warm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel wait: trace+lower | 0.0 | 0.1 | 0.0 | 3.7 | 4.9 | 4.7 | 54.2 | 43.1 | 4.3 |
| kernel wait: XLA compile | 15.6 | 9.7 | 0.0 | 10.3 | 10.6 | 0.9 | 0.0 | 51.5 | 55.7 |
| bound_domain | 0.0 | 0.0 | 0.0 | 15.4 | 14.4 | 10.5 | 3.2 | 3.2 | 1.7 |
| root bound: kernel run (first LP) | **42.3** | 0.5 | 2.8 | 3.8 | 3.5 | 4.6 | 0.0 | 0.0 | 0.0 |
| root dives + enum | 0.1 | 11.5 | 7.2 | 4.4 | 5.6 | 4.4 | 0.0 | 0.0 | 0.0 |
| root OBBT: kernel run | 0.0 | **203.8** | **115.4** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| probing: run | 0.0 | 0.0 | 0.0 | 21.7 | 36.8 | **77.0** | 0.0 | 0.0 | 0.0 |
| wall (60 s limit) | 62.2 | 232.3 | 129.3 | 62.7 | 80.3 | 106.3 | 60.3 | 103.4 | 64.7 |
| nodes | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| load (1 min) | 3.3 | 5.8 | 7.6 | 7.1 | 6.8 | 7.8 | 9.6 | 7.8 | 6.5 |

m20 / m21 (stage-2 runs): 0 nodes, no bound, no incumbent at 60 s, cold and warm.

Three causes.

**(a) OpenBLAS threads (every IPM; m441, m400).** One 16-box call of m441's bound kernel (lifted
quadratic LP: 316 columns, 1095 rows, 22 IPM iterations) took 42 s; the root OBBT call (2n = 452
LPs x 2 rounds) 115-204 s, far past the limit, since a kernel call cannot be interrupted. Each
IPM iteration factors a dense normal matrix with `jax.scipy.linalg.cho_factor`; jaxlib's CPU
backend sends it to SciPy's OpenBLAS, which by default runs one thread per core and busy-waits.
Next to minimize's own threads (dispatch threads, background XLA compiles, pounce) and other
processes, a factorization waits for its spinning helpers to be scheduled. Same 316 x 316 SPD
matrix, load ~7: `cho_factor` median 269 ms (min 1.3, max 519) with the default threads, 0.68 ms
with `OPENBLAS_NUM_THREADS=1`; 16 batched factorizations 442 ms vs 12 ms. m441's root bound (same
LP, same 22 iterations): 16.8 s -> 0.03 s for one box, 47 s -> 0.45 s for 16. `run.py --jobs 1`
sets no thread cap, so the whole stage-2b run was in this regime.

**(b) m47's reduced-space kernel does not fit.** m47 has no lifted form; its bound kernel is the
reduced-space cut LP (`lp.make_constrained_bounder`). Its trace had not finished after 60 s
(trace + lower 113 s on the tracer thread). Two parts:
- alphaBB: the traced interval Hessian `jacfwd(jacfwd(g))` has 60k live operations (volume 6e7)
  for the inequalities and 29k for the equalities (thresholds 15k / 2e5); the interpreter would
  need 38 GB and 27 GB (`TAYLOR_BYTES` 512 MB), so `alphabb.hessian` fell back to the traced one.
- cut generation: with alphaBB skipped the kernel compiles, but a 16-box call asks XLA for 84 GB
  (`RESOURCE_EXHAUSTED`). The forward-mode cut pass runs the relaxation (2059 tape slots) with
  n = 168 tangents at the center and 2n axis points, 337 points in one vmap: 4.6 GB per box. Of
  the 112871 (point, part) rows `_cut_rows` keeps 1881 (a part keeps the axis points of the
  variables it is nonlinear in); 67 of the 168 variables are linear in every part; the
  (points, parts, n) cut buffer alone is 150 MB per box per copy.

**(c) Root phases whose cost grows with n run whatever they cost.** Root OBBT is one call of 2n
LPs x 2 rounds (fixed 40 IPM iterations, no deadline inside); the guard only asked for 10 slowest
bound calls of time left. After (a) it still took m441 20-22 s (127 nodes in the remaining time),
m400 15 s (88 bound calls), m20 / m21 10-17 s. `domain.monotone_bounds` swept the whole tape
eagerly once per candidate variable: m400 12 sweeps, 12.7 s of its 15 s `bound_domain`.

## 2. Changes

1. `jaxglobal/blas.py` (new), `bnb.minimize`: `minimize` runs inside `blas.limited()`, which sets
   every loaded OpenBLAS (SciPy's and NumPy's; found through `/proc/self/maps`, or dyld on macOS,
   and driven by ctypes) to `JAXGLOBAL_BLAS_THREADS` threads (default 1; 0 leaves them alone) and
   restores the previous counts when the last concurrent `minimize` returns. `bnb.minimize` is a
   thin wrapper of `_minimize` (same signature and docstring via `functools.wraps`).
2. `domain.monotone_bounds`: on tapes swept eagerly (above `_JIT_MAX_NODES`), candidates none of
   whose rows holds another candidate's variable share one sweep (each gets exactly the row bounds
   of its own sweep, since a row's interval bound depends only on its own variables); groups are
   swept in turn on the box capped so far. Small tapes are unchanged. m400 `bound_domain` (cold,
   standalone): 23.4 s -> 9.3 s.
3. `alphabb.SKIP_UNAFFORDABLE` (default on): with `interpret="auto"`, a function whose traced
   Hessian is over the size thresholds and whose interpreter is over `TAYLOR_BYTES` gets no
   alphaBB (its McCormick / univariate relaxation alone, valid) instead of the traced Hessian
   anyway.
4. `lp.make_constrained_bounder`, `lp.make_obbt`:
   - cuts only at the center and at the axis points of variables some part is nonlinear in
     (`_axis_points`; at the others every part's cut repeats the center's, and `_cut_rows`
     already dropped those rows);
   - the bounder keeps its cuts as the flat list of kept LP rows (center / axis rows, then one slot
     of the nonlinear parts' rows per Kelley point) instead of a (points, parts, n) buffer; rows
     are gathered before they are built and normalized (also in OBBT);
   - the points run in sequential chunks (`lax.map`) when `_cut_bytes` (64 x slots x n per point)
     x points x 16 boxes exceeds `CUT_BYTES` (1 GiB).
   The LPs are the same row for row: old and new bounder give bounds equal to 0.0 relative
   difference on 8 near-root boxes each of ex1252, nvs03, ex8_4_5, alan, batch, ex6_2_6, st_e40.
   `tests/test_cut_points.py` checks the point selection (exact) and the chunking (chunks round
   differently: bound and reduced box to 1e-7). m47: XLA temp memory of a 16-box bound call
   74 GB -> 2.0 GB; compile 19 s.
5. `bnb._ROOT_OBBT_SHARE` (0.25): root OBBT also needs its estimated cost, slowest bound call x
   2 n obbt_rounds / batch size, to be at most a quarter of the time left. The estimate was within
   2x of the measured call on m400 (9 s estimated / 15 s) and m441 (31 s / 21 s). With n <= 50 it
   can only fire when one bound call takes more than ~1/50 of the time left. Evidence that skipping
   pays on these models (cold, 60 s, after (a)): m441 with OBBT 127 nodes, bound -5e-9; without
   1023 nodes, bound 0.059 (incumbent 0.909 both); m400 without OBBT 31 nodes and an incumbent
   78680 (with OBBT, cold: 1 node, no incumbent).

None of these touches a certificate: (1), (2), (4) compute the same quantities differently, (3)
drops an optional relaxation term that was combined by max with McCormick, (5) skips a
tightening.

## 3. After

`profile_root.py` on this branch (all five changes), 60 s; load 8-18 (another agent's sweep was
running), so wall times are inflated; nodes and bounds are the comparison:

| bucket (s) | m47 prime | m47 warm | m441 prime | m441 warm | m400 prime | m400 warm | m20 prime | m20 warm | m21 prime | m21 warm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel wait: trace+lower | 32.6 | 0.1 | 7.3 | 0.0 | 8.2 | 1.6 | 10.1 | 3.4 | 12.0 | 2.7 |
| kernel wait: XLA compile | 40.1 | 3.7 | 14.0 | 0.6 | 14.4 | 1.2 | 23.1 | 1.5 | 19.8 | 1.4 |
| bound_domain | 4.1 | 1.6 | 0.0 | 0.0 | 10.7 | 4.3 | 32.2 | 16.8 | 35.6 | 18.9 |
| root bound: kernel run (first LP) | 0.0 | **70.7** | 0.6 | 0.5 | 0.2 | 0.2 | 0.0 | 0.3 | 0.0 | 0.3 |
| root local solves + incumbent | 0.0 | 0.0 | 0.9 | 1.7 | 1.9 | 2.8 | 0.0 | 13.0 | 0.0 | 8.8 |
| root dives + enum | 0.0 | 0.2 | 12.7 | 8.6 | 4.5 | 4.3 | 0.0 | 0.1 | 0.0 | 0.1 |
| root OBBT: kernel run | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 17.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| probing: run | 0.0 | 0.0 | 5.9 | 5.1 | 10.1 | 5.3 | 0.0 | 2.9 | 0.0 | 6.4 |
| search | 0.0 | 0.0 | 19.8 | 42.1 | 8.7 | 20.8 | 0.0 | 19.1 | 0.0 | 18.9 |
| wall | 81.2 | 80.7 | 65.1 | 62.2 | 61.2 | 60.2 | 70.5 | 61.2 | 73.3 | 61.1 |
| nodes | 0 | 1 | 127 | 255 | 15 | 31 | 0 | 757 | 0 | 221 |
| bound (min form) | -inf | -1.0 | -7e-9 | -7e-9 | -0.0015 | 1840 | -inf | -4236 | -inf | -15470 |
| incumbent (min form) | inf | inf | 0.909 | 0.909 | inf | inf | -274.2 | -274.2 | -350.1 | -350.1 |
| load (1 min) | 8.4 | 11.4 | 14.9 | 14.5 | 14.2 | 17.3 | 18.2 | 18.0 | 18.5 | 17.8 |

At low load (2-4, `exp1`, the same code without changes 3-5 enabled but with root OBBT off for
m441 / m400): m441 cold 1023 nodes, bound 0.059, incumbent 0.909; m400 cold 31 nodes, incumbent
78680; m20 warm 637 nodes, bound -2181 (i.e. objective <= 2181 for this max problem, best known
391.6), incumbent 274.2; m21 warm 263 nodes, bound -7036, incumbent 350.1.

Every target now has a root bound and searches in the warm runs; m441, m20, m21 have incumbents.
m47 is only half fixed: its kernel now compiles (73 s trace + compile at load 8, so a cold run
still ends without a bound) and runs, but one 16-box call costs 70 s, almost all of it padding:
the root evaluates one box padded to the batch size of 16 (4.4 s per box). The warm run returns
the root bound 1.0 for this max problem (SCIP's dual bound at 60 s: 0.99997, best known 0.9697),
20 s after the limit.

`benchmarks/run.py --jobs 1`, 60 s, fresh cache (cold) then the same cache (warm), base
(`origin/claude/stage2b`) then this branch in one slot. The base arm ran at load 16-19, the new
arm at 7-9; the base warm run of the three stage-2b models was killed by the 6 GB memory watchdog
(m47's traced alphaBB Hessian), so its warm column is the base profile of section 1. Bounds and
values in the problem's own sense (m47, m20, m21 maximize: "dual" is an upper bound):

| instance | base cold | new cold | base warm | new warm | SCIP 60 s |
|---|---|---|---|---|---|
| m47_bcp8 (max, best 0.9697) | 0 nodes, no bound | 0 nodes, no bound | 0 nodes, no bound | 1 node, bound 0.99999 | bound 0.99996 |
| m400 (min, opt 40262) | 1 node, bound -0.002 | 7 nodes, bound -0.002 | 1 node, bound -0.001 | 63 nodes, bound 1840 | optimal 2.8 s |
| m441 (min, opt 0.3289) | killed past the limit, nothing | 255 nodes, 0.909 / -7e-9 | 1 node, 0.909 / -4e-8 | 511 nodes, 0.909 / 0.025 | optimal 3.7 s |
| m20 (max, best 391.6) | 0 nodes, 274.2, no bound | 1 node, 274.2 / 4899 | 1 node, 274.2 / 4899 | 2549 nodes, 274.2 / 4230 | 388.5 / 1309 |
| m21 (max, best 450.5) | 0 nodes, 350.1, no bound | 1 node, 350.1 / 15499 | 1 node, 350.1 / 15499 | 907 nodes, 350.1 / 15452 | 383.0 / 4777 |

## 4. Correctness and A/B

Unit tests (one file per process, `JAXGLOBAL_NO_CACHE=1`): test_domain 55 passed / 2 skipped,
test_blas 5, test_alphabb_budget 18, test_cut_points 4, test_constrained 23, test_time_limit 14,
test_tightening 76, test_integer 27, test_dispatch 4, test_kernels 21, test_quadratic 30,
test_cut_pool 20: all pass. New: `tests/test_blas.py`, `tests/test_cut_points.py`, a batched-caps
test in `tests/test_domain.py`, the skip path in `tests/test_alphabb_budget.py`
(`test_auto_path_memory_guard` documented the old fallback and now checks both settings).

`benchmarks/check_wrong.py --sample 12 --time-limit 20` (2 jobs, 1 thread each): seed 0 4 solved,
8 unsolved, 0 wrong; seed 1 6 solved, 6 unsolved, 0 wrong. (autocorr_bern35-26 returned after
82.7 s with its 20 s limit, without a bound: a kernel still compiling or a call that cannot be
interrupted, see section 5.)

MINLPLib sets, 60 s, `--jobs 1`, fresh cache per arm, both arms back to back in one slot per half
set (`origin/claude/stage2b` vs this branch); solved per `report.classify`; geometric mean of the
times (floored at 1 s) over the instances both solved:

| set | base solved | new solved | wrong | geo mean, both solved | load (1 min) base / new |
|---|---:|---:|---:|---:|---|
| int50 (50) | 44 | 47 | 0 / 0 | 19.0 s -> 15.2 s (44) | 13.0 and 12.0 / 13.0 and 7.8 |
| start50 (48 with rows) | 42 | 42 | 0 / 0 | 10.9 s -> 11.4 s (42) | 5.9 and 7.5 / 5.8 and 8.5 |

int50 gains: hadamard_5 (unsolved -> 31.6 s), smallinvDAXr3b010-011 (0 nodes -> solved 55.5 s),
smallinvDAXr5b150-165 (211 -> 863 nodes, solved 59.0 s); cvxnonsep_nsig30 29.4 -> 17.7 s, m3
33.4 -> 20.3 s, st_miqp1 18.5 -> 10.0 s, with the same node counts. The second int50 half's base
arm ran at a higher load than its new arm (12 vs 8), so part of the time gain is load; the node
counts (same tree, faster calls; more nodes on smallinvDAXr5b150-165) are not. start50: no
instance changed class or moved by more than 1.5x + 2 s; the 5 % geometric-mean difference is
within the noise of this machine. No losses on either set.

## 5. What is left

- m441 / m400 now search, but the root bound of these hull / cone reformulations is ~0 (SCIP closes
  them in 3-4 s, 350-600 nodes). That is bound strength (perspective / cone cuts) and node speed
  (a dense 316 x 316 Cholesky per IPM iteration), not root cost.
- m20 / m21: `bound_domain` still takes 16-25 s, spread over FBBT compiles, `used_variables`,
  feasible-point local solves and the LP bounds; no single item dominates.
- A kernel call still cannot be interrupted, so one call larger than its estimate overruns the
  limit. A root OBBT in chunks of variables (deadline checked between calls) would bound it.
- `JAXGLOBAL_BLAS_THREADS` handles OpenBLAS only (the NumPy / SciPy wheels' BLAS); MKL or Accelerate
  builds are left alone. Other agents' timings on this machine were taken with OpenBLAS threads on.

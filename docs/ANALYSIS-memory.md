# Memory: the OOM kills on m34_convex2 and m410_CLay0203HRS

jaxglobal was killed by the kernel (about 14 GB RSS, recorded as `error`) within 50-70 s on
minlp.org `m34_convex2` (100 vars, 60 binaries, 217 cons) and `m410_CLay0203HRS` (90 vars, 18
binaries, 132 cons). Machine: 4 cores, 15 GB, jaxlib 0.11, shared with other agents (load
averages 4-18 during these runs; given with each measurement).

## Where the memory went

Phase markers (`bnb._mark`) with RSS sampling (m34, before): setup, kernel tracing and
`make_local` stay under 0.3 GB; RSS climbs from 0.7 to 5+ GB inside `root_bound`, with the main
thread in `block_until_ready` of the first **bound_pool** kernel call. Compiling took 1-2.5 GB;
the rest is the executable's working memory. XLA's `memory_analysis()` of the compiled
16-box node kernel says it directly:

| kernel | temp before | temp after |
|---|---:|---:|
| m410 bound_pool (16 boxes) | 13,970 MB | 1,048 MB |
| m34 bound_pool (16 boxes) | 7,336 MB | 838 MB |
| m34 root OBBT (1 box) | 6,443 MB (with the axis-point fix alone) | 445 MB |
| m32_nonconvex2 bound_pool (16 boxes) | 5,594 MB | 1,043 MB |
| m75_General_Model_Case2 bound_pool (16 boxes) | 107,357 MB | 6,331 MB |

The biggest buffers in the optimized HLO (m410, 1 box) were `f64[181, 1838, 184]`, 46 times.
The node bound (`lp.make_constrained_bounder`) linearizes every convex part by `jax.jacfwd`
at the center and the 2n axis points (181 points), all in one `vmap`. The constraint tape
(732 scalar rows) is over `relax_interp.MIN_ROWS`, so its McCormick relaxation runs in the
**relaxation interpreter**, whose `fori_loop` carry is `(S, 4 + 2n)` float64 per point (lo, hi,
cv, cc and the gradients of cv and cc; S = 1838 laid-out slots: 732 rows, the inputs,
constants and the schedule's padding). Per box 181 x 1838 x 184 x 8 B = 490 MB; XLA keeps
two copies; 16 boxes per call: 14 GB. Not the cause: the expression DAG (the tapes are 1-2k
scalar rows, no tree blow-up), Hessians (alphaBB's Taylor interpreter already has a 512 MB
bytes budget, `alphabb.TAYLOR_BYTES`), cut pools (128 rows), kcache.

Two observations made the fix cheap:

* Of the 181 axis points, `_cut_rows` already **dropped every row** at the axis point of a
  variable that no part is nonlinear in (along such a variable a part's cut is the center's
  cut). On m410 only 18 of 90 variables enter nonlinearly, on m34 40 of 100: 144 and 120
  points were linearized only to be thrown away.
* The root OBBT kernel built the LP rows of each of its 2n (vmapped) LPs from the full
  `(points x parts, n)` cut tensor and only then dropped the duplicate rows: 6.4 GB on m34.

## The fix

1. `lp.axis_vars` / `lp.cut_points`: axis points only for the variables some part is
   nonlinear in (`_nonlinear_vars`; all n when unknown). `_cut_rows` follows the new layout.
   The LP has exactly the same rows in the same order (the dropped points contributed none);
   only `tlo`/`thi` (max over all points' objective cuts) can differ by rounding.
2. OBBT drops the duplicate rows before normalizing (row-wise operations: kept rows unchanged).
3. A memory budget (`jaxglobal/memory.py`):
   * `cut_point_bytes`: bytes per cut point and box of the forward-mode cut pass,
     `2 * S * (4 + 2n) * 8` summed over the **interpreted** tapes (S from the interpreter's
     layout). Straight-line tapes have no carry and are not counted: on the int50/start50
     instances with a large interpreted estimate (autocorr_bern20-10, smallinvDAX*) the node
     kernel takes the lifted path or measured <= 451 MB, and chunking does not change their
     compiled temp (9 MB / 451 MB / 4 MB with and without).
   * `point_chunk` / `map_points`: when `boxes * points * per_point > BUDGET`, the points are
     linearized in equal chunks with `lax.map(batch_size=...)` (same values) instead of one
     `vmap`. `boxes` is the call size (16; 1 for the root OBBT).
   * `BUDGET` = `JAXGLOBAL_MEMORY_BUDGET` (bytes, `K`/`M`/`G` suffix) or
     `min(1 GiB, 1/8 of physical RAM)`.
   * `bnb._memory_threads`: the dispatch threads that run a batch's kernel calls concurrently
     are capped so that `threads * temp_bytes <= BUDGET`, `temp_bytes` read from the compiled
     executable's `memory_analysis()` (exact). On m34 four concurrent 0.57-0.84 GB calls had
     taken the probe phase from 2.8 to 5.0 GB. The kernels do not depend on the thread count
     (a first version did and broke `tests/test_dispatch.py`'s reproducibility).
   * Decisions are logged on the `jaxglobal.memory` logger (INFO), and so is alphaBB's
     existing Taylor-interpreter bytes decision (`alphabb.TAYLOR_BYTES` unchanged: it already
     is a bytes budget, `taylor2_bytes = 2 S (2 + 2d + 2d^2) 8 batch`).

Estimate vs measured temp (16 boxes, all points at once): m410 3,055 MB estimated vs 3,078 MB
measured; m34 4,655 vs 2,975 MB. The estimate bounds the interpreted carry, not the whole
kernel: on m75_General_Model_Case2 (n = 148, 273 parts, 253 points) the remaining 6.3 GB per
call is about four copies of the `(points, parts, n + 1)` cut tensor, which chunking does not
touch (see "What is left").

## Results on the targets

| | before | after |
|---|---|---|
| m34, 60 s cold | `error` (OOM, ~14 GB) | load 3.9: time_limit, ub 2,722,191, lb 2,687,022 (gap 1.3%), 45 nodes, peak 5.2 GB (before the dispatch cap) |
| m34, 150 s cold (load 8-12) | | **optimal** 2,687,026.8 (SCIP 2,687,027), 55 nodes, finished at 97 s, peak 4.4 GB |
| m34, same with `MALLOC_ARENA_MAX=2` | | optimal, 55 nodes, peak **2.9 GB** (steady 2.1 GB) |
| m410, 60 s cold | `error` (OOM) | load 3.7: time_limit, no incumbent, lb 0, peak 3.2 GB |
| m410, 150 s cold (load 8) | | time_limit, ub 41,709 (opt 41,573), lb 2,830, 213 nodes, peak 4.1 GB |

What remains of the peak is mostly XLA compilation (1.5-2.5 GB per big kernel, up to four
compiling at once in the compiler pool) and glibc keeping freed memory in per-thread malloc
arenas: with `MALLOC_ARENA_MAX=2` the same m34 solve peaks at 2.9 GB instead of 4.4 GB with
the same nodes and bounds. That run took 139 s vs 97 s at a different load; there is no clean
timing A/B, so it is **not** a default: set `MALLOC_ARENA_MAX=2` in the environment of
memory-tight runs. At 60 s m34 and m410 are now limited by compile time (bound_pool lowering
plus compile 25-60 s depending on load), not by memory.

## Other instances

Compile-only scan of the node kernel (16 boxes, `memory_analysis` temp, base and new trees)
over the 34 minlp.org stage-2 / stage-2b instances whose pre-fix estimate exceeded 1.5 GB: only
four use the reduced-space node LP with interpreted tapes, and all four changed: m34
(7.3 GB -> 0.84), m410 (14.0 -> 1.05), m32_nonconvex2 (5.6 -> 1.04), m75_General_Model_Case2
(107 -> 6.3). The other 28 compile to the lifted quadratic / polynomial LP (`bound`, temp
3-69 MB), which this does not touch. m480_OPF_33_SBB and m47_bcp8 exceed 4 GB **while
compiling** on both trees (a separate, compile-memory problem). The 30 s peak-RSS scan of the
stage-2b list (`run.py --mem-limit-gb 6`, base / new) shows 0.9-2.3 GB on both sides: at 30 s
under load no big kernel had finished compiling, so it only shows that no setup phase blows
up. MINLPLib int50 / start50 peak RSS after: at most 2.75 GB (tspn05) / 2.44 GB (ramsey).

## MINLPLib A/B (int50, start50)

60 s cold, `--jobs 2 --threads-per-job 1`, base and new back to back in one slot per half set.
64 of the 100 instances get fewer cut points (axis points of linear variables dropped); the
chunking and the dispatch cap fire on none of them.

| set | solved base | solved new | geomean time (unsolved = 60 s) base / new |
|---|---:|---:|---|
| int50 | 43 | 41 | 23.7 / 27.1 |
| start50 | 35 | 33 | 24.4 / 26.5 |

All four "losses" (int50 m3, portfol_buyin; start50 ex8_4_5, kall_diffcircles_8) and the
slower instances were in halves where the new arm met a higher load (int50 half 1: 8-10 vs
13-15; start50 half 0: 10.6 vs 13). ex8_4_5's kernel is the same in both trees (every
variable is nonlinear). Reruns of those 9, alternating base / new twice at `--jobs 1` (load
5-9): all solved in all 4 runs with **identical node counts and bounds** (m3 43 nodes,
kall_diffcircles_8 23,473, ex8_4_5 5,521, kall_diffcircles_5b 6,701, st_e16 23, the others 1),
times within noise (m3 23.7 / 21.5 s base vs 24.1 s new; kall_diffcircles_8 30.6 / 34.7 vs
30.0 / 33.4 s). Files: `benchmarks/results/memory_ab_{int50,start50}_{base,new}.csv`.

check_wrong `--sample 12 --time-limit 20`, seeds 0 and 1: no wrong answers (5 solved, 7
unsolved each).

## Benchmarks record OOM kills

`benchmarks/run.py`: a solver process that dies of a signal the runner did not send is
recorded as `oom` (SIGKILL, what the kernel's OOM killer sends) or `killed` (any other
signal), with the signal and the peak RSS in the message, instead of a bare `error`. Every
row gets `peak_rss_mb` (the larger of `wait4`'s `ru_maxrss` and the polled RSS of the process
and its children). `--mem-limit-gb G` kills a solver above G GB itself (status `oom`). Old
CSVs keep their columns when appended to; `report.py`, `check_wrong.py` and
`minlporg/report.py` treat the new statuses like any other non-optimal one.
`tests/test_run_oom.py` covers it.

The old jaxglobal `error` rows of st_ph11 and ex2_1_3 (`results/rerun_heldout_cold.csv`) are
**not** memory kills: their message is a Python `_DeadlockError` on
`scipy.linalg.cython_blas` (a concurrent-import race) after 6 / 4 s. Rerun under the
watchdog: both optimal in 14 / 8 s, peak 838 / 726 MB. (`baseline.csv` has them as
`unsupported`, from before the unbounded-domain work.)

## What is left

* The non-chunkable part of the node kernel: the `(P0 + n_kelley, parts, n)` cut tensor is
  materialized about four times (cuts, while-loop carry, `rows()`'s reshape before `keep`,
  normalized rows). Storing only the kept rows (same rows, same order) would bring m75 from
  6.3 GB per call to roughly the interpreter chunk.
* XLA compile memory (m480_OPF_33_SBB, m47_bcp8 above 4 GB while compiling) and the compiler
  pool's four concurrent compiles.
* `MALLOC_ARENA_MAX=2` (or `mallopt(M_ARENA_MAX)` at import) as a default needs a timing A/B.
* The interpreter's layout padding (S = 1838 slots for 732 rows on m410) doubles its carry.

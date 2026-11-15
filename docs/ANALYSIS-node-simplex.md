# Node LP by a warm-started host dual simplex (Phase 3): results

Plan: `docs/PLAN-node-speed.md` (Phase 3); go / no-go numbers: `docs/ANALYSIS-node-speed.md`.
Machine: 4 cores (1 thread per core), jaxlib 0.11, one heavy process at a time, a fresh
`JAXGLOBAL_CACHE_DIR` per configuration.

**Summary.** `minimize(node_lp="simplex")` works and its bounds stay certified (tests and
`check_wrong` find no wrong answers). But it is **slower than the default on 4 cores**: on the
k-means models it reaches 0.47-0.75x the threaded IPM's node rate. With one dispatch thread it
reaches 1.66x the node rate of a one-thread IPM. The host side is the cause. A node's ~5 small
LPs cost 1.9-2.5 ms of host time per box, and that time does not spread over the dispatch
threads: SoPlex holds the GIL, and HiGHS releases it, but its runs slow each other down and the
Python around them holds the GIL. The threaded IPM already gets ~1.8 ms of wall time per box
from 4 cores. **Recommendation: keep it optional (off by default), do not make it the default.**
The int50 / start50 A/B was not run (agreed with the coordinator, see "Not done").

## Design

- `jaxglobal/simplex.py`, `lp_callback`: a `jax.pure_callback` that takes the place of
  `quadratic._ipm` in each round of `make_quadratic_bounder(node_lp="simplex")`. It uses
  `vmap_method="broadcast_all"`, so there is one host call per round per batch. The jitted,
  vmapped bound kernel, `_Padder`, the threaded dispatch in `bnb.minimize`'s `run` and the
  kernel signatures are unchanged. The LP rows are the ones `_lp` / `_unit` / the tangents /
  `_separate` already build in JAX.
- Solvers, both optional and imported lazily:
  - HiGHS through `highspy`, using the raw `_core._Highs` class (the Python subclass turns on
    callbacks, which makes runs slower). Settings: dual simplex, no presolve, `threads=1`, no
    scaling.
  - SoPlex through `pyscipopt.LP`.

  `node_lp="simplex"` without either package raises an ImportError naming them.
  `JAXGLOBAL_SIMPLEX=highs|soplex|auto` picks one. `auto` tries HiGHS first because it is
  faster under 4 dispatch threads; SoPlex is faster with one thread (below).
- Bounds come from the certificate, never from the simplex objective. The callback returns the
  row duals, and `quadratic._simplex_round` takes `_certificate(y)`: weak duality with the same
  `safe` margins.
  - Infeasible LP: the callback returns a dual ray. The certificate is taken at its multiples
    `RAY_SCALES` (both signs) and the best one is used, so a box is pruned only when a certified
    value reaches the cutoff.
  - Fallback to the IPM: when a solve fails (solver error, iteration limit, a singular basis it
    could not repair, nonfinite data), or when a ray does not reach the cutoff, the IPM runs for
    that box and round. The other boxes get zero IPM iterations.
  - Range reduction (DBBT) uses the reduced costs of the same certificate, as on the IPM path.
- Warm starts within a node:
  - Round 0 builds the LP from the node's nonempty rows. It starts from the basis its parent's
    round 0 ended with; dead or idle rows are forced basic and the count of basic variables is
    repaired (`_repair`).
  - Later rounds add the rows that became nonempty (that round's tangents and pool rows) to the
    same LP object and re-solve from its basis.
  - The LP object lives in a host session. The bounder carries the session's id through its
    `lax.scan` over rounds. The session closes after the last round or when the box goes
    inactive. A box that the call padding duplicates is solved once.
- Basis hand-over between nodes:
  - `bnb.minimize` keeps an object array of bases (`open_bs`) next to `open_lo` / `open_hi`, and
    children get their parent's basis (`evaluate(..., bases=)`).
  - The bases do not travel in the kernel's arguments. `simplex.Store.give` registers them under
    the bytes of each box the kernel is called with. The callback looks them up by those bytes:
    the box before FBBT, which is passed to the bounder as `key`. Each node's end basis is left
    for `Store.take`.
  - Any basis, including a stale one or one from another box, only moves the simplex's starting
    point.
- The default path is untouched. With `node_lp="ipm"` the kernels are the same functions, and
  the option is left out of the kernel-cache fingerprint (`tests/test_dispatch.py` passes
  bit-identically).
- Coverage: only the lifted quadratic / polynomial LP path uses the simplex. The reduced-space
  cut LP (`lp.py`), `lifted=True`, the lifted OBBT and the root OBBT keep the IPM.
- Bookkeeping:
  - `Result.simplex` counts host solves by status, simplex iterations and warm starts.
  - `benchmarks/run.py` stores it in a new `simplex` column.
  - `benchmarks/check_wrong.py --node-lp simplex` runs the check with the option on.
- A kernel with a host callback cannot be exported, so its trace and compile are not cached on
  disk: `kcache` falls back to tracing the kernel in every process.

## Measurements

### Single LPs (m424 node LPs: 90 columns, 199 rows, 858 nonzeros)

| | HiGHS | SoPlex (pyscipopt) |
|---|---|---|
| re-run of a solved LP, nothing changed | 130 us (default options), 55-65 us with `threads=1` | 5 us |
| LP warm-started from a neighbouring box's basis | 300-400 us, 17.6 iterations | ~100 us solve (23 iterations) + ~170 us of Python to build the rows |
| releases the GIL while solving | yes (a spinning Python thread keeps its speed) | no (a spinning Python thread runs at half speed) |
| 4 threads, cold LPs, throughput | 2.8x of one thread | none |

Under 4 threads, each HiGHS run gets slower. In a replay of one node's 5 LPs per box, the time
inside `run()` rises from 1.44 to 2.70 ms per node. The Python part (array building +
`passModel`) rises from 0.48 to 3.6 ms per node, waiting for the GIL.

The go / no-go estimate (91 us for a node's first LP, 158 us per Kelley round) timed only
SoPlex's `solve()` on a persistent LP. Building the LP, reading the duals and bases, and the
Python around those calls cost as much again or more. HiGHS has a large fixed cost per run and
takes ~20-30 us per iteration on these LPs.

Changing rows in place (`changeCoeff` / `chgCoef` per entry) instead of adding rows was slower
than rebuilding: 3.6 / 4.3 ms per node, against 2.1 ms for HiGHS rebuilt each round from the
basis.

### Real bound kernel on recorded boxes (m424, 256 boxes from a 25 s run, batch 16, cutoff 0.3737)

This is the kernel `profile_node.py` replays (`bnb._Kernels(...).bound.jitted`). Values are ms
of wall time per box, best of 2:

| | 1 thread | 4 dispatch threads |
|---|---:|---:|
| IPM (default) | 7.3-8.4 | 1.67-2.00 |
| simplex, HiGHS, LP rebuilt every round from the previous basis | 4.95-5.03 | 3.36 |
| simplex, HiGHS, one LP per node with rows added (committed) | 4.89-5.30 | 3.19 |
| simplex, SoPlex, one LP per node with rows added | 4.10-4.42 | 5.22-6.06 |

Ranges are separate runs.

- Host callback alone, one thread, 5 rounds of one batch: SoPlex 1.85 ms per box, HiGHS 2.50 ms
  per box (17 simplex iterations per LP, ~4.6 LPs per node).
- The JAX part of the kernel (FBBT, LP rows, certificates) is ~1.7 ms per box.
- cpu/wall with 4 threads: HiGHS ~2.0, SoPlex ~1.35. SoPlex is slower with 4 threads than with
  one (GIL convoy).

Bound quality: the first LP of a node is the same LP, and the bounds agree to 1e-6. Over the
rounds the simplex bounds are lower, because Kelley tangents are taken at the LP point. The
simplex returns a vertex of the optimal face; the IPM returns a point inside that face, and
tangents there cut better.

| | mean bound (min with cutoff) | boxes pruned |
|---|---:|---:|
| IPM | 0.0778 | 19 |
| simplex | 0.0742 (-4.7 %) | 19 (the same boxes) |

Per box, simplex minus IPM ranges from -0.017 to +0.0016.

### minlp.org k-means, 120 s (each configuration warmed by a 30 s run of itself)

4 dispatch threads (the default), HiGHS:

| model | IPM | nodes | nodes/s | simplex | nodes | nodes/s |
|---|---|---:|---:|---|---:|---:|
| m419 | optimal 21.0 s | 5 671 | 270 | optimal 41.2 s | 7 431 | 180 |
| m423 | optimal 67.5 s | 33 767 | 500 | limit, lb 0.10115, ub 0.10764 (opt 0.10116) | 28 391 | 236 |
| m424 | optimal 83.0 s | 31 009 | 373 | optimal 106.0 s | 28 463 | 269 |
| m425 | limit, lb 0.639 (ub 1.113) | 31 479 | 261 | limit, lb 0.294 | 22 205 | 185 |

- m423 with the simplex reached the optimal lower bound but did not find the optimal incumbent
  in time. Its time to solve depends on when the incumbent is found (see the node-speed
  analysis).
- Every node LP of these models goes through the simplex, with no failed solves and warm starts
  on all of them. For example, m424 ran 89 722 optimal and 7 762 infeasible LPs at 11.4 simplex
  iterations per LP.
- Warm-up runs (30 s, including the first compile; the simplex kernel is never disk-cached):
  45-118 nodes/s with the simplex against 67-186 with the IPM.

One dispatch thread (`JAXGLOBAL_DISPATCH_THREADS=1`), m424, 120 s:

| | status | nodes | nodes/s |
|---|---|---:|---:|
| IPM | limit, lb 0.221 | 12 223 | 102 |
| simplex (HiGHS) | limit, lb 0.313 | 20 293 | 169 |

The simplex pays only where each solve has a core to itself, for example `run.py --jobs 4` with
one thread per job. It does not pay with the default threaded dispatch on 4 cores.

## Validation

- `tests/test_node_simplex.py`: 15 tests, run with HiGHS and with SoPlex.
  - On random boxes of a nonconvex QCQP (Kelley rounds, `safe` on and off), no sampled feasible
    point is below the simplex bound. The one-LP bound equals the LP's value (scipy HiGHS) and
    is not below the IPM's certificate.
  - Infeasible boxes of Haverly's pooling problem are pruned by the dual-ray certificate; the
    box that holds the optimum is not.
  - Forced failures and useless rays fall back to the IPM and give exactly its bounds.
  - Bases from other boxes, random bases and malformed ones leave the bound at the LP's value.
  - On a nonconvex mixed-integer QCQP (`safe` on and off), the search with warm-started
    children matches the enumeration of its integers and the IPM search.
  - Option errors.
- Regression files, one per process, `JAXGLOBAL_NO_CACHE=1`, all pass: test_dispatch
  (bit-identical with the IPM default), test_quadratic, test_lifted_poly, test_binary_cuts,
  test_eigen_cuts, test_integer, test_time_limit, test_batching.
- `check_wrong.py --sample 12 --time-limit 20 --node-lp simplex` gave no wrong answers in any
  run:

| run | solved | wrong | instances that took the simplex path |
|---|---:|---:|---|
| seed 0 | 5 | 0 | st_test3, autocorr_bern25-03, st_e09, pointpack10, pooling_adhya1stp, kall_congruentcircles_c52 |
| seed 1 | 7 | 0 | prob09, st_fp7e, ex14_1_5, ex9_1_1, st_glmp_fp1 (st_e21, cvxnonsep_normcon40, ex1223: 0 solves) |
| seed 0, `--safe` | 5 | 0 | as seed 0 |
| seed 1, `--safe` | 4 | 0 | as seed 1 |

  For comparison, the IPM default solves 5 and 8 of the seed 0 and seed 1 samples
  (`docs/ANALYSIS-node-speed.md`). The difference is 1 on seed 1 (st_e21 at 19.9 s, on the
  limit, and not on the simplex path). There is no IPM `--safe` reference run.

## Not done

- The int50 / start50 A/B (60 s, 2 x 2 runs of ~40 min each). Under the default 4 dispatch
  threads the node rate is 25-53 % lower on the models where every node LP goes through the
  simplex, and every non-quadratic model keeps the IPM. The A/B could only show losses or noise,
  so it was skipped by agreement. `check_wrong` (above) shows which sampled instances take the
  simplex path.
- The reduced-space cut LP (`lp.py`, dense rows) has no simplex path.
- Two possible improvements were not tried:
  - taking the Kelley tangents at an interior point of the simplex's optimal face, to recover
    the 4.7 % of mean bound;
  - moving the LP glue out of Python.

## What would make it scale

The host LP must cost well under the IPM's ~1.8 ms of wall time per box, and it must run on
several cores at once. That means calling the solver without Python in between and without the
GIL. Two ways:

- a C/C++ XLA custom call that loops over the batch and keeps one solver per box between rounds;
- SoPlex through a binding that releases the GIL.

At the ~100-150 us per LP that SoPlex's `solve()` itself takes here, a node would cost ~0.6 ms of
CPU. That is about 2-3x the node rate of the threaded IPM on m424, as the go / no-go estimated.
Python glue around HiGHS (GIL-free but ~3x slower per LP, with fixed costs per run and slowdowns
when runs overlap) or around SoPlex (fast, but holds the GIL) does not get there.

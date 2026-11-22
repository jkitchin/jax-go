# HiGHS as the node LP of the lifted quadratic bounder, 2026-11-22

(Updated with the branching follow-up below: `NODE_LP = "auto"`, the default, now uses HiGHS when `highspy` is installed.)

**Question.** The batched interior-point LP costs about 7.2 ms of LP work per node, against about 0.6 ms for a warm-started dual simplex (ANALYSIS-node-speed, Phase 3). Its interior solutions also leave every binary fractional, so branching is uninformed (ANALYSIS-bigm). Does HiGHS's dual simplex, called from the JAX kernel, pay off, and what happens to differentiability?

## Design (`quadratic.NODE_LP = "highs"`, off by default)

**Unchanged.** Everything in the node kernel except the LP solve: box tightening, McCormick/RLT/binary rows, Kelley tangents, the certificate and the reduced-cost tightening stay in JAX.

**The solve.** Each Kelley round's unit-space LP (`min c.u, A u <= b, u in [0,1]^N`) goes through `jax.pure_callback` (`vmap_method="expand_dims"`: one callback per round for the whole batch) to `_highs_solve_host`. It solves each box with HiGHS:
- **Warm start:** from this thread's last optimal basis for the LP's shape.
- **Early stop:** the dual simplex stops at the box's pruning target (`objective_bound`), as the IPM's `target`.
- **Retry:** a failed solve is retried cold, then with HiGHS's IPM.

**The bound** is `_certificate` at the returned duals, as before: a weak-duality certificate whatever HiGHS returns. An infeasible LP returns its dual ray; when the ray's weak-duality value is positive (checked in JAX), the bound is `+inf`.

**Differentiability.** The callback's inputs are `stop_gradient`'ed, and its outputs have no derivative. The bound is `_certificate` at fixed duals, a JAX expression in the LP data. Its derivative is therefore the LP value's derivative by the envelope theorem, `dV = dc.u + y.(db - dA u)`, at a nondegenerate optimum.
- `jax.grad` works on this path. The IPM path does not support reverse mode (its `while_loop` with `custom_vmap` carries); `jacfwd` works on both.
- `_unit`'s zero-row `sqrt` gave NaN under reverse mode; fixed.
- Neither path's derivative of the *node bound* matches finite differences on a test box. The node bound is the best certificate over Kelley rounds, with tangent rows at earlier LP points, so it is not a smooth function of the box. The relaxations themselves, the basis of the design, are unaffected.

**Requirement:** `highspy` (scipy's HiGHS has no warm start).

## Measurements

**16-box bounder calls** (`lpcmp.py`, warm XLA): bounds equal, or tighter where the IPM stops early, e.g. m394 2229–2896 against 1920.

| model | Kelley rounds | IPM | HiGHS |
|---|---|---|---|
| m420 | 2 | 155 ms | 23 ms |
| m394 | 2 | 152 ms | 29 ms |
| m407 | 4 | 395 ms | 87 ms |
| m426 | 4 | 661 ms | 40 ms |

Per LP: m426 35 iterations, 1.3 ms; m407 ~100 iterations, ~3 ms. Over 8–10k solves almost all end optimal or infeasible.

**Certificates.** Certificate against HiGHS's objective over 3,688 m426 LPs: median gap 2e-16, worst 2e-10.

**Full sets, cold, 60 s, same container** (`benchmarks/results/hs_*`, `hs2_*`):

| set | IPM | HiGHS, run 1 | HiGHS, run 2 (final version) |
|---|---|---|---|
| stage 2b | 10 | 12 | 10 |
| int50 | 45 | 45 | – |
| start50 | 39 | 39 | – |
| stage 2c | 3 | 3 | 3 |

- **m420** (k-means, big-M): solved in 54.3 and 57.1 s in two runs; in run 2 the time limit, with incumbent 1.115. It solves once the optimal incumbent is found early enough. The bound reaches 1.08 in every HiGHS run, against 0.71–0.76 with the IPM.
- **m441:** solved at 58.1 s; then 60.7 s, just over the 61 s rule.
- **m400:** bound 3,025 → 26,178 in one run, 3,147 in the other.
- **Throughput:** the k-means models process 40–100% more nodes, but m421 and m422 end with lower bounds (1.28–1.34 against 1.52; 1.87–1.99 against 2.07). Branching sees vertex solutions now, and the rules were tuned on interior points.
- **Layout models are mixed:**
  - m394, m399, m410: same to +10 s.
  - m6 38.6 → 21–28 s.
  - Held-out m397 / m404 43 → 51–59 s, still solved.
  - m407: the optimal incumbent is found, but the bound is 26–27k, against 40,262 with the IPM.
  - Profiles of m407 put the time outside the LP: local solves (16 s), the OA heuristic (9 s), dives (8 s) against 3.8 s of kernel waits. The search spends it differently after an early optimal incumbent.

## Conclusion

**What works.** The node LP is 4–16× faster in isolation and at least as tight, with exact certificates and reverse-mode `grad`.

**What doesn't yet.** The B&B gains are not yet robust: stage 2b 10 → 10–12, the rest neutral, some layout models slower. The node LP is no longer the bottleneck on several models, and branching has not been adapted to vertex solutions. Left off by default.

**Next steps:**
- Branching on vertex solutions: integral binaries at the LP point, pseudocosts on exact LP values.
- Per-node warm starts from the parent's basis: requires carrying basis ids through the search; the thread's last basis is a poor start on the layout models, ~100 iterations.
- Moving the lifted OBBT kernel (`make_quadratic_obbt`) to HiGHS too.


## Follow-up: branching and dives on vertex LP points (now the default)

**Branching screen** on the HiGHS path (`brh_*`; width = the default rule, `hs_*` / `hs2_*`; 60 s cold):
- **pseudocost** is the best rule on the k-means big-M models. Bounds: m420 1.08 (solved, 59 s), m421 1.52, m422 2.12, against 1.08 / 1.28–1.34 / 1.87–1.99 with width. With the IPM it lost those models' bounds (ANALYSIS-bigm).
- **pseudocost** is neutral on the hull layout models.
- **reliability** is far worse everywhere: lost m391, m394, m410, and its bounds are near 0.

**Root dives from a vertex.** The LP point's binaries are at 0/1, and the dives (perturbation 0.1 × width) kept the vertex's assignment. On m397 / m404 they found no incumbent, and the outer-approximation heuristic then ran for 21 s at the root under its no-incumbent rules. That is the HiGHS path's slowdown on these models, measured as the root-OBBT stage at 21.9 s against 0.5 s. With perturbation 0.3: 59 / 52 s → 49 / 43 s.

**Changes:**
- **Default solver:** `quadratic.NODE_LP = "auto"` (HiGHS when `highspy` is installed, else the IPM; `node_lp()`). `highspy` is in the `dev` extra and a new `highs` extra.
- **Branching:** with vertex LP points (`vertex_lp` in `minimize`), every assignment model (`_assignment_bigm` > 0) gets pseudocost, not only those whose binaries switch linear rows.
- **Root dives:** perturbation `_DIVE_NOISE_VERTEX` = 0.3 with vertex LP points.
- **Fix:** `make_quadratic_bounder.lower` returns `+inf` for an infinite certificate (a verified dual ray). In safe mode the rounding margin made it inf − inf = NaN (`test_lifted_poly::test_bound_below_true_min[True-4]`).

**Full sets, cold, 60 s, same container** (IPM `hs_*`, HiGHS with these rules `hs3_*`):

| set | IPM | HiGHS (default now) |
|---|---|---|
| stage 2b | 10 | **12** (m420 55.3 s, m441 45.9 s) |
| stage 2c | 3 | 3 |
| int50 | 45 | 45 (no differences) |
| start50 | 39 | 39 (one swap at the limit, as in every run) |

**Stage 2b details:**
- **Times:** most solved models faster (m6 38.6 → 22.6 s, m399 35.1 → 31.0 s, m403 32.5 → 28.0 s, m410 56.1 → 49.1 s); m394 +5 s, m474 +7 s.
- **Bounds:** m400 3,025 → 26,178, m422 2.07 → 2.18, m472 (max) 810.5 → 735.9; m421 1.52 → 1.48.

**Stage 2c details:**
- **Times:** the held-out layout models at 49.5 / 41.6 / 44.4 s against 46.0 / 42.9 / 42.1 s.
- **m407:** the same bound as the IPM (40,262.33), but a worse incumbent.
- **m129:** a primal and a better bound.
- **m413:** a better bound.

**Tests:**
- **Full non-slow suite** (`-n 3`, cold): 5 failures.
  - Two were the NaN above, fixed.
  - Three (`test_xlogx`, `test_time_limit::test_root_local_budget`, `test_trace_sharing`) pass alone. They failed under memory pressure: an OOM-killed worker, a timeout.
- **Pass:** test_batching, including the multi-device subprocess test; test_highs_lp; test_lifted_poly; test_quadratic.

## Parent-basis warm starts (`HIGHS_PARENT`, off by default)

Each solved box's optimal basis is stored per LP shape and Kelley round, for the last
`HIGHS_PARENT` boxes. A new box's round-k LP starts from the round-k basis of the tightest stored
box that contains it, which is normally its parent and otherwise an ancestor. When no stored box
contains it, the LP starts from the thread's last basis as before.

**Offline check:** children of a box, branched on its most fractional binary. The figure is mean
dual simplex iterations per child LP.

| model | cold | parent's basis | an unrelated box's basis |
|---|---|---|---|
| m407 | 105.5 | 41.8 | 171.2 |
| m426 | 50.0 | 8.8 | 30.2 |
| m394 | 63.8 | 9.8 | 55.1 |

**In the search** (60 s solves): mean iterations per LP went from 46 to 14 (m407), 29 to 7 (m426)
and 31 to 6 (m394).

HiGHS's own time per LP only fell about 1.4–2×. Each LP still refactorizes its basis after
`setBasis`, and Python overhead in the callback (building the LP, waiting on other threads) is
about as large as the solve itself.

Finding the parent:
- Checking every stored box for containment costs about 0.1 ms per lookup at 512 boxes, done once
  per box. The first version took 6–9 s per run.
- 90–100 % of the parents are among the last 512 stored boxes.

**Full sets, cold, 60 s, same container** (off `p0_*`, on `pw_*`):

| set | solved off | solved on | geometric mean time, both solved: off / on |
|---|---|---|---|
| stage 2b | 13 | 13 | 36.5 / 34.6 s |
| stage 2c | 4 | 3 | 39.6 / 40.9 s |
| int50 | 47 | 46 | 13.4 / 14.0 s |
| start50 | 39 | 39 | 10.3 / 9.6 s |

The differences are all on borderline instances:
- m407 (stage 2c): 59.2 s off, the time limit on.
- tln4 (int50): 53 s off, the time limit on.
- m420 (stage 2b): 51.6 s off, 59.9 s on.

Two instances got faster: kall_circles_c7a (46 → 24 s) and ex8_4_5 (29 → 18 s).

The fewer iterations don't turn into node throughput, because most of the per-LP cost is fixed
overhead. The next step is that overhead (update the HiGHS model in place instead of
`passModel`, and keep the factorization by solving a box's rounds without resetting the basis).
After that, this is worth measuring again.

## Where the wall time goes, and the OA heuristic's cap (not adopted)

**Profile.** These are 60 s solves of m426, m407 and m394, sampled every 50 ms. The node search
runs for only about 18–24 s of each:
- **Before the search:** setup, root bound, dives and local solves take 12–20 s.
- **Root OA heuristic:** `MilpBound.propose` runs on the main thread with no node work alongside
  it, for 25 s on m426, 9 s on m407 and 9 s on m394.
- **During the search:** about 70% of node-thread time is in the HiGHS callback. HiGHS's own
  solve is half of that; rebuilding the LP in NumPy is about 30%.

**What OA contributed.**
- m426 had no incumbent at the root, so OA found all of them; its last improvement came at 27.8 s
  of a 36 s call.
- On m407, OA found nothing in 9 s.
- On m394, OA found the optimum after 4.7 s, then ran 4.5 s more.

**Single-run screen** (6 models, `_OA_SECONDS = 3` against 30): faster on every model it solved,
by 1.6–25 s. m400 was solved in 41.6 s against 60.3 s; with OA off entirely, m400 was not solved.

**Full sets, cold, 60 s, same container** (`b0_*` base, `oa3_*` capped). The first comparison
used the `p0_*` runs from before a container restart, about 10–25% faster, and was discarded.

| set | base | cap | geometric mean time, both solved |
|---|---|---|---|
| stage 2b | 12 | 11 (lost m441: 48.9 s → time limit) | 36.9 / 37.9 s |
| stage 2c | 3 | 3 | 44.3 / 50.6 s |
| int50 | 45 | 46 (gained smallinvDAXr5b150-165) | 15.6 / 15.7 s |
| start50 | 39 | 39 | 11.2 / 11.5 s |

The screen's gains did not hold up on the full sets, and the cap is not adopted. Single 60 s
runs vary by ±5–10 s on these models (m391: 46.3 s and 60.3 s, m420: 57.2 s and 41.7 s), so a
6-model single-run screen cannot detect effects of this size. A screen needs repeated runs, or
must measure something less noisy than wall time.

# HiGHS as the node LP of the lifted quadratic bounder, 2026-11-22

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

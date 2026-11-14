# Root bounds: jaxglobal against SCIP, and an integer-aware root bound (analysis, 2026-11-14)

Question: on the models where jaxglobal's bounds are weak (issue #87), how far is its root bound
from SCIP's, and can it get a better one?

## 1. Root gaps

`benchmarks/root_gap.py` (one solve per process, one at a time, 4-core VM): jaxglobal's bound
after root processing (`minimize(..., max_iter=0)`: root bound, local solves, dives, root OBBT),
and SCIP 10.0's root dual bound (`getDualboundRoot()`, 60 s solve). "best" is the better of
SCIP's final primal value and the instance's reference value; gaps are relative to it. Raw rows:
`benchmarks/results/root_gap.jsonl`.

| instance | sense | best | SCIP root | gap | jaxglobal root | gap |
|---|---|---:|---:|---:|---:|---:|
| m7_Model_P1 | min | 2260.3 | 1142.9 | 49% | 814.64 | 64% |
| m419_kClus_3_10_5_BM_1 | min | 1.8475 | 0 | 100% | 0 | 100% |
| m423_kClus_5_10_2_BM_1 | min | 0.10113 | 0 | 100% | 0 | 100% |
| m424_kClus_5_10_3_BM_1 | min | 0.37365 | 0 | 100% | 0 | 100% |
| m425_kClus_5_10_5_BM_1 | min | 0.99127 | 0 | 100% | 0 | 100% |
| m470_Adhya1_2Player_PriceTaker_FixedCharge1 | max | 545 | 1156 | 112% | 1241.3 | 128% |
| m19_MTG_EX1 | max | 352.45 | 1294.3 | 267% | 1429.9 | 306% |
| m20_MTG_EX1a | max | 391.61 | 1787.9 | 357% | 2257 | 476% |
| m21_MTG_EX1b | max | 450.55 | 5364.6 | 1091% | 7038 | 1462% |
| m31_nonconvex1 | min | 285,506 | 234,800 | 18% | 209,890 | 26% |
| m32_nonconvex2 | min | 2,687,027 | 2,686,800 | 0% | 2,423,600 | 10% |
| tln5 | min | 10.3 | (solved in presolve) | | 9.7 | 6% |
| kall_circlespolygons_c1p12 | min | 0.3396 | 0 | 100% | 0 | 100% |
| wastewater15m1 | min | 2446.4 | 975.48 | 60% | 975.39 | 60% |
| nvs17 | min | -1100.4 | -1105.1 | 0% | -9470.8 | 761% |

jaxglobal's root bound is never better than SCIP's. It is equal where both are trivial (the
k-means models m419-m425 and kall_circlespolygons: 0) or equally weak (wastewater15m1), and
worse everywhere else, most on nvs17 and m32, where SCIP's root is (nearly) the optimum. SCIP's
roots are also weak on most of these models, and it closes them by branching quickly (m7: root
49% gap, solved in 0.6 s with 231 nodes; m423: root 0, solved in 3 s with 2,524 nodes).

## 2. An integer-aware root bound: MILP over the relaxation's linear cuts

jaxglobal's node bound is an LP over linear cuts of the McCormick/alphaBB relaxations
(`lp.py`); integrality enters only through branching. Every such cut is valid on the box for
every point, integer or not, so a MILP over the same cuts with the integers required integral
also bounds the problem from below (its proven dual bound, not its solution). Outer
approximation adds cuts at each MILP solution and solves again; for convex models it
converges to the optimum.

`benchmarks/root_oa_prototype.py` builds the cuts from `jaxglobal.relax` (tangents of cv / cc at
the box center, 2n axis points, 2n random points, then at each round's LP and MILP solutions)
on the box from FBBT (and `bound_domain` when variables are free; no OBBT, no cutoff), and
solves them with HiGHS (scipy `linprog`/`milp`).

| instance | optimum | SCIP root | jaxglobal root | prototype LP | prototype MILP (rounds, time) |
|---|---:|---:|---:|---:|---:|
| m423 (k-means, convex MIQCP, big-M) | 0.10113 | 0 | 0 | 0 | **0.0418 -> 0.0974** (7 rounds, 152 s): 96% of the gap closed |
| m7 | 2260.3 | 1142.9 | 814.6 | 582 | 637 -> 663 (6, 3 s) |
| m32 | 2,687,027 | 2,686,800 | 2,423,600 | 2,420,600 | 2,564,266 (4, 12 s) |
| m470 (max) | 545 | 1156 | 1241 | 1342.6 | 1292.4 (4, 17 s) |
| m19 (max) | 352.45 | 1294 | 1430 | 8907 | 8491 (4, 17 s) |
| nvs17 | -1100.4 | -1105 | -9471 | -380,531 | -379,893 (4, 5 s) |

Findings:

* **Where the relaxation is good and only integrality is missing, the MILP bound is far better
  than SCIP's root.** On the k-means model m423 (convex, binaries in big-M constraints) the LP
  bound, SCIP's root and jaxglobal's root are all 0; the MILP over the same cuts reaches
  0.0974 of the optimum 0.1011 in 7 rounds. The other k-means instances (m419, m424, m425) have
  the same structure and the same zero roots.
* **Where the relaxation itself is weak, integrality adds little.** m7 (concave
  `sqrt(sum L y sigma^2)` risk pooling, `x * (c1 n + c3 d / n)` inventory terms): +13%. m470
  (bilinear pooling with fixed charges) and m19 (scheduling) gain nothing beyond the LP.
* **The prototype's box matters.** Without jaxglobal's OBBT and objective-cutoff tightening its
  LP is weaker than jaxglobal's real root on m7, m470, m19 and nvs17 (on nvs17 and m19 by
  orders of magnitude). A real implementation would run on jaxglobal's root box and cuts.
* **SCIP's strong roots on nvs17 and m32** come from something jaxglobal's relaxation does not
  have (nvs17: general integers in nonconvex quadratics; m32: a nonconvex MINLP whose root SCIP
  closes to 0.01%). Not investigated here.

## 3. What this suggests

1. An integer-aware root bound: a MILP (HiGHS) over the root's linear cuts, with a few outer
   approximation rounds, on jaxglobal's own root box. It pays off on models whose relaxation is
   tight except for integrality: convex MINLPs and big-M / on-off structures (here the k-means
   family). Costs and open points:
   * time: the MILPs took 15-20 s per round on m423 (22,000 rows from 2n + 1 + 2n points);
     cuts only at points that matter (as `lp.py` does per nonlinear variable) would cut rows
     by an order of magnitude;
   * validity: a MILP dual bound is not certified like `lp.py`'s weak-duality bound; it needs a
     safety margin, or a certified LP bound at the MILP's final node set (or `safe=True` only);
   * when to use it: at the root, and on models with integers in the relaxation's cuts; on a
     convex model whose MILP bound meets the incumbent the solve ends at the root.
2. For the nonconvex models (m7, m19-m21, m470, nvs17, m31, m32) the gap to SCIP is in the
   relaxation, not in integrality: structure-specific relaxations (concave functions of binary
   sums, perspective terms `x * phi(n, d)`, products of general integers) and search speed.

Not done: why SCIP's roots on nvs17 and m32 are tight.

## 4. The MILP root bound inside jaxglobal (`milp_root`)

`jaxglobal/milpbound.py`, `minimize(milp_root=True)`: cuts at the center, the axis points of
the nonlinear variables only (`lp._nonlinear_vars`; 10 of 70 on m423), random points in those
coordinates, the incumbent and the relaxation points, then at each MILP solution and its
midpoint with the previous one; run after root OBBT and probing on the hull of the open root
boxes. On m423 the first MILP has 3,280 rows (the prototype: 22,000) and the bound reaches 0.089
of 0.1011 in 10 rounds, 36 s.

Full solves, 180 s limit, rel_tol 1e-4, one at a time (4-core VM):

| instance | optimum | off: lb / ub / nodes / time | on: lb / ub / nodes / time |
|---|---:|---|---|
| m423 | 0.10113 | 0.10114 / 0.10115 / 5,459 / 56 s | same / same / 5,459 / 92 s |
| m419 | 1.8475 | 1.8474 / 1.8475 / 6,255 / 58 s | same / same / 6,255 / 96 s |
| m424 | 0.37365 | 0.316 / 0.3737 / 15,547 / limit | 0.250 / 0.3737 / 10,939 / limit |
| m425 | 0.99127 | 0.401 / 1.0033 / 9,197 / limit | 0.342 / 1.1134 / 7,671 / limit |

The search is identical (same node counts) and only loses the MILP's time. A root bound below
the incumbent fathoms nothing; the leaves still have to reach the optimum by themselves, and
children inherit the root's bound only as a floor that the search passes anyway. So a root MILP
bound pays only when it closes the gap (a convex model whose outer approximation converges
within the budget), which on these four it does not. The option stays, off by default.

What would make integrality count in the search: the same MILP at nodes (the root's cuts stay
valid on every child box, so a node MILP needs only the child's bounds; but at 0.2-3 s per MILP
against thousands of nodes it needs to be selective, e.g. when a node's LP point is integral
infeasible in many binaries), or cuts that carry integrality into the LP (big-M perspective /
disjunctive cuts), which SCIP's 2,524-node, 3 s solve of m423 suggests matter more than the
root bound.

## 5. What makes SCIP fast on the k-means models (ablation)

SCIP 10.0, 60 s, one component off at a time (`benchmarks` scratch script, settings as in
section 1). Nodes (time) to optimality; m425 never solves in 60 s, so its dual bound at 60 s:

| setting | m423 | m419 | m424 | m425 bound |
|---|---:|---:|---:|---:|
| default | 2,524 (2.8 s) | 6,478 (8.6 s) | 15,781 (15 s) | 0.503 |
| all separation off (`setSeparating(OFF)`) | 14,531 | 27,649 | 112,407 (52 s) | 0.005 |
| presolve off | 2,577 | 5,743 | 15,466 | 0.418 |
| propagation off | 4,422 | 6,280 | 16,024 | 0.458 |
| symmetry off | 2,524 | 6,478 | 15,781 | 0.519 |
| heuristics off | 3,510 | 9,231 | 20,062 | 0.680 |
| pscost / mostinf branching | 4,182 / 4,905 | 6,410 / 7,198 | 12,402 / 25,353 | 0.460 / 0.228 |
| separation, presolve, symmetry off | 10,146 | 26,505 | limit, 0.237 | 0 |

Which separation, on m424 (nodes; default 15,781):

| off | nodes |
|---|---:|
| every separator plugin (constraint handlers' separation kept) | 15,279 |
| `constraints/nonlinear` separation (gradient cuts of the convex constraints at the node LP point) | **80,143** |
| implied bounds | 30,801 |
| gomory, aggregation (cmir), flowcover | 11,032, 17,469, 14,230 |
| rlt, minor, zerohalf, knapsack, clique, mcf, eccuts, gauge, disjunctive, linear, varbound | 15,781 (no change) |

So SCIP's speed on these models is (1) outer-approximation cuts of the convex quadratic big-M
constraints at every node's LP solution, kept in the global cut pool (they are valid
everywhere), and (2) implied-bound cuts; no MIP cutting plane family matters. jaxglobal's node
LP already has the first kind (Kelley rounds at the node's LP point, `n_kelley`; its McCormick
relaxation of these sums of squares is exact), and its node counts are of the same order as
SCIP's (m423: 5,459 vs 2,524; m424 at 180 s: 15,547 nodes vs 15,781 to optimality): the gap
is nodes per second (jaxglobal ~90/s, SCIP ~1,000/s on m424).

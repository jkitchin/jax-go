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

Not done: the MILP bound inside jaxglobal (on its own root box and cuts); m419, m424, m425 with
the prototype; why SCIP's roots on nvs17 and m32 are tight.

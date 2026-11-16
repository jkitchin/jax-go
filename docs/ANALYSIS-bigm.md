# Big-M models: weak lower bounds on CLay BM and k-means BM (analysis, 2026-11-16)

Question: why do jaxglobal's lower bounds stay weak on the minlp.org big-M models with many binaries (constrained layout `m391_CLay0205BM`, `m394_CLay0305BM`; k-means `m420`-`m426`), where SCIP closes CLay in under 5 s (`benchmarks/results/minlporg_stage2b_cold.csv`)?

Short answer: **branching**. On these models the node relaxation is already the exact continuous relaxation, the convex constraints are cut at the node LP point, and tightening the big-M coefficients per node adds nothing measurable. The tree was large because integers were branched most-fractional-first at an interior-point LP point, where every free binary is fractional. Pseudocost branching with a fractionality weight solves m391 and m394. It loses on some int50 and k-means instances, so it is an option (`branch_var="pseudocost"`), not the default.

## 1. Diagnosis

### 1.1 The node bound is the exact continuous relaxation

`m391` (80 variables, 50 binaries) is a convex MIQCP:

- a linear objective (weighted distances `delta_ij >= |x_i - x_j|`);
- big-M non-overlap rows `x_i - x_j + 51 b <= r`;
- 4-way disjunctions `sum b = 1`;
- disk constraints `(x - a)^2 + (y - b)^2 + M b_c <= R`, with `M ~ 8000`.

jaxglobal bounds it with the lifted quadratic LP: one column per square, McCormick and tangent rows, and 4 Kelley rounds at the node's LP point.

On boxes with 5–40 binaries fixed at their optimal values, the node bound matches SCIP's continuous relaxation to 4–6 digits:

| binaries fixed | jaxglobal node bound | SCIP continuous relaxation |
|---:|---:|---:|
| 10 | 3777.5 | 3777.5 |
| 20 | 5954.97 | 5955 |
| 30 | 8092.48 | 8092.5 |

16 Kelley rounds or 200 IPM iterations change nothing. The root bound is 0 for jaxglobal, for SCIP's continuous relaxation, and for the hull reformulations m403 and m406. Any relaxation-based B&B starts from that bound.

### 1.2 Where SCIP's speed comes from (SCIP 10, m391)

| setting | nodes | time |
|---|---:|---:|
| default | 2,652 | 4.5 s |
| separation off | 8,833 | 3.7 s |
| separation, presolve, symmetry off ("bare") | 6,006 | 2.9 s |
| bare, propagation off too | 11,612 | 2.8 s |
| bare + most-infeasible branching | 15,750 | 5.5 s |
| bare, no propagation, most-infeasible / pscost | 23,772 / 15,117 | |
| default with most-infeasible / pscost | 5,756 / 4,288 | |
| default with one separator off (gomory, zerohalf, clique, aggregation, flowcover, impliedbounds, nonlinear) | 2,383–4,361 | |

On m394, default SCIP needs 2,802 nodes. Bare SCIP without propagation needs 13,986 with relpscost and 22,682 with most-infeasible branching.

No single separator dominates. Together, the MIP cuts give about 3x, propagation about 2x, and the branching rule about 1.5–2x.

Adding this structure's implied-bound and clique-aggregated cuts (`delta >= s (b_L + b_R)`) raises SCIP's root bound from 0 to 5,575. Even so, bare SCIP still needs 11.6–12.5k nodes (15.7k without the cuts), and default SCIP is unchanged (2,626).

In the stage2b run, jaxglobal processed 16,299 nodes in 60 s and was still 13% short.

### 1.3 Branching at an interior-point LP point

The branching point is the IPM's last iterate. In a 60 s m391 run, every free binary was fractional at every branching decision: 65,669 of 65,669, and 70,480 of 70,491 under pseudocost. All 3,541 branchings were on binaries. With every binary fractional, `width` (most fractional first) picks by IPM noise.

### 1.4 Per-node big-M coefficient tightening: no gain

`minimize(bigm=True)` (`quadratic.with_bigm`, `_bigm`) works on each inequality row that contains a binary `y`. Let `U` be the box maximum of the rest of the row (squares ranged exactly, plus a rounding margin). The coefficient `m` of `y` and the right-hand side `r` are then tightened to what the box allows:

- if `m > 0`: `m' = m - d`, `r' = r - d`, with `d = min(r - U, m)`;
- if `m < 0`: `m' = m + d`, with `d = min(r - m - U, -m)`.

This is valid for integral `y`, and the rows stay in the LP and in its certificate.

On m391 it tightens 4 of 80 rows once `x0 <= 30`. Around the optimum the node bounds do not change, and the nodes to solve barely move: 9,391 → 9,623 on m391, 12,215 → 11,177 on m394. The weakness of the big-M LP is that the disjuncts mix fractionally, not the size of M on the box. It is off by default.

## 2. Change: fractionality-weighted pseudocost for integers (an option)

With integer variables, an integer's pseudocost product score is multiplied by `min(f, 1 - f)` at the relaxation point. Without this weight, plain pseudocost lost badly on the 5-cluster k-means models. `branch_var="auto"` keeps `width` for integer problems (section 3).

Screening at a fixed budget of 60 iterations (about 13.8k nodes, independent of load). Each cell is the lower bound reached, or the nodes needed to solve:

| model (opt) | width (default) | plain pseudocost | weighted pseudocost |
|---|---|---|---|
| m391 (8092.5) | 6,780 | solved, 9,391 nodes | solved, 11,867 nodes |
| m394 (8092.5) | 7,557 | solved, 12,215 nodes | 7,915 |
| m420 (1.0836) | 0.728 | 0.704 | 0.849 |
| m423 (0.10113) | solved, 6,019 nodes | 0.0862 | 0.10113 (10,421 nodes) |
| m424 (0.3737) | 0.235 | 0.246 | 0.313 |

## 3. A/B: width vs weighted pseudocost as the integer default

60 s, `--jobs 1`, cold caches, both arms back to back in one slot. Load was 6–16: about 15 for the base arm of the first group, about 8 for the candidate.

| model (opt) | width: status, nodes, bound | weighted pseudocost |
|---|---|---|
| m391 (8092.5) | limit, 4,011, 5,035 | limit, 9,523, **7,695** |
| m394 (8092.5) | limit, 1,707, 1,980 | limit, 7,143, **6,940** |
| m420 (1.0836) | limit, 8,447, 0.641 | limit, 13,311, **0.849** |
| m421 (2.602) | limit, 9,215, **1.525** | limit, 11,007, 1.387 |
| m422 (4.734) | limit, 5,631, **2.106** | limit, 6,655, 1.568 |
| m423 (0.10113) | **solved 48.7 s**, 6,039 | limit, 11,189, 0.101137 |
| m424 (0.3737) | limit, 9,151, 0.180 | limit, 9,109, **0.222** |
| m425 (0.9913) | limit, 5,367, 0.252 | limit, 8,185, **0.390** |
| m426 (0.3299) | limit, 255, 0 | limit, 255, 0 |
| m389, m390, m392, m393, m417–m419 | solved | solved, similar nodes |

On a quieter machine in earlier runs, plain pseudocost solved m391 in 43–49 s and m394 in 52–64 s.

**int50: 47 solved in both arms.**

- Candidate gains: tln5 and smallinvDAXr4b150-165 are newly solved, and tln4 drops from 80,913 to 11,457 nodes.
- Candidate losses:
  - hadamard_5: base solved it in 60.4 s. Rerun alone, both arms time out, and the base bound is better (16 vs 10 on this max problem).
  - smallinvDAXr5b150-165: rerun, both solve, in 863 (base) vs 1,501 nodes.
  - ex1252a: 8,437 vs 11,043–18,723 nodes.
  - nous1: 2,585 vs 4,623 nodes.
- These are real losses, so `width` stays the integer default.

**start50 is unaffected by construction.** It has no integers, so `auto` was and stays pseudocost. The weight is `np.where(imask, ..., 1.0)`, and `bigm` is off and needs integers anyway.

check_wrong (`--sample 12 --time-limit 20`, seeds 0 and 1), with the defaults and with `branch_var="pseudocost", bigm=True`: no wrong answers.

## 4. What is left

- No rule yet picks pseudocost for big-M models only. Two ideas:
  - use the rows that `with_bigm` marks, when they cover many of the integers;
  - cheaper reliability branching. SCIP's relpscost needs 1.5–2x fewer nodes than its pscost here, but jaxglobal's `"reliability"` was worse in these runs.
- The node rate is about 150–270 nodes/s, against SCIP's 600–1,000.
- SCIP's 4–5x node advantage comes from MIP cuts working together (Gomory, zero-half, clique). Gomory cuts need a basic LP solution. The interior-point node LP does not give one; the simplex node LP on `claude/node-simplex` would.
- m426 is limited by node speed and by its incumbent: 3.23, against SCIP's 0.339. Under `width`, m420's incumbent stayed at 1.87, against the optimum 1.0836.

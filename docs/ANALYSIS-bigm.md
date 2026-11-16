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

# Part 2: automatic pseudocost, probe-seeded pseudocosts, fewer Kelley rounds (2026-11-16)

## Cause

- `auto` used `width` (most fractional at the interior-point LP point) on every integer problem, but the big-M layout models (CLay) need pseudocost.
- Pseudocosts started empty, although root probing already bounds every binary at 0 and at 1.
- On the k-means models, all 4 Kelley rounds run at almost every node, because the models have 60–100 convex quadratic big-M distance rows.
- Pre-search work takes 21–29 s of a 60 s run on m391 (domain bounding 4–8 s, root dives 7 s, outer-approximation heuristic 12–14 s), so the search gets about half the time.

## Fix (`jaxglobal/bnb.py`)

- **`_assignment_bigm`** detects disjunctive assignment models from the lifted quadratic form. A model qualifies when all integers are binary, there are at least 20 of them, and at least 90% of them sit both in an assignment row (`sum b = 1` or `<= 1`) and in a linking row. The function returns the share of binaries that switch a *linear* row. The form is computed once and cached (`_quadratic_form`).
- **`n_kelley=None`** (the new default) resolves to 2 rounds on these models (`_ASSIGN_KELLEY`) and to 4 otherwise. With 2 rounds a node costs about half, and the bound per box is about 7% weaker.
- **`branch_var="auto"`** picks pseudocost when at least half the binaries switch linear rows (`_ASSIGN_LINEAR`):

  | models | linear share |
  |---|---|
  | CLay big-M (m391, m394) | 0.73–0.80 |
  | hull models | 1.0 |
  | k-means big-M | ~0 |

  The k-means big-M models therefore keep `width`: pseudocost won on m420 but lost on m421 and m422.
- **`seed_pseudocosts`** (`_PC_PROBE`): root probing seeds the pseudocosts with each binary's two sides, recorded as a split at the root's LP value. Binaries where neither side raised the bound are left unobserved; seeding them with zero dropped m420 from 1.01 to 0.62.
- **`_PC_FRAC_POW`** exposes the exponent of the fractionality weight. It stays at 1.

## Measurements

**Fixed 60-iteration screens** (node counts, independent of load):

| model | `width` | pseudocost | pseudocost, probe-seeded |
|---|---|---|---|
| m391 | not solved (bound 7695) | 10.3–12.5k nodes | 5.9–7.2k nodes |
| m394 | not solved (bound 7595) | 9.9k nodes | 8.3k nodes |

**2 vs 4 Kelley rounds, `width`, 60 s, load about 14.** The bounds after 60 s, with 2 rounds processing 1.8x the nodes:

| model | 4 rounds | 2 rounds |
|---|---|---|
| m420 | 0.586 | 0.652 |
| m421 | 1.380 | 1.500 |
| m422 | 1.836 | 2.023 |

**A/B against `stage2b-merge`, 60 s, `--jobs 1`.** Both arms back to back with a shared warmed compile cache; load 7–9.6. The CSVs are in `benchmarks/results/bigm2_fin{A,B}_{base,cand}.csv`.

| model (opt) | base: status, dual, nodes | new: status, dual, nodes |
|---|---|---|
| m391 (8092.5) | limit, 7137, 7,355 | **optimal 49.0 s**, 7,061 |
| m394 (8092.5) | limit, 7060, 6,567 | **optimal 52.0 s**, 6,379 |
| m420 (1.0836) | limit, 0.652, 8,437 | limit, 0.723, 14,335 |
| m421 (2.602) | limit, 1.475, 7,679 | limit, 1.601, 14,335 |
| m422 (4.734) | limit, 2.012, 4,607 | limit, 2.206, 8,959 |
| m426 (0.330) | limit, 0.0014, 2,047 | limit, 0.0152, 4,095 |
| m399 | optimal 51.4 s, 467 | optimal 47.7 s, 465 |
| m406 | optimal 42.8 s, 465 | optimal 39.4 s, 457 |

**Other sets.** On the 7 int50 instances with at least 20 binaries, results are identical in both arms. The rest of int50, and all of start50, cannot reach the new rules. check_wrong (seeds 0 and 1) found no wrong answers.

## What is left

- m420 needs about 20–30k nodes; SCIP needs 16k, and we reach 14k in 60 s.
- The pre-search overhead on these models.
- m391 and m394 finish with a thin margin under load (49–52 s at load 8–10; m394 did not finish at load 13).
- The k-means branching rule is not understood: most-fractional beats pseudocost for d ≥ 3, but not for d = 2.

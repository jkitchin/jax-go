# The feasibility-tolerance tail, and a branching screen (minlp.org stage 2b), 2026-11-19

**Question.** SCIP solves four stage-2b models that jaxglobal does not: m400, m420, m441 and m472. The incumbents of m400, m420 and m441 are now optimal or close (ANALYSIS-noinc), so what holds their bounds?

## Branching screen

Existing rules (`JAXGLOBAL_OPTS='{"branch_var": ...}'`), cold, 60 s, one run each. The `width` row is the stage-2b run from ANALYSIS-noinc (`s2c_form2_cold.csv`). Each cell is incumbent / bound.

| model (optimum) | width (default) | pseudocost | reliability | strong |
|---|---|---|---|---|
| m400 (40262.39) | 40262.39 / 3257 | 41886 / 3257 | 41886 / 3025 | 40262.39 / **26682** |
| m420 (1.0836) | 1.2564 / 0.784 | 1.2564 / **0.960** | **1.0837** / 0.737 | **1.0837** / 0.192 |
| m441 (0.32892) | 0.32892 / **0.32709** | 0.32892 / 0.32709 | 0.32892 / 0.0 | 0.32892 / 0.0 |
| m472 (max 361.67) | 139.0 / **798** | 152.4 / 829 | 139.0 / 889 | **303.2** / 1016 |

No rule wins everywhere:
- `strong` raises m400's bound eightfold but loses m441's.
- `pseudocost` is best on m420's bound (consistent with ANALYSIS-bigm).

The default is unchanged. Raw rows: `benchmarks/results/s2c_br_*.csv`.

## m441: the bound stalls just below the incumbent

m441 (k-means, hull reformulation with rotated cones) gets close, but stalls:
- **Warm run.** It reaches a bound of 0.328886 at iteration 75 (1441 nodes). The incumbent is 0.3289229, so the gap is 1.1e-4 against `rel_tol` = 1e-4. It then splits that last box for another 3,250 nodes and returns `optimal` at 87–95 s.
- **Bound.** It ends at 0.3288870, and only because a point violating a row by 1e-6 turned up by chance (0.3288894, then repaired).

**Cause.**
- The search bounds the problem relaxed by `feas_tol` = 1e-6, whose optimum is about 0.328883.
- The local solves return strictly feasible points (pounce runs with `bound_relax_factor` 0), optimum 0.3289229.
- On this ill-conditioned model the two optima differ by 1.2e-4 relative, more than `rel_tol`, so no bound can close the gap to the strictly feasible incumbent.
- SCIP's own value is 0.3289199 (violation 5.5e-7), in between.

**Profile.** In the warm run's search, 54.7 s of the main thread is spent in bound-kernel calls (py-spy). The tail is node count, not overhead.

**Change** (`bnb._POLISH_RELAX` = 0.45).
- `polish`, which runs once for each new incumbent, re-solves the incumbent's continuous variables with Ipopt's `bound_relax_factor` set to 0.45 × `feas_tol`.
- pounce relaxes the variable bounds too, and clipping the point back into its box adds up to as much again to a row's violation, hence 0.45 rather than 0.9. With 0.9, the screen measured 1.8e-6 and rejected the point.
- The point passes `screen` (violation ≤ `feas_tol`), as any candidate must.
- `repair_incumbent` still reports a strictly feasible point. On m441 it moves 0.3289032 → 0.3289229, within `_REPAIR_SLACK`.

**m441, warm:** 4,693 nodes and 87–95 s → 1,329 nodes and 61.8 s.

**Follow-up: the relaxed solve runs ahead of `polish`'s time budget.** It sits ahead of the polish rounds, once per new incumbent (one local solve). In the first version it was inside the budgeted rounds, and on m441 cold the root's polish had spent its budget before reaching it, so the incumbent stayed strictly feasible and the bound stalled 1.15e-4 short. Now m441 solves cold: 57.6 s and 59.3 s standalone, 60.1 s in the stage-2b run.

## Checks

All cold, 60 s. "first" is the relaxed solve inside the polish budget (commit d613a4a); "final" is the solve run ahead of the budget (this commit).

| set | before | first | final | notes |
|---|---|---|---|---|
| stage 2b (`--jobs 1`) | 10 | 10 | **11** | m441 solved (60.1 s; the rule allows 61 s); no solved model slower than before by more than 1.6 s |
| int50 (`--jobs 2`, 2 threads each) | 47 | 48 | 47 | first: smallinvDAXr4b150-165 solved (55.3 s), not repeated; no wrong answers in either |
| start50 (`--jobs 2`, 2 threads each) | 43 | 42 | 41 | the losses solve in the base arm at 58.4 s (ex6_2_6) and 59.0 s (kall_circles_c7a) |

**Rerun alone** (`--jobs 1`), base against final:
- kall_circles_c7a: 36.4 s against 35.9 s.
- ex6_2_6: 50.1 s against 50.1 s.
- Both reach the same primal and dual values in both arms.

The start50 losses are models at the time limit under two concurrent jobs, not the change.

**Raw rows:**
- `s2c_relax_cold.csv`, `s2c_relax2_cold.csv`
- `s2c_int50_relax_cold.csv`, `s2c_int50_relax2_cold.csv`
- `s2c_start50_base_cold.csv`, `s2c_start50_relax_cold.csv`, `s2c_start50_relax2_cold.csv`
- `s2c_kc7a_*.csv`, `s2c_chk_*.csv`

The stage-2b and int50 "before" arms are the earlier runs `s2c_form2_cold.csv` and `s2c_int50_form2_cold.csv`. A container restart interrupted the first int50 run after 36 instances; `run.py` resumed it.

## What is left

- **m441's margin.** It solves at 57–60 s cold. The root takes about 30 s even warm; the outer-approximation heuristic at its end spends about 11 s proposing assignments that cannot beat the optimal incumbent the dives found. Capping its rounds at 3 did not help in one run.
- **m400.** `strong` branching's bound gain (3.3k → 26.7k in 60 s) suggests strong branching restricted to binaries, or to hull models. It needs an int50 A/B; `strong` lost m441's bound entirely.
- **m420.** Pseudocost's bound is the best (0.96 against 1.084), with reliability's incumbent.

## Tried after the stage-2b 11/19 run (not kept)

1. **Stopping the outer-approximation heuristic after non-improving proposals** (m441's margin).
   - On m441 the root call improves the incumbent 0.504 → 0.392. Then its MILPs grow by about 1,130 cut rows per round, and each takes 1.6–2.5 s.
   - Stopping after 3 non-improving proposals: 59.0 s and 58.8 s cold, against 57.6 s and 59.3 s, no change.
   - After 2: m441 times out with 2,873 nodes against about 1,330. The different incumbents change the search's path, and the tree grows.
2. **`strong` branching for m400.** It solves m400 in 95 s cold, against 139 s warm for `width` (5,018 against 14,619 nodes).
   - Not a default: on int50 it solved 41 against `width`'s 43 (ROADMAP, `branch_*` results).
   - Not a model-class rule either: in the screen it left m441's bound at 0, and m400 and m441 are both hull models.
3. **m472 (pooling, 2-player Nash–Cournot).** The heuristic's completions all come back 220–500 (minimization form, against an incumbent of 0 and an optimum of −361.7).
   - Adding 4 or 8 random continuous starts per assignment changes none of them: the proposed assignments themselves are poor.
   - Its MILP relaxation of the bilinear pooling terms does not steer toward profitable assignments.

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

**m441, warm:** 4,693 nodes and 87–95 s → 1,329 nodes and 61.8 s. Cold it still times out: its root takes about 30 s even warm.

## Checks

All cold, 60 s.

| set | before | after | notes |
|---|---|---|---|
| stage 2b (`--jobs 1`) | 10 | 10 | m394 49.6 → 43.9 s, m410 47.5 → 43.9 s; no solved model more than 1.9 s slower |
| int50 (`--jobs 2`, 2 threads each) | 47 | **48** | smallinvDAXr4b150-165 solved (55.3 s); no wrong answers |
| start50 (`--jobs 2`, 2 threads each) | 43 | 42 | kall_circles_c7a: base 59.0 s, new time limit. Rerun alone: 35.3 s base, 35.4 s new, the same nodes. Load at the margin |

- **Raw rows:** `s2c_relax_cold.csv`, `s2c_int50_relax_cold.csv`, `s2c_start50_{base,relax}_cold.csv`, `s2c_kc7a_*.csv`.
- **Arms:** the stage-2b and int50 "before" arms are the earlier runs `s2c_form2_cold.csv` and `s2c_int50_form2_cold.csv`; the start50 arms were run back to back.
- **Restart:** a container restart interrupted the int50 run after 36 instances; `run.py` resumed it.

## What is left

- **m441, cold.** The root takes about 30 s even warm (make_local 2.9 s, root local solves 3.6 s, root dives 5.4 s, root OBBT 4.2 s, the heuristic 2.8 s). The search then needs about 30 s.
- **m400.** `strong` branching's bound gain (3.3k → 26.7k in 60 s) suggests strong branching restricted to binaries, or to hull models. It needs an int50 A/B; `strong` lost m441's bound entirely.
- **m420.** Pseudocost's bound is the best (0.96 against 1.084), with reliability's incumbent.

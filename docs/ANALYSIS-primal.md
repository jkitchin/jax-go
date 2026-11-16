# Primal side on minlp.org: why the incumbents were poor, and an outer-approximation heuristic

Branch `claude/stage2b-primal`. Instances from `benchmarks/results/minlporg_stage2b_cold.csv`
(60 s, jaxglobal vs SCIP) whose bound was good but whose incumbent was not.

## 1. Diagnosis

How incumbents are found (`jaxglobal/bnb.py`): the bound kernel's candidate points are screened
(`screen`: box, integrality, `feas_tol` on every constraint, in float64); `update_incumbent` runs
local NLP solves (pounce) from relaxation points; with integers each start is solved as the NLP
relaxation, rounded at five thresholds (`ROUNDINGS`) and fix-and-solved, and a failed nearest
rounding is repaired against integer-only linear rows; propagation dives (`dive`) fix integers one
at a time under FBBT; up to 32 assignments are enumerated at the root; a new incumbent is
polished (`polish`: one-flip and swap neighbours, continuous part re-solved). Every point passes
`screen` before it becomes the incumbent.

SCIP's optimal solution read into the jaxglobal model, its integers fixed, local solve:

| instance | SCIP optimum passes `screen` | fixed-integer NLP from SCIP's x (default pounce) | other starts | cause |
|---|---|---|---|---|
| m399 CLay0303HRcone (convex) | yes (viol 9e-7) | 26669.11, feasible | zeros: 26669.11 | the optimal assignment is never tried |
| m420 kClus_3_20_2_BM (convex) | yes | 1.08365 | box mid, zeros: same | never tried |
| m441 kClus_3_10_2_HRc (convex) | yes | 0.32892 | same | never tried; and the run is stuck in its root stages (see 3) |
| m474 Bental5 (nonconvex, max) | yes | **6268.73 at violation 3.5e-4: rejected** | zeros: infeasible | pounce stops imprecise at max_iter 200; `mu_init=1e-6` or `mu_strategy=adaptive` (max_iter 1000) give 6271.6146, feasible |
| m472 Adhya1 (nonconvex, max) | yes | 360.71 (optimum 361.67) | zeros: -400 (another basin) | basin and precision; the warm options give 361.667 from the optimum's x |

So the feasibility tolerance does not reject good points. On the convex k-means / CLay models
the existing heuristics (roundings of relaxation points, dives, one-flip neighbours) never reach
the right binary assignment: the relaxations are weak in the binaries (big-M; the k-means LP bound
is 0), so their roundings carry little information. On the equilibrium models the fixed-integer
NLPs are hard for the default pounce settings, and the right assignment + basin is not found.

## 2. What changed

1. **Outer-approximation primal heuristic** (`MilpBound.propose` in `jaxglobal/milpbound.py`,
   `oa_primal` / `oa_start` in `minimize`). A MILP (HiGHS) over the relaxation's linear cuts on
   the root box (the cuts of `milp_root`, here only at the box center and the relaxation points),
   plus a cutoff row `t <= incumbent - tol` and a no-good cut per assignment already tried. Each
   MILP solution's integers are fixed and the continuous part local-solved (from the MILP point
   and the relaxation points; a failed solve again with the adaptive barrier); the local solutions
   are screened like every other candidate. Cuts at the MILP point and the local solutions are
   added and the MILP solved again: on a convex model this is outer approximation, which converges
   to the optimal assignment. It stops on a repeated assignment, an infeasible MILP, or its round
   / time budget. Its MILPs are never used as bounds (the cutoff and no-good rows make them no
   relaxation), and only screened points come out.
   * only problems whose integers are all binary (`oa_binary`): there the no-goods make every
     MILP propose a new assignment. On general integers it cost smallinvDAXr4b/r5b150-165 their
     solves in a first int50 A/B (tracing their 123k-equation cut program).
   * budget: at most `_OA_SHARE` = 15 % of the time limit in all, 30 s per call, each MILP half a
     call's budget; HiGHS stops at a 1 % MILP gap (`PROPOSE_GAP`).
   * when: once after the root stages if the gap is open, then every 10 % of the time limit
     (`_OA_EVERY`), resumed with its cuts and no-goods plus cuts at the best children's points.
   * compile: the cut program (`jacfwd` of the relaxations; 18k-91k jaxpr equations, 6-10 s of
     XLA compile on these models) compiles in a background thread started after the root stages
     (started with the root bound it slowed the dives / OBBT / probing under load); the root call
     waits up to half its budget, later calls run once it is ready. Points are evaluated
     `CHUNK` = 8 at a time so the program compiles once (`cut_values`). A cut program above
     `MAX_EQNS` = 100k equations is not compiled (smallinvDAXr3b010-011: 123k equations, 52 s
     and 4.2 GB to compile).
   * `MilpBound.bound` (`milp_root`) shares the cut building (`_initial`) and `cut_values`.
     HiGHS reports a "solve error" on some small MILPs whose objective variable has a finite lower
     bound; `propose` retries those without it.
2. **Warm re-solve of a new incumbent** (`polish`, `_WARM`): with integers, continuous variables
   and constraints, after the neighbour rounds and only with what is left of the polish budget,
   the incumbent's continuous part is solved again from itself with `mu_init=1e-6` and with
   `mu_strategy=adaptive` (max_iter 1000).
3. `_TRACE` hook (diagnostics, off by default): the incumbent value at each stage of `minimize`.

Gating: every new path needs integer variables, so continuous problems (start50) are untouched:
`oa_binary` is False without integers, and the warm re-solve tests `imask.any()`. No start50 A/B
was run for that reason.

## 3. Measurements

60 s, per instance base then new back to back in one benchmark slot (`--jobs 1`); base =
`origin/claude/stage2b` + the trace hook. Incumbents in the model's sense (m472/m474 maximize).
Load averages 3-14 on 4 cores (other agents), noted per run.

Final candidate (round 4):

| instance | optimum | base incumbent | new incumbent | base / new nodes | load base / new |
|---|---:|---:|---:|---|---|
| m399 CLay0303HRcone | 26669.11 | 28351.42 | **26669.11** (OA, 57 s; root took 40 s) | 129 / 13 | 3.2 / 11.5 |
| m420 kClus_3_20_2_BM | 1.08364 | 1.81442 | **1.08365** (OA, 26 s) | 8959 / 8181 | 13.6 / 14.3 |
| m421 kClus_3_20_3_BM | 2.6021 | 2.7944 | 2.6712 (root dives) | 10751 / 8703 | 10.5 / 9.4 |
| m422 kClus_3_20_5_BM | 4.734 | 4.94761 | 4.94761 | 8703 / 9471 | 7.0 / 7.7 |
| m474 Bental5 (max) | 6271.61 | 6146.61 | 6146.61 | 35 / 7 | 8.8 / 7.2 |
| m472 Adhya1 (max) | 361.67 | 176.89 | 176.89 | 511 / 127 | 6.0 / 6.8 |

Earlier rounds (same heuristic, compile in the background from the root bound, load 7-19):
m399 28351 -> 26669.11 twice (36 s), m420 1.65/1.81 -> 1.08365 / 1.256, m421 2.914 -> 2.855;
m422, m472, m474 unchanged every time. On m399 the optimum is then found but the bound is the
remaining gap (base's root bound at 60 s ranged from 1755 to 26669 across runs with the load).

Not changed: m474 and m472. The MILPs over McCormick cuts of these bilinear equilibrium models
are weak (MILP value about -9600 against the optimum 6271 on m474) and slow (m472: 19 s for the
first MILP), and their proposals land in poor basins. m426 (kClus_5_20_2, 100 binaries) and
m441 (kClus_3_10_2_HRc) did not get past their root stages in either arm: under load the root
incumbent's local solves and the probing took 30 s to several minutes (base m441 overran the 60 s
limit by minutes inside `probe`), so no search heuristic ran. That is a time-limit problem of the
root stages, present in the base.

MINLPLib int50 (60 s, `--jobs 1`, both arms per chunk of 10 in one slot, load 2.9-9.0, cold
kernel caches in both arms): base 48 solved, new 48 solved, no losses; geometric mean time over
the 48 solved by both 13.60 s (base) vs 14.11 s (new). (The candidate before the binary-only gate
lost smallinvDAXr4b150-165 and smallinvDAXr5b150-165 and won hadamard_5; 47 vs 46 solved.)

check_wrong (`--sample 12 --time-limit 20 --jobs 1`, final code): seed 0 5 solved / 7 unsolved,
seed 1 8 solved / 4 unsolved, no wrong answers, exit 0 (also no wrong answers on the earlier
candidate).

Unit tests: `tests/test_milpbound.py` (3 new: the optimal assignment found with no assignment
proposed twice, the cutoff respected, same answer with the heuristic off) and
`tests/test_incumbent.py` (the polish budget test now turns the heuristic off and allows the
polish budget's elapsed-time share) pass; `test_integer`, `test_nonfinite_incumbent`,
`test_time_limit` pass.

## 4. Risks and what is left

* The heuristic's time (up to 15 % of the limit) and its compile thread cost nodes on problems it
  does not help (m474: 35 -> 7 nodes; int50 geometric mean +4 %).
* The size gate is a jaxpr-equation count, not a memory estimate (compile peaks measured: 0.4-0.7
  GB at 18k-91k equations, 4.2 GB at 123k).
* Nonconvex models: the next steps would be the same MILP on a promising node's box (tighter
  McCormick cuts) or local branching around the incumbent, and robust fixed-integer NLP solves
  (the warm options from more starts).
* The root stages of m426 / m441 overrun the time limit under load; worth their own look.

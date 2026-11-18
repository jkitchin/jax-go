# No incumbent on m400 and m47 (minlp.org stage 2b), 2026-11-18

**Question.** Issue #87's next step 2: in 60 s cold, `main` reports no solution on m400_CLay0304HRcone or m47_bcp8. SCIP solves m400 in 2.8 s and finds no solution on m47 in 60 s. Why do we find none, and what fixes it?

## m400_CLay0304HRcone (MIQCP, 212 variables, 36 binaries)

The model is constrained layout in the hull (HR) reformulation, with rotated cones `2·W·T ≥ U_c² + UU_c²`.
- `W` assigns four rectangles to three circular areas.
- `Z` picks a non-overlap disjunct per pair.

**Diagnosis.**
1. **Completions work.** With SCIP's assignment fixed, a local solve reaches 40262.3875 from every start (0.2–2.7 s). Fixing a `W = 0` cone, which leaves a single feasible point, does not trouble pounce.
2. **The root heuristics find nothing.**
   - The root bound is −0.002: the HR relaxation lets every rectangle sit in the same place.
   - The relaxation point is useless for rounding and dives; their completions fail.
3. **The outer-approximation heuristic (`MilpBound.propose`) does reach the optimum, after about ten infeasible proposals.**
   - Offline, from the FBBT root box, the first feasible proposal is the 10th–11th: about 15 s at 1–2 s per round, almost all in the HiGHS MILP.
   - The early proposals put all four rectangles in one circle. With few tangent cuts each circle is a loose polygon. SCIP confirms every early proposal infeasible with its integers fixed.
   - More initial cut points (`few=False`) or ESH-style cuts at the boundary between the proposal and the relaxation point did not reduce the number of proposals.
4. **In the solve the heuristic never gets that far.**
   - The cut program (54k jaxpr equations, no cut pool on this quadratic model) is built only after the root stages. It is ready at 37–46 s cold.
   - Then each call is capped at 8 rounds and all calls together at `_OA_SHARE` × 60 s = 9 s.
   - Warm, it runs from 24 s and finds the optimum at 32 s.

**Change.** When there is no incumbent (the search cannot prune on the objective anyway):
- The heuristic gets `_OA_SHARE_NOINC` = 0.5 of the time limit instead of 0.15.
- Each call gets `_OA_ROUNDS_NOINC` = 64 MILP rounds instead of 8.
- `_OA_NOINC_START`: if the root dives leave no incumbent, the cut program is built right then, beside root OBBT and probing, instead of after the root stages.

A model with an incumbent after the root dives runs exactly as before.

Tried and not kept:
- `_OA_EARLY` (trace at the root bound) with or without the 35k-equation compile limit: proposals from 31–40 s, but no feasible one in the 9 s budget.
- The share alone: the program was ready too late (37.6 s or 45.8 s cold).

**Effect (m400, cold, 60 s).**

| | first proposal | incumbent |
|---|---|---|
| `main` | 36.7 s | 40964.4 at 44.4 s in one run; none in three others |
| new | 33.4 s | 40262.3875 (optimal) at 46.5 s |

The run.py row is `time_limit`, primal 40262.39, dual 3025. Before, the primal was `nan` with a dual of −0.002. The bound still needs about 26 s of search after the incumbent (warm: optimal at 58.5 s), so m400 is not solved in 60 s.

## m47_bcp8 (MINLP, 168 variables, 9 binaries)

This is the Belgian chocolate problem: Routh–Hurwitz stability of a degree-8 controller, with rational Routh-table equalities and polynomial constraints of degree up to 6.
- **Bound side.** Its first bound call takes about 67 s (ANALYSIS-compile: dense LPs of about 51k rows per box), so no root bound or root heuristic runs within 60 s.
- **Primal side.** 16 random-start local solves, relaxed and then with the binaries rounded and fixed, are all infeasible, with maximum violations of 10¹–10⁶.
- SCIP also finds no solution in 60 s (dual 0.99996; minlp.org's best known value is 0.9697).

The primal problem is hard in itself, not a gap in our heuristics: it needs either a much cheaper bound kernel (the issue's step 6, row-level pruning) or a problem-specific construction. Not changed here.

## Checks

All cold, 60 s. Raw rows: `benchmarks/results/s2c_noinc_cold.csv` and `s2c_int50_{base,noinc}_cold.csv`.

- **Stage 2b** (`--jobs 1`): 10 → 10 solved, the same models. m400 gains the optimal incumbent; m441's incumbent improves 0.3921 → 0.3289 (optimum 0.3289). No solved model is slower by more than 0.8 s. Most are 2–5 s faster, which is run-to-run variation: they have an incumbent after their root dives and do not take the new path.
- **int50** (`--jobs 2 --threads-per-job 2`, base and new arms back to back): 47 → 47 solved, no wrong answers. Only tln4 differs by more than 10 s (58.6 → 48.3 s).
- **start50:** continuous models only; the heuristic requires binaries, so nothing changes.
- **Tests:** test_milpbound, test_integer, test_compile_gate, test_incumbent, test_bigm and test_cut_pool pass.

## What is left

- **m400's bound.** After the incumbent, the search needs about 26 s. It is a hull model whose root relaxation is 0 (as for every CLay model here), so branching and node throughput decide.
- **Fewer infeasible proposals.** Each early proposal costs a 1–2 s MILP for a no-good cut on one full assignment. Two options:
  - Detect infeasibility on the `W` part alone: a local solve with `Z` relaxed, then a no-good cut on `W` only. This is a heuristic cut, allowed because `propose` never reports a bound.
  - A cheaper OA program without a cut pool (issue step 5), which would also start the heuristic earlier.

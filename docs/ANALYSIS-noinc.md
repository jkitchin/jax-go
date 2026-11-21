# No incumbent on m400 and m47 (minlp.org stage 2b), 2026-11-18

Two commits: three no-incumbent rules for the heuristic, then cuts from the quadratic form with any MILP solution accepted while there is no incumbent. The first commit's m400 result was a lucky run; the follow-up makes it reliable.

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

**Effect (m400, cold, 60 s), first commit.**

| | first proposal | incumbent |
|---|---|---|
| `main` | 36.7 s | 40964.4 at 44.4 s in one run; none in three others |
| these three rules | 33.4 s | 40262.3875 (optimal) at 46.5 s in one run |

That run was lucky. In six later cold runs (three with each cut source of the follow-up below, before its MILP-gap change) the heuristic found a solution once: 40929.6 at 57.6 s. Proposals took 1.5–3 s each in the solve, and 10–12 infeasible ones came first. The follow-up makes this reliable.

## Follow-up: cuts from the quadratic form, and any MILP solution while there is no incumbent

1. **`milpbound.form_tangents` (`bnb._OA_FORM`).**
   - A quadratic problem without a cut pool (m400, the k-means and 2-player models) takes the heuristic's cuts from its lifted quadratic form, `[f, g, h] = E (x, w) + c`. Each product `w = x_a x_b` is replaced by the McCormick plane active at the point, or by the tangent or secant of a square, chosen by the sign of its coefficient.
   - These are valid convex minorants of the rows, like `relax`'s, computed in numpy: build 0.1 s, against a 54k-equation trace and 10–20 s of XLA compile on m400.
   - Because nothing is compiled, the heuristic runs right after the root dives when they find no incumbent, before root OBBT and probing.
   - Plain linearizations, which are not valid for nonconvex rows, were tried first. They found a first feasible point sooner, but then cut off the optimum (the MILP became infeasible at 58014.8).
   - On a degree-3 polynomial form it returns None and the generic program is used.
2. **`milpbound.PROPOSE_GAP_NOINC` = 1.0.** While the cutoff is +inf, HiGHS stops at its first MILP solution; after that, the 1% gap as before.
   - Offline from m400's root box, the first feasible proposal comes at 4.3–4.6 s (0.2–0.5 s per MILP), against 15–22 s with the 1% gap.
   - The optimum comes at 6.5 s if the gap stays at 1.0, and at about 12.5 s with the switch back to 1%.

**m400, cold, three runs:** a first incumbent (41886.4) at 25 s each time, and the optimum (40262.39) in all three by the time limit; final duals 3257–3884. Still `time_limit`: the bound is now what is missing.

## m47_bcp8 (MINLP, 168 variables, 9 binaries)

This is the Belgian chocolate problem: Routh–Hurwitz stability of a degree-8 controller, with rational Routh-table equalities and polynomial constraints of degree up to 6.
- **Bound side.** Its first bound call takes about 67 s (ANALYSIS-compile: dense LPs of about 51k rows per box), so no root bound or root heuristic runs within 60 s.
- **Primal side.** 16 random-start local solves, relaxed and then with the binaries rounded and fixed, are all infeasible, with maximum violations of 10¹–10⁶.
- SCIP also finds no solution in 60 s (dual 0.99996; minlp.org's best known value is 0.9697).

The primal problem is hard in itself, not a gap in our heuristics: it needs either a much cheaper bound kernel (the issue's step 6, row-level pruning) or a problem-specific construction. Not changed here.

## Checks

All cold, 60 s. Raw rows: `benchmarks/results/s2c_noinc_cold.csv`, `s2c_form_cold.csv` (form cuts, before the gap change), `s2c_form2_cold.csv` (everything), `s2c_int50_{base,noinc,form2}_cold.csv`.

**Stage 2b** (`--jobs 1`), incumbent of the unsolved models:

| model (optimum) | `main` | three rules | everything |
|---|---|---|---|
| m400 (40262.39) | none | 40262.39 (lucky run) | 40262.39 |
| m420 (1.0836) | 1.6405 | 1.6405 | 1.2564 |
| m426 (0.3299) | 0.5144 | 0.5144 | 0.4181 |
| m441 (0.3289) | 0.3921 | 0.3289 | 0.3289 |
| m472, max (361.67) | 140.0 | 138.6 | 139.0 (252.4 in the run before the gap change) |
| m421, m422, m473 | unchanged | unchanged | unchanged |

- The same 10 models are solved in every arm.
- With everything, m394 is 0.6 s slower than `main` (49.6 against 49.0 s); the rest are 2–11 s faster: m399 43.0 → 32.0 s, m474 43.2 → 36.2 s, m406 36.2 → 31.6 s.
- The quadratic models without a cut pool no longer trace and compile the heuristic's 30–90k-equation program, so the gains on m399 and m474 are compile CPU saved. The others are partly run-to-run variation.

**int50** (`--jobs 2 --threads-per-job 2`, the `main`-equivalent arm and the new one run back to back):
- Three rules: 47 → 47 solved, no wrong answers; only tln4 differs by more than 10 s (58.6 → 48.3 s).
- Everything: 47 → 47, no wrong answers, no model more than 10 s apart.

**start50:** continuous models only; the heuristic requires binaries, so nothing changes.

**Tests:**
- test_milpbound has a new test: form-tangent cuts lie below their rows on random boxes, and `propose` runs with them.
- test_milpbound, test_integer, test_compile_gate, test_incumbent, test_bigm, test_cut_pool and test_quadratic pass.

## What is left

- **The bound.** m400, m420 and m441 now have their optimal or near-optimal incumbents; the dual is what keeps them open.
  - m400 needs about 26 s of search after the incumbent (warm). It is a hull model whose root relaxation is 0, as for every CLay model here.
  - On m441, node bounds with 12–24 binaries fixed are close to SCIP's root bounds on the same boxes; with 6 fixed, about half of SCIP's (0.062 against 0.135).
- **Symmetry: not a lever here.** The k-means models already order their centroids (`c(k-1,'1') ≤ c(k,'1')`), so their formulations have no relabelling symmetry left.
- **m472.** The incumbent swings between 139 and 252 from run to run; optimum 361.67.
- **Not useful:** no-goods over only the binaries of the rows a failed completion violates. With the binaries fixed, those rows hold only continuous variables, so the set is empty.

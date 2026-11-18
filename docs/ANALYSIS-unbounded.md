# Unbounded variables: m6_del_asl_hvar and m75_General_Model_Case2

jaxglobal refused two stage-2b minlp.org instances as `unsupported` ("no finite bounds found for
variables [...]"). `domain.bound_domain` has to give branch-and-bound a finite box, and it may only
remove points that are infeasible or worse than a known feasible point. The two cases are different.
m6 hit a gap in the interval arithmetic and a disjunction that FBBT cannot see. Both are fixed, and the
fix also makes m4_del_asl_hfix (stage 2) supported. m75 has no valid finite box: its infimum is not
attained.

## m6_del_asl_hvar (NLP, 151 variables): fixed

The model (Baumrucker & Biegler) is an MPCC solved as a penalty NLP. It splits each state into a
positive and a negative part, `x(i,j) = yp(i,j) - yn(i,j)` and `x0(i) = yp0(i) - yn0(i)`, sets
`yyp(i) = sum_j yp(i,j) + yp0(i)` (and `yyn` the same way), and penalizes the complementarity in the
objective:

    phi = (x_end - 5/3)^2 + sum h(i) x(i,j)^2 + 1000 sum_i (uun(i) yyn(i) + uup(i) yyp(i))

with `uup = 1 - u`, `uun = 1 + u` and `u in [-1, 1]`. The 60 variables yp, yn, yp0, yn0, yyp and yyn
(indices 20-39 and 81-120) are declared `>= 0`, and nothing else bounds them: only their differences
are fixed by the states, which are bounded. The cutoff from the feasible point (1.53642) is
`uun yyn + uup yyp <= 1.5e-3` per stage, and that does bound them. FBBT could not use it, for two
reasons.

1. **`0 * inf` in the interval product** (`relax.mc_mul`). For `uup * yyp` with `uup in [0, 2]` and
   `yyp in [0, inf]`, one endpoint product is `0 * inf = NaN`. The min and max of the four products then
   came out NaN, and FBBT reads NaN as no information, so the term was `[-inf, inf]` instead of
   `[0, inf]`. A cutoff on a sum of such terms then reached none of them: each term is bounded by the
   cutoff minus the *lower* bounds of the others, and those were `-inf`. `fbbt._imul` already used the
   convention `0 * inf = 0`, but `mc_mul` did not, and the forward sweep (and the tape interpreter) use
   `mc_mul`. `mc_mul` now maps a NaN endpoint product to 0 when no operand bound is NaN. On finite boxes
   nothing changes. This alone bounds 42 of the 60 variables of m6 (in the stages where the LP bounds
   fix the sign of u). It also bounds all 40 free variables of **m4_del_asl_hfix**, the same model with
   h fixed, which was also `unsupported`.
2. **A disjunction: FBBT on split halves** (`domain.split_bounds`, new). Where u can lie anywhere in
   `[-1, 1]`, `uup` or `uun` may be 0, so on the whole box the cutoff bounds neither `yyp` nor `yyn`.
   But `uup + uun = 2`:
   - On `uun >= 1`, the cutoff bounds `yyn`, hence `yn`, and `yp = x + yn` is bounded because `x` is.
   - On `uun <= 1`, `uup >= 1` bounds `yyp`, and then `yn = yp - x` is bounded.

   For each variable that is still free after the existing steps, `split_bounds` does the following:
   - It takes each bounded variable `c` that enters nonlinearly a row that also uses a free variable.
   - It runs FBBT, with the cutoff, on each of `c`'s two halves (for an integer, `[lo, m]` and
     `[m + 1, hi]`).
   - It replaces each infinite end of the box by the hull of the two results.

   Every feasible point with `f <= cutoff` lies in one of the halves and survives FBBT there, so the
   hull keeps it. An empty half contributes nothing. The step runs after the LP bounds and alternates
   with them (`_SPLIT_ROUNDS = 3`, `_SPLIT_SECONDS = 20`). It runs only while variables are still free,
   so it cannot fire on a model that the old code already bounded.

With both changes, `bound_domain` bounds all 151 variables of m6 in 6 s, and the formerly free
variables end up in `[0, 3.52]`. Nothing is assumed: like the earlier steps, both changes keep every
point of the box that is feasible (to `feas_tol`) and no worse than the feasible point found.

**SCIP's lower values are tolerance artifacts.** SCIP reports 1.5358754 on m6 (minlp.org: 1.5364) and
1.4735037 on m4 (minlp.org: 1.4738). Both SCIP points violate the declared bounds by 1e-8 to 7e-8,
within SCIP's 1e-6 feasibility tolerance, for example `uun = -7e-8`. They pair those values with very
large `yyn` (the variables the model leaves unbounded), so a bound violation of 1e-8 gains 5e-4 in the
penalty term. Projected onto the declared bounds, SCIP's points evaluate to **1.5364152** (m6) and
**1.4738038** (m4), which equal jaxglobal's incumbents 1.5364152 and 1.4737778 up to the equality
tolerance. jaxglobal enforces the variable bounds exactly. Its m4 result (optimal 1.4737778, bound
1.4737508) is therefore not contradicted by SCIP's 1.4735, and check_wrong-style classification against
SCIP would wrongly flag it.

## m75_General_Model_Case2 (MINLP, 148 variables): no finite box exists

The 16 free variables (89-104) are the log-mean temperature differences of the 10 exchangers, 5
coolers and 1 heater of the Yee & Grossmann superstructure (`lmtd`, `lmtdcu` and `lmtdhu`, declared
`>= 0`). They appear only in these equations:

    lmtd = (dt1 - dt2) / log(dt1 / (dt2 + 1e-6))          (lmeq, lmcueq, lmhueq)
    area = q (1/h1 + 1/h2) / (lmtd + 1e-2)                 (aeq, acueq, ahueq)
    cost = ... + 1200 (area + 1e-2)^0.6 + ...

The model's authors added the `+ 1e-6` to avoid 0/0 at `dt1 = dt2`. It puts a pole at
`dt1 = dt2 + 1e-6`, where the numerator is `1e-6` and the logarithm is 0. As `dt1` approaches
`dt2 + 1e-6` from above, `lmtd` goes to `+inf`, the area goes to 0 and the cost falls. The approach
temperatures `dt`, `dtcu` and `dthu` appear elsewhere only in `dt >= 10` and in upper-bound
inequalities (`dt <= th - tc + gamma (1 - z)`). They can therefore be moved onto the pole without
changing any other variable.

Every point with an active exchanger therefore has a feasible neighbour with a larger `lmtd` and a
strictly lower cost. The infimum is approached as `lmtd -> inf` and never attained, so any finite upper
bound on `lmtd` cuts off better feasible points. No sound box exists, and `unbounded_domain` is the
correct answer. No interval rule, cutoff or LP can change that, because the set `cost <= C` is
unbounded in `lmtd` for every `C` above the infimum.

Measured with the scratch script `pole75.py` (float64, the model as read from the `.nl`; the move is
`dt1 = dt2 + 1e-6 + eta`):

| start point | cost | cost after the move (eta = 1e-9 / 1e-11) | max violation |
|---|---|---|---|
| jaxglobal's feasible point from `bound_domain` | 784503.65 | 737547.90 / 736601.54 | 9.1e-13 |
| SCIP's incumbent (60 s) | 685832.27 | 652602.36 / 651691.42 | 1.8e-12 |

The cost keeps falling as eta goes to 0. The first row, starting from jaxglobal's own local solution,
already beats minlp.org's best known value of 739019.02.

On the sense question: the model minimizes `cost`, and the `.nl` reads as minimization (sense +1).
SCIP's 686019.59 includes slack in its objective epigraph variable: the same point evaluates to
685832.27 in jaxglobal's reader and is feasible to 1.8e-12. It is below the best known value because
minlp.org's value is a local SBB solution, not because of a sense or conversion error. SCIP's incumbent
does not use the pole: its `lmtd` values are 16-213, which are physical. SCIP's dual bound (583567) is
consistent with the infimum.

What m75 is missing is a bound that the model does not state. The physical log-mean lies between the
two approach temperatures, `lmtd <= max(dt1, dt2)`. Adding that bound, or removing the 1e-6 (for
example with the Chen approximation that the model's own report section uses), is a modelling change.
It is not something jaxglobal may derive, so it was not added. **m76_General_Model_Case1** (stage 2, 12
free variables, 68-79) uses the same formulation and stays `unsupported` for the same reason.

## Other unsupported instances

Free variables after `bound_domain` alone, baseline (`origin/claude/stage2b`) against this branch:

| instance | baseline | `mc_mul` fix only | this branch |
|---|---|---|---|
| m6_del_asl_hvar (stage 2b) | 60 | 18 | 0 |
| m4_del_asl_hfix (stage 2) | 40 | 0 | 0 |
| m75_General_Model_Case2 (stage 2b) | 16 | 16 | 16 (pole) |
| m76_General_Model_Case1 (stage 2) | 12 | 12 | 12 (pole) |
| m19_MTG_EX1, m77, m78 (stage 2; unsupported in older runs) | 0 | 0 | 0 |
| MINLPLib ex8_5_3, launch (start50 "unsupported" group; unsupported in issue14) | 0 | 0 | 0 |
| MINLPLib least (start50) | 2 | 2 | 2 |

Earlier work on the base branch had already bounded m19, m77, m78, ex8_5_3 and launch. `least` is
`min sum_i (d_i - x0 - x1 exp(x2 t_i))^2` with `x0` and `x1` free and `x2 in [-5, 5]`, and it stays
free. Near `x2 = 0`, `x0 + x1 exp(x2 t)` expands to `(x0 + x1) + x1 x2 t + ...`, so on every piece
that contains `x2 = 0`, `x0 = -x1 -> inf` cancels in the model but not in interval arithmetic.
Splitting `x2` into halves cannot remove that piece. Whether the cutoff bounds `x1` at all depends on
the best affine fit being worse than the optimum. Proving it would take a reformulation (for example
`s = x0 + x1`), not a bound.

## Results (60 s)

Load averages were 7 to 14 during these runs (other agents), so times are noisy.

| instance | before | after (cold) | after (warm) | SCIP (stage-2b / stage-2 CSV) |
|---|---|---|---|---|
| m6_del_asl_hvar | unsupported (9 s) | time limit, 1.5364152, bound 1.44254, 69 nodes | time limit, 1.5364152, bound 1.44015, 21 nodes (load 14) | optimal 1.07 s, 1.5358754 (1.5364152 projected) |
| m4_del_asl_hfix | unsupported (7 s) | optimal 19.0 s, 1.4737778 (bound 1.4737508), 1 node | optimal 5.0 s | optimal 0.05 s, 1.4735037 (1.4738038 projected) |
| m75_General_Model_Case2 | unsupported (33 s) | unsupported (41 s, correct: see above) | | time limit, 686019.6, bound 583567 |

On m6 the box is now finite, but the root bound is 1.44 against 1.5364. The relaxation of
`1000 uup yyp` over `yyp in [0, 3.5]` is weak, and the search did 21-69 nodes in 60 s on the loaded
machine. Closing that gap is ordinary branch-and-bound work, not a domain problem.

### A/B on MINLPLib (60 s, cold, 4 jobs; baseline and branch back to back in one slot)

The machine was heavily shared during these runs (load average 6-18 on 4 cores), so times and
solved counts are dominated by noise. A separate baseline start50 run at load 4-6 solved 36.

| set | baseline | this branch | wrong | losses |
|---|---|---|---|---|
| start50 (50; load 6-12 / 12-14) | 25 solved, 1 unsupported (least), SGM 45.3 s | 29 solved, 1 unsupported (least), SGM 40.5 s | 0 / 0 | none |
| int50 (37 of 50 each\*; load 14-18 / 15-16) | 18 solved, SGM 45.8 s | 27 solved, SGM 38.0 s | 0 / 0 | none (hadamard_4 5.0 s -> 11.7 s) |

\* Both int50 arms were stopped by the 8 GB process-group watchdog after 37 instances (4 parallel
jobs). That happened at the same point in both arms, so the 37 are the same instances.

Every difference is a gain for the branch, but the gains are what load noise between two
consecutive cold runs looks like. The new code paths cannot explain them:
- `split_bounds` fires only on a model with free variables after the LP bounds. On start50 that is
  only `least`, which stays free.
- The `mc_mul` change alters only products with an infinite endpoint.

The result to take from this: no regressions, and no wrong answers.

## Correctness

- Unit tests: `tests/test_domain.py` (55 + 3 new: `0 * inf` in `mc_mul` and through FBBT, the hull of
  halves bounding a complementarity pair while FBBT alone does not, with sampled points checked to be
  kept, and an end-to-end solve), `tests/test_relax.py` (58) and `tests/test_fbbt_interp.py` (15), all
  passing.
- `check_wrong.py --sample 12 --time-limit 20`: seed 0, 5 solved and 7 unsolved; seed 1, 8 solved and 4
  unsolved; no wrong answers.
- Risks: the `mc_mul` change also applies inside branch-and-bound wherever an intermediate interval is
  infinite (for example `1/x` on `[0, U]` times a factor with a zero endpoint). There, `[-inf, inf]`
  becomes the correct, tighter interval. On finite operands nothing changes.

Raw rows: `benchmarks/results/minlporg_unbounded_{cold,warm}.csv` (target instances) and
`benchmarks/results/unbounded_ab_{base,new}_{start50,int50}.csv` (A/B).

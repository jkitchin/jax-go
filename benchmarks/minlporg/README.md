# minlp.org benchmark (issue #87)

Smoke tests of jaxglobal on instances from the [minlp.org](https://www.minlp.org/) library (the
CMU-IBM Cyber-Infrastructure for MINLP), against SCIP, to find where the approach falls short of
equation-based solvers.

* `fetch.py` lists the library, keeps the nonlinear instances (not LP/MILP) with at most 60
  variables and 60 constraints, downloads their models to `src/` and converts them to `nl/` with
  GAMS CONVERT (the GAMS system in the `gamspy` package; its demo license is enough at this size).
  MPS models are read by SCIP and written as GAMS first. `instances.csv` lists the 16 instances,
  `skipped.csv` the 3 that are data files or multi-file algorithms rather than a model.
* `python benchmarks/run.py --lib minlporg --solvers jaxglobal,scip --max-vars 60 --tag <tag>` runs
  them (`--lib` is new; everything else is the MINLPLib harness: same tolerances, rel. gap 1e-4,
  abs. gap 1e-6, feasibility 1e-6).
* `report.py` prints the table below.

## Stage 1 results (2026-11-13)

Machine: 4-core cloud VM, two solves at a time, 60 s limit; SCIP 10.0 through `pyscipopt` 6.3.
"cold" is the first run (compilation included), "warm" a second run with jaxglobal's kernel cache
filled. *reference* is the better of minlp.org's best known value and SCIP's primal value; *class*
is `benchmarks/report.classify` against it and SCIP's dual bound (1e-3 relative tolerance);
*pass* means solved within max(2 x SCIP time, 10 s), the rule of `docs/PLAN-scip-gap-v2.md`.
Raw rows: `benchmarks/results/minlporg_stage1_cold.csv` (jaxglobal and SCIP) and
`minlporg_stage1_warm.csv` (jaxglobal).

| instance | type | vars (bin) | cons | reference | SCIP | jaxglobal cold | jaxglobal warm | class | pass (cold / warm) |
|---|---|---:|---:|---:|---|---|---|---|---|
| m7_Model_P1 | MINLP | 27 (21) | 27 | 2260.26 | optimal 0.73 s, 2260.26 | time_limit 60.2 s, 2260.34 | time_limit 60.2 s, 2260.26 | unsolved | - / - |
| m8_Model_P2 | MINLP | 27 (3) | 30 | 2260.26 | optimal 0.208 s, 2260.26 | optimal 19.7 s, 2260.26 | optimal 13.1 s, 2260.26 | solved | - / - |
| m245_Model_E2 | NLP | 20 (0) | 27 | 2068.2 | optimal 0.0672 s, 2068.2 | unsupported | unsupported | unsupported | - / - |
| m266_Formulation_MINLP | MINLP | 50 (50) | 1 | 3760.72 | time_limit 60 s, 3909.73 | time_limit 60 s,  | time_limit 60 s,  | unsolved | - / - |
| m267_Formulation_MIQP | MIQP | 50 (50) | 1 | 3760.72 | time_limit 60 s, 3909.73 | time_limit 60 s,  | time_limit 60 s,  | unsolved | - / - |
| m383_single_obj | MINLP | 30 (15) | 30 | 335.137 | optimal 0.284 s, 335.137 | optimal 18.9 s, 335.144 | optimal 9.3 s, 335.144 | solved | - / PASS |
| m389_CLay0203BM | MIQCP | 30 (18) | 54 | 41573.3 | optimal 0.496 s, 41573.3 | optimal 25.4 s, 41573.3 | optimal 10.1 s, 41573.3 | solved | - / - |
| m417_kClus_3_10_2_BM_1 | MIQCP | 46 (30) | 45 | 0.328894 | optimal 1.72 s, 0.328919 | optimal 48.5 s, 0.328923 | optimal 23.6 s, 0.328923 | solved | - / - |
| m418_kClus_3_10_3_BM_1 | MIQCP | 49 (30) | 45 | 0.983639 | optimal 4.16 s, 0.983676 | optimal 35.3 s, 0.983679 | optimal 22.3 s, 0.983679 | solved | - / - |
| m419_kClus_3_10_5_BM_1 | MIQCP | 55 (30) | 45 | 1.84745 | optimal 12.2 s, 1.84753 | time_limit 60.2 s, 1.84754 | time_limit 60.2 s, 1.84754 | unsolved | - / - |
| m465_Haverly_2Player_PriceTaker_FixedCharge0 | MIQCP | 37 (0) | 43 | 1750 | optimal 0.0732 s, 1750 | optimal 13.2 s, 1750 | optimal 5.05 s, 1750 | solved | - / PASS |
| m466_Haverly_2Player_PriceTaker_FixedCharge1 | MIQCP | 43 (6) | 55 | 1525 | optimal 0.181 s, 1525 | optimal 30.1 s, 1525 | optimal 14.9 s, 1525 | solved | - / - |
| m467_Haverly_2Player_NashCournot_FixedCharge0 | MIQCP | 37 (0) | 43 | 1062.5 | optimal 0.0716 s, 1062.5 | optimal 12 s, 1062.5 | optimal 5.42 s, 1062.5 | solved | - / PASS |
| m468_Haverly_2Player_NashCournot_FixedCharge1 | MIQCP | 43 (6) | 55 | 837.5 | optimal 0.229 s, 837.5 | optimal 27.9 s, 837.5 | optimal 16.9 s, 837.5 | solved | - / - |
| m478_MINLP_example_case1_2 | MINLP | 30 (0) | 29 | 0.12788 | time_limit 60 s, 0.127887 | unsupported | unsupported | unsupported | - / - |
| m479_MINLP_example_case2_1 | MINLP | 30 (7) | 29 | 0.0675034 | time_limit 60 s, 0.0675034 | unsupported | unsupported | unsupported | - / - |

classes (cold): solved 9, unsolved 4, unsupported 3

**Summary.** No wrong answers. jaxglobal solves 9 of 16 (SCIP: 12), passes 1 (m383, warm), and is
much slower where both solve: the time ratio to SCIP on the 9 has a median of 95 cold and 63 warm
(5 to 180). SCIP solves three instances jaxglobal does not (m245, m7, and m419 in 12 s); neither
solves m266/m267 or m478/m479 in 60 s.

## Gaps found

1. **Unbounded variables (3 instances: m245, m478, m479). Fixed; see below.** Unlike MINLPLib,
   minlp.org models often leave variables without bounds. jaxglobal stopped with
   `unbounded_domain` when it could not derive finite bounds ("no finite bounds found for
   variables [9, ..., 19]" on m245), while SCIP solves m245 in 0.07 s. Three more instances (m7,
   m383, m389) have unbounded variables in the model but their bounds were already inferred.
2. **Fixed cost per solve. Partly fixed; see below.** Where both solve, jaxglobal needs 5 to 48 s against SCIP's 0.07 to
   4 s. The four Haverly pooling instances (37 to 43 variables) take 5 to 17 s warm and 12 to 30 s
   cold against under 0.25 s, so most of it is setup and compilation, not search.
3. **Dense 0-1 quadratic programs (m266, m267: 50 binaries, one cardinality constraint).**
   jaxglobal does not finish the root node in 60 s (0 nodes; it reports no incumbent and no
   bound, written as `nan` in the CSV). SCIP does not solve them either but has an incumbent
   (3909.7, best known 3760.7) and a bound (1956).
4. **Weak bounds with many binaries (m7: 21 binaries; m419: k-means clustering, 30 binaries).**
   On m7 jaxglobal has the optimal incumbent (warm) but its lower bound stays at 2224.5 against
   2260.26 in both runs; on m419 the bound reaches 1.813 against 1.8475. SCIP solves them in
   0.7 s and 12 s.

## Gap 1 fix: unbounded variables

Three changes, each of which keeps every optimal point (the box is never guessed):

* **`1/x` on a box touching 0** (`relax.mc_recip`). For `x in [0, U]` the interval was the whole
  line; it is now `[1/U, +inf]` (and `[-inf, 1/L]` for `[L, 0]`), since `x = 0` is outside the
  domain of `1/x`, the convention `log` already uses. On m245 (`D2 >= 1/X1 + X2^2 + ...`, `X1 in
  [0, 10]`), the unbounded `1/X1` let the LP bounds of `domain.LinearRelaxation` make `X2^2`
  arbitrarily large; with it the objective cutoff bounds every variable.
* **The feasible-point search** (`domain._feasible_points`) that gives that cutoff starts off the
  bounds (a start at `X1 = 0` made SLSQP stop at once on `1/0`) and asks for inequalities with a
  margin of half the feasibility tolerance (points ended at a violation of 1.5e-6 against 1e-6).
* **Variables seen only through `sin`/`cos`** (`domain.periodic_bounds`, new). In m478/m479 the
  voltage angles enter only as `cos(d(N) - d(NP) - phi)` and `sin(...)`. A forward pass over the
  scalar tape checks that a variable reaches every output only through `sin`/`cos` of an affine
  argument with commensurate coefficients; shifting it by the period then changes nothing, so it
  is limited to one period (here `[-pi, pi]`).

Rerun after the fix (jaxglobal cold, 60 s; SCIP rows from stage 1; raw rows in
`benchmarks/results/minlporg_gap1_cold.csv`):

| instance | before | after | SCIP |
|---|---|---|---|
| m245_Model_E2 | unsupported | optimal 27 s, 2068.2006 | optimal 0.07 s, 2068.2005 |
| m478_MINLP_example_case1_2 | unsupported | time limit, no incumbent at 60 s | time limit, 0.127887 (bound -228) |
| m479_MINLP_example_case2_1 | unsupported | time limit, no incumbent at 60 s | time limit, 0.0675034 (bound -228) |

jaxglobal now solves 10 of 16 (SCIP 12), with none unsupported and no wrong answers; the other 13
instances give the same results as before. m478 and m479 are now hard rather than unsupported:
with 300 s, m478 has the best known value 0.12788818 from the first iteration, but after 16,000
nodes its bound is only -2065 (the angle windows of width 2 pi are wide for differences of about 1e-3, and the
trigonometric relaxations on them are weak). Solve times in this rerun vary with the state of the
kernel cache (m465 and m467 now pass at 7 s), so compare times within one run.

## Gap 2: fixed cost per solve

**Where the time went.** Profiles of single warm solves (the kernel caches filled) on m466, m8,
m389, m418 and m245, with `benchmarks/profile_root.py` and the JAX compilation-cache events:

* The local NLP solves (pounce, for incumbents, dives and integer-fixed solves) were the largest
  part everywhere, 5 to 8 s per solve: on m466 65 local solves, 3,900 interior-point iterations,
  31,000 Python callbacks. Per iteration, half the time was the callbacks and half pounce itself.
  The callbacks were slow because pounce's `from_jax` converts every `x` with `jnp.asarray`
  (an eager JAX op, about 0.1 to 0.3 ms per call inside a solve) and evaluates the Jacobian
  and the Hessian densely (one direction per variable: 120 us on m466).
* `bound_domain` took 2 to 5 s when a model has a free variable (m466 has one: the objective
  variable SCIP's GAMS writer adds), mostly compiling its FBBT kernels.
* Tracing, lowering and reading the XLA cache for the ~170 functions a solve compiles cost 4 to
  5 s (m245: 173 compiles, all cache hits), spread thinly: no function above 1 s.
* Not a cost: re-tracing the model function. Replaying its jaxpr instead of re-running the .nl
  reader made no measurable difference, and was not kept.

**What changed** (`jaxglobal/local.py`, `_Callbacks`): the local solves use jaxglobal's own
callbacks with pounce's sparsity structure. They pass NumPy `x` straight to the compiled
functions, and evaluate the Jacobian and the Lagrangian Hessian compressed, one forward
direction per color of a greedy column coloring of their patterns (Curtis-Powell-Reid), when
that needs at most half as many directions as there are variables (m466: 5 for the Jacobian
instead of 43; 12 against 121 us per call). Values agree with pounce's dense path to rounding
(`tests/test_local_callbacks.py`). Running the batch of local solves sequentially instead of on
pounce's thread pool made no difference (the callbacks hold the GIL) and was not changed.

**Result.** Warm runs of the whole set, before (the gap 1 commit) and after, each after a
priming run, same settings (`benchmarks/results/minlporg_gap2_base_warm.csv`,
`minlporg_gap2_warm.csv`):

| instance | before | after | SCIP | pass before / after |
|---|---:|---:|---:|---|
| m8_Model_P2 | 13.6 s | 9.1 s | 0.21 s | - / PASS |
| m245_Model_E2 | 6.0 s | 6.0 s | 0.07 s | PASS / PASS |
| m383_single_obj | 13.0 s | 7.5 s | 0.28 s | - / PASS |
| m389_CLay0203BM | 13.7 s | 9.5 s | 0.50 s | - / PASS |
| m417_kClus_3_10_2_BM_1 | 30.4 s | 24.7 s | 1.72 s | - / - |
| m418_kClus_3_10_3_BM_1 | 24.3 s | 22.8 s | 4.16 s | - / - |
| m465 (Haverly PT, continuous) | 5.7 s | 5.2 s | 0.07 s | PASS / PASS |
| m466 (Haverly PT, mixed) | 16.7 s | 13.2 s | 0.18 s | - / - |
| m467 (Haverly NC, continuous) | 6.0 s | 6.0 s | 0.07 s | PASS / PASS |
| m468 (Haverly NC, mixed) | 17.4 s | 15.7 s | 0.23 s | - / - |

On the 10 instances solved in both runs the geometric mean time falls from 12.7 s to 10.4 s
(-18%), and 6 pass instead of 3. The six unsolved instances are unsolved in both, with the same
or better incumbents (m478 and m479 now have incumbents at 60 s: 0.127888 and 0.0567866; the
latter is below SCIP's 0.0675 and a tenth of minlp.org's 0.5679). m383's reported optimum is
now 335.1370, SCIP's value, instead of the 335.1435 that used the feasibility tolerance (a
different local-solve path).

**What remains.** jaxglobal is still 20 to 200 times slower than SCIP on these instances. In
order of size: the local solves (still 3 to 6 s; pounce's own per-iteration time is now half
of it), `bound_domain`'s kernels on models with free variables, and the per-function JAX
trace/lower/cache-read overhead. The last two are compile costs that a solve pays for every new
model; reducing them means compiling fewer distinct functions (or caching more of them with
`kcache`), which is a larger change.

## Stage 2 results: larger instances (2026-11-13)

Instances with more than 60 and at most 300 variables or constraints, at most the 3 smallest of
each minlp.org problem (`python benchmarks/minlporg/fetch.py --max-vars 300 --max-cons 300
--above 60 --per-problem 3 --out instances_stage2.csv`): 45 selected from 20 problems, 32 convert
(`instances_stage2.csv`); the other 13 are data files, archives, an AMPL source or GAMS files that
do not compile on their own (`skipped_instances_stage2.csv`). Same machine and settings as stage 1
(60 s, two solves at a time), with the gap 1 and gap 2 changes; raw rows in
`benchmarks/results/minlporg_stage2_cold.csv` (jaxglobal and SCIP) and `minlporg_stage2_warm.csv`.
Sizes in the table are those of the converted models.

| instance | type | vars (bin) | cons | reference | SCIP | jaxglobal cold | jaxglobal warm | class | pass (cold / warm) |
|---|---|---:|---:|---:|---|---|---|---|---|
| m3_del_asl_miqp | MIQP | 71 (10) | 79 | 1.47378 | optimal 0.00775 s, 1.47378 | optimal 22 s, 1.47378 | optimal 12 s, 1.47378 | solved | - / - |
| m4_del_asl_hfix | NLP | 121 (0) | 99 | 1.4735 | optimal 0.0474 s, 1.4735 | unsupported | unsupported | unsupported | - / - |
| m5_del_asl_minlp | MINLP | 81 (10) | 100 | 1.5364 | optimal 0.544 s, 1.53642 | time_limit 61.1 s, 1.53642 | time_limit 60.5 s, 1.53642 | unsolved | - / - |
| m19_MTG_EX1 | MINLP | 125 (48) | 196 | 352.446 | optimal 48.9 s, 352.446 | time_limit 61.1 s,  | unsupported | unsolved | - / - |
| m20_MTG_EX1a | MINLP | 194 (93) | 257 | 391.613 | time_limit 60 s, 388.489 | time_limit 63.1 s,  | time_limit 62.6 s,  | unsolved | - / - |
| m21_MTG_EX1b | MINLP | 194 (93) | 257 | 450.548 | time_limit 60 s, 382.95 | time_limit 61.3 s,  | time_limit 61.7 s,  | unsolved | - / - |
| m31_nonconvex1 | MINLP | 34 (12) | 67 | 285506 | optimal 0.628 s, 285507 | time_limit 72.2 s, 403047 | time_limit 61.1 s, 374648 | unsolved | - / - |
| m32_nonconvex2 | MINLP | 76 (36) | 205 | 2.68703e+06 | optimal 0.752 s, 2.68703e+06 | time_limit 60 s,  | time_limit 69.6 s, 2.94911e+06 | unsolved | - / - |
| m33_convex1 | MINLP | 46 (24) | 73 | 285506 | optimal 0.403 s, 285507 | optimal 46.5 s, 285507 | optimal 14.4 s, 285507 | solved | - / - |
| m48_bcp7 | MINLP | 134 (8) | 153 | 0.9629 | time_limit 60 s,  | time_limit 60 s,  | time_limit 60 s,  | unsolved | - / - |
| m49_bcp6 | MINLP | 118 (7) | 134 | 0.9609 | time_limit 60 s,  | time_limit 147 s,  | time_limit 60 s,  | unsolved | - / - |
| m50_bcp5 | MINLP | 90 (6) | 103 | 0.9519 | time_limit 60 s,  | time_limit 130 s, 0.95098 | time_limit 105 s, 0.95098 | unsolved | - / - |
| m60_GMA_ethanol_model_BigM | MINLP | 37 (24) | 72 | -157.587 | optimal 0.226 s, -157.587 | time_limit 60 s,  | time_limit 60 s,  | unsolved | - / - |
| m61_GMA_ethanol_model_CH | MINLP | 61 (24) | 80 | -157.587 | optimal 0.134 s, -157.587 | time_limit 60 s,  | time_limit 60 s,  | unsolved | - / - |
| m76_General_Model_Case1 | MINLP | 112 (12) | 120 | 170750 | time_limit 60 s, 170750 | unsupported | unsupported | unsupported | - / - |
| m77_Specilized_Model_Case2_NLP_Improvement | MINLP | 76 (16) | 90 | 634978 | optimal 38.8 s, 634978 | unsupported | unsupported | unsupported | - / - |
| m78_Specilized_Model_Case1 | MINLP | 56 (12) | 64 | 154997 | time_limit 60 s, 154997 | unsupported | unsupported | unsupported | - / - |
| m126_generalizedpooling_lee1 | MIQCP | 49 (9) | 82 | -4640.08 | optimal 2.09 s, -4640.08 | optimal 43.8 s, -4640.08 | optimal 26.6 s, -4640.08 | solved | - / - |
| m127_generalizedpooling_lee2 | MIQCP | 53 (9) | 92 | -3849.27 | optimal 4.46 s, -3849.27 | time_limit 61.3 s, -3849.27 | optimal 53.9 s, -3849.27 | unsolved | - / - |
| m128_generalizedpooling_meyer4 | MIQCP | 118 (55) | 141 | 1.24981e+06 | time_limit 60 s, 1.24981e+06 | time_limit 62.3 s, 1.12278e+06 | time_limit 60.8 s, 1.11235e+06 | unsolved | - / - |
| m274_MPBP_029 | MIQCP | 102 (36) | 213 | 13.3594 | optimal 2.22 s, 13.3594 | optimal 43.1 s, 13.3594 | optimal 18.6 s, 13.3594 | solved | - / - |
| m384_pareto_linear_convex | MINLP | 49 (40) | 96 | 0.992899 | optimal 0.0215 s, 0.992899 | optimal 32.6 s, 0.992899 | time_limit 60.1 s, 0.990331 | solved | - / - |
| m390_CLay0204BM | MIQCP | 52 (32) | 90 | 6545 | optimal 1.01 s, 6545 | optimal 43.7 s, 6545 | optimal 19.7 s, 6545 | solved | - / - |
| m392_CLay0303BM | MIQCP | 33 (21) | 66 | 26669.1 | optimal 0.627 s, 26669.1 | optimal 32.3 s, 26669.1 | optimal 12.9 s, 26669.1 | solved | - / - |
| m393_CLay0304BM | MIQCP | 56 (36) | 106 | 40262.4 | optimal 1.63 s, 40262.4 | optimal 45.9 s, 40262.4 | optimal 25.3 s, 40262.4 | solved | - / - |
| m423_kClus_5_10_2_BM_1 | MIQCP | 70 (50) | 69 | 0.101129 | optimal 3.97 s, 0.10115 | time_limit 60.3 s, 0.126211 | time_limit 60.2 s, 0.107641 | unsolved | - / - |
| m424_kClus_5_10_3_BM_1 | MIQCP | 75 (50) | 69 | 0.373649 | optimal 20.2 s, 0.373697 | time_limit 60.3 s, 0.515463 | time_limit 60.2 s, 0.515463 | unsolved | - / - |
| m425_kClus_5_10_5_BM_1 | MIQCP | 85 (50) | 69 | 0.991268 | time_limit 60 s, 0.999356 | time_limit 60.5 s, 1.43309 | time_limit 60.4 s, 1.24299 | unsolved | - / - |
| m469_Adhya1_2Player_PriceTaker_FixedCharge0 | MIQCP | 75 (0) | 99 | 1089.18 | time_limit 60 s, 1089.18 | time_limit 60.4 s, 1087.72 | time_limit 60.6 s, 1087.72 | unsolved | - / - |
| m470_Adhya1_2Player_PriceTaker_FixedCharge1 | MIQCP | 85 (10) | 119 | 545 | optimal 43.2 s, 545 | time_limit 60.6 s, 363.199 | time_limit 60.3 s, 363.199 | unsolved | - / - |
| m471_Adhya1_2Player_NashCournot_FixedCharge0 | MIQCP | 75 (0) | 99 | 894.175 | time_limit 60 s, 894.175 | time_limit 60.5 s, 892.724 | time_limit 60.3 s, 892.724 | unsolved | - / - |
| m480_OPF_33_SBB | MINLP | 134 (33) | 133 | 0.7279 | time_limit 60 s, 2.10987 | time_limit 60 s,  | time_limit 60 s,  | unsolved | - / - |

classes (cold): solved 8, unsolved 20, unsupported 4

**Summary.** No wrong answers. SCIP solves 20 of 32; jaxglobal 9 (8 in the cold run, plus m127
warm), none within the pass time, and 4 (warm: 5) are unsupported. Where both solve, jaxglobal
takes 12 to 54 s warm against SCIP's 0.01 to 4.5 s. Among the instances
neither finishes, jaxglobal's incumbent is the better one only on m128 (1.11e6 against SCIP's
1.25e6, minimization); on m469/m471 SCIP's is better by 0.1%.

**Gaps found in stage 2**, by number of instances:

1. **Weak bounds / no incumbent within 60 s on mixed-integer models (11 instances)**: the
   periodic scheduling models m19-m21 (48 to 93 binaries) and m60/m61 (24 binaries) end without
   an incumbent or a bound; m31/m32, m423-m425 (k-means, 50 binaries) and m470 end with poor
   incumbents (m31: 374,648 against 285,506; m470: 363 against 545). SCIP solves m19, m31, m32,
   m60, m61, m423, m424 and m470. This is stage 1's gap 4 at larger size, and the main gap here.
2. **Unbounded variables (4 instances, 5 in the warm run)**: m4 (an NLP with 61 free variables),
   and the heat-exchanger-network models m76-m78 (synheat type: the free variables are probably
   the approach temperatures `dt`, bounded above only through big-M constraints
   `dt <= th - tc + G (1 - z)`, which FBBT does not use while `z` is free). SCIP solves m4 in
   0.05 s and m77 in 39 s.
3. **The time limit is overrun on some models**: m49 took 147 s and m50 130 s in the cold run
   (105 s warm) against a 60 s limit, m31 72 s and m32 70 s; at least one solve was killed for
   lack of memory during the run (the run kept going and no row reports an error).
4. **Nondeterminism between runs**: m19 is unsupported in the warm run but not in the cold one,
   and m384 is solved cold (33 s) but not warm (time limit, incumbent 0.990 against 0.993).
   Both look time-dependent (bound tightening and local solves are given deadlines).
5. **Fixed cost (stage 1's gap 2)** is still there: on the 9 instances jaxglobal solves, it is 8
   to 1,500 times slower than SCIP.

The Belgian chocolate problems m48-m50 are unsolved by both solvers, and m20/m21/m425/m480 by
SCIP as well.

## Notes on the instances

* **mps2gms changed the quadratic MPS models.** Converted with GAMS `mps2gms`, the four Haverly
  instances (m465 to m468) had optimal values about 20% below minlp.org's (1407.03 against 1750
  for m465), in both jaxglobal and SCIP; SCIP reading the MPS files directly gives minlp.org's
  values. `fetch.py` now converts MPS through SCIP's reader and GAMS writer, which reproduces them.
* **m383**: jaxglobal's optimum, 335.1435, is above SCIP's proven maximum, 335.1370 (2e-5
  relative). The point puts 12 equality residuals at about 1e-6, the feasibility tolerance, so this
  is the tolerance being used, not a wrong bound; within the 1e-3 classification tolerance.
* **m479**: SCIP finds 0.0675, well below minlp.org's best known 0.5679 (minimization);
  minlp.org's value is not optimal for this converted model, or SCIP's point uses its tolerances.
* **m267** solves the same model three times with different option files; the converted model is
  the last solve (the MIQCP), as for every model with several solve statements.

## Next steps

* Stage 1 follow-ups, by expected gain: the root-node cost on dense binary quadratics; the fixed
  cost per solve on small models; tighter angle bounds and trigonometric relaxations for the
  power-flow models (m478/m479).
* Stage 2 follow-ups (see *Stage 2 results*): bounds and incumbents on the mixed-integer models
  (the largest gap), bounds for the `dt` variables of the heat-exchanger-network models from
  their big-M constraints, the time-limit overruns, and run-to-run nondeterminism.

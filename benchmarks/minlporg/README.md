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

1. **Unbounded variables (3 instances: m245, m478, m479).** Unlike MINLPLib, minlp.org models often
   leave variables without bounds. jaxglobal stops with `unbounded_domain` when bound tightening
   does not find finite bounds ("no finite bounds found for variables [9, ..., 19]" on m245),
   while SCIP solves m245 in 0.07 s. Three more instances (m7, m383, m389) also have unbounded
   variables in the model but their bounds are inferred.
2. **Fixed cost per solve.** Where both solve, jaxglobal needs 5 to 48 s against SCIP's 0.07 to
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

* Stage 1 follow-ups, by expected gain: bounds for variables left unbounded (e.g. from linear
  equalities and the objective, or a bounded-domain fallback that is reported as such); the
  root-node cost on dense binary quadratics; the fixed cost per solve on small models.
* Stage 2: larger instances (`python benchmarks/minlporg/fetch.py --max-vars 500 --max-cons 500`).

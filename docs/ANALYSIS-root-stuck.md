# Stage-2b models stuck at or near the root (m34, m410, m400, m441), 2026-11-16

**Question.** Why do m34_convex2, m410_CLay0203HRS, m400_CLay0304HRcone and m441_kClus_3_10_2_HRc_1 end at or near the root in 60 s, when SCIP solves them in 0.4 to 4 s?

## SCIP's root bounds

pyscipopt 6 / SCIP 10, via `benchmarks/root_gap.py`:

| instance | SCIP root | SCIP result |
|---|---:|---|
| m34 | 2686799 (optimum 2687027) | 1 node |
| m410 | 0 | 92 nodes, 0.8 s |
| m400 | 26.6 | 586 nodes, 4.1 s |
| m441 | 0 | 351 nodes, 4.8 s |

- m34's continuous relaxation is 2537156, which matches our root bound of 2533578. SCIP's tight root comes from its cuts: with separation off, its root is 2467066.
- On m410, m400 and m441, SCIP's root is no better than ours. SCIP wins there by branching fast.

## Profiles

`benchmarks/profile_root.py`, primed cache, 60 s:

- **m410**: 63 s waiting for the bound kernel (18.5 s trace, 44.5 s compile, still unfinished), 0 nodes.
  - Its 24 constraints `x²/(1e−6 + 0.999999 y) + x'²/(1e−6 + y) + …` were not certified convex: the monomial rule required every factor to be > 0, and `x` starts at 0.
  - Without the certificate, they went through alphaBB (`jacfwd` on 4313 equations).
- **m34**: the bound kernel (41k HLO lines) took 40 s to compile and 13 s to trace; the root bound was ready at 54 s. Each 16-box bound call took 3.5 s, so probing (8 calls) took 27 s.
- **m441**: bound-kernel compile 17 s; root dives plus probing 17 s; 2.3 s per 16-box call (a dense lifted LP with 316 columns). 63 nodes, bound about 0.
- **m400**: domain bounding 13 s, compile 13 s, probing 10 s, 15 nodes.

## Causes

1. **Missing convexity certificate (m410).** The perspective `x^p · y^q` with `p ≥ 1`, `q < 0` is convex on the closure where `x = 0` (by continuity, with a finite gradient), but the rule required `x > 0`.
2. **Kernel size of the reduced-space bound (m34, m410).**
   - The relaxation's forward-mode cut pass (about 11k HLO lines) and the LP each appeared twice: once for the first LP, once in the Kelley loop. XLA's compile time grows faster than linearly with kernel size.
   - Every part was linearized at every cut point, although `_cut_rows` keeps only the center row of an affine part. m34 has 240 affine parts out of 242; m410 has 108 out of 157.
3. **Node throughput (m400, m441).**
   - These use the lifted quadratic LP: a dense normal-equations Cholesky on about 316 columns per IPM iteration.
   - The rotated cones `x² + y² ≤ i·z` are relaxed with McCormick on `i·z`.

## Fixes

- **`convexity._apply_mono`.** Factors with exponent ≥ 1 now need only be ≥ 0, in both the curvature conditions and the sign check; all other factors still need > 0. m410's constraints are now all certified, so the exact-certificate path skips the base relaxation and alphaBB is no longer built.
- **`lp.make_constrained_bounder`, `SHARED_CUTS`.** One while loop now covers the first LP and the Kelley rounds.
  - Round 0 linearizes the initial points in passes of `SHARED_CHUNK` (32) points.
  - Round k linearizes the last LP point, in slot 0 of one pass; the pass's other slots are discarded.
  - The objective range `[tlo, tlo + T]` and the infeasibility flag accumulate over the passes.
- **`lp.make_constrained_bounder`, `SPLIT_AFFINE`.** Affine parts are linearized once, at the center, and the cut points linearize only the nonlinear parts. Parts are selected by static slices, so XLA drops the computation of the other parts. The LP has the same rows.
- **Gating.** Both switches default to `"auto"`, which requires a joint tape of at least 700 equations. The split also requires at least half of the parts to be affine.

## Effect on kernels

Standalone microbenchmarks, 16 boxes:

| case | HLO lines | compile | bound call (16 boxes) |
|---|---:|---:|---:|
| m410, before | 41k | 24 s | 0.75–0.9 s |
| m410, shared only | 15k | 8–11 s | about the same |
| m410, shared + split | 27.6k | 14 s | 0.25 s |
| m34, before | 40k | 19–24 s | 2.6–3.9 s |
| m34, shared + split | 26.8k | 15 s | 0.85–1.2 s |

## Effect on solves

| instance | before | after |
|---|---|---|
| m34, primed cache | solved, 68.5 s (load 5–7), 25 nodes | solved, 52 s (load 10), 25 nodes |
| m34, cold | solved, 125 s | solved, 138 s (load 8), 54 s of it waiting for the compile |
| m410, primed cache | no root bound in 60 s; at 128 s, 55 nodes, bound 1360 | solved, 88 s (load 14–15), 255 nodes |
| m410, cold | — | solved, 235 s (load 14–15). At 60 s (load 8): a root bound and incumbent 42098 |

The m34 cold times were taken at different loads, so they are not directly comparable.

## Validity

No bound changes in kind:

- The convexity change certifies only functions that are convex on the box. Tests check the midpoint inequality and the tangent cuts on random boxes.
- Shared and split give the same LP rows and weak-duality certificate. Tests compare bounds against the old layout to 1e−6, and range-reduced boxes to 1e−3, because range reduction amplifies IPM rounding.

## What is left

- **Cold compile time under load** still limits m34 and m410: about 13 s of trace plus 15–50 s of compile. Next steps:
  - the same split/shared treatment for `make_obbt` (its kernel takes 25–31 s to compile);
  - compiling OBBT only after the bound kernel is ready;
  - reducing trace time.
- **m400 and m441** need two things:
  - node throughput: a structured or sparse lifted LP;
  - rotated-cone (SOC) outer approximation through the norm reformulation `‖(2x, 2y, i − z)‖ ≤ i + z`, which the convexity module already certifies.

  Root probing also needs a time budget.
- **m47** was not addressed.

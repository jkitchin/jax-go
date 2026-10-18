# Roadmap: from prototype to a state-of-the-art solver

"State of the art" here means two separate claims, and the plan treats them
separately:

1. **Competitive on standard benchmarks.** On the continuous part of MINLPLib,
   solve about as many instances as SCIP, Couenne, EAGO and MAiNGO within a
   time limit, measured by performance profiles and shifted geometric mean time.
   Matching BARON is a stretch goal, not the bar.
2. **Best in class where equation-based solvers cannot go.** This covers
   models defined by code: neural-network and Gaussian-process surrogates, ODE
   and implicit-function models, `jnp.where` logic, and batches of parametric
   instances solved at once on GPU. jaxglobal can lead here by construction,
   provided it can still certify optima at a useful scale.

Claim 2 is the publishable niche. Claim 1 is what makes reviewers take it
seriously.

## Where the prototype stands

| area | today | state of the art (BARON, SCIP, Couenne, MAiNGO, EAGO) |
|---|---|---|
| relaxations | reduced-space McCormick plus per-primitive αBB | extended formulations with auxiliary variables; multivariate McCormick; convex/concave envelopes of bilinear, trilinear, fractional and edge-concave terms |
| bound tightening | FBBT every node; OBBT at the root; duality-based range reduction | FBBT every node; OBBT at the root and selectively in the tree; duality-based (marginals) range reduction; probing |
| cuts | Frank–Wolfe/Kelley linearizations | outer approximation, RLT, SDP/eigenvalue cuts for QCQP, perspective cuts |
| LP | dense IPM, fixed iterations | warm-started sparse simplex (dual simplex in the tree) |
| branching | widest nonlinear variable, bisection | violation-weighted, pseudocost/reliability branching, branching at the relaxation solution |
| node selection | best-bound | best-bound with diving and depth-first plunging |
| upper bounds | local NLP from box points | NLP from many starts, feasibility pump, local search in the tree |
| presolve | linear-variable detection | structure detection (quadratic, polynomial, separable), reformulation, symmetry |
| integers | none | full MINLP |
| numerics | floating-point intervals (not rigorous) | non-rigorous in practice, but with feasibility tolerances and safe LP bounds |
| scale | ≤ 9 variables tested | hundreds to thousands of variables |

## Design principle: keep it all JAX, and make batching pay

Every component below should be a pure function of fixed-shape arrays, so it
can be `jit`-compiled, `vmap`-ped over boxes or instances, and run on device.
Two existing pieces make this unusually clean:

- **A bound that tolerates inexact solves.** The weak-duality certificate
  (`lp.py`) is valid for *any* dual vector. So node LPs can be solved by a
  first-order method (PDLP-style PDHG) with a fixed iteration count on GPU, and
  an inexact solve can only weaken the bound, never make it wrong.
- **Relaxations as JAX functions.** FBBT, OBBT and cut generation are all
  passes over the same jaxpr, so they get `vmap` and `grad` for free.

## Phases

Each phase ends with an exit criterion measured on a fixed benchmark set.

### Phase 0 — benchmark harness (2–3 weeks)

The rest of the plan cannot be judged without this.

- **`.nl` → JAX translator.** pounce already parses `.nl` files
  (`read_nl`, `NlExpr`). Turn its expression trees into JAX functions, so all of
  MINLPLib and GlobalLib become jaxglobal inputs automatically.
- **Benchmark set.** Choose the continuous MINLPLib instances with ≤ 50
  variables (several hundred), and tag them by structure: bilinear, polynomial,
  trigonometric, exp/log.
- **Runner.** Generalize `examples/compare.py` to run jaxglobal, SCIP,
  Couenne, BARON (GAMS), EAGO and MAiNGO with equal tolerances and time limits.
  Report Dolan–Moré performance profiles, shifted geometric mean time, and
  solved/unsolved counts.
- **Exit:** a reproducible baseline table. Expect jaxglobal to fail on most
  instances with more than 10 variables; that is the point of measuring.

**Unbounded variables (after phase 2).** The phase-0 baseline skipped 150 of
322 instances for unbounded variables. `minimize` now derives finite bounds
itself (`jaxglobal/domain.py`): FBBT from the declared box, fixing unused
variables, and FBBT against the value of a feasible point from local solves.
It reports `unbounded_domain` only when a variable stays unbounded.
`benchmarks/run.py` no longer rejects these instances up front. How many of
the 150 become solvable has to be measured on MINLPLib.

### Phase 1 — bound tightening (4–6 weeks; the largest single gain)

Mature solvers attribute a large share of their performance to domain
reduction.

- **FBBT on the jaxpr.** A backward (reverse-interval) rule per primitive,
  iterated forward and backward to a fixed point. It is batched over all boxes
  in a node batch and runs before bounding at every node.
- **OBBT.** Minimize and maximize each variable over the relaxation LP at the
  root. In the tree, only do it for variables with large impact. The LP is
  batched, so 2n LPs per box are one `vmap` call: this is where GPU batching
  pays most directly.
- **Duality-based range reduction.** Reuse the LP duals already computed at
  every node to shrink variable ranges, at almost no cost.
- **Exit:** node counts on the current 21 problems drop at least 3×; at least
  half of the ≤ 10-variable MINLPLib subset is solved.

**Status (implemented).** `jaxglobal/fbbt.py` (FBBT on the flattened jaxpr,
5 rounds per node in a `fori_loop`), duality-based range reduction in both
bounders (`lp.reduce_ranges`), and batched OBBT (`lp.make_obbt`, root by
default, `obbt="tree"` for every node). Node counts on the 21 problems
(defaults, same tolerances as `examples/03-test-problems.ipynb` and the tests):

| problem | before | after | ratio |
|---|---:|---:|---:|
| gramacy_lee | 23 | 5 | 4.6× |
| six_hump_camel | 407 | 167 | 2.4× |
| mccormick | 41 | 11 | 3.7× |
| goldstein_price | 8883 | 7309 | 1.2× |
| shubert | 2399 | 1781 | 1.3× |
| michalewicz_2d | 65 | 9 | 7.2× |
| holder_table | 391 | 183 | 2.1× |
| eggholder | 153 | 47 | 3.3× |
| hartmann3 | 1769 | 515 | 3.4× |
| styblinski_tang_3d | 451 | 333 | 1.4× |
| shekel10_4d | 133 | 83 | 1.6× |
| hartmann6 | 6515 | 2261 | 2.9× |
| piecewise_2d | 87 | 31 | 2.8× |
| tanh_net_2d | 387 | 323 | 1.2× |
| G06 | 119 | 25 | 4.8× |
| G08 | 99 | 33 | 3.0× |
| G11 | 127 | 35 | 3.6× |
| G24 | 43 | 37 | 1.2× |
| G04 | 67 | 5 | 13.4× |
| haverly | 229 | 21 | 10.9× |
| reactor | 2587 | 541 | 4.8× |

Geometric mean 3.0×; 11 of 21 reach 3×. The constrained problems gain most
(geometric mean 4.6×). Unconstrained problems with a flat objective
landscape (Goldstein–Price, Shubert, tanh net) barely change: there the
cutoff `f <= UB` removes little until the gap is already small. Wall time is
currently higher on these small problems because the extra kernels must be
compiled. The MINLPLib half of the exit criterion has not been measured yet.
Remaining phase-1 items: feeding FBBT's intermediate ranges back into the
relaxations, choosing tree-OBBT variables by impact, and probing.

### Phase 2 — stronger relaxations (6–8 weeks)

- **Extended (lifted) formulation as an option.** Introduce auxiliary
  variables for the intermediate values of the jaxpr; mature solvers use this
  representation. The jaxpr already *is* the DAG, so lifting is mechanical.
  Keep reduced space as the default where it is tighter or smaller (Bongartz &
  Mitsos show either can win).
- **Better envelopes.** Multivariate McCormick (Tsoukalas–Mitsos);
  convex-envelope rules for trilinear, fractional and edge-concave terms;
  tighter univariate envelopes, with secant-tangent rules in place of αBB where
  convexity is known.
- **Structure detection from the jaxpr.** Recognize quadratic forms, products,
  and sums of univariate terms, and route them to specialized relaxations, for
  example eigenvalue/SDP-based cuts for quadratic parts.
- **Better αBB.** Scaled Gershgorin and Hertz-type α in place of a single
  uniform α, with interval Hessians of multivariate subexpressions rather than
  only per primitive.
- **Exit:** root gaps close at least 2× on the pooling and polynomial
  instances; Adhya and Ben-Tal pooling instances solve.

**Status (started).** Two pieces are implemented.

- *Univariate envelopes* (`jaxglobal/univariate.py`, default on): a structural
  per-element dependency analysis of the jaxpr finds smooth sub-expressions of
  a single variable that are built from more than one exactly relaxed
  operation. Each one gets a grid-hull envelope: the samples are lowered by a
  per-cell curvature bound from interval `jvp(jvp(g))`, then the lower convex
  hull is taken. That envelope is intersected with McCormick. This covers the
  "sums of univariate terms" structure and "tighter univariate envelopes"
  above. Root lower bounds (whole box, no tightening):

  | problem | f* | McCormick | + univariate |
  |---|---:|---:|---:|
  | styblinski_tang_3d | −117.50 | −612 | −118.64 |
  | six_hump_camel | −1.032 | −156.1 | −7.08 |
  | michalewicz_2d | −1.801 | −2.000 | −1.823 |
  | goldstein_price | 3 | −8.79e7 | −7.48e7 |

  Node counts change only where such blocks exist (phase-1 tightening on in
  both columns; other problems are unchanged):

  | problem | without | with |
  |---|---:|---:|
  | styblinski_tang_3d | 333 | 9 |
  | six_hump_camel | 167 | 31 |
  | shubert | 1781 | 305 |
  | michalewicz_2d | 9 | 1 |
  | G24 | 39 | 3 |
  | G08 | 33 | 31 |

- *Lifted LP* (`jaxglobal/lifted.py`, `lifted=True`): one column per
  nonlinear jaxpr node, linked to its inputs by cuts from the node's own
  McCormick rule (bilinear terms get the four McCormick facets) at FBBT
  bounds, with Kelley rounds and the same safe certificate. With FBBT bounds
  for both, the root bound is at most ~8% tighter than reduced space on the
  21 problems. Node counts are mixed: G11 47 → 31, but reactor 747 → 2491.
  So it stays opt-in until it also uses the univariate envelopes and
  structure-specific cuts.

- *Scaled-Gershgorin alphaBB* (`jaxglobal/alphabb.py`, default on): for every
  function built only from smooth primitives, the interval Hessian
  (`jacfwd(jacfwd(g))` evaluated by the interval interpreter, once per box)
  gives per-variable α by the scaled Gershgorin rule, and
  `g + Σ αᵢ(xᵢ−lᵢ)(xᵢ−uᵢ)` / `g − Σ βᵢ(...)` are combined with McCormick
  by max/min. Its gap shrinks with the square of the box width, so it pays
  off deep in the tree. Node counts (univariate envelopes on in both):

  | problem | without | with |
  |---|---:|---:|
  | goldstein_price | 7309 | 1445 |
  | hartmann3 | 515 | 155 |
  | hartmann6 | 2261 | 1083 |
  | reactor | 747 | 439 |
  | six_hump_camel | 31 | 23 |
  | tanh_net_2d | 323 | 287 |

  No problem got worse. Nonsmooth functions (holder_table, eggholder,
  piecewise) are skipped.

- *Quadratic-form detection with eigenvalue cuts* (`alphabb="eigen"`,
  opt-in): a point interval Hessian means an exact quadratic
  `Q = V diag(λ) V^T`; each negative-curvature direction `λₖ tₖ²`
  (`t = Vᵀx`) is replaced by its secant over the range of `tₖ`. It is sound
  and convex, and tightens about 4 in 10 random indefinite quadratics. On the
  21 problems it changes no root bound (their quadratics are separable or
  bilinear, where the eigen-secant equals McCormick). It made reactor worse
  (439 → 761 nodes, from different cut points), so it is off by default.

- *Lifted LP with the new envelopes* (`lifted=True`): the lifted LP also gets
  the reduced-space cuts of the whole functions (univariate hull + αBB) on
  the `x` columns, plus the hull's interval floor, so its root bound is at
  least as tight as reduced space on all 21 problems. In the tree it is
  mixed against the default:

  | problem | default | lifted |
  |---|---:|---:|
  | goldstein_price | 1445 | 561 |
  | gramacy_lee | 5 | 1 |
  | six_hump_camel | 23 | 59 |
  | shubert | 305 | 681 |
  | hartmann3 | 155 | 313 |
  | reactor | 439 | 897 |

  It is also 2–30× slower per solve (dense IPM over the extra columns), so it
  stays opt-in.
- *Fractional envelope* (`relax.mc_frac`): for `a/b` with `a ≥ 0`, `b > 0`,
  the Tawarmalani–Sahinidis convex envelope
  `((a + √(aᴸaᵁ))/(√aᴸ + √aᵁ))² / b`, composed with `a.cv` and `b.cc`,
  tightens McCormick of `a·(1/b)`. On `x/y` over `[0.5,4]×[1,3]` it cuts
  the mean relaxation gap by about a third. None of the 21 problems has a
  nonnegative-over-positive quotient, so their node counts do not change.

- *Multivariate McCormick for products* (`relax.mc_mul`, opt-in
  `multivariate=True`). The Tsoukalas–Mitsos bound
  `min over [a.cv, a.cc] × [b.cv, b.cc] of max(F₁, F₂)` is evaluated through
  its dual: `max over λ` of the single facet `λF₁ + (1−λ)F₂` minimized over
  the rectangle. That is concave piecewise-linear in λ, with breakpoints
  where a combined coefficient changes sign; these depend only on the box.
  So the rule is the two standard facets plus at most two extra facets
  (only when a factor's interval straddles zero), and the cv is a max of
  convex facets with ordinary autodiff subgradients.

  A first version selected the primal minimizer among corner and edge
  candidates. It failed cut-validity tests twice. First, autodiff through the
  argmin mixed in the concave `a.cc` slope where the rectangle degenerates.
  Second, with `custom_jvp` subgradients, XLA under `jit` with reverse mode
  re-evaluated knife-edge comparisons differently in the forward and backward
  passes. The dual form has neither problem. Node counts with it:
  shubert 305 → 159, reactor 439 → 353, piecewise 31 → 29, six_hump_camel
  23 → 25, others unchanged. Wall time is 10–70% higher (mostly compilation),
  so it stays opt-in.

**Phase 2 outcome.** Root gaps on the polynomial problems close far more
than 2×: styblinski_tang −612 → −118.6 (f* −117.5) and six_hump_camel
−156 → −7.1 (f* −1.03). On goldstein_price, αBB gives about 2.7× smaller
gaps on small boxes. Node counts against phase 1, with no regressions:

| problem | phase 1 | phase 2 |
|---|---:|---:|
| styblinski_tang_3d | 333 | 9 |
| goldstein_price | 7309 | 1445 |
| shubert | 1781 | 305 |
| six_hump_camel | 167 | 23 |
| hartmann3 | 515 | 155 |
| hartmann6 | 2261 | 1083 |
| reactor (post-main baseline 747) | 541 | 439 |
| G24 | 37 | 3 |

**Not done:**
- *Pooling exit criterion (Adhya, Ben-Tal).* Not measured: MINLPLib is
  unreachable from the development environment. Nothing implemented here
  tightens the root bound of separable bilinear pooling constraints:
  haverly stays at −1550 under every relaxation. Pooling needs
  formulation-level structure (pq-cuts, RLT on the flow balances), which
  belongs with structure detection in a later phase.

  *Issue #12 (measured on MINLPLib).* Two costs, one of them now fixed.
  First, the node LP was mostly duplicate rows: every convex part got a cut
  at all 2n + 1 axis points, but moving along a variable a part is linear in
  does not change its cut, and a pooling balance is nonlinear in only two
  variables. Each part now keeps the axis points of its own nonlinear
  variables (`lp._nonlinear_vars`, `_cut_rows`): the root LP of
  `pooling_adhya1stp` drops from 11,252 to 756 rows and of `nous1` from
  8,925 to 781, with the same bound, and a 16-box bound call on
  `pooling_adhya1stp` from 17.6 s to 2.3 s (both at load 40 to 50). OBBT's
  second round now gives each LP the cuts at its own first-round solution
  instead of all 2n of them: same tightening on `pooling_adhya1stp`, about
  7x faster. With a long limit (load about 100) `pooling_rt2tp` then
  solves in 71 nodes and `pooling_adhya1stp` in 737; at 60 s, warm, the
  `rt2tp` bound reaches -4578 (was -5528, optimum -4392)
  (`results/issue12_*.csv`, loaded machine, 0 wrong). Second, the
  relaxation itself is too weak on `nous1`, and no reformulation of
  McCormick fixes that: with the binaries fixed at the optimum (1, 1) the
  exact lifted McCormick LP gives -0.27 against 1.567 (scipy check), four
  rounds of exact OBBT only raise it to -0.16, and 4,737 nodes of our search
  reach -0.39. RLT products of the linear balances with the nonlinear
  variables raise the root to 0.83, but only when the product columns are
  shared with the original constraints (a lifted LP with about 1,000
  columns); added as redundant constraints in reduced space they change
  nothing. That needs a lifted, sparse LP, beyond the dense IPM.

  *Issue #14 (lifted LP with RLT rows, `quadratic.py`).* Problems whose
  objective and constraints are quadratic polynomials (read off the
  scalarized tape) are now bounded in the lifted space: one column per
  product, McCormick facets on each, tangents and Kelley cuts on squares,
  and RLT rows (a linear equality times a variable) kept only when all
  their products are already columns, which on `nous1` is 34 rows and the
  whole RLT gain. The LP is solved by an interior point method on a fixed
  sparse pattern (Cholesky on the N x N normal matrix, about 1.5 ms per box
  at batch 16), certified by the same weak-duality bound as `lp.py`, and
  OBBT runs on the same LP. Its root bounds equal HiGHS on the same rows
  and are at least the reduced-space bound on all 11 `start50` QCQPs.
  Branching on these problems scores relative width times the McCormick
  gap at the LP point, and failed local searches and dives back off
  (`_LOCAL_BACKOFF`, `_DIVE_BACKOFF`). With both binaries of `nous1` fixed
  at (1, 1) the bound after OBBT is 0.86 (the other three assignments are
  infeasible by FBBT or OBBT); the "root bound at least 0.8" of issue #14
  meant this fixed-binary bound, the 0.83 above. `nous1` solves in 2,597
  nodes, 36 s alone on a quiet machine (load 4.5), 57.8 s in the benchmark
  with four jobs; about 30 s of that is the root fix-and-solve of the
  infeasible assignments, the next cost to remove. Cold, 60 s, baseline
  (`ace5724`) against this change, each started at 15-minute load below 8:
  `start50` 25 to 31 solved, SGM 40.9 to 32.6 s, all four pooling
  instances solved (`rt2tp` 21 s, `adhya1stp` 39 s, `foulds2tp` 16 s,
  `haverly2stp` 13 s); `int50` 25 to 29, SGM 49.2 to 35.9 s (`nous1`,
  `ex4`, `tloss`, `meanvarx`, `spring` newly solved; `batch` timed out at
  the limit under four parallel jobs, 23.5 s alone against 24.8 s for the
  baseline, and does not take the quadratic path); 0 wrong on both
  (`results/issue14_*_cold.csv`).
- *Trilinear envelopes* (Meyer–Floudas facets): not implemented. No test
  problem has a trilinear term to validate against; trilinear terms are
  relaxed as nested bilinear McCormick (and lifted, with `lifted=True`).

### Compile time

After phase 2, wall time on most built-in problems was XLA compilation, not
search. Changes:

- **Kernel cache.** `minimize` keeps an LRU of compiled kernels keyed on the
  functions and options (as `jit` keys on the function), so a repeat solve of
  the same problem, on any box, skips tracing and compilation. Relaxation
  tapes are also cached per input shape, so each kernel traces `f` once.
- **Two kernels, not four.** FBBT, the node bound and DBBT are one kernel;
  root (or tree) OBBT plus its leading FBBT is a second. The node sequence is
  unchanged: FBBT, OBBT, FBBT, bound. Merging everything into one kernel
  with `lax.cond` was about 2× slower, because `cond` under `vmap` runs
  both branches.
- **Kelley rounds as `lax.scan`** over a fixed-size cut buffer (unused slots
  are dead rows), so the bound's LP is compiled once, not `n_kelley + 1`
  times. The OBBT rounds stay unrolled: with the default 2 rounds a scan
  saves little, and its dead rows moved the IPM's box bounds at the 1e-7
  level, which on haverly roughly doubled the node count (the search is that
  sensitive: `feas_tol` 1e-5 / 1e-6 / 1e-7 gives 19 / 45 / 87 nodes).

`benchmarks/builtin.py` (each problem in a fresh process, solved twice):

| | before | after |
|---|---:|---:|
| total, first solve | 206 s | 130 s |
| total, repeat solve | 173 s | 18 s |
| gramacy_lee, first / repeat | 4.8 / 3.9 s | 4.0 / 0.06 s |
| G08, first / repeat | 28.4 / 25.4 s | 13.0 / 0.8 s |

Node counts are unchanged except reactor (1007 → 797) and michalewicz_2d
(11 → 13). The first solve is still mostly compilation, now dominated by
the size of the relaxation jaxpr (`prepare` emits thousands of equations for
an 8-equation `f`). A persistent compilation cache and leaner `prepare`
tracing are the remaining options.

Second pass, on the .nl-read MINLPLib instances (first solve is trace plus
XLA compile; load was heavy, so times are +-50%):

- **Lean .nl tracing.** `ineq` evaluated every constraint and the objective
  and then gathered; zero coefficients became `0.0*x` and sums began with
  `0.0 +`. The reader now traces only the requested rows, drops zero terms
  and caches variable reads.
- **FBBT.** `propagate` used to trace the forward pass three times; it now
  gets the slot structure from `eval_shape` and traces it twice. The meet
  emptiness flags are folded into one reduction. The tape is CSE'd and
  DCE'd (no measurable effect yet).
- **Root OBBT skipped** when the root bound already closes the gap, which
  avoids compiling the second kernel (about 1.5 s) on st_e01 and rbrock.

| | before | after |
|---|---:|---:|
| st_e01, first / repeat | 6.9 / 0.006 s | 1.9-2.3 / 0.006 s |
| rbrock, first / repeat | 4.1 / 0.007 s | 1.1-1.6 / 0.007 s |
| ex4_1_7, first / repeat | 3.9 / 0.010 s | 3.5-3.6 / 0.010 s |
| st_e01 FBBT eqns / all kernels | 11605 / 20.8k | 2549 / 9.2k |

Nodes and bounds are unchanged on the three (1 node each). On a 5-problem
guard (gramacy_lee, six_hump_camel, G08, G24, haverly) nodes and bounds are
the same except haverly 45 -> 43 nodes and a 2e-7 shift in G08's lower
bound. The builtin totals were not re-measured.

Remaining: ex4_1_7 needs its root OBBT, so both kernels (about 0.7 s trace
and 0.7 s compile each) still build; trace is about 45% and compile 55% of
each kernel. Options are a persistent compilation cache, leaner `prepare`
and alphaBB tracing, fewer padded batch sizes, and a lower XLA optimisation
level (tried: small compile gain, about 5x slower solves).

### Scaling (synthetic benchmark)

`jaxglobal/synthetic.py` generates seeded instances of four families
(`univariate_sum`, `qcqp`, `pooling`, `nn_surrogate`); `benchmarks/synthetic.py`
runs each in a fresh process (cold solve, warm repeat, root-only solve) and
writes JSON (`benchmarks/results/synthetic_*.json`). The machine was heavily
loaded (load ~200), so times are indicative only; nodes and gaps are not.
Runs were cut short (limits of 300 s, then abandoned), so large-n rows are
partial. Defaults, 300 s limit:

| instance | n | m | nodes | status | final gap | root gap |
|---|---:|---:|---:|---|---:|---:|
| univariate_sum_5 / 10 / 20 | 5-20 | 0 | 21 / 73 / 511 | optimal | <2e-4 | 7e-3 / 2e-2 / 1e-1 |
| univariate_sum_50 | 50 | 0 | 4197 | optimal | 3e-4 | 0.40 |
| univariate_sum_100 | 100 | 0 | 4827 | time_limit | 0.58 | 0.86 |
| qcqp_5_2 / 10_5 | 5-10 | 2-5 | 15 / 109 | optimal | 0 | 0.7 / 3.7 |
| qcqp_20_10 | 20 | 10 | 16361 | time_limit | 0.03 | 9.3 |
| nn_tanh 2 / 5 inputs (8x2) | 2-5 | 0 | 243 / 883471 | optimal | <2e-6 | 2.4 / 3.1 |
| nn_tanh 10 inputs | 10 | 0 | 764927 | time_limit | 0.12 | 0.13 |
| pooling (binding spec) 2_4_2 | 14 | 14 | 17-31k | time_limit | ~1e-2 (1e-5 rel) | n/a |
| pooling (binding spec) 3_4_3 | 24 | 19 | 10k | time_limit | ~16 | n/a |

(Pooling rows are from 200-300 s runs of the final generator; an earlier
version with non-binding quality specs solved at the root up to n = 24 and
was not kept. `pooling_2_4_2` in the JSON is a 20 s smoke run.)

Single-kernel measurements (batch 16, `make_jaxpr` / compile / call):

| instance | n | jaxpr eqns | trace + compile | bound call, per box |
|---|---:|---:|---:|---:|
| univariate_sum | 5 / 20 / 50 / 100 | 12331 (all) | ~2 s | 0.4 / 1.1 / 3.7 / 8.8 ms |
| qcqp | 5 / 10 / 20 / 50 | 10796 (all) | ~4 s | 0.3 / 1.6 / 8.1 / 72 ms |
| pooling_4_6_4 | 44 | 19697 | ~6 s | 51 ms |
| nn_tanh (8x2) | 2-10 | 6583 | ~2 s | 0.1 ms |

Root OBBT on one box: univariate_sum_100 about 12 s, qcqp_50_25 1.6 s.

What dominates:

- **Not compile time.** The jaxpr is vectorized, so equation count depends on
  the structure of `f`, not on n; trace + compile is a few seconds per kernel
  at every size, and cold - warm is small once solves take minutes. (The
  cold solve of tiny instances is 10-30 s, many kernels and batch shapes.)
- **Search size from weak relaxations first.** nn_tanh needs 883k nodes at n = 5
  although each node costs 0.1 ms: composed tanh layers lose tightness through
  interval propagation, root gap 3.1. qcqp root gaps grow with n (0.7, 3.7,
  9.3) and nodes with them (15, 109, 16k); univariate_sum is separable, yet
  the tree grows 21, 73, 511, 4197 nodes with n because branching one variable
  at a time cannot exploit separability (each term must be refined inside
  every combination of the others' boxes).
- **Per-node cost next.** The dense LP / IPM per node grows roughly as
  n^2.4 on constrained problems (qcqp 8 ms -> 72 ms from n = 20 to 50) and root
  OBBT (2n LPs per round) becomes seconds by n = 100.
- **Upper bound is not the problem.** Best of 30 random-start SLSQP runs
  matched the incumbent on qcqp_20_10 (-21.33481) and nn_tanh 5 / 10
  (-1.7174 vs -1.7174; -1.6583 vs -1.6583), so the gap is on the lower-bound
  side. The first local solve (box center) can be poor (nn_tanh_5 root
  incumbent -0.75 vs -1.72), which only affects early pruning.

Not done: no run past the time limits above; no n = 50 / 100 B&B for qcqp,
pooling or nn_tanh; no memory measurement; no check of relaxation tightness
against exact envelopes. Candidate fixes: separable-aware branching and bounds
(branch where the per-term gap is largest), tighter composition rules for
tanh layers (or an extended-space `lifted` bound), and sparse / structured LP
for constraint-heavy problems.

### Persistent compilation cache

`tests/conftest.py` and `benchmarks/*.py --cache-dir` turn on JAX's persistent
cache (the library does not). `builtin.py --only gramacy_lee,G08,haverly`,
first vs second process (cold): 7.1 / 31.9 / 20.7 s -> 3.4 / 8.5 / 7.1 s.
Hits do occur across processes; what is left is tracing and the
non-compiled Python. Timings are from a loaded machine.

### Setup cost

On larger models the integer stage 5 profile found the search hardly starts
in 60 s: setup takes the time. Two synchronous steps on the main thread were
the largest, on `spectra2` (69 variables) and `syn05m02m` (60): finding the
variables that enter nonlinearly (`nonlinear_variables` compiles a dense
Hessian per function and samples it at 3 points: 33.8 and 5.1 s) and building
the pounce local solver (26.5 and 10.1 s), almost all of it pounce probing the
Jacobian and Lagrangian Hessian sparsity (compiling a dense probe of the
Hessian).

`jaxglobal/structure.py` reads all three off the scalarized tape instead:
each scalar slot carries a bitmask of the variables it depends on, and a
non-affine row marks its variables nonlinear and all their pairs as Hessian
entries. That gives the nonlinear mask, the constraint Jacobian pattern
(exact) and a superset of the Lagrangian Hessian pattern, with no compilation.
On 8 MINLPLib instances both patterns contain every nonzero dense AD finds (0
missing), the nonlinear masks equal Hessian sampling, and the analysis takes
0.07 to 0.85 s. `minimize` uses the mask for branching and passes the patterns
to pounce (`jac_pattern`, `hess_pattern`), which then skips probing: building
the local solver went from 20 to 26 s to 0.2 s on `spectra2` and from 4 to 17
s to 0.1 s on `nous1`, with the same time per solve (alternated A/B). Dense
evaluation is kept: pounce's colored AD (`sparse=True`) did not measure faster
per solve. Tapes that cannot be scalarized (vectorized code) keep the old
paths: `structural_nonlinear` (`univariate._dep_masks`), then Hessian
sampling, and pounce's own probing.

*asdex, evaluated for the vectorized case.* [asdex](https://pypi.org/project/asdex/)
(0.5.2) detects global Jacobian and Hessian sparsity by abstract
interpretation of the jaxpr, which would cover vectorized code too. On the 9
`.nl` models tried it is correct (0 missing) and tighter than
`structure.py` (exact Hessian patterns: 112 against 178 entries on `nous1`, 70
against 99 on `edgecross10-010`), but 1.1 to 3.6 times slower. It did not
help where it was needed: its primitive dispatch is a fixed `match` with no
way to register handlers, and an unknown primitive raises
`NotImplementedError` (a few linear-algebra primitives fall back
conservatively). It fails on `lax.one_minus_square`, which JAX 0.11 uses in
the derivative of `tanh` (so any `tanh` network), and on jaxglobal's own
primitives (`matern52`, `matern52_d1` in `gp_mean`, `gp_std`, EI). So it was not
adopted. It becomes useful here if upstream adds (1) a `one_minus_square`
handler and (2) either a registry for custom primitive handlers or an opt-in
conservative fallback instead of the error. The alternative without it is to
extend `univariate._dep_masks`, which already follows vectorized tapes and
falls back conservatively, to produce the two patterns.

### SCIP gap: root profile

Workstream 0 of `docs/PLAN-scip-gap.md`: where the wall time goes on 7
instances that are stuck or slow against SCIP. The script is
`benchmarks/profile_root.py` and the data is in
`benchmarks/results/scipgap_profile.csv`, one row per measured item.

How it was measured:

- Each instance ran in its own process, one at a time, with a 120 s limit and
  `uptime` load recorded per run.
- Each instance ran in three modes:
  - cold: `JAXGLOBAL_CACHE=0`;
  - prime: an empty kernel cache;
  - warm: the same cache again.
- `minimize` marks each setup and root stage through an opt-in hook,
  `bnb._PROFILE`, which is `None` by default and changes nothing. These marks
  split the wall time.
- Wrappers time the work inside each stage:
  - `bound_domain`, `make_local` and every local solve;
  - the kernel trace and lowering (tracer thread) and the XLA compile
    (compiler pool);
  - each kernel wait and each kernel run.
- Time spent waiting for a kernel is split, in 10 ms samples, by what the
  background threads were doing: trace and lowering, or XLA compile.
- The buckets sum to the wall time within 0.1 s for every run (exit criterion:
  10 %).

**Caveat:** every run here, including the follow-ups and the 4-job run (all finished by 11:18), ran while two unrelated processes pinned 2 CPU cores (until about 11:20). Absolute timings are inflated by an unknown amount; the ratios (compile against first LP, alphaBB on or off) and the bugs found do not depend on it.

**Cold runs** (no cache), 120 s limit, one instance at a time:

| instance | n / int | rows | relaxation; alphaBB interval Hessian | load (1 min) | wall, status, nodes | first bound at | wait: trace+lower / XLA compile | first LP run | largest other bucket |
|---|---|---|---|---|---|---|---|---|---|
| ex6_2_6 | 3 / 0 | 184 | straight-line; jacfwd² (1063 eqns) | 8 | 120 s, limit, 109k | 6.4 s | 2.1 / 5.4 s | 0.1 s | search 111 s |
| ramsey | 33 / 0 | 176 | straight-line; jacfwd² (753) | 10 | 8.9 s, **optimal**, 1 | 6.2 s | 2.0 / 4.5 s | 0.3 s | make_local 0.7 s |
| portfol_buyin | 17 / 8 | 943 | interpreter; taylor2 over budget (0.5M), jacfwd² (3193) | 13 | 14.9 s, **optimal**, 91 | 9.2 s | 2.7 / 3.6 s | 0.2 s | search 2.6 s, bound_domain 2.3 s |
| smallinvDAXr3b010-011 | 31 / 30 | 1583 | interpreter; taylor2 over budget (5.0M), jacfwd² (10600) | 12 | 135 s, limit, 291 | 16.7 s | 4.4 / 5.9 s | 1.8 s | root dives 36 s; search 77 s |
| batch | 46 / 24 | 388 | straight-line; jacfwd² (855) | **124** | 24.5 s, **optimal**, 19 | 13.8 s | 6.7 / 5.4 s | 0.5 s | root local solves 5.0 s |
| hadamard_5 | 26 / 25 | 505 | straight-line; jacfwd² (8344) | **110** | 131 s, limit, 0 | never | 20.6 / ≥ 106 s (compile unfinished) | none | bound_domain 3.6 s |
| cvxnonsep_nsig30 | 30 / 15 | 122 | straight-line; jacfwd² (1242) | 14 | 140 s, limit, 1023 | 38.5 s | 2.3 / 41.5 s | 2.1 s | search 86 s |

Notes on the cold table:

- **rows** is the number of scalarized tape rows.
- **relaxation** is `relax_interp.use_relax_interpreter`: it uses the tape
  interpreter for a function with at least 600 rows. FBBT is interpreted on all
  seven.
- **jacfwd²** means alphaBB builds its interval Hessian by tracing
  `jacfwd(jacfwd(f))` (the eqns count is that jaxpr's size). It does this in
  two cases:
  - always on the straight-line path;
  - on the interpreted path when the `make_taylor2` size is over
    `TAYLOR_BUDGET = 30000`.
- No instance takes the quadratic or lifted path.

**Warm runs** (kernel cache primed, same instance order):

| instance | wall, status | first bound at | XLA compile wait |
|---|---|---|---|
| ex6_2_6 | 120 s, limit (119k nodes) | 0.7 s | 0.1 s |
| ramsey | 2.2 s, optimal | 1.3 s | 0 s |
| portfol_buyin | 7.6 s, optimal | 2.6 s | 0.3 s |
| smallinvDAXr3b010-011 | 263 s, limit (291 nodes) | 8.1 s | 0.3 s |
| batch | 9.2 s, optimal | 2.4 s | 0 s |
| hadamard_5 | 121 s, limit (0 nodes) | never | 118 s |
| cvxnonsep_nsig30 | 134 s, limit (1535 nodes) | 7.6 s | 5.6 s |

The prime runs match cold within the noise, except that the kernel cache's
`jax.export` re-traces in the background. That re-trace took 9 s on
smallinvDAX and 22 s on hadamard_5, overlapping the wait for the bound kernel.

Load:

- Other jobs started during the batch runs, which then ran at load 110-124.
- hadamard_5 cold also ran at 110, prime at 29 and warm at 18. All other runs
  were at 8-19 on this 14-core machine (10 performance, 4 efficiency).
- hadamard_5 does not depend on the load: its result is the same at every
  load, and the follow-up runs below were done at load 8-12.

**Follow-up runs** (cold, in the same CSV with mode `cold_tl900`, `cold+noalphabb`
and `cold+interp`):

- hadamard_5 with a 900 s limit (load 12):
  - The bound kernel's XLA compile takes 273 s and the OBBT kernel's 301 s.
  - The first bound arrives at 286 s.
  - The search then runs 88k nodes in 577 s without closing: bound -7.68
    against -5 (a maximization).
- Is it the relaxation or the alphaBB Hessian? Measured by toggling each:

| instance | default: compile, first bound, nodes | `alphabb=False` | relaxation interpreter on (`MIN_ROWS=0`) |
|---|---|---|---|
| hadamard_5 | 273 s, 286 s, 0 in 120 s | 38.5 s, 48 s, 9215 nodes (bound -14.4) | > 118 s, never: taylor2 is 0.7M, over budget, so still jacfwd² |
| cvxnonsep_nsig30 | 31.6 s, 38.5 s, 1023 nodes (bound 30.5) | 16.9 s, 21.8 s, 2047 nodes (bound 35.1) | 12.0 s, 15.2 s, 7167 nodes (bound 44.8) |
| ex6_2_6 | 3.4 s, 6.4 s, 109k nodes | 2.1 s, 4.5 s, 121k nodes | 1.6 s, 4.2 s, 56k nodes (node rate halved) |

- 4 jobs, 60 s, cold (`benchmarks/results/scipgap_w0_jobs4_cold.csv`, load
  8-17), to compare with the `issue14_*_cold` baseline:
  - ramsey solves in 11.8 s, batch in 15.1 s and portfol_buyin in 20.7 s.
  - In the baseline all three hit the limit after 1 node, and ex6_2_6 stopped
    after 1 node with dual -1098. Here ex6_2_6 reaches dual -8.8e-4.
  - So the baseline was taken on a much more loaded machine. Its setup times
    are inflated, and three of Workstream A's six target instances already
    solve cold on a quiet machine.

Conclusions:

1. **The decision rule: Workstream A stays ahead of D.**
   - The rule was: if compile is under 30 % of wall time everywhere and the
     first LP dominates, D moves ahead of A. Neither condition holds.
   - Kernel compile (trace, lowering and XLA wait) is 30 % or more of the cold
     wall time on 5 of 7 instances:

     | instance | compile share |
     |---|---|
     | ramsey | 73 % |
     | batch | 50 % |
     | portfol_buyin | 42 % |
     | cvxnonsep_nsig30 | 31 % |
     | hadamard_5 | 97 % (100 % of the 286 s before its first bound) |
     | smallinvDAX | 8 % |
     | ex6_2_6 | 6 % |

   - The first bound kernel call (the dense LP) takes 0.1-2.1 s, at most
     1.6 % of the wall time. It never dominates.
2. **The largest bucket is XLA compile of the bound and OBBT kernels.**
   - It totals 172 s of kernel wait over the 7 cold runs, against 41 s for
     trace and lowering.
   - The cost is heavy-tailed. It does not depend on the row count:
     - hadamard_5: 273 s for 505 rows;
     - cvxnonsep_nsig30: 32 s for 122 rows;
     - smallinvDAX: 6 s for 1583 rows (interpreted).
   - The driver is alphaBB's `jacfwd(jacfwd(f))` interval Hessian, compiled
     straight-line. With alphaBB off, hadamard_5's compile drops 7×
     (273 → 38.5 s) and nsig30's 2× (31.6 → 16.9 s).
   - The OBBT kernel (batch 1) compiles as slowly as the bound kernel
     (batch 16): 301 s against 273 s. A smaller batch does not make a kernel
     cheap to compile.
   - The XLA persistent cache writes an executable only when its compile
     finishes. hadamard_5 never finishes within the limit, so its warm runs pay
     the full compile again.
3. **The second largest is root dives on smallinvDAX: 36 s cold, 50 s warm.**
   - This is one call of the dive kernel: 16 boxes, 30 integers fixed one at
     a time, two roundings each, and interpreted FBBT over 1583 rows each time.
   - In the search, a periodic dive costs the same again.
   - Next in size, all much smaller, summed over the 7 cold runs:

     | bucket | total |
     |---|---|
     | root local solves (mostly pounce jit on first use) | 14 s |
     | `bound_domain` | 10 s |
     | `make_local` | 5 s |
     | first LP | 5 s |
     | structure / nonlinear / fingerprint | under 1 s per instance |

4. **The time limit is not enforced inside an iteration or a dive.**
   - `minimize` checks the deadline once per iteration.
   - One iteration evaluates up to `batch = 256` boxes, which is 16 calls of
     the 16-box kernel: about 21 s on nsig30 (1.3 s per call) and 28 s on
     smallinvDAX (1.75 s per call).
   - The dive kernel run has no deadline.
   - Measured overruns: 14-20 s on nsig30, 15 s on smallinvDAX cold, and 143 s
     on smallinvDAX warm, where a periodic dive started at 119 s and ended at
     263 s.
   - This is a bug, and it inflates the SGM time on limit-hit instances.
5. **Two of the seven are not setup problems.**
   - ex6_2_6 has its first bound at 6.4 s and runs 109-121k nodes in 120 s,
     but stops at a gap of 1.1e-5 against an absolute tolerance of 1e-6. Its
     objective is about 0, so this is bound convergence near a zero optimum.
   - cvxnonsep_nsig30's bound kernel is slow per call (1.3 s per 16 boxes for
     122 rows) and its bound weak (30.5 against 130.65). Workstream B (convex
     OA cuts) and the relaxation interpreter matter more there than compile
     time.

What this means for Workstream A's design items:

1. **Cheap root bound first: keep, but drop alphaBB rather than shrink the
   batch.**
   - A batch-1 kernel is not cheaper to compile, as the OBBT kernel shows.
   - First compile a kernel without the alphaBB term and with the relaxation
     interpreter, then swap in the full kernel when it is ready.
   - Measured on first bound: hadamard_5 286 → 48 s, nsig30 38.5 → 15-22 s.
2. **Relaxation-interpreter threshold: keep, but choose on graph size, not
   rows.**
   - On nsig30 the interpreter cuts compile 31.6 → 12.0 s and raises the node
     rate 7×.
   - On ex6_2_6 it halves the node rate, so a lower row threshold alone is
     wrong.
   - It does nothing while alphaBB still traces jacfwd² (hadamard_5).
3. **Kelley as `lax.scan` in the lifted path: drop from A.** None of the seven
   uses the lifted or quadratic path.
4. **Dense Hessian and Hessian fingerprint in setup: drop.**
   - Structure, nonlinear, quadratic and fingerprint take under 1 s on every
     instance; `K.fingerprint` is 0.35 s on hadamard_5.
   - The background `jax.export` re-trace for the kernel cache (9-22 s) is
     worth moving off the tracer thread, but it overlaps a wait rather than
     adding to it.
5. **Local-solver build: lower priority.** `make_local` is at most 2.2 s and the
   root local solves at most 5 s (batch, at load 124).
6. **Cache hit rate: keep as a check.**
   - The warm runs show the cache works: first bound in 0.7-2.6 s instead of
     6-14 s.
   - Two gaps remain:
     - A compile that never finishes is never cached (hadamard_5).
     - nsig30 still waits 5.6 s on compile when warm.
7. **New, highest payoff: the alphaBB interval Hessian.** Stop tracing
   `jacfwd(jacfwd(f))` straight-line for large functions. Options:
   - skip alphaBB above a jaxpr-size × n² budget;
   - an interval Hessian restricted to `structure.py`'s Hessian pattern, so
     taylor2 fits its budget;
   - per-row alphaBB on small rows only;
   - alphaBB in a second kernel added after the first bound.
8. **New: enforce the deadline between kernel calls and around dives.**
   - Check the deadline in `run` between 16-box calls.
   - Skip a dive whose expected time (the last dive's time) exceeds the time
     left.
9. **New: a time budget for dives.**
   - On large integer problems (smallinvDAX), one dive call costs as much as
     20 bound calls.
   - Options: fewer boxes per dive, a cap on integers fixed per call, or
     scheduling by measured cost.

Re-baseline first: the `issue14` cold CSVs were taken under heavy load. Run
start50 and int50 cold again, with load recorded, before using them to measure
A.

#### R0 baseline

R0 of `docs/PLAN-scip-gap-v2.md`: the reference numbers every later item of the
plan is measured against. **Pending**: the machine has stayed at load 10-16
(other projects benchmarking), above the plan's gate of 10, so no run has been
made yet. Everything is scripted:

```
python benchmarks/scipgap_r0.py run       # all of R0, resumable, ~2.5-3 h at low load
python benchmarks/scipgap_r0.py summary   # the tables for this section
```

`run` takes the shared benchmark lock (`mkdir
/private/tmp/claude-501/jaxglobal-bench.lock`) for each step and waits for the
1-min load to fall below 10 (at most 30 min, then it proceeds and flags the step
`gate=timeout` in `scipgap_r0_loads.csv`). Steps:

- start50 and int50 with `benchmarks/run.py --solvers jaxglobal --jobs 4
  --time-limit 60`: cold (`JAXGLOBAL_CACHE=0`), prime (an empty
  `JAXGLOBAL_CACHE_DIR`) and warm (the primed directory), written to
  `benchmarks/results/scipgap_r0_{start50,int50}_{cold,prime,warm}.csv`.
  `run.py` now records the 1-min load at the start of each instance in a `load`
  column (additive; CSVs without the column are appended to with their own
  header, and `report.classify` is unchanged).
- The seven W0 instances, one at a time, `benchmarks/profile_root.py --modes
  cold` at 60 s and 120 s, 3 repeats each, in
  `benchmarks/results/scipgap_r0_seven_cold.csv` (mode `cold_tl60_r1` …
  `cold_tl120_r3`).

`summary` gives solved / wrong / SGM per set and mode beside SCIP on the same
instances (start50: 44 solved, 1 wrong, SGM 1.9 s from `baseline.csv`; int50:
45 solved, 0 wrong, SGM 1.3 s from `int50_scip.csv`), and per seven-instance
run the medians of wall time, overrun past the limit, nodes, primal, dual and
first-bound time, with the load of each repeat.

#### A3 per-node kernel ablation and interpreter thresholds

**Status: tooling and rule in place; thresholds not changed** (no valid timing:
1-min load 10-77 and the shared lock busy for the 2 h measurement window).

- `benchmarks/profile_root.py --ablate` runs the fixed variant list `ABLATE`,
  cold, and prints per (instance, variant) the median over `--repeats` of: node
  bound compile s, kernel run s per call, nodes/s, bound at the limit, status
  and the interpreted relaxer rows, with the 1-min load of each run and its
  gate (`ok`, `timeout` after `--max-wait`, `rose`). Variants: base,
  `+opsrule`, `+noalphabb`, `+nounivariate`, `+nokelley`, `+fbbt1`, `+noobbt`
  (`obbt=False`; the plan's `"none"` is not an accepted value), `+interp` /
  `+straight` (relaxer always / never interpreted), `+fbbtrows` / `+fbbtsteps`
  (`SCHEDULE_MIN_ROWS` 0 / 1e9). It waits for the load without holding the lock,
  then holds the lock for one run at a time.
- `relax_interp.kernel_ops(tape, n) = rows (n + 1) (2n + 1 + n_kelley)`: scalar
  McCormick rules per box of the straight-line node bound (each row with its
  `jacfwd` tangents at each cut point). `fbbt_interp.fbbt_ops(rows, rounds) =
  rows (2 rounds + 1)`.
- `relax_interp.RULE = "ops"` (opt-in) interprets when `rows >= MIN_ROWS` (600)
  **or** `kernel_ops >= MIN_OPS` (candidate 600,000). The rows term stays because
  compile time of the straight-line kernel grows with rows even at small n
  (kriging_peaks-red030: 1,162 rows at n = 2 is only 31k ops). The default
  `RULE = "rows"` is the former rule. `fbbt_interp`'s default op rule makes the
  same choice as the former `rows >= 50` at the default 5 rounds.
- Tests: `tests/test_interp_rule.py`.

Which tapes the candidate flips to the interpreter (from the tape statistics; no
timing needed). Among the 7 W0 instances: hadamard_5 g (503 rows, n 26, 774k)
and batch g (261 rows, n 46, 1.19M). Among 20 sampled start50/int50 instances
(seed 1): tloss g (514 rows, n 48, 2.54M). Elsewhere: st_rv7 g (967k),
pooling_rt2tp g (1.11M), clay0203m, tln5, flay04m, pooling_adhya1stp,
kall_circlespolygons_c1p12, wastewater15m1, nous1. Only the joint
(objective + constraints) tape flips on cvxnonsep_pcon40, launch, meanvarx and
pooling_foulds2tp. Their per-function tapes keep their previous choice, so the
effect is limited to the paths that relax the joint tape. Only these need
node-rate measurement. No tape changes from interpreted to straight-line.

Partial data (cold, one run, `a3_kbench`: `bnb._Kernels` node bound for 16 boxes):

| instance | relaxer | lower s | compile s | run s/call | load before / after |
|---|---|---|---|---|---|
| hadamard_5 | straight-line | 10.5 | 80.5 | 0.134 | 9.6 / 15.4 |

To finish (each run takes the lock; about 1.5 h at load < 10):

```
python benchmarks/profile_root.py --ablate --repeats 3 --time-limit 60 \
    --names cvxnonsep_nsig30,smallinvDAXr3b010-011,ex6_2_6,hadamard_5 \
    --out benchmarks/results/scipgap_a3.csv
python benchmarks/profile_root.py --ablate --variants base,+opsrule --repeats 3 --time-limit 60 \
    --names batch,tloss,st_rv7,pooling_rt2tp,clay0203m,tln5 \
    --out benchmarks/results/scipgap_a3_flips.csv
```

Make `RULE = "ops"` the default only if hadamard_5's McCormick-only kernel
compiles in < 15 s, no flipped instance loses > 10 % node rate or gains > 10 %
cold wall, and bounds at the limit agree within 1e-6; otherwise raise `MIN_OPS`
above the losing instances.

### Phase 3 — search (4 weeks)

- **Branching.** Choose the variable by violation-weighted or pseudocost
  scores, initialized with strong branching (which is cheap batched); split at
  the relaxation solution, not the midpoint.
- **Node selection.** Best-bound with periodic dives.
- **Upper bounds.** Use the batched local NLP solves (pounce, or optax on
  device) from many starts drawn from the relaxation solutions of promising
  boxes, plus a feasibility pump for hard equality-constrained instances.
- **Exit:** performance profile within 2× of Couenne on the ≤ 20-variable
  subset.

**Status (branching, started).** `bnb.py` had opt-in `branch_var=
"pseudocost"` and `"reliability"` (strong branching for variables with
fewer than `_ETA` observations, at most `_M` per box) and
`branch_point="relax"`, never benchmarked. On the 21 built-in problems plus
`univariate_sum_10/20`, `nn_tanh_5x8x2` and `qcqp_10_5` (defaults otherwise,
all certified), node counts relative to the default widest-variable
bisection, as geometric means: `relax` 0.98 and `pseudocost` 0.97 (on
the 18 problems all runs finished), `reliability` 0.90 (25). On 2-6 variables the choice of
variable matters little.

The literature points the same way. Strong branching gives the fewest
nodes but is slow on CPU solvers, because each candidate is a separate LP
(Couenne: fewest nodes on 17 of 33 instances, fastest on 1)
[belotti2009branching, achterberg2005branching]. Gradient-times-width
direction rules beat widest-side in interval B&B, but can starve a
variable [ratz1995selection, csendes1997subdivision]. Splitting a node on
several strong-branched variables at once, building the orthants from the
two halves already bounded, cuts tree depth [langiu2025musebb].

`branch_var="strong"` (new, opt-in) combines these for the batched kernel:

- Every selected parent is split on its best-scored variable (pseudocost
  product score, width before any pseudocost is known), so it never splits
  fewer parents per iteration than `"width"`.
- Further candidates are strong-branched only in the padding slots of the
  16-box calls the iteration makes anyway. Filling the batch to 256 instead
  was 2-3× slower in wall time (univariate_sum_20: 29-39 s vs 15.5 s),
  because extra boxes are extra calls.
- Only variables at least 1/`_BAL` = 1/10 as wide as the widest are
  candidates. Without this, gain-based scores starved one variable of the
  nonsmooth problems: holder_table 191 → 1831 nodes, eggholder 47 → 843.
- Orthant multisection: each parent is split on its `_Q` = 2 best
  candidates (those scoring at least half the best) at once, into up to 4
  children. Each child is the intersection of one tightened half per
  variable, so its bound is the largest of theirs, and nothing is bounded
  twice.

Against the default on the same 25 problems (all certified): nodes 0.80,
iterations 0.82 (fewer or equal on 23; styblinski_tang_3d 5 → 7 and
nn_tanh_5x8x2 +6% are the exceptions). Largest gains: hartmann3 157 → 46
nodes, univariate_sum_20 513 → 201 (97 → 71 iterations), reactor 1053 →
654, hartmann6 1099 → 634. Wall time, warm and with configurations
interleaved on a loaded machine (median of 2), on six problems: 0.84 of
the default, with no problem more than 4% slower. In an earlier version
that filled the whole batch, `_Q` = 3 cut iterations further than `_Q` = 2
(0.59 vs 0.67 of the default on nine deep searches) but raised nodes and
hurt haverly (73 → 124 nodes), so 2 is the setting.

It stays opt-in: making it the default would change every published
result table, and the wall-time evidence is six problems (decision
tracked in issue #11). Not done:
gap-attribution scores (αBB separation α_j w_j²/4, Ratz's
gradient-times-width) to rank candidates instead of pseudocosts, branching
at the incumbent, and a `_LAM` sweep for `branch_point="relax"` (SCIP and
Couenne weight the midpoint more than jaxglobal does). For GP surrogates,
branching is not the limit: notebook 06's GP-mean solves need 1,079 to
91,289 nodes for N = 20 to 400 because the Matérn product form and the
GP standard deviation relax poorly. `jaxglobal/kernels.py` now provides the
kernels as primitives of the squared distance: one convex, nonincreasing
function each, relaxed by the envelopes of [schweidtmann2021deterministic],
with derivative primitives so αBB's Hessian is finite (Matérn 5/2, SE). The
N = 200 Matérn 5/2 solve goes from 23,191 to 2,343 nodes (98 to 19
iterations), matching a hand-written kernel-sum bound prototype (2,335).
`jaxglobal/gp.py` adds `gp_std`, a primitive for the posterior standard
deviation whose rule combines the sum-of-squares form `sf2 - ||Wᵀk||²`
(`K⁻¹ = WWᵀ`) with the bound `|σ(x) − σ(c)| ≤ sd_prior(f(x) − f(c))`, which
holds because the posterior is a GP and does not involve `K⁻¹`. Bounds that
go through `K⁻¹` (centering the quadratic form at the box center, or the
tangent of `√` at a kernel-sum upper bound on the variance) were tried and
were sound but stayed loose: their constants scale with `K⁻¹` (≈ 1/noise).
The chance-constrained example now certifies (2,807 nodes, 11 s; previously
not closed in 120 s); LCB 703 → 333 nodes, EI 1,893 → 1,401, PI 815 → 727.
A non-constant version (the same bound at the box corners, each composed
with the secant of the distance) tightened σ's relaxation by about 20% but
left the solves unchanged, so it was not kept. An oracle that clamps σ to its
sampled range on each box (not rigorous, diagnostic only) showed why: even
near-exact σ bounds would cut EI only 18% and PI 13%. Most of their looseness
was in how EI and PI combine μ and σ. `expected_improvement` and
`probability_of_improvement` are now primitives: EI as a jointly convex,
monotone function (the perspective of `zΦ(z) + φ(z)`) with the concave
envelope over the (μ, σ) box; PI from its monotonicity plus exact `Φ`
envelopes on `z = (t − μ)/σ`. EI 1,401 → 765 nodes, PI 727 → 451 (warm solves
about 2× and 1.4× faster).

Found on the way and fixed: the generic relaxation of `Φ((t − μ)/σ)`
through `erf` put its convex side up to 4e-8 above `Φ` on boxes where σ is
tiny, so `z` spans ~1e3. Cause: the quotient's relaxation was one ulp below
the quotient (ordinary rounding in default mode), and αBB's added slope
`a (hi − lo) / 2`, which `f` itself does not have, amplified it ~10⁵-fold.
`alphabb_rule` now widens its input by a few ulps before composing; the worst
excess is 1e-16 (rounding), and `safe=True` was already exact. Also added: a rule for `erfc`, which
`jax.scipy.stats.norm.cdf` uses on jax 0.11.

### Phase 4 — device-resident B&B (4–6 weeks)

- **Node pool on device.** Fixed-capacity arrays with masks, and the whole
  loop inside `lax.while_loop`. No host sync per iteration.
- **GPU LP.** A first-order solver (PDHG/PDLP-style) for the batched node LPs,
  with the weak-duality certificate making it safe. Keep the IPM for small
  dense cases.
- **Sparse jaxpr relaxations** for the larger instances (segment sums in place
  of dense Jacobians).
- **Multi-GPU:** shard node batches with `jax.sharding`.
- **Exit:** at least 10× node throughput on GPU over CPU on 20–100-variable
  instances; a full solve is `jit`-able and `vmap`-able over parameters.

### Phase 5 — the niche (in parallel with phases 3–4)

- **Neural-network surrogates.** Use bound-propagation relaxations (CROWN-style
  linear bounds) as a primitive rule for `dense → activation` chains, which are
  much tighter than generic McCormick. Benchmark against MAiNGO and OMLT on
  NN-embedded problems.
- **ODE models.** Relaxations through `diffrax` solves (unrolled steps,
  discretize-then-relax).
- **Parametric batches.** Solve 1,000 instances of a problem family in one
  call. No equation-based solver can do this today.
- **Differentiable optimal value.** `jax.grad` of the certified optimum with
  respect to parameters, via the envelope theorem at the solution.
- **Exit:** at least one application paper-quality result where jaxglobal is
  the only tool that certifies the optimum, or is at least 10× faster.

**CROWN bounds (Phase 5).** `crown.py` adds backward linear bound
propagation (CROWN, Zhang et al. 2018 [zhang2018efficient]) as an extra node lower bound,
`minimize(..., crown=True)`; the node bound is the max of it and the existing
one. Every nonlinear tape node gets linear lower/upper relaxations on the
interval of its input (tanh, sigmoid, erf, atan, asinh by convexity region with
tangents from bisection; relu/`max`/`min` with a constant; exp, log, sqrt,
even powers, abs; bilinear x*y by the McCormick plane tighter at the box
center); affine nodes are exact (`vjp`). A backward pass from the output gives
`f >= A.x + c`, bounded in closed form over the box. Intermediate bounds are
CROWN's own (a backward pass from every neuron of a layer, vmapped; `deep`).
An unsupported primitive makes the bound `-inf`, decided at trace time.
`crown="auto"` is on when `f` is fully supported and contains an activation.
The default stays `False`: it is evidence-limited (below), and it adds compile
time and a few percent per node.

| problem | root lb existing | root lb CROWN | nodes off | nodes on |
|---|---|---|---|---|
| tanh_net_2d | -2.80 | -2.09 | 291 | 191 |
| nn_tanh 2x8x2 | -2.74 | -2.12 | 243 | 87 |
| nn_tanh 5x8x2 | -3.86 | -2.75 | 883,471 | 13,411 |
| nn_tanh 10x8x2 | -1.79 | -3.76 | 30 s, gap 0.123 | 30 s, gap 0.123 |
| nn_relu 5x8x2 | -14.62 | -12.88 | 559 | 135 |
| six_hump_camel | -7.02 | -168 (unused) | 25 | 25 |

(Root bounds are of the single root box before OBBT; nodes are full solves
with the other options at their defaults.) On the 5-input net CROWN is worth
65x in nodes. At 10 inputs the existing whole-function alphaBB root bound is
already tighter than CROWN, so CROWN does nothing there: the remaining gap is a
search problem. Polynomial benchmarks are supported but CROWN is far looser than
the existing relaxations (`auto` leaves it off). Not done: alpha-CROWN
(optimized lower slopes), and the affine underestimator's minimizer is not used
as a branching point (it is a box vertex).

### Phase 6 — integers and rigor (later)

- **Integer variables:** branch on integrality alongside spatial branching,
  giving MINLP. Relaxations are unchanged; the work is box rounding, integer
  branching and incumbent search. Detailed plan: *Integers: plan* below.
- **Correctness status (implemented):** `minimize(..., safe=True)` rounds every
  quantity that feeds a lower bound outward by one mechanism at the
  primitive-dispatch level (`relax.apply_rule` widens each primitive's
  `lo`/`hi`/`cv`/`cc` by a few ulps scaled to the operands; sums by `n` times
  the magnitude sum). The same wrapper serves intervals, McCormick, FBBT's
  forward pass, univariate envelopes (grid values are enclosures, hull lines
  lowered), alphaBB (Hessian enclosure, enlarged alpha, shifted values) and the
  lifted rule cuts; LP cut constants, row normalization, the change of
  variables and the weak-duality certificate carry explicit margins; CROWN lines
  carry a heuristic margin. Cuts stay valid because margins only lower a
  minorant's constant. Measured on five built-in problems the node count is
  unchanged and warm solves cost about the same, but tracing and compiling the
  extra ops makes cold solves about 1.7x slower (about 3.7x without the XLA
  disk cache), so it is **opt-in** (`bnb.SAFE_DEFAULT`). Independently of
  `safe`, the incumbent is re-evaluated outside jit in float64 after every
  search (`Result.verified`, `Result.max_violation`; a failed check turns
  `optimal` into `unverified`), and `tests/test_fuzz.py` plus
  `benchmarks/check_wrong.py` compare bounds with dense samples and MINLPLib
  references.
  *Guaranteed (modulo the assumptions):* a returned point is checked to
  `feas_tol` in float64; with `safe`, a bound is not undercut by operation
  rounding in the arithmetic above. *Assumed:* XLA transcendentals are accurate
  to a few ulps (the margin is `SAFE_K` = 4 eps relative, not a proof); autodiff
  subgradients are used only through margins on the constant, so their own
  rounding is covered only by that factor; local NLP solvers are untrusted and
  verified a posteriori. *Not rigorous:* plain-float evaluation of tangent
  lines in CROWN and of grid points where a heuristic margin stands in for
  interval evaluation, FBBT's relative 1e-10 padding, the IPM duals (safe by
  weak duality only up to the certificate's float evaluation), constants of
  the affine composition in the lifted LP, `reduce_ranges`' absolute 1e-9
  slack, and feasibility is to `feas_tol`, not exact. A fully rigorous mode
  still needs verified transcendental enclosures and an exact LP-dual check.
- **Rigorous mode (remaining):** directed rounding in interval and LP-bound computations.
  JAX has no rounding-mode control, so this means outward-rounding by 1 ulp per
  operation (`nextafter`) and a Neumaier–Shcherbina correction of the LP bound.

### Integers: plan

**Goal.** Solve MINLPs of modest size natively: up to about 50 variables, of
which up to about 30 are binary or general integer, including integers that
enter nonlinear terms. MINLPLib has 220 such instances (no semicontinuous or
SOS variables, `.nl` available): 193 with a reference solution closed to 1e-4,
106 convex, 125 with general integers, and 166 with integers inside nonlinear
terms (counts from `instancedata.csv`). Within 100 variables there are 328.

**Approach.** Integrality is a property of boxes, so it fits the existing
spatial B&B. The continuous relaxation of a box is still a valid bound, so
FBBT, the McCormick/alphaBB relaxations, the LP and its certificate, OBBT and
batching are unchanged. Integer logic goes in two places only: host-side
bookkeeping in `bnb.py` (branching, incumbents, termination), and a static
integer mask inside the node kernel (bound rounding), which keeps every
kernel shape fixed. External MILP machinery (HiGHS, SCIP) stays out: HiGHS has
no nonlinear relaxation, and SCIP as the host would bound one node per
callback and lose batching. SCIP as the host is the fallback only if
combinatorial instances (many binaries in big-M constraints) turn out to
matter, since the plan below has no MILP cutting planes.

**Stage 0: stop silent relaxation (done).** `read_nl` parsed integer models
and only counted the integers (`NLModel.n_int`), so `minimize` on a MINLP
`.nl` file solved the continuous relaxation and could report it as `optimal`.
The count also missed integers in nonlinear terms: it summed the linear
binary and integer fields of the header and not `nlvbi`, `nlvci`, `nlvoi`
(MINLPLib `nvs01` and `nvs20` counted 0, `tln2` 2 of 8). `parse_nl` and
`read_nl` now count all five fields and raise `NLError` for integer models
unless called with `relax_integers=True`.

**Stage 1: model, reader and harness (done; items 1 and 3 moved to stage
2).** `read_nl(..., integers="mask")` returns `NLModel.integer`
(`_integer_mask`). It decodes all 220 instances correctly: the counts match
`instancedata.csv`, every masked variable has integral bounds, and every
binary is in [0, 1] except the four that `tanksize` fixes, exactly as its
`.gms` file does. A test checks the decoding against Pyomo's `NLWriter`
(with and without objective-only nonlinear variables). The `.gms` cross-check
of item 2 was not needed: the `.gms` files name variables, the `.nl` files do
not, so there is no mapping to compare. `fetch.py --integer` writes
`instances_int.csv`; `run.py`, `report.py` and `check_wrong.py` read both
reference files (`fetch.references`); `benchmarks/sets/int50.txt` comes from
`make_int50.py`. SCIP is measured through GAMS (`scip-gams`), the only SCIP on
this machine. The `minimize` argument (item 1) and the kernel fingerprint
(item 3) only mean something with integer branching, so they move to stage 2.
The plan as first written:

1. `minimize(..., integer=None)` takes a boolean mask or index array. Integer
   bounds are rounded on entry, `lo = ceil(lo - int_tol)` and
   `hi = floor(hi + int_tol)`, after `bound_domain` for unbounded ones;
   `lo > hi` returns `infeasible`. `_verify` also checks
   `|x_j - round(x_j)| <= int_tol` (`int_tol = 1e-6`), and the returned `x` has
   its integer entries rounded exactly before verification.
2. The `.nl` reader returns the mask. In Gay's variable ordering the integer
   variables are the last ones of each block: the last `nlvbi` of the
   variables nonlinear in both constraints and objectives, the last `nlvci` of
   those nonlinear in constraints only, the last `nlvoi` of those nonlinear in
   the objective only, then the `nbv` linear binaries and the `niv` linear
   integers at the end. A test checks the mask against the `Binary
   Variables` / `Integer Variables` declarations of the `.gms` files for all
   220 instances.
3. The mask goes into the kernel cache key and the disk fingerprint
   (`_Kernels`, `kcache.fingerprint`), since stage 2 compiles it into the
   node kernel.
4. `benchmarks/minlplib/fetch.py` gets a second selection (discrete, at most
   50 variables, `.nl` available, no semicontinuous/SOS) written to
   `instances_int.csv`. `run.py` and `report.py` read it, and `run.py` passes
   the mask. A set file `benchmarks/sets/int50.txt` takes 50 of them,
   stratified by number of integers (1 to 5, 6 to 10, 11 to 20, 21 to 50),
   binary versus general integer, and convex versus not, with at most 3 per
   family.

**Stage 2: correct and complete integer B&B (done).** `minimize(...,
integer=...)` rounds integer bounds on entry, in the node kernel after FBBT,
range reduction and OBBT (`round_box`, `INT_TOL = 1e-6`), splits integer
variables at `floor(v)` / `floor(v) + 1` (fractional integers first in the
`width` rule), evaluates incumbent candidates at the LP point with its integers
rounded, and fix-and-solves them (per-start bounds in every `local.py`
backend; pounce reuses its compiled functions). Integer variables are always
branchable, also where they enter linearly. `_verify` checks integrality.
`bound_domain` takes the mask: the point whose value bounds the objective must
be integral (rounded, then the continuous variables re-solved), since a
relaxation point cut off every integer solution of `gear4` (reported
infeasible). Its local solves also check the deadline inside the SLSQP
callbacks, and running out of time while deriving bounds now reports
`time_limit`. Tests (`tests/test_integer.py`): hand-made pure and mixed
problems, infeasibility by integrality, linear-only integers, enumeration of
every integer assignment on random problems, the 11 tiny MINLPLib instances,
and the `gear4` failure mode.
*Measured* on all 220 instances at 60 s (`results/int220_stage2.csv`; start
load 3.1, end about 100 from the run's own jobs and other work): 81 solved
(80 optimal, `ball_mk3_10` correctly infeasible), **0 wrong**, 104 time
limits, 35 unsupported. All `st_miqp` (5) and 9 / 10 `st_test` solved, but
only 9 / 14 `ex12*` and 15 / 24 `nvs`; `nvs05`, a time limit here, solves alone
in 50 s, so part of the gap is load and the rest is speed, which stages 3 and 4
address. The 35 unsupported get no finite bound for some variable: 25 are
`smallinvDAX*`, whose objective variable is bounded only by an integer
feasible point, and rounding breaks their budget constraint (incumbent search,
stage 4). 13 runs were killed for running past the limit: tracing very large
models (the `eg_*` GP surrogates, 868 KB `.nl` files) in the bound-deriving
local solve cannot be interrupted. The plan as first written:

1. *Rounding in the node kernel.* After FBBT, after the duality-based range
   reduction (`reduce_ranges`) and after OBBT, round the masked bounds inward
   as above, and mark the box empty if `lo > hi`. The mask is static, so this
   is a closure in `node` / `node_obbt` with no shape change. `int_tol` must
   exceed FBBT's relative padding (1e-10) and the 1e-9 slack of
   `reduce_ranges`, so rounding never removes an integer point that floating
   point nudged past a bound.
2. *Integer branching.* For an integer variable with `lo < hi`, split at
   `p = floor(v)` with `v` the relaxation point `xr_j` clipped to
   `[lo, hi - 1]`, or `floor((lo + hi) / 2)` when `xr_j` is integral. The
   children are `[lo, p]` and `[p + 1, hi]`. In the child construction
   (`bnb.py`, after the variable choice) this is `right_lo = pt + 1` for
   integer variables. A fixed variable (`lo == hi`) is never a candidate,
   which already holds (`branchable & (phi > plo)`).
3. *Variable choice.* The first rule: integer variables whose `xr_j` is
   fractional by more than `int_tol`, most fractional first, and spatial
   branching on continuous variables otherwise. The pseudocost and strong
   rules then work unchanged, except that an integer split's gain is
   normalized per unit of fractionality (as in MIP pseudocosts), not per
   squared relative width.
4. *Termination.* A box whose integers are all fixed is a continuous problem,
   and spatial branching continues on its continuous variables. In a pure
   integer problem it is a point, where the relaxations are exact (zero-width
   intervals), so its bound equals `f` there and the box is pruned or gives
   the incumbent.
5. *Incumbents.* The node kernel evaluates `f` and the violation at the LP
   point. Make it evaluate at the point with integer coordinates rounded to
   the nearest integer (simple rounding, batched, free). In `update_incumbent`,
   the best few candidates get a fix-and-solve: integers fixed at the rounded
   values (`lo = hi`) and the local NLP over the continuous variables. This
   needs per-start bounds in `local.py`. The optax backend already takes
   `lo, hi` as arguments (vmap them with `in_axes=0`). The pounce backend
   builds a problem with fixed bounds (`from_jax(..., lb, ub)`); measure
   whether rebuilding per assignment recompiles. If it does, cache by
   assignment, or add per-solve bounds upstream in pounce.
6. *Tests.* Compare with MINLPLib references on tiny instances (at most
   5 variables): `ex1221`, `ex1222`, `ex1226`, `nvs03`, `nvs04`, `nvs06`,
   `st_e13`, `st_e36`, `prob03`, `gbd`, `st_miqp3`. A fuzz test builds random
   small mixed problems, enumerates every integer assignment, solves each
   continuous subproblem with `minimize`, and checks the optimum. Further
   tests cover integrality in `_verify` and an infeasible integer box.

*Exit:* 0 wrong on all 220 instances at 60 s, and the small `ex12xx`, `nvs*`,
`st_miqp*` and `st_test*` instances solved.

**Stage 3: baseline and profile (done).** `int50` at 60 s, 4 jobs, each run
started at load average below 8 (2.4 to 4.4; the runs' own jobs raise it to
about 35):

| run | solved | wrong | unsolved | unsupported | SGM time (s) |
|---|---:|---:|---:|---:|---:|
| jaxglobal cold (`int50_cold.csv`) | 24 | 0 | 22 | 4 | 36.7 |
| jaxglobal warm (`int50_warm.csv`) | 29 | 0 | 17 | 4 | 17.3 |
| SCIP via GAMS (`int50_scip.csv`) | 45 | 0 | 5 | 0 | 1.35 |

jaxglobal solves nothing SCIP does not; it solves 29 of SCIP's 45 (64%), and of
the 14 SCIP-solved instances with at most 5 integers all but `nous1`.
(`run.py` counted GAMS model status 8, "integer solution", as an error; a
solve with integers that ends at the gap limit reports it. Statuses 2 and 8
now count as solved when the gap is closed.)

A profile of the 16 instances SCIP solves and jaxglobal does not (warm cache,
one at a time, load 4.4 to 6.7) sorts them by cause:

* *No incumbent* (7): `cvxnonsep_normcon30` / `40`, `nous1` and `tln5` end
  with no integer-feasible point (up to 8,191 nodes), `tln4`, `gear2` and
  `flay04m` find one only at iteration 8 to 19. Without a cutoff there is no
  pruning, OBBT or range reduction.
* *No feasible point while deriving bounds* (3, unsupported): `clay0203m`,
  `m3`, `smallinvDAXr5b150-165`. The same weakness, earlier.
* *Compile* (3): `autocorr_bern20-10`, `autocorr_bern30-04`, `hadamard_5`
  (dense polynomials): trace and lowering take 20 to 35 s, or the first
  compile more than 55 s, and no bound is computed.
* *Weak bound or slow calls* (4): `ex4` (bound -13 against incumbent 81,838),
  `flay04m`, `smallinvDAXr3b010-011` and `nous1`, with kernel calls of 0.6 to
  6 s: big-M and cardinality structure, where MILP cuts would help.

`gear2` and `meanvarx` solve when run alone (43 s and 12 s): near the limit,
lost to the run's variation. So incumbent search comes first in stage 4,
before the propagation items it was listed after. The plan as first written: Run `int50` with jaxglobal
(cold and warm) and SCIP at 60 s. Profile every unsolved instance the way the
continuous no-bound instances were profiled: no bound yet (compile), no
incumbent, slow kernel calls, or too many nodes. The profile sets the order
of stage 4.

**Stage 4: performance (done; `nous1` waived, issue #12).** In the order the
stage 3 profile set:

* *Incumbent search.* The node kernel evaluates the LP point rounded at five
  thresholds (`ROUNDINGS`: nearest, up, down, 1/4, 3/4), keeping the best
  feasible or least violated one. Fix-and-solve starts from the local solution
  of the continuous relaxation, not the LP point: on `cvxnonsep_normcon30` the
  LP point lies far outside the ball constraint (nine integers at their upper
  bound), and no rounding of it is feasible. Propagation dives
  (`_Kernels.dive`, a batched kernel: fix the most fractional integer to its
  rounding, FBBT, take the other rounding if the box empties) from the root
  and from the best children every few iterations; tln4 and tln5 then have an
  incumbent at the root. With at most `_ENUM` = 32 integer assignments, all are
  fix-and-solved at the root from four starts each (`nous1` reaches its
  optimum 1.567072 this way). `bound_domain` also tries all five roundings.
* *Rounding inside FBBT.* `make_fbbt(..., integer=...)` rounds the integer
  bounds after every round, in the compiled sweep and both interpreter sweeps.
* *Root probing on binaries* (`probe`): one batch of 2 boxes per binary,
  fixing it to 0 and 1; a side that cannot beat the incumbent fixes the
  variable, every variable shrinks to the hull of the two sides, and the root
  bound rises to the larger side minimum. A first version formed the hull with
  inf arithmetic; a side box with an infinite bound made it NaN and `spring`
  was "solved" at 3.0097 against 0.8462 (now a test).
* *Reduced-cost fixing* comes from the stage 2 rounding of the range-reduced
  box.
* *Integer pseudocosts.* The pseudocost rules learn an integer split's gain
  per unit moved (fractional part down, its complement up) instead of per
  squared relative width, and score integer candidates with them. On `nvs05`
  `pseudocost` and `reliability` solve in 6.5 s and about 350 nodes against
  33.7 s and 1,277 nodes for `width`, but over `int50` (warm) `width` stays
  best: 31 solved against 27 (`pseudocost`, 0.94 of the time on the 27
  both solve) and 23 (`reliability`, 3.9 times slower: strong-branching trials
  cost kernel calls). `width`, fractional integers first, remains the default.
* *Plunging* (item 6) is covered by the dives from the best children; it is
  not implemented as a node selection rule.
* *Coefficient tightening* (item 7) is not done: the model is code, so
  tightening a big-M row means rewriting the scalarized tape, a larger change
  than this stage; root probing gets part of its effect.

*Measured* (`int50`, 60 s, runs started at load 2.1 to 4.0):

| run | solved | wrong | of SCIP's 45 | SGM time (s) |
|---|---:|---:|---:|---:|
| stage 3, cold / warm | 24 / 29 | 0 / 0 | 24 / 29 | 36.7 / 17.3 |
| stage 4, cold (`int50_probe_cold.csv`) | 29 | 0 | 29 | 29.2 |
| stage 4, warm (`int50_probe_warm.csv`) | 31 | 0 | 31 (69%) | 21.8 |

The exit criterion asks for half of SCIP's instances (met), 0 wrong (met) and
every SCIP-solved instance with at most 5 integers: all but `nous1`, waived
(user decision) as a continuous-bound limitation. Its
integer part is solved (the root enumeration finds the optimum), but proving
it is a continuous problem: with its two binaries fixed it is a 50-variable
pooling model whose bound calls take 6.5 s and whose relaxation stays at
-0.75 against 1.567 after the first nodes. That is the continuous solver's
bound on pooling (the same weakness that leaves the pooling instances of
`start50` unsolved), not something the items below address; tracked in issue #12, and waived from this exit criterion. The lifted
quadratic LP of issue #14 has since lifted the waiver: `nous1` solves in 36 s
(57.8 s in the four-job benchmark). The remaining
unsolved instances are weak bounds on big-M, cardinality and
convex nonlinear constraints (`ex4`, `flay04m`, `smallinvDAX*`,
`cvxnonsep_*`, `tln*`) and compile time on dense polynomials
(`autocorr_bern*`, `hadamard_5`).

The plan as first written, in the expected order of payoff (the stage 3 profile moves item 4, incumbent search, to the front):

1. *Rounding inside FBBT rounds.* Round the variable slots after each sweep,
   in both the compiled `propagate` and the interpreter, so a fixing
   propagates within the same call. Binaries are often fixed by constraint
   propagation alone.
2. *Root probing.* Bound the `2 nb` boxes with one binary at 0 or 1 in a
   single batched kernel call. Fix a binary when a side is empty or its bound
   reaches the incumbent, and shrink every variable to the hull of the two
   sides. This is one batch, the same mechanism as strong branching; it can
   be repeated in the tree with a budget.
3. *Reduced-cost fixing.* `reduce_ranges` already confines each variable from
   the certificate's reduced costs; with rounding it fixes binaries. This
   needs no new computation.
4. *Incumbent search.* Reserve part of each batch for a dive: depth-first
   children of the most promising box, rounding and fixing as they go, plus
   fix-and-solve from LP points. A feasibility-pump variant (alternate
   rounding and NLP projection) comes later. Pruning, OBBT and range
   reduction all need a cutoff, so an early incumbent matters more here than
   in continuous problems.
5. *Branching.* Integer pseudocosts per unit change, with reliability
   branching through the existing batched strong branching, whose candidates
   fill the padding slots of each call.
6. *Node selection.* Best-bound with periodic plunging.
7. *Linear integer structure.* Big-M constraints give weak LP relaxations.
   Coefficient tightening from FBBT bounds is cheap. Cover and MIR cuts are a
   separate project and are deferred; on instances that need them, SCIP will
   stay ahead.

*Exit:* on `int50`, solve at least half of the instances SCIP solves at 60 s
(warm cache), every SCIP-solved instance with at most 5 integers, and 0
wrong.

**Stage 5: toward 100 to 200 variables (measured; the work it points to is
continuous).** `fetch.py --integer --max-vars 100` gives 328 instances (all
decode; `instances_int.csv` now holds them). jaxglobal at 60 s, cold
(`results/int328_stage5.csv`; started at load 2.9, then 60 to 88 from other
work for most of the run, so the counts are pessimistic): 73 solved, **0
wrong**, 229 time limits, 26 unsupported (no finite bound for an unbounded
variable). By size: 72 of 220 with at most 50 variables (81 in the stage 2
run of the same instances), 1 of 108 with 51 to 100. The SCIP run on the 328
was skipped. A profile of 10 of the 51-100 instances (one per family, cold,
load 63 to 149, so times are inflated) shows the search hardly starts: 0 or 1
nodes in 60 s. The time goes to setup: tracing and lowering the bound and
OBBT kernels (9 to 37 s each) and compiling them (9 to 25 s each), first bound
calls of 7 to 50 s (the dense LP, rows times n squared per iteration),
building the local solver (up to 24 s), the Hessian-sampled
`nonlinear_variables` (18 s on `spectra2`) and deriving bounds (33 s on `m6`,
still unbounded). Where the root is reached the incumbent is good
(`cvxnonsep_normcon30r` ends at -34.55 to -34.23 after one node). So the next
gains at this size are the continuous ones already on this roadmap: the LP
(sparse or first-order, fewer cut points), compile time (one relaxation trace
per cut site), and setup work outside the kernels; issue #12 tracks the
pooling bound. The plan as first written: Re-measure on the 328
instances with at most 100 variables. The known costs grow as in the
continuous case: the dense IPM (rows times n squared per iteration; linear
constraints already contribute one row each), the kernel compile time (one
relaxation trace per cut call site), and host-side Python bookkeeping over
many small boxes.

**Risks.**
- **Incumbents:** finding them is the most likely bottleneck, and the
  continuous solver's local search was not designed for it.
- **Big-M constraints:** without MILP cuts, instances dominated by them will
  lag SCIP.
- **Integer tolerance:** rounding interacts with FBBT padding and the range
  reduction slack. A rounding bug removes the optimum silently, which is why
  the fuzz test enumerates.
- **Variable ordering:** errors in the `.nl` integer mask would mislabel
  variables. The `.gms` cross-check covers them.

**Effort.** Stages 0 to 2 take about 2 weeks and give correct MINLP support.
Stage 3 takes a few days. Stage 4 takes 3 to 5 weeks to reach modest
problems at useful speed, so about 6 to 8 weeks in total.

## Risks

- **JIT compile time** grows with jaxpr size and the number of rule variants.
  Mitigated by the kernel cache and scanned Kelley rounds (see *Compile
  time*); the first solve of a new problem still pays it.
- **Fixed shapes.** Masks and padding waste work when the batch is
  half-empty. Use a few size buckets.
- **The extended formulation on GPU** produces large sparse LPs. First-order
  LP may converge slowly. The safe bound keeps it correct, but bounds may be
  weak.
- **Competing with decades of engineering.** Do not aim to beat BARON on
  GlobalLib broadly. Aim to be competitive there and dominant on code-defined
  and batched problems.

## Order of work and rough effort

Phase 0 → 1 → 2 → 3 are sequential and give the largest gains (about 4–5
months for one person). Phases 4 and 5 can overlap with 3. A methods paper is
realistic after phases 0–2 plus one phase-5 application; a "state of the art"
claim needs phases 0–4.

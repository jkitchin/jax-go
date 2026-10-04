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
(defaults, same tolerances as `examples/benchmark.py` and the tests):

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

### Phase 6 — integers and rigor (later)

- **Integer variables:** branch on integrality alongside spatial branching,
  giving MINLP. Mostly B&B bookkeeping; relaxations are unchanged.
- **Rigorous mode:** directed rounding in interval and LP-bound computations.
  JAX has no rounding-mode control, so this means outward-rounding by 1 ulp per
  operation (`nextafter`) and a Neumaier–Shcherbina correction of the LP bound.

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

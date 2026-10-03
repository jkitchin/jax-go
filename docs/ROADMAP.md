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
| bound tightening | interval infeasibility only | FBBT every node; OBBT at the root and selectively in the tree; duality-based (marginals) range reduction; probing |
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
  Mitigation: cache compiled kernels per problem structure, and use
  `jax.lax.scan` in place of unrolled loops.
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

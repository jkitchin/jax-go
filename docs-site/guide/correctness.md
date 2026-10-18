# Correctness and limitations

## Correctness checks

`minimize(..., safe=True)` widens every relaxation outward by a few ulps so
floating-point rounding cannot cut off the global minimum (off by default, about
1.7× slower). Every result is re-evaluated in float64; a point that violates the
constraints is reported with status `"unverified"` and its `max_violation`.
`tests/test_fuzz.py` checks bounds against sampled points on random problems
(`JAXGLOBAL_FUZZ=N` for more cases), and `benchmarks/check_wrong.py` exits
nonzero on any wrong or unverified MINLPLib answer.

## Lessons and gotchas

* **Autodiff subgradients need tie-aware rules.** At points where cv == cc (box
  vertices, which is exactly where Frank–Wolfe iterates go), `jnp.minimum`
  splits the gradient 50/50 between its arguments. In the product rule and in
  the composition `mid(·)` this blends the *concave* relaxation's gradient into
  a "subgradient" of the convex one. The result is invalid cuts and a wrong
  certificate (Eggholder reported −935 as optimal). The fix is to select
  branches with `jnp.where` on sign or ordering. `test_cuts_valid` guards this
  and fails on 9 problems if the fix is reverted.
* **A cut must use one evaluation of its value and slope.** Under `jit`, XLA
  may recompute a value separately for each of its uses, and at a near-tie
  between two McCormick facets the copies can pick different facets. A cut that
  takes its coefficients from one facet and its offset from the other is
  invalid; on jax 0.11 this made the lifted LP infeasible and its bound ~1e277.
  Values and slopes of every cut go through `jax.lax.optimization_barrier`.
* **αBB amplifies rounding.** `f + a/2 (z − lo)(z − hi)` has a slope of about
  `a (hi − lo) / 2` that `f` does not, so on a wide range (a quotient with a
  tiny denominator) a one-ulp error in `z`'s relaxation became 4e-8 in the
  bound. `alphabb_rule` widens its input by a few ulps first.

## Limitations

* **Scale.** On 328 mixed-integer MINLPLib instances with up to 100
  variables (60 s, cold), jaxglobal solved 72 of the 220 with at most 50
  variables and 1 of the 108 with 51 to 100. At that size the time goes to
  setup (tracing, compiling, the first dense LP) before the search starts. The
  node LP is a dense interior point method whose cost grows as rows × n² per
  iteration. Only quadratic problems get a sparse lifted LP.
* **Speed.** Shifted geometric mean time at 60 s, cold: 32.6 s against
  SCIP's 1.9 s on `start50`, and 35.9 s against 1.35 s on `int50`. Much of
  jaxglobal's cold time goes to tracing and compiling the bounding kernels. The kernels are cached in memory and on disk
  under `~/.cache/jaxglobal` (`$JAXGLOBAL_CACHE_DIR`; `JAXGLOBAL_CACHE=0`
  turns the cache off), so a repeat solve of the same model takes about
  0.3-1 s on the built-in problems instead of 2-7 s.
* **Relaxations.** The relaxations are reduced-space McCormick and αBB,
  univariate envelopes, scaled-Gershgorin αBB and interval arithmetic. They
  are weak on big-M, cardinality and some bilinear structure. There are no
  MILP cuts, no trilinear envelopes and no probing on continuous variables.
  FBBT's tightened intermediate ranges are not yet fed back into the
  relaxations. The general lifted LP (`lifted=True`) is opt-in and slower.
* **Floating point.** The default mode is not rigorous. `safe=True` rounds
  lower bounds outward by a few ulps per operation, but it assumes XLA
  transcendentals are accurate to a few ulps and does not cover every margin. Feasibility holds to `feas_tol`, not exactly.
* **Supported JAX.** `dot_general` between two variable-dependent arrays is
  supported only for 1-D inner products. `lax.cond`, `while_loop` and `scan`
  with variable-dependent control flow are not supported; Python control flow
  on static values is fine. An unsupported primitive raises
  `NotImplementedError` naming the primitive.
* **Unbounded variables.** Finite bounds are derived first
  (`jaxglobal/domain.py`): FBBT from the declared box, unused variables fixed,
  then FBBT against `f <= UB` once a local solve finds a feasible point. A
  variable that nothing bounds gives status `"unbounded_domain"` rather than a
  guessed box. FBBT cannot prove bounds that need joint reasoning over several
  constraints.
* **Hardware.** Development and benchmarks are on CPU. The bounding step is a
  pure `jit(vmap)` kernel, but the branch-and-bound loop runs on the host, and
  GPU throughput has not been measured.

## Next steps

Planned work:

* a sparse or first-order LP for the node bound beyond quadratic problems;
* lower setup and compile cost for models with 50 to 100 variables;
* tighter relaxations: FBBT ranges fed into McCormick, and more structure
  detected from the jaxpr;
* a device-resident B&B loop for GPU;
* a fully rigorous mode with verified transcendental enclosures and an exact
  LP-dual check.

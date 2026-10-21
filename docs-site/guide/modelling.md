# Modelling

A jaxglobal model is plain Python. There is no modelling language: the
objective and the constraints are functions of one 1-D array `x`, written with
`jax.numpy`, and the search space is a box `lo <= x <= hi`.

```python
minimize(f, lo, hi, ineq=g, eq=h, integer=[...])
```

## Writing the objective

`f(x)` must be traceable by `jax.make_jaxpr` and return a scalar. Inside it you
can use:

* arithmetic, `jnp.exp`, `log`, `sqrt`, powers, `abs`, trigonometric and
  hyperbolic functions, `erf`, `jax.nn.*` activations, and any other smooth
  primitive with a known range (its relaxation is derived automatically by
  αBB from interval bounds of its second derivative);
* `jnp.where`, `jnp.maximum`/`minimum`, `clip`, comparisons: these are
  resolved on boxes where the interval decides the branch, and relaxed by the
  hull otherwise;
* array operations: indexing, slicing, reshape, broadcast, concatenate,
  reductions, `dot`/`matmul`/`einsum` with constant matrices, and 1-D inner
  products between two variable-dependent arrays;
* Python loops and helper functions, and `jit`/`custom_jvp` sub-calls:
  tracing removes them;
* `jax.lax.scan`, and `jax.lax.fori_loop` with Python `int` bounds (it lowers
  to a scan): the bounding interpreters unroll the loop (at most 10,000
  iterations per scan), so trace and compile time grow with the trip count,
  as with a Python loop; `lax.cond`/`switch` on a constant index.

Not supported: `lax.while_loop` (and `fori_loop` with traced bounds, which
lowers to one), `lax.cond` whose predicate depends on `x` (use `jnp.where`),
and `dot_general` between two variable-dependent arrays beyond 1-D inner
products. An unsupported primitive raises `NotImplementedError` naming the
primitive. Python control flow on static values is fine.

```python
import jax
import jax.numpy as jnp

W1, b1, W2, b2 = ...                     # trained weights (constants)

def surrogate(x):                        # a tanh MLP is just a function
    return (W2 @ jnp.tanh(W1 @ x + b1) + b2)[0]

def cost(x):
    friction = jnp.where(x[0] < 2300.0, 64.0 / x[0], 0.316 * x[0] ** -0.25)
    return friction * x[1] ** 2 + surrogate(x)
```

## Bounds

`lo` and `hi` are sequences or arrays of length `n`. Bounds may be infinite:
finite ones are derived first (`jaxglobal/domain.py`) by FBBT from the
declared box, fixing unused variables, and then FBBT against `f <= UB` once a
local solve finds a feasible point. A variable that nothing bounds gives status
`"unbounded_domain"` (with `message` naming the variables) rather than a
guessed box. FBBT cannot prove bounds that need joint reasoning over several
constraints, so give finite bounds whenever you know them: tighter boxes give
tighter relaxations.

## Constraints

`minimize(f, lo, hi, ineq=g, eq=h)` handles `g(x) <= 0` and `h(x) == 0`, where
`g` and `h` are plain JAX functions returning a scalar or a 1-D array.

* **Lower bound** (`jaxglobal/lp.py`): the relaxations give convex functions
  `cv_f`, `cv_g`, `cv_h`, `-cc_h`. `jax.jacfwd` turns them into affine cuts at
  the box center, 2n axis points, and 4 Kelley points. The small LP is solved by
  a primal–dual interior point method in JAX (Mehrotra predictor–corrector
  steps in a `while_loop` that stops early once converged), so it `vmap`s over
  boxes. The bound **does not trust the LP solver**. Weak duality gives a valid
  bound for *any* multipliers y ≥ 0, and the best value over all IPM iterates is
  kept. A badly converged LP gives a weaker bound, never a wrong one.
  Interval arithmetic prunes boxes that are provably infeasible.
  Problems whose functions are all quadratic, or polynomial up to degree 6, are
  bounded instead over a lifted sparse LP with one column per product,
  McCormick facets and RLT rows (`jaxglobal/quadratic.py`, `quadratic=True` by
  default). With binary variables the lifted LP also gets cuts on products of
  binaries (static rows plus a separation pool) and bound-factor RLT rows.
  Functions that are structurally convex on the box (`jaxglobal/convexity.py`)
  are cut by themselves, which is outer approximation.
* **Upper bound**: a local NLP solve with JAX gradients and Jacobians, started
  from the LP solutions: pounce (default when installed), optax or scipy
  SLSQP (`local_solver=`). A point counts as feasible when the violation is
  ≤ `feas_tol`.
* **Branching** only on variables that appear nonlinearly, read off the jaxpr
  (sampled Hessians as a fallback).
* `status` is `"optimal"`, `"infeasible"`, `"max_iter"`, `"time_limit"`,
  `"unbounded_domain"`, `"unverified"` or `"unproven"` (see [Correctness](correctness.md#correctness-checks)).

| problem | n | f\* | found | proven LB | nodes | time |
|---|---|---|---|---|---|---|
| G06 | 2 | −6961.813876 | −6961.813876 | −6961.817327 | 11 | 18.0s |
| G08 | 2 | −0.095825 | −0.095825 | −0.095830 | 15 | 28.2s |
| G11 (equality) | 2 | 0.750000 | 0.750000 | 0.749999 | 7 | 6.2s |
| G24 (disconnected feasible set) | 2 | −5.508013 | −5.508013 | −5.508016 | 43 | 8.7s |
| G04 | 5 | −30665.538672 | −30665.538672 | −30665.540938 | 19 | 28.3s |
| Haverly pooling | 9 | −400.000000 | −400.000000 | −400.000076 | 17 | 11.3s |
| Reactor network | 6 | −0.388811 | −0.388811 | −0.388821 | 159 | 19.7s |

Equality optima can sit slightly below f\* because of `feas_tol` (1e-6).

## Integer variables

`minimize(f, lo, hi, ineq=g, integer=[0, 3])` (indices or a boolean mask)
makes those variables integer. Their bounds are rounded inward, FBBT rounds
their ranges, branching splits them at integer values, and integer dives and
root probing look for incumbents. The returned `x` is integral, and
`Result.verified` checks that. The node bounds are the relaxations of the
continuous problem; only the lifted LP of polynomial problems adds cuts on
products of binaries. Integer polynomial problems also get symmetry
(lex-leader) pruning (`jaxglobal/symmetry.py`), and an unconstrained binary
polynomial of small induced width is solved exactly by variable elimination
(`jaxglobal/pseudoboolean.py`) with no search. `.nl` files with integer
variables are read with `read_nl(path, integers="mask")`, which puts the mask
in `m.integer`; the default raises `NLError` rather than silently solve the
continuous relaxation.

## Gaussian-process surrogates

A GP prediction written in `jax.numpy` is already a reduced-space model
([Schweidtmann et al. 2021](../references.md#ref-schweidtmann2021deterministic)). Write its kernel with `jaxglobal.kernels`
(`matern12`, `matern32`, `matern52`, `se`, each a function of the weighted
squared distance `d`) instead of the `jnp` formula:

```python
from jaxglobal import kernels

def gp_mean(x):
    d = jnp.sum(((x - X_train) / lengthscales) ** 2, axis=-1)
    return y_mean + alpha @ kernels.matern52(d)
```

Each kernel is a JAX primitive that is one convex, nonincreasing function of
`d`, so it is relaxed by its envelopes; written in `jnp`, the Matérn kernels
are relaxed factor by factor. Its derivatives are primitives too, so αBB gets
a finite Hessian at the training points (Matérn 5/2 and SE). On the GP mean of
the peaks function (Matérn 5/2, abs/rel gap 1e-3), same answers:

| training points | nodes, `jnp` formula | nodes, `kernels.matern52` | time |
|---:|---:|---:|---|
| 20 | 1079\* | 81 | — |
| 50 | 3477 | 411 | 13.8 s → 1.2 s |
| 100 | 6165\* | 647 | — |
| 200 | 22449 | 2253 | 56.4 s → 11.9 s |

Nodes and times are the stored outputs of notebook 06, re-run on 2026-10-20 on
current `main` (after PRs #76, #77 and #81) on a 4-core cloud VM (one session;
the `kernels` solve at N = 50 reuses kernels compiled earlier in it). \*Not
reproduced by the stored notebook: from runs on or before 2026-10-17, as is
the `jnp` count for Matérn 3/2 at N = 50, 1917 nodes (429 with
`kernels.matern32`, as in the notebook).
SE and Matérn 1/2 are unchanged: their `jnp` formulas were already relaxed
exactly.

The posterior standard deviation needs more: it is a quadratic form in the
kernel vector, and bounds through `K⁻¹` inherit its conditioning.
`jaxglobal.gp.gp_std(x, Z, lengthscale, A, sf2, kernel=...)`, with `A = K⁻¹`, has the same
value, and its relaxation uses the GP's structure: the quadratic form as a sum
of squares, and the bound `|σ(x) − σ(c)| ≤ sd_prior(f(x) − f(c))`, which holds
because the posterior is itself a Gaussian process and does not involve
`K⁻¹`. On a 95% chance constraint (`mean + 1.96 std ≤ c`) the solve is now
certified in 2759 nodes and 31.0 s (notebook 06's stored output), where the
formula did not close in 120 s (140k nodes; a run on or before 2026-10-17,
not reproduced by the stored notebook).

`jaxglobal.gp.gp_mean(x, Z, lengthscale, alpha, sf2, kernel=...)` is the
matching mean, `k(x)ᵀα`. Relaxed term by term, the signed sum loses the
cancellation between its terms. For Matérn 5/2 and SE its relaxation is also
intersected with a first-order Taylor model around the box center, whose
remainder is bounded by the mean's RKHS norm:
`|μ(x) − μ(c) − ∇μ(c)·δ| ≤ ‖μ‖_H e(δ)`, with `e` in closed form and of order
`|δ|²`. On notebook 06's problems it gives the same answers in fewer nodes:
2253 → 497 for the N = 200 GP mean, 2759 → 1359 for the chance constraint,
683 → 559 for EI. A few cases are unchanged or within 3%
([`benchmarks/results/gp_mean_ab.csv`](https://github.com/jkitchin/jax-go/blob/main/benchmarks/results/gp_mean_ab.csv)).

The acquisition functions are primitives too:
`expected_improvement(mu, sigma, fmin)` is relaxed as the jointly convex,
monotone function it is (the perspective of `zΦ(z) + φ(z)`), with the concave
envelope over the `(mu, sigma)` box, and
`probability_of_improvement(mu, sigma, target)` from its monotonicity plus the
exact envelopes of `Φ`, instead of `Φ`, `φ` and the division one by one. On
the 20-point GP of notebook 06, with `gp_std`: EI 1401 → 683 nodes, PI 727 →
491. The 683 and 491 are in the notebook's stored outputs; the formula counts
(1401, 727) and the warm times measured with them (EI 0.9 s → 0.4 s, PI
0.55 s → 0.4 s) are from runs on or before 2026-10-17 and are not reproduced
by the stored notebook. [Notebook 06](../notebooks/06-gaussian-processes.ipynb)
repeats the paper's case studies: GP means, acquisition functions, a chance
constraint and a Bayesian-optimization loop.


## Reading AMPL `.nl` files

Any AMPL `.nl` model, text or binary, as written by AMPL, Pyomo, GAMS (convert)
or downloaded from MINLPLib, can be loaded with `jaxglobal.nl.read_nl(path)`.
It returns an `NLModel` with `f`, `ineq`, `eq`, `lo`, `hi` in the form
`minimize` takes. The expression trees are compiled to plain `jax.numpy` code,
so the solver sees an ordinary traced function. Integer variables are read as a
mask with `integers="mask"` (the default raises an error on integer models;
`integers="relax"` reads the continuous relaxation). Imported functions,
suffixes, logical constraints and complementarity are not supported.

The reverse direction, writing a JAX model to `.nl` or to SCIP, is covered in
[Export to SCIP and .nl](export.md).

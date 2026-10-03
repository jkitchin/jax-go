# jaxglobal — global optimization of plain JAX functions (proof of concept)

Write an ordinary Python function with `jax.numpy`, give a box, and get a
**certified** global minimum: an upper bound (a feasible point) and a proven
lower bound. You write no equations and no algebraic model.

```python
import jax.numpy as jnp
from jaxglobal import minimize

def f(x):                       # Eggholder: very rugged, optimum on the boundary
    a, b = x[0], x[1]
    return -(b + 47) * jnp.sin(jnp.sqrt(jnp.abs(a / 2 + b + 47))) \
           - a * jnp.sin(jnp.sqrt(jnp.abs(a - (b + 47))))

r = minimize(f, [-512, -512], [512, 512])
# Result(fun=-959.64066, lower_bound=-959.64144, x=[512. 404.231805], converged=True, nodes=153, ...)
```

## How it works

1. **Trace.** `jax.make_jaxpr(f)` turns the Python function into a flat list of
   primitives (`sin`, `mul`, `integer_pow`, `dot_general`, `slice`, …).
   Python loops, helper functions, `jnp.where`, `jax.nn.*`, and `jit`/`custom_jvp`
   sub-calls all disappear into this list.
2. **Re-interpret the jaxpr on McCormick objects** `MC(lo, hi, cv, cc)` in
   `jaxglobal/relax.py`. Every primitive has a rule:
   * *structural / positive-linear* primitives (slice, reshape, add, reduce_sum,
     concatenate, gather, …) apply component-wise;
   * *signed linear* (neg, sub, mul or `dot_general` by a constant) swap by sign;
   * *bilinear* `mul` uses the McCormick product rule;
   * *univariate* primitives use McCormick composition with known envelopes
     (exp, log, sqrt, xⁿ, |x|, 1/x, pow, cosh, max/min) **or an automatically
     derived αBB relaxation**. For sin, cos, tanh, logistic, erf, atan, …, the
     interpreter traces `jax.grad(jax.grad(prim))`, evaluates *that* jaxpr in
     interval mode to bound f″, and builds `f(z) ± α/2 (z−lo)(z−hi)`.
     Adding a new smooth primitive needs only its range function.
   * comparisons / `select_n` (from `jnp.where`) are resolved when the interval
     decides the branch and fall back to the hull otherwise.
3. **The relaxation is itself a JAX function**, so it can be `jit`-ed,
   `vmap`-ed over hundreds of boxes, and differentiated. `jax.grad(cv)` gives
   subgradients of the convex relaxation. A few Frank–Wolfe steps on cv give
   affine cuts `cv(x) + g·(v − x)`, whose minimum over the box is a valid lower
   bound (`jaxglobal/bnb.py`).
4. **Batched spatial branch-and-bound.** Each iteration bisects the 128
   lowest-bound boxes and bounds all 256 children in one `jit(vmap(...))` call.
   L-BFGS-B (with `jax.grad`) polishes the incumbent.

![relaxations](figures/relaxations.png)

## Results (`python examples/benchmark.py`)

Tolerance: gap ≤ max(1e-5, 1e-6·|f|). *Root LB* is plain interval arithmetic
on the whole box. *mstart* is the fraction of 20 random L-BFGS-B starts that
reach f\*, which shows how multimodal each problem is.

| problem | n | f\* (lit.) | found | proven LB | root LB | mstart | nodes | time |
|---|---|---|---|---|---|---|---|---|
| gramacy_lee | 1 | −0.869011 | −0.869011 | −0.869016 | −1 | 15% | 23 | 0.4s |
| six_hump_camel | 2 | −1.031628 | −1.031628 | −1.031638 | −156 | 95% | 407 | 0.4s |
| mccormick | 2 | −1.913223 | −1.913223 | −1.913228 | −13.5 | 55% | 41 | 0.2s |
| goldstein_price | 2 | 3 | 3.000000 | 2.999990 | −8.8e7 | 60% | 8883 | 0.6s |
| shubert | 2 | −186.7309 | −186.7309 | −186.7311 | −225 | 0% | 2399 | 0.5s |
| michalewicz_2d | 2 | −1.801303 | −1.801303 | −1.801308 | −2 | 5% | 65 | 0.5s |
| holder_table | 2 | −19.20850 | −19.20850 | −19.20851 | −33.2 | 0% | 415 | 0.7s |
| eggholder | 2 | −959.6407 | −959.6407 | −959.6414 | −1070 | 0% | 153 | 0.5s |
| hartmann3 | 3 | −3.862782 | −3.862780 | −3.862790 | −8.4 | 50% | 1769 | 0.3s |
| styblinski_tang_3d | 3 | −117.4985 | −117.4985 | −117.4986 | −638 | 15% | 451 | 0.2s |
| shekel10_4d | 4 | −10.536443† | −10.536443 | −10.536452 | −35.4 | 25% | 133 | 0.5s |
| hartmann6 | 6 | −3.322368 | −3.322368 | −3.322378 | −8.4 | 55% | 6515 | 0.3s |
| piecewise_2d (where/abs/max) | 2 | — | −0.823199 | −0.823201 | −3.7 | 25% | 87 | 0.5s |
| tanh_net_2d (16-unit MLP) | 2 | — | −0.553102 | −0.553111 | −2.8 | 30% | 387 | 0.3s |

† The usual −10.5364 is f(4,4,4,4); the true minimizer is slightly off that point.
Times are wall-clock on a laptop CPU and include JIT compilation. For the two
problems without a published optimum, the tests check the certificate against
a 401×401 brute-force grid.

![convergence](figures/convergence.png)

Excluded on purpose: sum-of-squares problems with f\* = 0 (Rosenbrock,
Himmelblau, Rastrigin, Ackley, Levy) and Branin. Plain interval arithmetic
already proves their optimum at the root node, so they show nothing about the
relaxations.

## Constraints

`minimize(f, lo, hi, ineq=g, eq=h)` handles `g(x) <= 0` and `h(x) == 0`, where
`g` and `h` are plain JAX functions returning a scalar or a 1-D array.

* **Lower bound** (`jaxglobal/lp.py`): the relaxations give convex functions
  `cv_f`, `cv_g`, `cv_h`, `-cc_h`. `jax.jacfwd` turns them into affine cuts at
  the box center, 2n axis points, and 4 Kelley points. The small LP is solved by
  a fixed-iteration primal–dual interior point method in JAX, so it `vmap`s over
  boxes. The bound **does not trust the LP solver**. Weak duality gives a valid
  bound for *any* multipliers y ≥ 0, and the best value over all IPM iterates is
  kept. A badly converged LP gives a weaker bound, never a wrong one.
  Interval arithmetic prunes boxes that are provably infeasible.
* **Upper bound**: scipy SLSQP with JAX gradients and Jacobians, started from
  the LP solutions. A point counts as feasible when the violation is ≤ `feas_tol`.
* **Branching** only on variables that appear nonlinearly, detected from
  sampled Hessians.
* `status` is `"optimal"`, `"infeasible"` or `"max_iter"`.

| problem | n | f\* | found | proven LB | nodes | time |
|---|---|---|---|---|---|---|
| G06 | 2 | −6961.81388 | −6961.81388 | −6961.8200 | 61 | 2.2s |
| G08 | 2 | −0.0958250 | −0.0958250 | −0.0958257 | 99 | 3.8s |
| G11 (equality) | 2 | 0.75 | 0.7499996 | 0.7499989 | 127 | 1.3s |
| G24 (disconnected feasible set) | 2 | −5.508013 | −5.508013 | −5.508016 | 43 | 1.7s |
| G04 | 5 | −30665.539 | −30665.539 | −30665.552 | 67 | 5.8s |
| Haverly pooling | 9 | −400 | −400 | −400.0004 | 157 | 20s |
| Reactor network | 6 | −0.388811 | −0.3888114 | −0.3888124 | 1935 | 12.7s |

Equality optima can sit slightly below f\* because of `feas_tol` (1e-6).

## Comparison with equation-based solvers

`examples/compare.py` builds Pyomo models from **the same Python functions**,
using a `jnp`→Pyomo shim on numpy object arrays. It then runs SCIP and Couenne
(via `.nl` files) and BARON and LindoGlobal (via GAMS). `where`/`maximum` and
`tanh` had to be reformulated by hand. BARON rejects every model containing
`sin`/`cos` ("Cannot handle function 'sin'"). Results go to
`examples/compare_results.json`, and `examples/compare_plot.py` draws
`figures/benchmark.png`. The slides in `slides/jaxglobal.md` summarize this.

## Run

```bash
pip install -e '.[dev]'
pytest -q                      # soundness, cut validity, global optima (57 tests, ~30 s)
python examples/benchmark.py
python examples/figures.py
```

## Lessons / gotchas found so far

* **Autodiff subgradients need tie-aware rules.** At points where cv == cc (box
  vertices, which is exactly where Frank–Wolfe iterates go), `jnp.minimum`
  splits the gradient 50/50 between its arguments. In the product rule and in
  the composition `mid(·)` this blends the *concave* relaxation's gradient into
  a "subgradient" of the convex one. The result is invalid cuts and a wrong
  certificate (Eggholder reported −935 as optimal). The fix is to select
  branches with `jnp.where` on sign or ordering. `test_cuts_valid` guards this
  and fails on 9 problems if the fix is reverted.

## Limitations (it's a PoC)

* Not rigorous in floating point: no outward rounding, and αBB minimizers come
  from 60-step bisection.
* Constrained bounds use a dense LP per node and SLSQP upper bounds: fine for
  ≤ 10 variables, not beyond.
* `dot_general` between two variable arrays only for 1-D inner products.
  `lax.cond`/`while`/`scan` with variable-dependent control flow are not
  supported; Python control flow on static values is fine.
* Unsupported primitives raise `NotImplementedError` naming the primitive.
* Bounds are first-order McCormick/αBB plus interval arithmetic. There is no
  bound tightening, no reduced-cost or optimality-based tightening, and no LP
  over multiple cuts, so node counts are far from the state of the art.

## Natural next steps

* Harder constrained sets (larger pooling problems, MINLPLib `ex8_*`).
* Feasibility-based bound tightening by backward propagation over the jaxpr.
* Multivariate McCormick / tighter envelopes for common patterns
  (x·exp(x), x·log(x), bilinear `dot_general`).
* Outward rounding (e.g. `nextafter`) for rigorous bounds.
* GPU: the bounding step is already a pure `jit(vmap)` kernel.

<p align="center"><img src="assets/figures/logo.svg" alt="jaxglobal" width="560"></p>

# jaxglobal

**Global optimization of plain JAX functions.**

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
# Result(status='optimal', fun=-959.64066, lower_bound=-959.64066, gap=2.06e-07,
#        x=[512. 404.23181], nodes=47, iterations=10, time=4.82s)
```

## Status

jaxglobal is past the proof-of-concept stage. It is a research solver: usable,
tested, and measured on MINLPLib, but not yet competitive with mature global
solvers on general problems (see [Correctness and limitations](guide/correctness.md#limitations)).

* **Problems:** continuous and mixed-integer, with inequality and equality
  constraints, on finite or partly unbounded boxes. Models can be plain JAX
  functions or AMPL `.nl` files (`jaxglobal.nl.read_nl`).
* **Correct answers so far:** 0 wrong answers on the MINLPLib sets below, all
  checked against the library's reference bounds. Every returned point is
  re-checked in float64, and an opt-in `safe=True` mode rounds the bounds
  outward.
* **MINLPLib, 60 s limit, cold start (compilation included):**

  | set | jaxglobal | SCIP |
  |---|---:|---:|
  | `start50`: 50 continuous instances, ≤ 50 variables, weighted toward hard ones | 31 | 44 |
  | `int50`: 50 mixed-integer instances, ≤ 50 variables | 29 | 45 |

  jaxglobal solves no `int50` instance that SCIP does not. Details:
  [Benchmarks and results](guide/benchmarks.md).
* **What it is for:** problems defined by code, which equation-based solvers
  cannot read at all or need rewritten by hand: `jnp.where` logic,
  neural-network and Gaussian-process surrogates, ODE kinetics written as
  Python loops.

Narrated, executed [example notebooks](examples/index.md) walk through
the relaxations, branch and bound, the test problems, the comparison with
equation-based solvers, batched throughput and Gaussian-process surrogates.

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
5. **Bound tightening against the incumbent** (`f <= UB`), all batched:
   * *FBBT* (`jaxglobal/fbbt.py`): the jaxpr is flattened into a tape and swept
     forward (interval rules) and backward (inverse rules: `x ∈ Z − Y` for
     add, `x ∈ log Z` for exp, the two-sided root for `x²ⁿ` and `|x|`, …) to
     a fixed number of rounds before every node is bounded;
   * *duality-based range reduction*: the cut / LP-dual certificate
     `f ≥ lb + Σ r_j (x_j − lo_j)` confines each variable to where it could
     still beat the incumbent, at no extra cost;
   * *OBBT* (`lp.make_obbt`): min and max of every variable over the
     outer-approximation LP, as 2n `vmap`-ed LPs with weak-duality-certified
     results; on the root by default (`obbt="tree"` for every node).
   Children are split from the tightened box.
6. **Univariate envelopes** (`jaxglobal/univariate.py`, on by default). Every
   smooth sub-expression whose elements each depend on one variable (found
   structurally on the jaxpr: separable sums, polynomial factors,
   `sin(x)·sin(x²)^20`, ...) is sampled on a 65-point grid. The samples are
   lowered by `max(g″, 0)·h²/8`, with `g″` bounded per cell by interval
   evaluation of `jvp(jvp(g))`, and their lower convex hull is a nearly exact
   convex envelope (symmetrically for the concave one). It is intersected
   with the McCormick relaxation, so it never loosens it. On Styblinski–Tang
   the root bound goes from −612 to −118.6 (f* = −117.5).
   An extended-space LP bound (`jaxglobal/lifted.py`, `lifted=True`) is also
   available: one LP column per nonlinear jaxpr node, cut from that node's own
   McCormick rule at FBBT-tightened bounds.

![Relaxations of a JAX function](assets/figures/relaxations.png)

## Where to go next

* [Installation and quickstart](installation.md)
* [Modelling](guide/modelling.md): objectives, constraints, integers, GP surrogates, `.nl` files
* [Solver options](guide/solver-options.md): every argument of `minimize`, including branching
* [Benchmarks and results](guide/benchmarks.md)
* [API reference](api/index.md)

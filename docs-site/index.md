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
# Result(status='optimal', fun=-959.64066, lower_bound=-959.64071, gap=4.78e-05,
#        x=[512. 404.231805], nodes=41, iterations=8, time=1.97s)
```

## Status

jaxglobal is past the proof-of-concept stage. It is a research solver: usable,
tested, and measured on MINLPLib, but not yet competitive with mature global
solvers on general problems (see [Limitations](guide/correctness.md#limitations)).

* **Problems:** continuous and mixed-integer, with inequality and equality
  constraints, on finite or partly unbounded boxes. Models can be plain JAX
  functions or AMPL `.nl` files (`jaxglobal.nl.read_nl`).
* **Correct answers so far:** 0 wrong answers on the MINLPLib sets below, all
  checked against the library's reference bounds. Every returned point is
  re-checked in float64, and an opt-in `safe=True` mode rounds the bounds
  outward.
* **MINLPLib against SCIP, October 2026** (`main` at 58bb551). An instance
  passes if jaxglobal proves it optimal within max(2 x SCIP's time, 10 s).
  "Cold" includes JAX compilation; "warm" reuses the compile caches from a
  previous run. SCIP has no compile step, so the cold numbers are the fair
  comparison.

  | set | instances | pass, cold | pass, warm | where SCIP takes ≥ 0.5 s (warm) |
  |---|---:|---:|---:|---:|
  | development: the targets the recent solver work was tuned on | 79 | 68 | 74 | 20 / 23 |
  | held out: same families, never used during development | 184 | 151 | 160 | 10 / 26 |

  - **Development set.** It is drawn from `start50` and `int50` (41 of the 79 have
    integer variables). Recent PRs fixed these instances one by one, so the
    numbers above are optimistic. The 5 still failing, and why, are listed in
    [`docs/PLAN-scip-gap-v2.md` §10](https://github.com/jkitchin/jax-go/blob/main/docs/PLAN-scip-gap-v2.md).
  - **Held-out set.** Every MINLPLib instance with a SCIP reference time
    that was not used during development: not in `start50`, `int50` or the
    `check_wrong` sample, and not named in the code, tests or docs. The set is
    easier than the development set: all continuous, with a median of 6
    variables and a median SCIP time of 0.03 s.
    - Easy instances (SCIP < 0.5 s) generalize: 150 of 158 pass, warm.
    - Harder instances do not: 10 of 26, against 20 of 23 on the development set.
    - 11 are rejected as unsupported, because bound propagation finds no finite
      bounds for some variables.
  - **Speed.** jaxglobal beats SCIP's time on only 8 of the development
    instances and 3 of the held-out ones. Most passes rely on the 10 s floor.
  - **Wrong answers:** 0 in either set.
  - **To reproduce:** run `benchmarks/run.py --solvers jaxglobal,scip --names
    <list> --time-limit 60` once with `JAXGLOBAL_CACHE=0` (cold) and again on a
    primed cache (warm), then apply the pass rule per instance against SCIP's
    time. `run.py` takes a single time limit, so the rule is applied
    afterwards. The development and held-out instance lists are not checked
    in; the held-out set is defined above.

* **global50, 60 s limit** (the 50-instance BARON comparison panel from
  discopt, `discopt_benchmarks/config/baron_global50.txt`). All solvers ran on
  the same machine. 48 of the 50 ran; casctanks and hda are missing from this
  repo's MINLPLib tables.

  | | solved in 60 s | wrong answers | not solved in 60 s |
  |---|---:|---:|---|
  | jaxglobal, cold | 45 / 48 | 0 | clay0303hfsg, cvxnonsep_psig30, tanksize |
  | jaxglobal, warm | 45 / 48 | 0 | the same three |
  | SCIP (command line, `.nl`) | 45 / 48 | 0 | cvxnonsep_nsig30, st_e36, tspn05 |
  | BARON (GAMS) | 47 / 48 | 1 (nvs05: reports 5.887 as optimal; the optimum is 5.471) | — |

  - **Development overlap.** 11 of the 48 are in the development set above, and
    every solver solves all 11. On the other 37, jaxglobal and SCIP each solve
    34.
  - **Under the 2 x SCIP / 10 s rule:** 40 of 48 pass cold and 45 warm.
    nvs09 takes 44 s cold and 2.6 s warm, so compilation dominates the cold
    misses.
  - **Speed.** SCIP and BARON solve most of these instances in under 0.1 s;
    jaxglobal takes 0.5–8 s warm. jaxglobal solves cvxnonsep_nsig30, st_e36 and
    tspn05, which SCIP leaves unsolved, but BARON solves all three faster.

* **MINLPLib, 60 s limit, cold start (compilation included)**, measured on
  2026-10-17, before the SCIP-gap work, and not re-run since:

  | set | jaxglobal | SCIP |
  |---|---:|---:|
  | `start50`: 50 continuous instances, ≤ 50 variables, weighted toward hard ones | 31 | 44 |
  | `int50`: 50 mixed-integer instances, ≤ 50 variables | 29 | 45 |

  The sets are defined in [`benchmarks/sets/README.md`](https://github.com/jkitchin/jax-go/blob/main/benchmarks/sets/README.md);
  the 31 and 29 are the issue #14 cold runs in [`docs/ROADMAP.md`](https://github.com/jkitchin/jax-go/blob/main/docs/ROADMAP.md)
  (`results/issue14_*_cold.csv`).
  Branching comparisons ([Solver options](guide/solver-options.md#branch_var-which-variable-to-split)) were separate runs, so their counts differ.
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
   subgradients of the convex relaxation. Frank–Wolfe steps on cv (`n_fw=20`) give
   affine cuts `cv(x) + g·(v − x)`, whose minimum over the box is a valid lower
   bound (`jaxglobal/bnb.py`).
4. **Batched spatial branch-and-bound.** Each iteration splits the `batch // 2`
   (128 by default) lowest-bound boxes and bounds the children with a
   `jit(vmap(...))` kernel, in calls of `batch_sizes` boxes (16 by default).
   A local NLP solve (pounce when installed, otherwise optax) polishes the
   incumbent.
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
* [jaxglobal and discopt](guide/discopt.md): how this solver differs from discopt
* [API reference](api/index.md)

To cite jaxglobal, use [doi:10.5281/zenodo.23478503](https://doi.org/10.5281/zenodo.23478503)
(see [Citing and license](references.md#citing-and-license)).

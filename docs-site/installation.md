# Installation and quickstart

## Requirements

* Python 3.10 or newer
* JAX 0.11.x (`jax>=0.11.2,<0.12`): older jaxlib versions segfault when running a
  deserialized `jax.export` kernel with LAPACK calls in a fresh process, and JAX
  0.12 lowers `erfc` with `bitcast_convert_type`, which has no relaxation rule. NumPy,
  SciPy, optax and flatbuffers. These are installed automatically.
* Optional: [pounce](https://pypi.org/project/pounce-solver/) (`pounce-solver>=0.12`),
  the default local NLP solver for upper bounds when it is installed. Without
  it jaxglobal uses optax (pure JAX); `local_solver="scipy"` selects SciPy
  L-BFGS-B / SLSQP.
* Optional, for exporting models: PySCIPOpt (`jaxglobal.export.to_scip`).

## Install

jaxglobal is not on PyPI. Install it from a clone of the repository:

```bash
git clone https://github.com/jkitchin/jax-go.git
cd jax-go
pip install -e .              # core
pip install -e '.[pounce]'    # with the pounce local solver
pip install -e '.[dev]'       # tests, notebooks: pytest, matplotlib, mpmath, pounce, pyomo
```

Importing `jaxglobal` turns on 64-bit floats in JAX (`jax_enable_x64`), which
global optimization needs for meaningful tolerances.

## Quickstart

Write the objective as an ordinary Python function of a 1-D array using
`jax.numpy`, give lower and upper bounds, and call `minimize`:

```python
import jax.numpy as jnp
from jaxglobal import minimize

def f(x):                       # Eggholder: very rugged, optimum on the boundary
    a, b = x[0], x[1]
    return -(b + 47) * jnp.sin(jnp.sqrt(jnp.abs(a / 2 + b + 47))) \
           - a * jnp.sin(jnp.sqrt(jnp.abs(a - (b + 47))))

r = minimize(f, [-512, -512], [512, 512])
print(r)
# Result(status='optimal', fun=-959.64066, lower_bound=-959.64071, gap=4.78e-05,
#        x=[512. 404.231805], nodes=41, iterations=8, time=1.97s)
```

The result is a certificate: `r.fun` is the objective at a feasible point
`r.x` (an upper bound on the global minimum), and `r.lower_bound` is a proven
lower bound. `r.gap = r.fun - r.lower_bound`. `status == "optimal"` means the
gap is within `max(abs_tol, rel_tol * |fun|)`.

### Constraints and integers

```python
def g(x):                       # inequality constraints, g(x) <= 0
    return jnp.array([x[0] ** 2 + x[1] ** 2 - 4.0, 1.0 - x[0] * x[1]])

def h(x):                       # equality constraint, h(x) == 0
    return x[0] - 2 * x[1] + x[2]

r = minimize(f3, [0, 0, 0], [3, 3, 5], ineq=g, eq=h, integer=[2])
```

`ineq` and `eq` return a scalar or a 1-D array. `integer` takes indices or a
boolean mask. See [Modelling](guide/modelling.md).

### AMPL `.nl` models

```python
from jaxglobal import minimize
from jaxglobal.nl import read_nl

m = read_nl("model.nl")         # text or binary .nl, e.g. from MINLPLib
r = minimize(m.f, m.lo, m.hi, ineq=m.ineq, eq=m.eq)

# a model with integer variables: read the mask and pass it on
m = read_nl("minlp.nl", integers="mask")
r = minimize(m.f, m.lo, m.hi, ineq=m.ineq, eq=m.eq, integer=m.integer)
```

A maximization model is negated into `m.f`; `m.sense` records the original sense.

### First call vs later calls

The first `minimize` call on a new function traces it and compiles the
bounding kernels and the local solver, which usually dominates the time of a
small problem (2-7 s on the built-in problems). The kernels are cached in
memory and on disk under `~/.cache/jaxglobal` (set `JAXGLOBAL_CACHE_DIR` to
move it, `JAXGLOBAL_CACHE=0` to turn it off), so a repeat solve of the same
model takes about 0.3-1 s.

## Running the tests

```bash
pytest -q          # soundness, cut validity, global optima, integers
```

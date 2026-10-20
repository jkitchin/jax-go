# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05. Models that are awkward to write in Pyomo
#
# Pyomo is an algebraic modelling language: a model is a set of `Var`, `Param`, `Constraint` and `Objective` objects built from expressions.
# jaxglobal takes a Python function. That makes a few things easy that Pyomo does not have a direct form for. This notebook tries four of them,
# says which worked and which did not, and shows what a Pyomo user would do instead.
#
# 1. A fixed-iteration loop inside the model (`lax.fori_loop`), here the Colebrook friction-factor iteration.
# 2. Reusing existing `jax.numpy` / `jax.scipy` library functions inside the model.
# 3. Vectorized linear-algebra code (`einsum`, `@`) with no index sets.
# 4. A parametric family: the same model solved for several parameter values.
#
# Several of these fail in jaxglobal today; the failures are shown, not hidden. Solver limits for everything: relative gap 1e-4, at most 60 s.

# %%
import numpy as np
import pyomo.environ as pyo

from nbutil import loc, model_size, save_results, solve_jax, solve_pyomo, table  # imports jaxglobal first (x64 enabled)

import jax
import jax.numpy as jnp

# %% [markdown]
# ## 1. A fixed-iteration loop: Colebrook friction factor
#
# Pressure drop in turbulent pipe flow needs the Darcy friction factor $f$ from the implicit Colebrook equation [colebrook1939turbulent]
# $1/\sqrt f = -2\log_{10}\!\big(\varepsilon/(3.7D) + 2.51/(Re\sqrt f)\big)$. In a program, you solve it by a few fixed-point iterations. Problem: choose the
# flow rate $Q \in [10^{-3}, 10^{-2}]$ m3/s and diameter $D\in[0.02, 0.3]$ m to maximize revenue minus pumping and pipe cost
# (the cost coefficients are made up for the example; the fluid has density 900 kg/m3 and viscosity 0.05 Pa s, 200 m of commercial steel pipe).
#
# **jaxglobal.** The iteration is a `lax.fori_loop` with a fixed trip count of 4, started from the Haaland explicit formula.

# %%
EPS, RHO, MU, LEN = 4.5e-5, 900.0, 0.05, 200.0


def friction(Re, D, iters=4):
    x0 = -1.8 * jnp.log10((EPS / D / 3.7) ** 1.11 + 6.9 / Re)  # Haaland start for x = 1/sqrt(f)
    x = jax.lax.fori_loop(0, iters, lambda i, x: -2.0 * jnp.log10(EPS / (3.7 * D) + 2.51 * x / Re), x0)
    return 1.0 / x**2


def pipe(x):
    Q, D = x[0], x[1]
    v = 4 * Q / (jnp.pi * D**2)
    Re = RHO * v * D / MU
    dP = friction(Re, D) * (LEN / D) * RHO * v**2 / 2
    return -4e5 * Q + 0.8 * dP * Q / 0.7 + 2.4e4 * D**1.5


r_pipe_jax = solve_jax(pipe, [1e-3, 0.02], [1e-2, 0.3])
table([r_pipe_jax])

# %% [markdown]
# **Pyomo.** There is no loop construct that is part of the model. Two honest options:
#
# * unroll the iteration in Python, as an expression (what jaxglobal's trace does anyway), producing a deeply nested expression;
# * add $x = 1/\sqrt f$ as a variable and impose the Colebrook equation itself as a constraint. This is the standard approach, and it is exact (no fixed iteration count),
#   but it adds a variable and a nonconvex equality to the model.
#
# The second is written below. Note the two models are therefore not identical: Pyomo solves the implicit equation exactly, jaxglobal approximates it with 4 iterations.

# %%
def pipe_pyomo():
    m = pyo.ConcreteModel()
    m.Q = pyo.Var(bounds=(1e-3, 1e-2), initialize=5e-3)
    m.D = pyo.Var(bounds=(0.02, 0.3), initialize=0.1)
    m.x = pyo.Var(bounds=(1.0, 20.0), initialize=5.0)  # 1/sqrt(f)
    v = 4 * m.Q / (np.pi * m.D**2)
    Re = RHO * v * m.D / MU
    m.colebrook = pyo.Constraint(expr=m.x == -2.0 * pyo.log10(EPS / (3.7 * m.D) + 2.51 * m.x / Re))
    dP = (1.0 / m.x**2) * (LEN / m.D) * RHO * v**2 / 2
    m.obj = pyo.Objective(expr=-4e5 * m.Q + 0.8 * dP * m.Q / 0.7 + 2.4e4 * m.D**1.5)
    return m


size_pipe = model_size(pipe_pyomo())
print(f"Pyomo: {size_pipe[0]} variables, {size_pipe[1]} constraint")
rows_pipe = [solve_pyomo(pipe_pyomo(), s) for s in ("scip", "couenne", "baron")] + [r_pipe_jax]
table(rows_pipe)

# %% [markdown]
# The two objectives agree to the accuracy of the 4-iteration approximation. Pyomo's version has a different (arguably better) property, an exact friction factor,
# at the price of an additional nonconvex constraint for the solver to branch on. In jaxglobal the iteration count is a modelling choice you control, but a
# `lax.scan`/`while` whose behaviour depends on the variables is not supported (see notebook 10).

# %%
save_results("12", "Colebrook fori_loop", loc(pipe_pyomo), loc(pipe, friction), size_pipe, rows_pipe,
             note="Pyomo: implicit equation; jaxglobal: 4 fixed iterations")

# %% [markdown]
# ## 2. Reusing library functions
#
# If the building blocks already exist as `jax.numpy` / `jax.scipy` functions (property correlations, special functions, a library from another project),
# a jaxglobal model can call them. A Pyomo model cannot call them at all, because Pyomo expressions are not numbers; every function has to be re-expressed with
# `pyo.exp`, `pyo.log`, ... or wrapped as an external function (compiled code). Which library functions work in jaxglobal depends on whether the primitives they trace to
# are supported. We test three, each on a small problem; failures are printed.

# %%
def try_solve(name, f, lo, hi):
    try:
        r = solve_jax(f, lo, hi, tlim=30)
        r["solver"] = name
        return r
    except Exception as e:  # noqa: BLE001
        print(f"{name}: {type(e).__name__}: {str(e)[:150]}")
        return {"solver": name, "obj": np.nan, "bound": np.nan, "time": np.nan, "status": f"unsupported ({type(e).__name__})"}


# NASA-style heat capacity polynomial evaluated with jnp.polyval; minimize cost-like function of T
CP_COEF = jnp.array([1.2e-9, -3.0e-6, 2.5e-3, -0.4, 80.0])


def f_polyval(x):
    return jnp.polyval(CP_COEF, x[0]) * jnp.sin(x[0] / 80.0) + 0.01 * x[0]


# jax.nn.softplus, a smooth max, from the jax library
def f_softplus(x):
    return jax.nn.softplus(3 * x[0] - 1) + jax.nn.softplus(-2 * x[1] + x[0]) + (x[0] - x[1]) ** 2


# jax.scipy normal CDF (a newsvendor expected-profit model, which needs the normal loss function)
from jax.scipy.stats import norm


def f_newsvendor(x):
    q, p = x[0], x[1]
    mu = 150.0 - 6.0 * p
    z = (q - mu) / 20.0
    short = 20.0 * (norm.pdf(z) - z * (1 - norm.cdf(z)))
    sales = mu - short
    return -(p * sales + 1.0 * (q - sales) - 3.0 * q)


rows_lib = [
    try_solve("jnp.polyval", f_polyval, [250.0], [450.0]),
    try_solve("jax.nn.softplus", f_softplus, [-3.0, -3.0], [3.0, 3.0]),
    try_solve("jax.scipy.stats.norm.cdf", f_newsvendor, [0.0, 5.0], [250.0, 20.0]),
]
table(rows_lib)

# %% [markdown]
# Two of three work unchanged: `jax.nn.softplus`, and the newsvendor model built on the normal CDF (`erfc`), which is certified at -664.459 in about 2,400 nodes.
# `jnp.polyval` fails with a shape error inside the interpreter (it traces to a loop over the coefficient array). That is an implementation gap, not a fundamental limit; the polynomial can be written as an explicit Horner expression instead.
# The Pyomo-side work would be the same either way: no jnp function can be called from Pyomo at all. The Pyomo alternative for the newsvendor model is to write
# the normal CDF with `pyo.erf`-free approximations or an external function, which no global solver here can then bound.

# %%
save_results("12", "library functions", None, None, (None, None), rows_lib,
             note="Pyomo cannot call jnp functions; jaxglobal works if primitives are supported")

# %% [markdown]
# ## 3. Vectorized linear algebra
#
# Dense vector/matrix code, with no index sets: minimize $x^\top Q x + \sum_i \sin(3x_i)$ over $[-2,2]^3$ with a fixed positive semidefinite $Q$.
# (Random $Q$ with seed 0; the example is synthetic.)

# %%
rng = np.random.default_rng(0)
Bm = rng.standard_normal((3, 3))
Qm = Bm @ Bm.T / 3


def quad(x):
    return jnp.einsum("i,ij,j->", x, Qm, x) + jnp.sum(jnp.sin(3 * x))


def quad_pyomo():
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 2)
    m.x = pyo.Var(m.I, bounds=(-2, 2), initialize=0.0)
    m.obj = pyo.Objective(expr=sum(Qm[i, j] * m.x[i] * m.x[j] for i in m.I for j in m.I) + sum(pyo.sin(3 * m.x[i]) for i in m.I))
    return m


size_quad = model_size(quad_pyomo())
rows_quad = [solve_pyomo(quad_pyomo(), s) for s in ("scip", "couenne", "baron")] + [solve_jax(quad, [-2.0] * 3, [2.0] * 3)]
table(rows_quad)

# %%
save_results("12", "einsum quadratic + sin", loc(quad_pyomo), loc(quad), size_quad, rows_quad, note="3 variables")
print(f"lines of model code: Pyomo {loc(quad_pyomo)}, jaxglobal {loc(quad)}")

# %% [markdown]
# Pyomo is fine here: the double sum is easy to write and the solvers are fast. The difference is that the jaxglobal version has no index sets, so the same
# function works for any dimension (`Qm` of size n) with no rewriting, whereas in Pyomo the problem size is baked into the `RangeSet` and the expression generation.
# For large $n$, Pyomo's expression-generation time and memory become the issue, while jaxglobal's cost depends on the number of traced operations.
# We do not test large $n$ here.

# %% [markdown]
# ## 4. A parametric family
#
# Solve the same model for several values of a parameter $p$: minimize the six-hump camel function plus $p(x+y)$.
# jaxglobal has no batching of independent solves (no GPU claim is made here): the family is a Python loop over `minimize` calls.
# Each solve is a separate B&B run. We pass $p$ as a third variable fixed by `lo == hi`, so the traced function is the same for every value; this avoids recompilation.

# %%
def camel_p(z):
    x, y, p = z[0], z[1], z[2]
    return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2 + p * (x + y)


PS = [0.0, 0.5, 1.0]
rows_fam = []
for p in PS:
    r = solve_jax(camel_p, [-3.0, -2.0, p], [3.0, 2.0, p])
    r["solver"] = f"jaxglobal p={p}"
    rows_fam.append(r)
table(rows_fam)

# %% [markdown]
# Pyomo does this naturally with a mutable `Param`: build the model once, change the value, and solve again (a solver call writes a new file each time, but the model is not rebuilt).

# %%
def camel_pyomo_param():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=0.0, mutable=True)
    m.x = pyo.Var(bounds=(-3, 3), initialize=0.5)
    m.y = pyo.Var(bounds=(-2, 2), initialize=0.5)
    m.obj = pyo.Objective(expr=(4 - 2.1 * m.x**2 + m.x**4 / 3) * m.x**2 + m.x * m.y + (-4 + 4 * m.y**2) * m.y**2 + m.p * (m.x + m.y))
    return m


mp = camel_pyomo_param()
rows_fam_pyo = []
for p in PS:
    mp.p = p
    r = solve_pyomo(mp, "scip")
    r["solver"] = f"scip p={p}"
    rows_fam_pyo.append(r)
table(rows_fam_pyo)

# %%
save_results("12", "parametric family (3 values)", loc(camel_pyomo_param), loc(camel_p), model_size(camel_pyomo_param()), rows_fam_pyo + rows_fam,
             note="loop over parameter values; no batching")

# %% [markdown]
# ## What was hard
#
# **Worked in jaxglobal**
# * The `fori_loop` model (fixed 4 iterations; one solve took 17 to 45 s on the loaded machine versus about 2 s for the Pyomo solvers), `jax.nn.softplus`, the normal-CDF newsvendor model, the einsum quadratic, and the parametric loop. Both are shorter to write than the Pyomo equivalents; the Pyomo versions are also not equivalent in the Colebrook case (implicit equation versus a fixed iteration).
#
# **Did not work, or worked only partly**
# * Library functions are only reusable when every primitive they trace to is supported. `jnp.polyval` fails with a shape error (see the error above); `jax.nn.softplus` and the normal CDF work.
# * BARON (through GAMS) returned no solution for the sine-containing einsum quadratic, while SCIP and Couenne solved it. Solver coverage of functions differs.
# * Parametric families are a Python loop. Passing the parameter as a fixed variable (`lo == hi`) worked: the first solve paid the compile cost (about 5 s) and the later ones took about 0.1 s.
#   Closing over a new Python constant for each value would retrace the function.
# * Loops whose trip count or body depends on the variables (`while_loop`, `scan` with variable-dependent state; see 03) are not supported at all.
#
# **What a Pyomo user does instead**
# * Colebrook: the implicit equation as a constraint (done above).
# * Library functions: re-express with Pyomo functions, or an `ExternalFunction` (which global solvers cannot bound).
# * Families: `Param(mutable=True)` and a Python loop (done above), which is as simple as in jaxglobal and does not suffer from the compile cost.
# * `jaxglobal.export.to_scip` / `to_nl` now write a SCIP or `.nl` model straight from a JAX function (unrolling fixed loops, rewriting `tanh`/`softplus`, reformulating `where`/`max`) and refuse what they cannot represent, such as `erfc` or variable-dependent loops. That removes the hand translation for the supported subset; see `notebooks/export_scip.py`.
#
# **Verdict.** The advantage of the function-based approach is real for models written as numerical code, but this notebook shows that it applies to the subset of JAX that
# the interpreter supports. For models with many integer or logical decisions, many constraints or large sparse structure, Pyomo remains the better choice.

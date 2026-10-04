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
# # 01. The basics: two problems where both tools are easy
#
# Before looking at cases where the two approaches differ, we need a baseline.
# Here are two problems that a Pyomo user would write without any trouble:
#
# 1. **Six-hump camel**: an unconstrained, polynomial, nonconvex function of two variables.
# 2. **Haverly pooling**: a small constrained problem with bilinear terms [haverly1978studies]; the usual global optimum is -400).
#
# For each we write the model twice, once as a Pyomo `ConcreteModel` and once as a plain
# Python function of a vector `x` using `jax.numpy`, and solve both with the same relative gap (1e-4) and a 60 s limit.
# The aim is to show that the two agree, and to see what the syntax looks like in the easy case.
#
# The helper module `nbutil.py` (next to the notebooks) holds the solver wrappers and the line counter.

# %%
import inspect

import jax.numpy as jnp
import numpy as np
import pyomo.environ as pyo

import nbutil
from nbutil import loc, model_size, save_results, solve_jax, solve_pyomo, table

print(inspect.getsource(nbutil.loc))

# %% [markdown]
# `loc` counts the lines of a model-building function, skipping blank lines, comments and docstrings.
# It is used for every comparison in this series. It counts *model code only*: not imports, not the solver call, not result printing.
#
# ## 1. Six-hump camel
#
# $f(x_1,x_2) = (4 - 2.1x_1^2 + x_1^4/3)x_1^2 + x_1x_2 + (-4+4x_2^2)x_2^2$ on $[-3,3]\times[-2,2]$.
# Two global minima, $f^*=-1.031628$, at $\pm(0.0898,-0.7126)$, plus four more local minima
# (Dixon and Szego, 1978; Surjanovic and Bingham, Virtual Library of Simulation Experiments).

# %%
def camel_pyomo():
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(-3, 3), initialize=0.5)
    m.x2 = pyo.Var(bounds=(-2, 2), initialize=0.5)
    m.obj = pyo.Objective(
        expr=(4 - 2.1 * m.x1**2 + m.x1**4 / 3) * m.x1**2 + m.x1 * m.x2 + (-4 + 4 * m.x2**2) * m.x2**2
    )
    return m


# %%
def camel(x):
    a, b = x[0], x[1]
    return (4 - 2.1 * a**2 + a**4 / 3) * a**2 + a * b + (-4 + 4 * b**2) * b**2


# %% [markdown]
# The two bodies are nearly the same expression. The Pyomo version needs the model object, the declaration of each variable and its bounds, and the `Objective` wrapper;
# the JAX version takes the bounds as arguments of the solver call.

# %%
m = camel_pyomo()
size = model_size(m)
rows = [solve_pyomo(camel_pyomo(), s) for s in ("scip", "couenne", "baron")]
rows.append(solve_jax(camel, [-3, -2], [3, 2]))  # first call includes JIT compilation
rows.append({**solve_jax(camel, [-3, -2], [3, 2]), "solver": "jaxglobal (2nd call, cached)"})
t = table(rows)
t

# %%
save_results("01", "six-hump camel", loc(camel_pyomo), loc(camel), size, rows)
print(f"lines of model code: Pyomo {loc(camel_pyomo)}, jaxglobal {loc(camel)};  Pyomo model size: {size[0]} vars, {size[1]} constraints")
print("jaxglobal minimizer:", rows[3]["x"], " (literature: +-(0.0898, -0.7126))")

# %% [markdown]
# All four solvers find $-1.031628$. Notes on the columns:
#
# * *obj* is the best feasible value found; *bound* is the proven lower bound. Couenne's `.nl` interface does not return its final bound to Pyomo, so it is blank.
# * Times are wall-clock for the whole call: for Pyomo solvers that includes writing the model file and starting the solver process; for jaxglobal the first call includes tracing and XLA compilation (the second call reuses the compiled kernels). The machine is loaded by other work, so only large ratios mean anything.
# * `status` is whatever each solver reports. For SCIP we map "gap limit reached" to `optimal`, since the 1e-4 relative gap is the stopping criterion we set.
#
# ## 2. Haverly pooling problem
#
# Two sources A (3% sulfur, cost 6) and B (1%, cost 16) feed a pool; the pool blends with a direct feed C (2%, cost 10) into products x (max 2.5% sulfur, price 9, demand <= 100) and y (max 1.5%, price 15, demand <= 200).
# Pooling means the sulfur concentration of the pool $p$ is a variable, so the quality balances contain products $p \cdot P_x$, which are bilinear.
# The variables are the flows and the pool quality. This is the standard formulation used in `jaxglobal/problems.py`, with the standard global optimum -400.
#
# In Pyomo we can name every constraint and give the variables names, which makes the model self-documenting.

# %%
def haverly_pyomo():
    m = pyo.ConcreteModel()
    m.A = pyo.Var(bounds=(0, 300), initialize=0)    # flow from source A to pool
    m.B = pyo.Var(bounds=(0, 300), initialize=0)    # flow from source B to pool
    m.Cx = pyo.Var(bounds=(0, 100), initialize=0)   # direct feed C to product x
    m.Cy = pyo.Var(bounds=(0, 200), initialize=0)   # direct feed C to product y
    m.Px = pyo.Var(bounds=(0, 300), initialize=0)   # pool to x
    m.Py = pyo.Var(bounds=(0, 300), initialize=0)   # pool to y
    m.x = pyo.Var(bounds=(0, 100), initialize=0)    # product x
    m.y = pyo.Var(bounds=(0, 200), initialize=0)    # product y
    m.p = pyo.Var(bounds=(0, 3), initialize=1)      # sulfur fraction in the pool (percent)
    m.pool_balance = pyo.Constraint(expr=m.Px + m.Py == m.A + m.B)
    m.x_balance = pyo.Constraint(expr=m.x == m.Px + m.Cx)
    m.y_balance = pyo.Constraint(expr=m.y == m.Py + m.Cy)
    m.pool_quality = pyo.Constraint(expr=m.p * (m.Px + m.Py) == 3 * m.A + m.B)
    m.x_spec = pyo.Constraint(expr=m.p * m.Px + 2 * m.Cx <= 2.5 * m.x)
    m.y_spec = pyo.Constraint(expr=m.p * m.Py + 2 * m.Cy <= 1.5 * m.y)
    m.obj = pyo.Objective(expr=-(9 * m.x + 15 * m.y - 6 * m.A - 16 * m.B - 10 * (m.Cx + m.Cy)))
    return m


# %% [markdown]
# In the JAX version the variables are one vector; constraints are two plain functions returning arrays, one for `<= 0` and one for `== 0`.
# The names live in the tuple unpacking on the first line of each function.

# %%
def haverly(z):
    A, B, Cx, Cy, Px, Py, x, y, p = z
    return -(9 * x + 15 * y - 6 * A - 16 * B - 10 * (Cx + Cy))


def haverly_ineq(z):
    A, B, Cx, Cy, Px, Py, x, y, p = z
    return jnp.stack([p * Px + 2 * Cx - 2.5 * x, p * Py + 2 * Cy - 1.5 * y])


def haverly_eq(z):
    A, B, Cx, Cy, Px, Py, x, y, p = z
    return jnp.stack([Px + Py - A - B, x - Px - Cx, y - Py - Cy, p * (Px + Py) - 3 * A - B])


hlo = np.array([0.0] * 8 + [0.0])
hhi = np.array([300.0, 300, 100, 200, 300, 300, 100, 200, 3])

# %%
m = haverly_pyomo()
size = model_size(m)
rows = [solve_pyomo(haverly_pyomo(), s) for s in ("scip", "couenne", "baron")]
rows.append(solve_jax(haverly, hlo, hhi, ineq=haverly_ineq, eq=haverly_eq))
table(rows)

# %%
lp, lj = loc(haverly_pyomo), loc(haverly, haverly_ineq, haverly_eq)
save_results("01", "Haverly pooling", lp, lj, size, rows)
print(f"lines of model code: Pyomo {lp}, jaxglobal {lj};  Pyomo model size: {size[0]} vars, {size[1]} constraints")

# %% [markdown]
# ## What was hard
#
# Nothing. Both formulations are direct transcriptions of the algebra, and all solvers agree on -400 (Haverly) and -1.0316 (camel).
# Observations that matter for the rest of the series:
#
# * **Line counts are similar.** Pyomo is a bit longer because of variable declarations, but it also gives you named variables and constraints for free. The JAX version has no names for constraints; you get an array of values and rely on comments and ordering.
# * **Model size.** Pyomo's model has one variable per decision variable and one constraint object per equation, which is the same size as the algebra. This will not hold once the model contains loops or tables.
# * **Tooling.** On these problems the mature solvers (BARON, SCIP) are as fast as, or faster than, jaxglobal, which pays a JIT-compilation cost on first use. jaxglobal is a proof-of-concept solver; its bounds are not rigorous in floating point (no outward rounding), and it handles only continuous variables.
# * **Fairness check passed.** Same optimum, same gap tolerance. Differences in later notebooks come from model structure, not from different problems.

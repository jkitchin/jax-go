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
# # 06. Many switches, few variables: a motor price list
#
# Notebook 02 showed that a `jnp.where` in Python is a disjunction in an algebraic model. There the equation-based solvers were still
# much faster, because the model had only a handful of switches. This notebook looks for the case where representing a switch natively
# should pay off: **a model with few continuous variables and many switches.**
#
# The argument is about how each approach scales:
#
# * **jaxglobal** branches on the continuous variables. A `where` costs nothing extra in the search: once a box is small enough,
#   interval arithmetic decides which branch applies on the whole box. The cost of the search grows with the number of continuous
#   variables, not with the number of switches.
# * **An equation-based model** needs one binary variable per switch, plus the rows that tie it to the continuous variables. The
#   branch-and-bound tree now runs over the binaries as well.
#
# The model is a branched water main whose pump motor is bought from a price list. In code, the price list is one line.
#
# Solver limits for the live comparison: relative gap 1e-4, at most 60 s. The scaling study at the end uses 600 s and is read from
# `benchmarks/results/water_main.csv` (how to regenerate it is given there).

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from nbutil import loc, model_size, save_results, solve_jax, solve_pyomo, table  # imports jaxglobal first (x64 enabled)

from jaxglobal.applications import logic

INK, MUTED, BLUE, ORANGE = "#1a1a19", "#77766f", "#2a78d6", "#eb6834"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6, "figure.dpi": 110})  # fmt: skip

# %% [markdown]
# ## The model
#
# A pump at the source drives water through a 1500 m trunk to a junction. From there, two branches serve two users: 0.12 m3/s
# delivered 15 m up through 1000 m of pipe, and 0.08 m3/s delivered 5 m up through 2500 m. The decision variables are the three
# diameters, each between 0.15 and 0.6 m.
#
# * Head loss in each pipe uses Darcy-Weisbach with the Swamee-Jain friction factor [swamee1976explicit]. The flow is turbulent
#   everywhere on this box.
# * The pump has to serve the harder branch: head $H = h_{trunk} + \max(h_1 + z_1,\ h_2 + z_2)$, power $W = \rho g Q H / \eta$.
# * Cost per year: energy ($0.8/W), pipe ($100 \cdot L D^{1.5}$), and the **motor**. Motors come in standard ratings. You pay for the
#   smallest rating that covers $W$. The ratings are the 26 IEC 60034 values from 0.37 to 315 kW; the price curve
#   $900\,P^{0.75} + 300$ \$/yr is illustrative.
#
# Making the pipes larger lowers the pumping power. Near a rating boundary, a slightly larger pipe can let you buy the next smaller
# motor, so the optimum tends to sit exactly on a boundary.
#
# ### jaxglobal
#
# The motor price is the base price plus the step up at every rating the load exceeds. That is one `jnp.where` over the rating table.

# %%
RHO, MU, EPS, G = 1000.0, 1e-3, 4.5e-5, 9.81
Q = np.array([0.20, 0.12, 0.08])  # m3/s: trunk, branch 1, branch 2
L = np.array([1500.0, 1000.0, 2500.0])  # m
Z = np.array([15.0, 5.0])  # m, delivery elevations
ETA, E_COST, C_PIPE = 0.7, 0.8, 100.0
KW = np.array([0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3, 4, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110,
               132, 160, 200, 250, 315])  # IEC 60034 motor ratings, kW  # fmt: skip
PRICE = 900.0 * KW**0.75 + 300.0  # $/yr, illustrative
LO, HI = np.full(3, 0.15), np.full(3, 0.6)


def heads(D):
    v = 4 * Q / (jnp.pi * D**2)
    Re = RHO * v * D / MU
    f = 0.25 / jnp.log10(EPS / (3.7 * D) + 5.74 / Re**0.9) ** 2
    return f * L / D * v**2 / (2 * G)


def power(D):
    h = heads(D)
    return RHO * G * Q[0] * (h[0] + jnp.maximum(h[1] + Z[0], h[2] + Z[1])) / ETA


def cost(D):
    W = power(D)
    motor = PRICE[0] + jnp.sum(jnp.where(W / 1000 > KW[:-1], np.diff(PRICE), 0.0))  # smallest motor that covers W
    return motor + E_COST * W + C_PIPE * jnp.sum(L * D**1.5)


def big_enough(D):  # g(D) <= 0: the largest motor covers the load
    return jnp.array([power(D) / 1000 - KW[-1]])


# %% [markdown]
# ### Pyomo
#
# The natural formulation buys exactly one rating: a binary $y_k$ per rating, $\sum_k y_k = 1$, and the chosen rating must cover the
# load, $W \le \sum_k P_k y_k$. This is tighter than a big-M translation of the `where` steps (that version is what
# `jaxglobal.export` would produce). The `max` over the two branches becomes two inequalities on a head variable $H$. That is exact
# here because the cost increases with $H$, but the modeller has to notice it.

# %%
def water_pyomo():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(3), bounds=(0.15, 0.6), initialize=0.35)
    m.H = pyo.Var(bounds=(0, 1e4))
    m.y = pyo.Var(range(len(KW)), domain=pyo.Binary)
    v = [4 * Q[i] / (np.pi * m.x[i] ** 2) for i in range(3)]
    Re = [RHO * v[i] * m.x[i] / MU for i in range(3)]
    h = [0.25 / pyo.log10(EPS / (3.7 * m.x[i]) + 5.74 / Re[i] ** 0.9) ** 2 * L[i] / m.x[i] * v[i] ** 2 / (2 * G) for i in range(3)]
    m.c = pyo.ConstraintList()
    m.c.add(m.H >= h[0] + h[1] + Z[0])
    m.c.add(m.H >= h[0] + h[2] + Z[1])
    W = RHO * G * Q[0] * m.H / ETA
    m.c.add(sum(m.y[k] for k in m.y) == 1)
    m.c.add(W / 1000 <= sum(KW[k] * m.y[k] for k in m.y))
    m.obj = pyo.Objective(expr=sum(PRICE[k] * m.y[k] for k in m.y) + E_COST * W + C_PIPE * sum(L[i] * m.x[i] ** 1.5 for i in range(3)))
    return m


# %% [markdown]
# The same model, for every price-list length used below, is `water_main_<K>` in `jaxglobal/applications/logic.py`. That copy is
# what the flagship benchmark runs. Check that it is the model written here:

# %%
f26, g26 = logic.water_main(26)
D = jnp.asarray(np.random.default_rng(0).uniform(LO, HI, (2000, 3)))
assert np.allclose(jax.vmap(cost)(D), jax.vmap(f26)(D), rtol=1e-12)
assert np.allclose(jax.vmap(big_enough)(D), jax.vmap(g26)(D), rtol=1e-12)

# %% [markdown]
# The exact optimum, for checking the solvers: for each rating $k$, minimize the smooth part of the cost subject to $W \le P_k$,
# then take the cheapest rating. Each of these subproblems is smooth and is solved by multistart SLSQP (`logic._water_exact`).

# %%
FSTAR = logic._water_exact(26)
print(f"exact optimum, 26 IEC ratings: {FSTAR:.2f} $/yr")

# %% [markdown]
# ## The 26-rating price list

# %%
m = water_pyomo()
size = model_size(m)
print(f"Pyomo: {size[0]} variables ({len(KW)} binary), {size[1]} constraints")
rows = [solve_pyomo(water_pyomo(), s) for s in ("scip", "couenne", "baron")]
rows.append(solve_jax(cost, LO, HI, ineq=big_enough))
table(rows)

# %%
xj = rows[-1]["x"]
Wj = float(power(jnp.asarray(xj))) / 1000
print(f"jaxglobal: D = {np.round(xj, 4)} m, pump power {Wj:.3f} kW, motor {KW[np.searchsorted(KW, Wj)]} kW")
print(f"jaxglobal lower bound {rows[-1]['bound']:.2f} <= exact optimum {FSTAR:.2f}: {rows[-1]['bound'] <= FSTAR * (1 + 1e-9)}")

# %%
lp, lj = loc(water_pyomo), loc(heads, power, cost, big_enough)
save_results("06", "motor price list, 26 ratings", lp, lj, size, rows, note=f"{len(KW)} binaries, one per rating")
print(f"lines of model code: Pyomo {lp}, jaxglobal {lj}")

# %% [markdown]
# With 26 ratings, a 60 s budget and this machine, read the table above for the actual timings. In the runs used to write this
# notebook, SCIP was faster at this size. The interesting question is what happens as the list gets longer.
#
# ## Scaling with the number of switches
#
# Real price lists are rarely this short once options are included: frame size, enclosure, efficiency class, drive, installed cost.
# To see the trend, the same model is solved with price lists of 26, 104, 416 and 1664 ratings, geometrically spaced over the same
# 0.37 to 315 kW range (the 26-rating case is the IEC list). The continuous model does not change; only the number of switches does.
#
# These runs take up to 600 s each, so they are read from a file produced by
#
# ```
# python benchmarks/flagship.py --classes logic --only water_main_26,water_main_104,water_main_416,water_main_1664 \
#     --solvers jaxglobal,scip --time-limit 600 --out results/water_main.csv
# ```

# %%
res = pd.read_csv("../benchmarks/results/water_main.csv")
res["K"] = res["name"].str.split("_").str[-1].astype(int)
res["gap"] = (res["primal"] - res["dual"]) / res["primal"].abs()
res.pivot_table(index="K", columns="solver", values=["status", "time", "gap", "verdict"], aggfunc="first")

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for solver, col in (("jaxglobal", BLUE), ("scip", ORANGE)):
    d = res[res.solver == solver].sort_values("K")
    done = d.status == "optimal"
    ax.loglog(d.K[done], d.time[done], "o-", color=col, lw=2, label=f"{solver}, proved optimal")
    ax.loglog(d.K[~done], d.time[~done], "x", color=col, ms=9, mew=2, label=f"{solver}, time limit")
ax.set_xlabel("ratings in the price list (= binaries in the Pyomo model)")
ax.set_ylabel("time to prove optimality, s")
ax.legend(frameon=False, fontsize=9)
plt.show()

# %% [markdown]
# ## What this shows, and what it does not
#
# * **Representation.** The price list is one line of `jnp.where`. In Pyomo it is one binary per rating plus two constraints, and the
#   `max` over the branches needs an argument about monotonicity before it can be written as inequalities.
# * **Scaling.** jaxglobal's time is roughly independent of the length of the list, because its search is over three diameters. With
#   the binary formulation, the search also runs over the ratings, and past a few hundred of them SCIP no longer closes the gap within
#   the limit. On short lists SCIP is faster.
# * **This is a favorable case for jaxglobal**: few continuous variables, and many switches that each depend on one quantity of the
#   model. Not every model with switches is like this. A controller whose on/off state is carried from one time step to the next (a
#   `where` inside a loop over time) is the opposite case. Interval bounds lose all information after a few steps, and jaxglobal
#   cannot certify such models while SCIP solves them quickly.
# * **Ties.** The optimum sits exactly on a rating boundary, where the step is discontinuous. Writing the test as `W > rating` means
#   the boundary point gets the cheaper motor, so the minimum is attained. Equation-based solvers return points that satisfy
#   $W \le P_k$ only up to their feasibility tolerance, so evaluate the code at the returned point before trusting the cost.
# * **Two bugs found while building this comparison**, both fixed: the SCIP export padded exact 0/1 intervals, so big-M constants of
#   about 1e-9 let SCIP's presolve cut off valid points; and at near-ties the compiled gradient of `jnp.maximum` inside the
#   bounding loop was not a subgradient, which once let jaxglobal report a wrong optimum on this model.

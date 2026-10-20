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
# # 06. Many switches, few variables: motor price lists
#
# Notebook 02 showed that a `jnp.where` in Python is a disjunction in an algebraic model. There the equation-based solvers were still
# much faster, because the models had only a handful of switches. This notebook looks for the case where representing a switch
# natively pays off: **few continuous variables and many switches.** It also shows a closely related model where it does not, and a
# kind of switching model where jaxglobal is the one that fails.
#
# The argument is about how each approach scales:
#
# * **jaxglobal** branches on the continuous variables only. Once a box is small enough, interval arithmetic decides which side of a
#   `where` applies on the whole box, so a switch adds work to each bound evaluation but no new dimension to the search.
# * **An equation-based model** needs a binary variable per switch, plus constraints that tie it to the continuous variables. The
#   search then runs over the binaries too, unless presolve can fix them.
#
# The models are pipe networks whose pump motors are bought from a price list. In code, a price list is one line.
#
# Solver limits for the live comparison: relative gap 1e-4, at most 300 s (not the 60 s of the other notebooks: jaxglobal needs
# about 75 s here, most of it compiling on the first call). The scaling runs use 600 s and are read from
# `benchmarks/results/many_switches.csv` (the command that regenerates it is given below).

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
# ## The model: two parallel lines, each with its own pump
#
# Water (0.3 m3/s in total) is delivered through two parallel steel lines, 2000 m and 3000 m long. Each line has its own pump. The
# decision variables are the fraction $s$ of the flow sent through the first line and the two diameters:
#
# * $s \in [0.1, 0.9]$: each line must carry at least 10% of the flow, so that either can be taken out of service gradually.
# * $D_1, D_2 \in [0.15, 0.6]$ m.
#
# Head loss uses Darcy-Weisbach with the Swamee-Jain friction factor [swamee1976explicit]; the flow is turbulent everywhere on this
# box. Cost per year is energy ($0.8/W), pipe ($100 \cdot L D^{1.5}$), and **the two motors**. Motors come in standard ratings, and
# for each line you pay for the smallest rating that covers its pump power. The ratings are the 26 IEC 60034 values from 0.37 to
# 315 kW. The price curve $900\,P^{0.75} + 300$ \$/yr is illustrative.
#
# A slightly larger pipe can let a line use the next smaller motor, so the optimum tends to sit exactly on a rating boundary.
#
# ### jaxglobal
#
# A motor's price is the base price plus the step up at every rating its load exceeds: one `jnp.where` over the rating table.

# %%
RHO, MU, EPS = 1000.0, 1e-3, 4.5e-5  # water; commercial steel roughness, m
QT, L = 0.3, np.array([2000.0, 3000.0])  # m3/s; m
ETA, E_COST, C_PIPE = 0.7, 0.8, 100.0
KW = np.array([0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3, 4, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110,
               132, 160, 200, 250, 315])  # IEC 60034 motor ratings, kW  # fmt: skip
PRICE = 900.0 * KW**0.75 + 300.0  # $/yr, illustrative
LO, HI = np.array([0.1, 0.15, 0.15]), np.array([0.9, 0.6, 0.6])


def power(x):  # W, for each line
    q = QT * jnp.array([x[0], 1 - x[0]])
    D = x[1:]
    v = 4 * q / (jnp.pi * D**2)
    Re = RHO * v * D / MU
    f = 0.25 / jnp.log10(EPS / (3.7 * D) + 5.74 / Re**0.9) ** 2
    return f * L / D * RHO * v**2 / 2 * q / ETA


def cost(x):
    W = power(x)
    motors = PRICE[0] + jnp.sum(jnp.where(W[:, None] / 1000 > KW[:-1], np.diff(PRICE), 0.0), axis=1)  # smallest motor covering W
    return jnp.sum(motors + E_COST * W) + C_PIPE * jnp.sum(L * x[1:] ** 1.5)


def big_enough(x):  # g(x) <= 0: the largest motor covers each load
    return power(x) / 1000 - KW[-1]


# %% [markdown]
# ### Pyomo
#
# The natural formulation buys exactly one rating per line: binaries $y_{ik}$, $\sum_k y_{ik} = 1$, and the chosen rating must
# cover the load, $W_i \le \sum_k P_k y_{ik}$. This is tighter than a big-M translation of the `where` steps (that version is what
# `jaxglobal.export` would produce), so it is the fair comparison.

# %%
def lines_pyomo():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(3), bounds=lambda m, i: (LO[i], HI[i]), initialize=[0.6, 0.4, 0.3])
    m.y = pyo.Var(range(2), range(len(KW)), domain=pyo.Binary)
    m.c = pyo.ConstraintList()
    W = []
    for i in range(2):
        q = QT * (m.x[0] if i == 0 else 1 - m.x[0])
        D = m.x[1 + i]
        v = 4 * q / (np.pi * D**2)
        Re = RHO * v * D / MU
        f = 0.25 / pyo.log10(EPS / (3.7 * D) + 5.74 / Re**0.9) ** 2
        W.append(f * L[i] / D * RHO * v**2 / 2 * q / ETA)
        m.c.add(sum(m.y[i, k] for k in range(len(KW))) == 1)
        m.c.add(W[i] / 1000 <= sum(KW[k] * m.y[i, k] for k in range(len(KW))))
    m.obj = pyo.Objective(expr=sum(PRICE[k] * m.y[i, k] for i in range(2) for k in range(len(KW))) + E_COST * sum(W)
                          + C_PIPE * sum(L[i] * m.x[1 + i] ** 1.5 for i in range(2)))  # fmt: skip
    return m


# %% [markdown]
# The same model, for every price-list length used below, is `motor_lines_<K>` in `jaxglobal/applications/logic.py`. That copy is
# what the flagship benchmark runs. Check that it is the model written here:

# %%
f26, g26 = logic.motor_lines(26)
X = jnp.asarray(np.random.default_rng(0).uniform(LO, HI, (2000, 3)))
assert np.allclose(jax.vmap(cost)(X), jax.vmap(f26)(X), rtol=1e-12)
assert np.allclose(jax.vmap(big_enough)(X), jax.vmap(g26)(X), rtol=1e-12)

# %% [markdown]
# ## The 26-rating price list

# %%
m = lines_pyomo()
size = model_size(m)
print(f"Pyomo: {size[0]} variables ({2 * len(KW)} binary), {size[1]} constraints")
TLIM = 300
rows = [solve_pyomo(lines_pyomo(), s, tlim=TLIM) for s in ("scip", "couenne", "baron")]
rows.append(solve_jax(cost, LO, HI, ineq=big_enough, tlim=TLIM))
table(rows)

# %%
xj = rows[-1]["x"]
Wj = np.asarray(power(jnp.asarray(xj))) / 1000
print(f"jaxglobal: s = {xj[0]:.4f}, D = {np.round(xj[1:], 4)} m")
for i in range(2):
    print(f"  line {i + 1}: pump power {Wj[i]:.4f} kW, motor {KW[np.searchsorted(KW, Wj[i] - 1e-9)]} kW")

# %%
lp, lj = loc(lines_pyomo), loc(power, cost, big_enough)
save_results("06", "A: two lines, 26 motor ratings", lp, lj, size, rows, note=f"{2 * len(KW)} binaries, one per rating and line")
print(f"lines of model code: Pyomo {lp}, jaxglobal {lj}")

# %% [markdown]
# The flow split sits at its bound: the shorter line carries 90%. Each pump runs at a standard rating (to within 0.02%), so each
# motor is fully loaded. With 26 ratings, SCIP is the faster solver here.
#
# ## Scaling with the number of switches
#
# Real price lists are often longer once options are included (frame size, enclosure, efficiency class, drive, installed cost). To
# see the trend, the same model is solved with price lists of 26, 104, 416 and 1664 ratings, geometrically spaced over the same
# 0.37 to 315 kW range; the 26-rating case is the IEC list. The continuous model is unchanged; only the number of switches grows.
#
# These runs take up to 600 s each, so they are read from a file produced by
#
# ```
# cd benchmarks
# python flagship.py --classes logic --solvers jaxglobal,scip --time-limit 600 --out results/many_switches.csv \
#     --only motor_lines_26,motor_lines_104,motor_lines_416,motor_lines_1664,water_main_26,water_main_104,water_main_416,water_main_1664
# ```
#
# The jaxglobal times include JIT compilation (cold cache). SCIP is called through `pyscipopt` if the `scip` binary is not installed.

# %%
res = pd.read_csv("../benchmarks/results/many_switches.csv")
res["model"] = res["name"].str.rsplit("_", n=1).str[0]
res["K"] = res["name"].str.rsplit("_", n=1).str[1].astype(int)
res["rel_gap"] = (res["primal"] - res["dual"]) / res["primal"].abs()
cols = ["status", "verdict", "time", "rel_gap"]
res[res.model == "motor_lines"].pivot_table(index="K", columns="solver", values=cols, aggfunc="first")[cols].round(4)


# %%
def plot(model, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for solver, col in (("jaxglobal", BLUE), ("scip", ORANGE)):
        d = res[(res.model == model) & (res.solver == solver)].sort_values("K")
        done = d.status == "optimal"
        ax.loglog(d.K, d.time, "-", color=col, lw=1.5, alpha=0.5)
        ax.loglog(d.K[done], d.time[done], "o", color=col, ms=7, label=f"{solver}, proved optimal")
        ax.loglog(d.K[~done], d.time[~done], "x", color=col, ms=9, mew=2, label=f"{solver}, hit the time limit")
    ax.set_xlabel("ratings in the price list")
    ax.set_ylabel("time, s")
    ax.set_title(title, fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9)
    plt.show()


plot("motor_lines", "two parallel lines, two motors")

# %% [markdown]
# jaxglobal's time hardly changes with the length of the list, because its search is over the same three variables. With the binary
# formulation, the search also runs over the ratings: SCIP is faster on the short lists and does not close the gap within 600 s on
# the long ones (see the gap column above).
#
# ## A counterexample: one pump on a branched main
#
# The advantage depends on the structure of the model, not just on the number of switches. `water_main_<K>` in
# `jaxglobal/applications/logic.py` uses the same price lists for the single pump of a branched water main: a 1500 m trunk feeds two
# users, 0.12 m3/s lifted 15 m through 1000 m of pipe and 0.08 m3/s lifted 5 m through 2500 m. The decision variables are the three
# diameters. The pump serves the harder branch, so the head is a `jnp.maximum` of the two; in Pyomo that is two inequalities on a head
# variable, which is exact only because the cost increases with the head.

# %%
res[res.model == "water_main"].pivot_table(index="K", columns="solver", values=cols, aggfunc="first")[cols].round(4)

# %%
plot("water_main", "branched main, one motor")

# %% [markdown]
# Here SCIP stays fast for every list length. The static lift puts a floor under the pump power and the largest motor puts a ceiling
# on it, so SCIP's presolve can fix most of the rating binaries before the search starts (in the runs for this notebook, 1664
# ratings left 147 binaries). jaxglobal's time is again roughly flat, but now it is the slower solver throughout.
#
# ## Where jaxglobal fails: switches inside a time loop
#
# The opposite case is a switch whose result feeds the next switch. A tank refilled by an on/off pump with a hysteresis controller,
# `on = where(h < lo * V, 1, where(h > hi * V, 0, on))` inside a loop over hours, is natural code, and the decision variables (tank
# volume, pump capacity, two set points) are few. But the controller state after a few steps depends on every earlier branch, and
# interval bounds on it soon cover both branches everywhere. In runs made while writing this notebook (12 hourly steps, 4
# variables), jaxglobal's best point after 120 s cost 98.60 against an optimum of 96.79, and its lower bound never rose above the
# capital-cost term (60.59), a 38% gap. SCIP, on the model from `jaxglobal.export` with one binary per comparison, proved 96.79 in
# 2.5 s.
#
# ## Summary
#
# * **Representation.** A price list is one line of `jnp.where`. In Pyomo it is a binary per rating plus two constraints, and a
#   `max` needs an argument about monotonicity before it can be written as inequalities.
# * **When native switches pay off:** few continuous variables, and many switches that each depend on a quantity the solver cannot
#   bound tightly in presolve. Then jaxglobal's time stays roughly flat while the binary formulation grows (two parallel lines).
# * **When they do not:** if presolve can fix most of the binaries (branched main), or if switches are chained through time (the
#   controller), the equation-based solver is faster, in the second case by a wide margin.
# * **Ties.** The optimum sits exactly on a rating boundary, where the cost jumps. Writing the test as `W > rating` means the boundary
#   point gets the cheaper motor, so the minimum is attained. Equation-based solvers satisfy $W \le P_k$ only up to their feasibility
#   tolerance, so evaluate the code at their returned point before trusting the cost.
# * **Bugs found while building this comparison.** At near-ties, the compiled gradient of `jnp.maximum` inside the bounding loop was
#   not a subgradient, which let jaxglobal report a wrong optimum on the 26-rating two-line model (fixed in PR #76). The SCIP export
#   padded exact 0/1 intervals, so big-M constants of about 1e-9 let SCIP's presolve cut off valid points of the controller model
#   (PR #77).

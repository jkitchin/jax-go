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
# # 03. Kinetic parameter estimation with an ODE model
#
# Parameter estimation for dynamic models is a standard global-optimization problem: least squares is nonconvex in the rate constants,
# and local solvers can stall in a poor local minimum.
#
# The model is a first-order series reaction $A \xrightarrow{k_1} B \xrightarrow{k_2} C$:
#
# $$\dot A = -k_1 A, \qquad \dot B = k_1 A - k_2 B, \qquad A(0)=1,\ B(0)=0.$$
#
# We estimate $k_1, k_2 \in [0.1, 3]$ from concentrations of A and B measured at $t = 1, 2, 3, 4$.
#
# **The data are synthetic.** They are generated below from the analytic solution with true values $k_1 = 1.0$, $k_2 = 0.5$ plus Gaussian noise
# (standard deviation 0.01, `numpy` seed 0). Nothing here is experimental data.
#
# **Both models use the same discretization.** To make the optimum identical, both use one classical fixed-step RK4 step per measurement interval
# ($h = 1$, four steps). The step function `rk4_step` is plain Python arithmetic, so the *same* function can be called on JAX numbers and on Pyomo variables.
# The difference is what surrounds it: in jaxglobal the states are local Python variables inside the objective, while in Pyomo every state at every step
# is a `Var`, linked to the previous one by a `Constraint`. (Step size 1 is coarse for $k$ near 3, but it is the same discrete model on both sides.
# The reason for so few steps is a limit of jaxglobal, discussed at the end.)

# %%
import numpy as np
import pyomo.environ as pyo

from nbutil import loc, model_size, save_results, solve_jax, solve_pyomo, table

K1_TRUE, K2_TRUE, SIGMA = 1.0, 0.5, 0.01
N, H = 4, 1.0
T_OBS = H * np.arange(1, N + 1)
rng = np.random.default_rng(0)
Y_A = np.exp(-K1_TRUE * T_OBS) + SIGMA * rng.standard_normal(N)
Y_B = K1_TRUE / (K2_TRUE - K1_TRUE) * (np.exp(-K1_TRUE * T_OBS) - np.exp(-K2_TRUE * T_OBS)) + SIGMA * rng.standard_normal(N)
print("synthetic data (A, B):")
print(np.round(np.c_[T_OBS, Y_A, Y_B], 4))


# %%
def rk4_step(A, B, k1, k2, h=H):
    """One RK4 step for the series reaction; works on floats, JAX values or Pyomo expressions."""

    def rhs(a, b):
        return -k1 * a, k1 * a - k2 * b

    a1, b1 = rhs(A, B)
    a2, b2 = rhs(A + h / 2 * a1, B + h / 2 * b1)
    a3, b3 = rhs(A + h / 2 * a2, B + h / 2 * b2)
    a4, b4 = rhs(A + h * a3, B + h * b3)
    return A + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4), B + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)


# %% [markdown]
# ## jaxglobal model
#
# Two decision variables. The ODE solution is computed inside the objective by a Python loop.

# %%
def sse(k):
    A, B, err = 1.0, 0.0, 0.0
    for n in range(N):
        A, B = rk4_step(A, B, k[0], k[1])
        err = err + (A - Y_A[n]) ** 2 + (B - Y_B[n]) ** 2
    return err


# %% [markdown]
# ## Pyomo model: explicit RK4 as algebraic equations
#
# One `Var` per state per time step, one equality constraint per state per step, and the objective summed over the measurements.

# %%
def rk4_pyomo():
    m = pyo.ConcreteModel()
    m.k1 = pyo.Var(bounds=(0.1, 3), initialize=1.0)
    m.k2 = pyo.Var(bounds=(0.1, 3), initialize=1.0)
    m.n = pyo.RangeSet(0, N)
    m.A = pyo.Var(m.n, bounds=(0, 1), initialize=0.5)
    m.B = pyo.Var(m.n, bounds=(0, 1), initialize=0.2)
    m.A[0].fix(1.0)
    m.B[0].fix(0.0)
    m.steps = pyo.ConstraintList()
    for n in range(N):
        An, Bn = rk4_step(m.A[n], m.B[n], m.k1, m.k2)
        m.steps.add(m.A[n + 1] == An)
        m.steps.add(m.B[n + 1] == Bn)
    m.obj = pyo.Objective(expr=sum((m.A[n + 1] - Y_A[n]) ** 2 + (m.B[n + 1] - Y_B[n]) ** 2 for n in range(N)))
    return m


# %% [markdown]
# Solving. jaxglobal needs reduced options here: with the defaults (OBBT, univariate envelopes, alphaBB, FBBT) the first JIT compilation of the
# unrolled RK4 graph took minutes already for three steps on this machine, so we turn those features off and use plain McCormick relaxations with interval arithmetic.
# That is a real cost of this style of model in the current prototype (see the end).

# %%
JAX_OPTS = dict(univariate=False, alphabb=False, fbbt=False, dbbt=False, obbt=False)
m = rk4_pyomo()
size = model_size(m)
print(f"Pyomo explicit RK4: {size[0]} free variables, {size[1]} constraints (two more fixed initial conditions)")
rows = [solve_pyomo(rk4_pyomo(), s) for s in ("scip", "couenne", "baron")]
rows.append(solve_jax(sse, [0.1, 0.1], [3.0, 3.0], **JAX_OPTS))
table(rows)

# %%
mm = rk4_pyomo()
solve_pyomo(mm, "baron")
print(f"Pyomo (BARON) k1, k2 = {pyo.value(mm.k1):.4f}, {pyo.value(mm.k2):.4f}")
print(f"jaxglobal     k1, k2 = {rows[-1]['x'][0]:.4f}, {rows[-1]['x'][1]:.4f}   (true values {K1_TRUE}, {K2_TRUE}; the data are noisy)")
print(f"jaxglobal objective at the BARON optimum: {float(sse(np.array([pyo.value(mm.k1), pyo.value(mm.k2)]))):.6e}")

# %%
lp, lj = loc(rk4_pyomo, rk4_step), loc(sse, rk4_step)
save_results("10", "RK4 kinetics (explicit)", lp, lj, size, rows, note="same RK4 discretization")
print(f"lines of model code (including the shared rk4_step): Pyomo {lp}, jaxglobal {lj}")

# %% [markdown]
# ## Pyomo model: `pyomo.dae`
#
# The idiomatic Pyomo way is not to write the integrator but to declare the ODE with a `ContinuousSet` and `DerivativeVar`
# and let a transformation discretize it. Here: orthogonal collocation on finite elements (3-point Radau, one element per measurement interval).
# This is a different discretization (order 5 and implicit) than the RK4 above, so the optimum differs slightly. It is closer to the exact solution.

# %%
from pyomo.dae import ContinuousSet, DerivativeVar


def dae_pyomo():
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=[0.0, *T_OBS])
    m.k1 = pyo.Var(bounds=(0.1, 3), initialize=1.0)
    m.k2 = pyo.Var(bounds=(0.1, 3), initialize=1.0)
    m.A = pyo.Var(m.t, bounds=(0, 1), initialize=0.5)
    m.B = pyo.Var(m.t, bounds=(0, 1), initialize=0.2)
    m.dA = DerivativeVar(m.A, wrt=m.t)
    m.dB = DerivativeVar(m.B, wrt=m.t)
    m.odeA = pyo.Constraint(m.t, rule=lambda m, t: m.dA[t] == -m.k1 * m.A[t])
    m.odeB = pyo.Constraint(m.t, rule=lambda m, t: m.dB[t] == m.k1 * m.A[t] - m.k2 * m.B[t])
    m.A[0].fix(1.0)
    m.B[0].fix(0.0)
    m.obj = pyo.Objective(expr=sum((m.A[T_OBS[n]] - Y_A[n]) ** 2 + (m.B[T_OBS[n]] - Y_B[n]) ** 2 for n in range(N)))
    pyo.TransformationFactory("dae.collocation").apply_to(m, nfe=N, ncp=3, scheme="LAGRANGE-RADAU")
    return m


# %%
m = dae_pyomo()
size_dae = model_size(m)
print(f"Pyomo dae (Radau, 4 elements x 3 points): {size_dae[0]} free variables, {size_dae[1]} constraints")
rows_dae = [solve_pyomo(dae_pyomo(), s) for s in ("scip", "baron")]
for r in rows_dae:
    r["solver"] += " (dae)"
table(rows_dae)

# %%
md = dae_pyomo()
solve_pyomo(md, "baron")
print(f"pyomo.dae (BARON) k1, k2 = {pyo.value(md.k1):.4f}, {pyo.value(md.k2):.4f}  vs RK4 {pyo.value(mm.k1):.4f}, {pyo.value(mm.k2):.4f}")
save_results("10", "pyomo.dae (collocation)", loc(dae_pyomo), loc(sse, rk4_step), size_dae, rows_dae, note="different discretization")

# %% [markdown]
# The estimates differ in the third decimal, which is the discretization difference (RK4 with $h=1$ versus Radau collocation), not a solver disagreement.
# The `pyomo.dae` model is declared in about the same number of lines as the explicit one; the transformation generates the many variables and constraints.
#
# ## Does `lax.scan` work for the loop?
#
# Python loops are unrolled in the trace, which is why compile time grows with the number of steps. A `lax.scan` would keep the graph small, if the interpreter supported it.

# %%
import jax


def sse_scan(k):
    def body(carry, y):
        A, B = rk4_step(carry[0], carry[1], k[0], k[1])
        return (A, B), (A - y[0]) ** 2 + (B - y[1]) ** 2

    _, errs = jax.lax.scan(body, (1.0, 0.0), np.c_[Y_A, Y_B])
    return errs.sum()


try:
    print(solve_jax(sse_scan, [0.1, 0.1], [3.0, 3.0], tlim=30, **JAX_OPTS))
except Exception as e:  # noqa: BLE001
    print(f"{type(e).__name__}: {e}")

# %% [markdown]
# ## What was hard
#
# **In Pyomo**
#
# * Writing the model is not hard, and `pyomo.dae` is the better abstraction for dynamic models than hand-written equations: the discretization is a one-line choice, and it scales to PDEs.
# * The size is the issue. Every state at every time point is a decision variable and every step is a constraint (10 variables here, 52 with collocation, for 2 parameters).
#   For an experiment with 100 time points and 10 species that is thousands of variables, each tied by a nonconvex equation; a global solver must branch on them.
#   Here SCIP, Couenne and BARON all solve it in about a second because the problem is tiny.
# * Everything must be a variable or an expression of variables. If the rate law needs an `if`, a table lookup or a call to an existing integrator (e.g. `scipy.integrate`), it cannot be used.
#
# **In jaxglobal**
#
# * The model is the program: 2 variables, the integrator is a loop, and changing the integrator (RK4 for RK2, a different step) means editing one function.
# * The bound is computed through the unrolled computation graph by McCormick relaxations, which suffer from the dependency problem: bounds on the states of step *n* are computed from bounds on step *n-1*, and the overestimation compounds. This is why the node count is large for a 2-parameter problem (compare with the Pyomo solvers, which branch on a few variables with strong relaxations of each bilinear term).
# * **Compile time grows quickly with the number of steps.** The loop is unrolled, and with the default bound-tightening options (OBBT, FBBT, univariate and alphaBB relaxations) the first compilation did not finish in several minutes for three RK4 steps on this machine. With those options off, four steps solve in tens of seconds. `lax.scan` over the steps is not supported (see the cell above), so long time horizons are not practical today. That is the main limitation shown by this notebook.
# * Timings above are wall-clock on a heavily loaded machine; the Pyomo solvers are an order of magnitude faster in this case.
#
# **Verdict.** For a problem this small, Pyomo with a mature global solver is the better tool today. The jaxglobal advantage in this notebook is in what the model looks like and how little must be restated
# (and in the fact that the same code can be differentiated or run in a simulator without changes), not in solve time.

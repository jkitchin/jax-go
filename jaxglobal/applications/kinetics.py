"""ODE kinetic parameter estimation: least squares on a fixed-step RK4 integrator.

Data sources (verbatim measurement tables, ODE right-hand sides, initial conditions):

* ``gasoil``: catalytic cracking of gas oil, Tjoa & Biegler
  [tjoa1991simultaneous], as distributed in GAMS Model Library ``gasoil.gms`` (COPS
  [dolan2004benchmarking] #12, 21 observations x 2 states, tau = 0 ... 0.95). Also test problem in
  Floudas et al. [floudas1999handbook] and Esposito & Floudas
  [esposito2000global].
* ``methanol``: methanol-to-hydrocarbons, Maria [maria1989adaptive],
  as distributed in GAMS Model Library ``methanol.gms`` (COPS 2.0 #13, 17
  observations x 3 states, tau = 0 ... 1.122).

Parameter bounds are our own choice (the benchmark tables in the papers were not
available to us), see each problem's ``note``. The integrator is plain RK4 with
substeps of at most ``hmax`` between consecutive measurement times, so the model
being optimized is the discretized one; ``fstar`` refers to that discretization and
the continuous-time optimum (``scipy.integrate.solve_ivp``) is quoted in ``source``.

Series reactions, reversible series and Lotka-Volterra from Esposito & Floudas
are *not* included: their data tables could not be found verbatim.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from . import AppProblem

# ----------------------------------------------------------------- data (GAMS COPS 2.0)
GASOIL_TAU = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4,
                       0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95])  # fmt: skip
GASOIL_Z = np.array([
    [1.0000, 0], [0.8105, 0.2000], [0.6208, 0.2886], [0.5258, 0.3010], [0.4345, 0.3215], [0.3903, 0.3123],
    [0.3342, 0.2716], [0.3034, 0.2551], [0.2735, 0.2258], [0.2405, 0.1959], [0.2283, 0.1789], [0.2071, 0.1457],
    [0.1669, 0.1198], [0.1530, 0.0909], [0.1339, 0.0719], [0.1265, 0.0561], [0.1200, 0.0460], [0.0990, 0.0280],
    [0.0870, 0.0190], [0.0770, 0.0140], [0.0690, 0.0100],
])  # fmt: skip
METHANOL_TAU = np.array([0.0, 0.05, 0.065, 0.08, 0.123, 0.233, 0.273, 0.354, 0.397, 0.418, 0.502, 0.553, 0.681,
                         0.75, 0.916, 0.937, 1.122])  # fmt: skip
METHANOL_Z = np.array([
    [1.0000, 0, 0], [0.7085, 0.1621, 0.0811], [0.5971, 0.1855, 0.0965], [0.5537, 0.1989, 0.1198],
    [0.3684, 0.2845, 0.1535], [0.1712, 0.3491, 0.2097], [0.1198, 0.3098, 0.2628], [0.0747, 0.3576, 0.2467],
    [0.0529, 0.3347, 0.2884], [0.0415, 0.3388, 0.2757], [0.0261, 0.3557, 0.3167], [0.0208, 0.3483, 0.2954],
    [0.0085, 0.3836, 0.2950], [0.0053, 0.3611, 0.2937], [0.0019, 0.3609, 0.2831], [0.0018, 0.3485, 0.2846],
    [0.0006, 0.3698, 0.2899],
])  # fmt: skip


def schedule(tau, hmax):
    """RK4 substeps per measurement interval: [(n_i, h_i)], n_i = ceil(dt_i / hmax)."""
    return [(int(np.ceil(dt / hmax - 1e-9)), float(dt / np.ceil(dt / hmax - 1e-9))) for dt in np.diff(tau)]


# ----------------------------------------------------------------- gas oil cracking
def make_gasoil(hmax=0.025, lo=(0.0, 0.0, 0.0), hi=(20.0, 20.0, 20.0), fstar=None, source="", note=""):
    sched, Z, Y0 = schedule(GASOIL_TAU, hmax), GASOIL_Z, [1.0, 0.0]

    def f(th):
        """A -> Q (k1), Q -> S (k2), A -> S (k3); second-order in A. RK4, sum of squared errors."""

        def rhs(y):
            return jnp.stack([-(th[0] + th[2]) * y[0] ** 2, th[0] * y[0] ** 2 - th[1] * y[1]])

        y, sse = jnp.array(Y0), 0.0
        for i, (n, h) in enumerate(sched):

            def step(y, _, h=h):
                k1 = rhs(y)
                k2 = rhs(y + h / 2 * k1)
                k3 = rhs(y + h / 2 * k2)
                k4 = rhs(y + h * k3)
                return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

            y, _ = lax.scan(step, y, None, length=n)
            sse = sse + jnp.sum((y - Z[i + 1]) ** 2)
        return sse

    def pyomo():
        import pyomo.environ as pe

        m = pe.ConcreteModel()
        m.x = pe.Var(range(3), bounds=lambda m, i: (lo[i], hi[i]), initialize=lambda m, i: 0.5 * (lo[i] + hi[i]))
        steps = [h for n, h in sched for _ in range(n)]
        end = np.cumsum([n for n, _ in sched])  # node index of each measurement
        m.y = pe.Var(range(1, len(steps) + 1), range(2), bounds=(0, 1), initialize=0.5)
        m.k = pe.Var(range(len(steps)), range(4), range(2), bounds=(-100, 100), initialize=0.0)
        m.c = pe.ConstraintList()
        node = lambda n, s: Y0[s] if n == 0 else m.y[n, s]  # noqa: E731
        rhs = lambda y: [-(m.x[0] + m.x[2]) * y[0] ** 2, m.x[0] * y[0] ** 2 - m.x[1] * y[1]]  # noqa: E731
        for n, h in enumerate(steps):
            y = [node(n, s) for s in range(2)]
            for j, c in enumerate([0, h / 2, h / 2, h]):
                arg = [y[s] + (c * m.k[n, j - 1, s] if j else 0) for s in range(2)]
                for s, v in enumerate(rhs(arg)):
                    m.c.add(m.k[n, j, s] == v)
            for s in range(2):
                m.c.add(m.y[n + 1, s] == y[s] + h / 6 * (m.k[n, 0, s] + 2 * m.k[n, 1, s] + 2 * m.k[n, 2, s] + m.k[n, 3, s]))
        m.obj = pe.Objective(expr=sum((m.y[end[i], s] - Z[i + 1, s]) ** 2 for i in range(len(end)) for s in range(2)))
        return m

    return AppProblem(f"gasoil_h{hmax:g}", f, np.array(lo), np.array(hi), None, None, fstar, source, pyomo, note)


# ----------------------------------------------------------------- methanol to hydrocarbons
def make_methanol(hmax=0.05, lo=(0, 0.1, 0, 0, 0), hi=(5, 5, 5, 5, 5), fstar=None, source="", note=""):
    sched, Z, Y0 = schedule(METHANOL_TAU, hmax), METHANOL_Z, [1.0, 0.0, 0.0]

    def f(th):
        """Rational-rate three-state kinetics [maria1989adaptive]; RK4, sum of squared errors."""

        def rhs(y):
            den = (th[1] + th[4]) * y[0] + y[1]
            r = th[0] * y[1] / den
            return jnp.stack([
                -(2 * th[1] - r + th[2] + th[3]) * y[0],
                th[0] * y[0] * (th[1] * y[0] - y[1]) / den + th[2] * y[0],
                th[0] * y[0] * (y[1] + th[4] * y[0]) / den + th[3] * y[0],
            ])  # fmt: skip

        y, sse = jnp.array(Y0), 0.0
        for i, (n, h) in enumerate(sched):

            def step(y, _, h=h):
                k1 = rhs(y)
                k2 = rhs(y + h / 2 * k1)
                k3 = rhs(y + h / 2 * k2)
                k4 = rhs(y + h * k3)
                return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

            y, _ = lax.scan(step, y, None, length=n)
            sse = sse + jnp.sum((y - Z[i + 1]) ** 2)
        return sse

    def pyomo():
        import pyomo.environ as pe

        m = pe.ConcreteModel()
        m.x = pe.Var(range(5), bounds=lambda m, i: (lo[i], hi[i]), initialize=lambda m, i: 0.5 * (lo[i] + hi[i]))
        steps = [h for n, h in sched for _ in range(n)]
        end = np.cumsum([n for n, _ in sched])
        m.y = pe.Var(range(1, len(steps) + 1), range(3), bounds=(1e-9, 1), initialize=0.3)
        m.k = pe.Var(range(len(steps)), range(4), range(3), bounds=(-60, 60), initialize=0.0)
        m.c = pe.ConstraintList()
        node = lambda n, s: Y0[s] if n == 0 else m.y[n, s]  # noqa: E731
        th = m.x

        def rhs(y):
            den = (th[1] + th[4]) * y[0] + y[1]
            r = th[0] * y[1] / den
            return [
                -(2 * th[1] - r + th[2] + th[3]) * y[0],
                th[0] * y[0] * (th[1] * y[0] - y[1]) / den + th[2] * y[0],
                th[0] * y[0] * (y[1] + th[4] * y[0]) / den + th[3] * y[0],
            ]  # fmt: skip

        for n, h in enumerate(steps):
            y = [node(n, s) for s in range(3)]
            for j, c in enumerate([0, h / 2, h / 2, h]):
                arg = [y[s] + (c * m.k[n, j - 1, s] if j else 0) for s in range(3)]
                for s, v in enumerate(rhs(arg)):
                    m.c.add(m.k[n, j, s] == v)
            for s in range(3):
                m.c.add(m.y[n + 1, s] == y[s] + h / 6 * (m.k[n, 0, s] + 2 * m.k[n, 1, s] + 2 * m.k[n, 2, s] + m.k[n, 3, s]))
        m.obj = pe.Objective(expr=sum((m.y[end[i], s] - Z[i + 1, s]) ** 2 for i in range(len(end)) for s in range(3)))
        return m

    return AppProblem(f"methanol_h{hmax:g}", f, np.array(lo, float), np.array(hi, float), None, None, fstar, source, pyomo, note)


_GAS_SRC = (
    "Data: Tjoa and Biegler [tjoa1991simultaneous] gasoil cracking (21 points), as in the COPS 3.0 gasoil model "
    "[dolan2004benchmarking]; Esposito and Floudas [esposito2000global]."
)
_GAS_NOTE = (
    "fstar = best of 10 multistart L-BFGS-B runs on the RK4 objective (one RK4 step per measurement interval): "
    "0.0052331591; not certified by this repo. The same parameters evaluated with an accurate "
    "scipy solve_ivp (LSODA, rtol 1e-11) give SSE 0.0052367 and the continuous-time optimum is 0.0052366 at "
    "(11.853, 8.350, 0.9948); the RK4 optimum is at (11.856, 8.348, 0.9935). The published optimum and "
    "the E&F parameter bounds could not be retrieved verbatim, so theta in [0, 20]^3 is our own choice."
)
_METH_SRC = (
    "Data: Maria [maria1989adaptive] methanol-to-hydrocarbons conversion, 17 points, as in the COPS 3.0 methanol model "
    "[dolan2004benchmarking]; Esposito and Floudas [esposito2000global]."
)
_METH_NOTE = (
    "fstar = best of 10 multistart L-BFGS-B runs on the RK4 objective (hmax 0.2, 16 steps): 0.0090491111; "
    "not certified by this repo. SSE of the same parameters by accurate solve_ivp: 0.0090246; continuous-time "
    "optimum 0.0090223 at (1.773, 2.167, 1.859, 1.803, 0). Published optimum and E&F bounds were not found "
    "verbatim; theta in [0, 5]^5 (theta2 >= 0.1) is our own choice."
)

PROBLEMS = [
    make_gasoil(0.1, fstar=0.005233159124563396, source=_GAS_SRC, note=_GAS_NOTE),
    make_methanol(0.2, fstar=0.009049111059327009, source=_METH_SRC, note=_METH_NOTE),
]

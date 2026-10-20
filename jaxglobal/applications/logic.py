"""Small engineering models whose natural code uses control flow and tables.

Each model is what an engineer would write in plain JAX: a laminar/turbulent
``jnp.where`` friction-factor switch, ``jnp.max`` over parallel branches,
a piecewise-linear property table (hinge form of ``jnp.interp``), ``jnp.minimum`` for volume-discount
tariffs. The Pyomo versions are the honest equation-based counterparts: every
``where``/``max``/``min`` becomes a ``pyomo.gdp`` disjunction or binary
variables (big-M), and every ``interp`` becomes ``pyomo.environ.Piecewise``
with binaries.

* ``pipes_parallel``: two parallel pipes of different length share a viscous
  oil flow; pump head is the larger of the two pressure drops.
* ``hx_water``: tube-side design of a water-cooled exchanger; water properties
  from a table, Nusselt number by regime (``max``), friction by regime (``where``).
* ``tariff_blend``: buying four raw materials with volume-discount tariffs
  (``jnp.minimum`` of two linear schedules) and an overage surcharge
  (``jnp.maximum``) subject to demand and a blend-quality constraint.
* ``motor_lines_<K>``: two parallel water lines share a flow; the flow split and
  the two diameters are chosen, and each line's pump motor is bought from a price
  list of ``K`` standard ratings, written as one ``jnp.where`` line. The Pyomo
  twin picks one rating per line with ``K`` binaries each. ``K = 26`` is the IEC
  60034 rating list; larger ``K`` are finer lists over the same range, to show
  how each side scales with the number of switches.
* ``water_main_<K>``: the same price list for the single pump of a branched water
  main (a trunk feeding two users at different elevations). The static lift
  bounds the pump power from below, and SCIP's presolve removes most of the
  binaries, so here the equation-based solver stays fast as ``K`` grows.

``fstar`` for the first two is a dense grid followed by a local polish
(``fstar_by_grid``); for ``tariff_blend`` it is exact (enumeration of the 2^4
tariff choices, one LP each). For ``motor_lines_<K>`` and ``water_main_<K>`` it is
the best point found by jaxglobal or SCIP (Pyomo twin), evaluated in JAX; see
each entry's ``source`` for which solvers proved it optimal.
"""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from . import AppProblem

RE_C = (64.0 / 0.3164) ** (4.0 / 3.0)  # Re where laminar 64/Re meets Blasius [blasius1913aehnlichkeitsgesetz] 0.3164 Re^-1/4 (continuous switch)


# ----------------------------------------------------------------- pipes_parallel
_P_RHO, _P_MU, _P_Q, _P_ETA = 900.0, 0.08, 0.004, 0.7  # oil kg/m3, Pa s; m3/s; pump efficiency
_P_L = np.array([200.0, 350.0])  # pipe lengths, m
_P_CAP, _P_EC = 600.0, 0.8  # $/yr per m * m^1.5 of diameter; $/yr per W of pumping power


def pipes_parallel(x):
    d, phi = x[:2], x[2]
    q = jnp.array([phi, 1.0 - phi]) * _P_Q
    v = 4 * q / (jnp.pi * d**2)
    re = _P_RHO * v * d / _P_MU
    f = jnp.where(re < RE_C, 64.0 / re, 0.3164 * re**-0.25)  # laminar / Blasius turbulent
    dp = f * _P_L / d * _P_RHO * v**2 / 2
    return _P_CAP * jnp.sum(_P_L * d**1.5) + _P_EC * _P_Q * jnp.max(dp) / _P_ETA


def pipes_parallel_pyomo():
    import pyomo.environ as pe
    from pyomo.gdp import Disjunct, Disjunction

    m = pe.ConcreteModel()
    m.x = pe.Var(range(3), bounds=lambda m, i: [(0.02, 0.2), (0.02, 0.2), (0.05, 0.95)][i], initialize=[0.08, 0.08, 0.5])
    m.re, m.f, m.dp = pe.Var(range(2), bounds=(1.0, 2e5)), pe.Var(range(2), bounds=(0.0, 1.0)), pe.Var(range(2), bounds=(0, 1e9))
    m.dpmax = pe.Var(bounds=(0, 1e9))
    flow = [m.x[2] * _P_Q, (1 - m.x[2]) * _P_Q]
    m.c = pe.ConstraintList()
    m.lam, m.turb = Disjunct(range(2)), Disjunct(range(2))
    for i in range(2):
        v = 4 * flow[i] / (math.pi * m.x[i] ** 2)
        m.c.add(m.re[i] * _P_MU == _P_RHO * v * m.x[i])
        m.lam[i].c = pe.Constraint(expr=m.re[i] <= RE_C)
        m.lam[i].f = pe.Constraint(expr=m.f[i] * m.re[i] == 64.0)
        m.turb[i].c = pe.Constraint(expr=m.re[i] >= RE_C)
        m.turb[i].f = pe.Constraint(expr=m.f[i] * m.re[i] ** 0.25 == 0.3164)
        m.add_component(f"regime{i}", Disjunction(expr=[m.lam[i], m.turb[i]]))
        m.c.add(m.dp[i] == m.f[i] * _P_L[i] / m.x[i] * _P_RHO * v**2 / 2)
        m.c.add(m.dpmax >= m.dp[i])
    m.obj = pe.Objective(expr=_P_CAP * sum(_P_L[i] * m.x[i] ** 1.5 for i in range(2)) + _P_EC * _P_Q * m.dpmax / _P_ETA)
    pe.TransformationFactory("gdp.bigm").apply_to(m)
    return m


# ----------------------------------------------------------------- hx_water
# Liquid water at 0.5 MPa, generated with the `iapws` package [gomera2024iapws] (IAPWS-IF97 density and cp
# [wagner2000iapws], IAPWS R12-08 viscosity [huber2009new], IAPWS R15-11 thermal conductivity
# [huber2012new]), rounded as printed here.
_TT = np.arange(10.0, 100.0, 10.0)  # deg C
_RHO_T = np.array([999.89, 998.39, 995.83, 992.40, 988.22, 983.38, 977.95, 971.98, 965.50])  # kg/m3
_CP_T = np.array([4193.9, 4183.6, 4178.9, 4177.6, 4178.6, 4181.9, 4187.2, 4194.6, 4204.1])  # J/kg/K
_MU_T = 1e-3 * np.array([1.3055, 1.0015, 0.7972, 0.6528, 0.5466, 0.4661, 0.4037, 0.3542, 0.3143])  # Pa s
_K_T = np.array([0.5790, 0.5982, 0.6146, 0.6287, 0.6408, 0.6512, 0.6600, 0.6672, 0.6730])  # W/m/K
_H_M, _H_QD, _H_TP, _H_ETA = 2.0, 2.0e5, 120.0, 0.7  # kg/s water, duty W, process temperature C, pump eff.
_H_CA, _H_CE, _H_CT, _H_CN = 150.0, 0.8, 40.0, 3.0  # $/yr per m2 tube area; per W; per K below 50 C coolant; per tube


def _table(t, ys):
    """Piecewise-linear table lookup in hinge form (equal to ``jnp.interp(t, _TT, ys)`` on [10, 90]).

    ``jnp.interp`` is built on ``searchsorted`` (integer indices), which the McCormick
    relaxation cannot interpret, so the table is written as y0 + slope*(t-t0) plus a
    ``maximum`` hinge per interior knot.
    """
    sl = np.diff(ys) / np.diff(_TT)
    out = ys[0] + sl[0] * (t - _TT[0])
    for j in range(1, len(sl)):
        out = out + (sl[j] - sl[j - 1]) * jnp.maximum(t - _TT[j], 0.0)
    return out


def hx_water(x):
    d, v, tw = x
    rho, cp, mu, k = (_table(tw, ys) for ys in (_RHO_T, _CP_T, _MU_T, _K_T))
    re, pr = rho * v * d / mu, cp * mu / k
    nu = jnp.maximum(3.66, 0.023 * re**0.8 * pr**0.4)  # laminar limit vs Dittus-Boelter
    h = nu * k / d
    area = _H_QD / (h * (_H_TP - tw))
    n_tubes = 4 * _H_M / (rho * v * jnp.pi * d**2)
    length = area / (n_tubes * jnp.pi * d)
    f = jnp.where(re < RE_C, 64.0 / re, 0.3164 * re**-0.25)
    dp = f * length / d * rho * v**2 / 2
    return _H_CA * area + _H_CN * n_tubes + _H_CE * _H_M * dp / rho / _H_ETA + _H_CT * jnp.maximum(0.0, 50.0 - tw)


def hx_water_pyomo():
    import pyomo.environ as pe
    from pyomo.gdp import Disjunct, Disjunction

    m = pe.ConcreteModel()
    m.x = pe.Var(range(3), bounds=lambda m, i: [(0.01, 0.05), (0.3, 5.0), (10.0, 90.0)][i], initialize=[0.02, 1.0, 40.0])
    d, v, tw = m.x[0], m.x[1], m.x[2]
    tab = {"rho": (_RHO_T, (960, 1000)), "cp": (_CP_T, (4170, 4210)), "mu": (_MU_T, (3e-4, 1.4e-3)), "k": (_K_T, (0.57, 0.68))}
    for name, (vals, bnd) in tab.items():  # property tables -> piecewise-linear with binaries
        setattr(m, name, pe.Var(bounds=bnd, initialize=sum(bnd) / 2))
        setattr(m, name + "_pw", pe.Piecewise(getattr(m, name), tw, pw_pts=list(_TT), f_rule=list(vals),
                                              pw_constr_type="EQ", pw_repn="INC"))  # fmt: skip
    m.re, m.nu, m.f = pe.Var(bounds=(1.0, 5e5), initialize=1e4), pe.Var(bounds=(3.0, 5e3), initialize=50), pe.Var(bounds=(0, 1), initialize=0.03)
    m.c = pe.ConstraintList()
    m.c.add(m.re * m.mu == m.rho * v * d)
    pr = m.cp * m.mu / m.k
    db = 0.023 * m.re**0.8 * pr**0.4
    m.nuA, m.nuB = Disjunct(), Disjunct()  # max(3.66, Dittus-Boelter)
    m.nuA.c = pe.Constraint(expr=m.nu == 3.66)
    m.nuA.g = pe.Constraint(expr=db <= 3.66)
    m.nuB.c = pe.Constraint(expr=m.nu == db)
    m.nuB.g = pe.Constraint(expr=db >= 3.66)
    m.nuD = Disjunction(expr=[m.nuA, m.nuB])
    m.lam, m.turb = Disjunct(), Disjunct()  # laminar / turbulent friction
    m.lam.c = pe.Constraint(expr=m.re <= RE_C)
    m.lam.f = pe.Constraint(expr=m.f * m.re == 64.0)
    m.turb.c = pe.Constraint(expr=m.re >= RE_C)
    m.turb.f = pe.Constraint(expr=m.f * m.re**0.25 == 0.3164)
    m.fD = Disjunction(expr=[m.lam, m.turb])
    h = m.nu * m.k / d
    area = _H_QD / (h * (_H_TP - tw))
    n_tubes = 4 * _H_M / (m.rho * v * math.pi * d**2)
    length = area / (n_tubes * math.pi * d)
    dp = m.f * length / d * m.rho * v**2 / 2
    m.cool = pe.Var(bounds=(0, 40), initialize=0.0)  # max(0, 50 - tw): minimised, so >= both branches suffices
    m.c.add(m.cool >= 50.0 - tw)
    m.obj = pe.Objective(expr=_H_CA * area + _H_CN * n_tubes + _H_CE * _H_M * dp / m.rho / _H_ETA + _H_CT * m.cool)
    pe.TransformationFactory("gdp.bigm").apply_to(m)
    return m


# ----------------------------------------------------------------- tariff_blend
_T_A1 = np.array([10.0, 8.0, 6.5, 5.0])  # $/t, first tariff
_T_A2 = np.array([7.5, 6.0, 5.0, 4.0])  # $/t, discount tariff ...
_T_F = np.array([60.0, 50.0, 70.0, 80.0])  # ... with a fixed fee
_T_CAP, _T_S = np.array([40.0, 40.0, 40.0, 40.0]), np.array([3.0, 3.0, 3.0, 3.0])  # supply cap, overage $/t
_T_IMP = np.array([0.02, 0.05, 0.09, 0.14])  # impurity fraction
_T_DEM, _T_MAXIMP = 100.0, 0.07  # t of blend, max blend impurity


def tariff_blend(x):
    cost = jnp.minimum(_T_A1 * x, _T_A2 * x + _T_F) + _T_S * jnp.maximum(x - _T_CAP, 0.0)
    return jnp.sum(cost)


def tariff_blend_ineq(x):
    return jnp.array([_T_DEM - jnp.sum(x), jnp.dot(_T_IMP - _T_MAXIMP, x)])


def tariff_blend_pyomo():
    import pyomo.environ as pe

    m = pe.ConcreteModel()
    m.x = pe.Var(range(4), bounds=(0, 80), initialize=25.0)
    m.z, m.over = pe.Var(range(4), bounds=(0, 1e3)), pe.Var(range(4), bounds=(0, 80))
    m.delta = pe.Var(range(4), domain=pe.Binary)  # tariff chosen: 1 = first schedule
    M = 1e3
    m.c = pe.ConstraintList()
    for i in range(4):
        m.c.add(m.z[i] >= _T_A1[i] * m.x[i] - M * (1 - m.delta[i]))  # min() of two linear schedules
        m.c.add(m.z[i] >= _T_A2[i] * m.x[i] + _T_F[i] - M * m.delta[i])
        m.c.add(m.over[i] >= m.x[i] - _T_CAP[i])  # max(x - cap, 0), convex so no binary
    m.c.add(sum(m.x[i] for i in range(4)) >= _T_DEM)
    m.c.add(sum((_T_IMP[i] - _T_MAXIMP) * m.x[i] for i in range(4)) <= 0)
    m.obj = pe.Objective(expr=sum(m.z[i] + _T_S[i] * m.over[i] for i in range(4)))
    return m


def _tariff_exact():
    """Exact optimum: for each choice of tariff per supplier the problem is an LP."""
    from scipy.optimize import linprog

    best = np.inf
    for pick in itertools.product((0, 1), repeat=4):
        a = np.where(np.array(pick) == 0, _T_A1, _T_A2)
        fee = np.where(np.array(pick) == 0, 0.0, _T_F)
        # variables x (4), o (4) >= max(x - cap, 0)
        c = np.concatenate([a, _T_S])
        A = np.zeros((2 + 4, 8))
        b = np.zeros(6)
        A[0, :4], b[0] = -1, -_T_DEM
        A[1, :4] = _T_IMP - _T_MAXIMP
        for i in range(4):
            A[2 + i, i], A[2 + i, 4 + i], b[2 + i] = 1, -1, _T_CAP[i]
        r = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 80)] * 4 + [(0, None)] * 4, method="highs")
        if r.status == 0:
            best = min(best, r.fun + fee.sum())  # fee paid whenever tariff 2 chosen (x may be 0: min() then takes tariff 1)
    return best


# ----------------------------------------------------------------- water_main
_W_RHO, _W_MU, _W_EPS, _W_G = 1000.0, 1e-3, 4.5e-5, 9.81  # water kg/m3, Pa s; commercial steel roughness, m
_W_Q = np.array([0.20, 0.12, 0.08])  # m3/s: trunk, branch 1, branch 2
_W_L = np.array([1500.0, 1000.0, 2500.0])  # m
_W_Z = np.array([15.0, 5.0])  # m, delivery elevation of the two users
_W_ETA, _W_EC, _W_CAP = 0.7, 0.8, 100.0  # pump efficiency; $/yr per W; $/yr per m * m^1.5 of diameter
_W_IEC = np.array([0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3, 4, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110,
                   132, 160, 200, 250, 315])  # kW, IEC 60034 standard motor ratings  # fmt: skip
_W_LO, _W_HI = np.full(3, 0.15), np.full(3, 0.6)


def _water_ratings(k):
    return _W_IEC if k == len(_W_IEC) else np.geomspace(_W_IEC[0], _W_IEC[-1], k)


def _water_price(kw):
    return 900.0 * kw**0.75 + 300.0  # $/yr, illustrative annualized motor cost


def _water_power(x):
    """Pump power in W: Darcy-Weisbach with Swamee-Jain friction; the pump serves the harder branch."""
    h = _water_heads(x)
    return _W_RHO * _W_G * _W_Q[0] * (h[0] + jnp.maximum(h[1] + _W_Z[0], h[2] + _W_Z[1])) / _W_ETA


def water_main(k):
    kw = _water_ratings(k)
    step = np.diff(_water_price(kw))

    def f(x):
        w = _water_power(x)
        motor = _water_price(kw[0]) + jnp.sum(jnp.where(w / 1000 > kw[:-1], step, 0.0))  # smallest motor that covers w
        return motor + _W_EC * w + _W_CAP * jnp.sum(_W_L * x**1.5)

    def ineq(x):
        return jnp.array([_water_power(x) / 1000 - kw[-1]])  # the largest motor must cover the load

    return f, ineq


def water_main_pyomo(k):
    def build():
        import pyomo.environ as pe

        kw = _water_ratings(k)
        m = pe.ConcreteModel()
        m.x = pe.Var(range(3), bounds=(_W_LO[0], _W_HI[0]), initialize=0.35)
        m.H = pe.Var(bounds=(0, 1e4))  # pump head, m
        m.y = pe.Var(range(k), domain=pe.Binary)  # motor rating chosen
        v = [4 * _W_Q[i] / (np.pi * m.x[i] ** 2) for i in range(3)]
        re = [_W_RHO * v[i] * m.x[i] / _W_MU for i in range(3)]
        h = [0.25 / pe.log10(_W_EPS / (3.7 * m.x[i]) + 5.74 / re[i] ** 0.9) ** 2 * _W_L[i] / m.x[i] * v[i] ** 2 / (2 * _W_G)
             for i in range(3)]  # fmt: skip
        m.c = pe.ConstraintList()
        m.c.add(m.H >= h[0] + h[1] + _W_Z[0])  # max() of the two branches: the cost increases with H, so >= is exact
        m.c.add(m.H >= h[0] + h[2] + _W_Z[1])
        w = _W_RHO * _W_G * _W_Q[0] * m.H / _W_ETA
        m.c.add(sum(m.y[j] for j in range(k)) == 1)
        m.c.add(w / 1000 <= sum(kw[j] * m.y[j] for j in range(k)))  # the chosen motor covers the load
        price = _water_price(kw)
        m.obj = pe.Objective(expr=sum(price[j] * m.y[j] for j in range(k)) + _W_EC * w
                             + _W_CAP * sum(_W_L[i] * m.x[i] ** 1.5 for i in range(3)))  # fmt: skip
        return m

    return build


def _water_heads(x):
    v = 4 * _W_Q / (np.pi * x**2)
    re = _W_RHO * v * x / _W_MU
    return 0.25 / jnp.log10(_W_EPS / (3.7 * x) + 5.74 / re**0.9) ** 2 * _W_L / x * v**2 / (2 * _W_G)


# ----------------------------------------------------------------- motor_lines
_M_QT = 0.3  # m3/s, total flow split over the two lines
_M_L = np.array([2000.0, 3000.0])  # m
_M_LO, _M_HI = np.array([0.1, 0.15, 0.15]), np.array([0.9, 0.6, 0.6])  # each line carries at least 10% of the flow


def _motor_lines_power(x):
    """Pump power of each line in W (Darcy-Weisbach with Swamee-Jain friction)."""
    q = _M_QT * jnp.array([x[0], 1.0 - x[0]])
    d = x[1:]
    v = 4 * q / (np.pi * d**2)
    re = _W_RHO * v * d / _W_MU
    f = 0.25 / jnp.log10(_W_EPS / (3.7 * d) + 5.74 / re**0.9) ** 2
    return f * _M_L / d * _W_RHO * v**2 / 2 * q / _W_ETA


def motor_lines(k):
    kw = _water_ratings(k)
    step = np.diff(_water_price(kw))

    def f(x):
        w = _motor_lines_power(x)
        motors = _water_price(kw[0]) + jnp.sum(jnp.where(w[:, None] / 1000 > kw[:-1], step, 0.0), axis=1)
        return jnp.sum(motors + _W_EC * w) + _W_CAP * jnp.sum(_M_L * x[1:] ** 1.5)

    def ineq(x):
        return _motor_lines_power(x) / 1000 - kw[-1]  # the largest motor must cover each load

    return f, ineq


def motor_lines_pyomo(k):
    def build():
        import pyomo.environ as pe

        kw, price = _water_ratings(k), _water_price(_water_ratings(k))
        m = pe.ConcreteModel()
        m.x = pe.Var(range(3), bounds=lambda m, i: (_M_LO[i], _M_HI[i]), initialize=[0.6, 0.4, 0.3])
        m.y = pe.Var(range(2), range(k), domain=pe.Binary)  # motor rating chosen for each line
        m.c = pe.ConstraintList()
        w = []
        for i in range(2):
            q = _M_QT * (m.x[0] if i == 0 else 1 - m.x[0])
            d = m.x[1 + i]
            v = 4 * q / (np.pi * d**2)
            re = _W_RHO * v * d / _W_MU
            f = 0.25 / pe.log10(_W_EPS / (3.7 * d) + 5.74 / re**0.9) ** 2
            w.append(f * _M_L[i] / d * _W_RHO * v**2 / 2 * q / _W_ETA)
            m.c.add(sum(m.y[i, j] for j in range(k)) == 1)
            m.c.add(w[i] / 1000 <= sum(kw[j] * m.y[i, j] for j in range(k)))  # the chosen motor covers the load
        m.obj = pe.Objective(expr=sum(price[j] * m.y[i, j] for i in range(2) for j in range(k)) + _W_EC * sum(w)
                             + _W_CAP * sum(_M_L[i] * m.x[1 + i] ** 1.5 for i in range(2)))  # fmt: skip
        return m

    return build


def fstar_by_grid(f, lo, hi, n=121, top=20, chunk=200_000):
    """Dense grid over the box, then a bounded Powell polish of the best ``top`` grid points."""
    from scipy.optimize import minimize

    axes = [np.linspace(a, b, n) for a, b in zip(lo, hi)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(lo))
    fv = jax.jit(jax.vmap(f))
    vals = np.concatenate([np.asarray(fv(jnp.asarray(pts[i : i + chunk]))) for i in range(0, len(pts), chunk)])
    best = (np.inf, None)
    for j in np.argsort(vals)[:top]:
        r = minimize(lambda x: float(f(jnp.asarray(x))), pts[j], method="Powell", bounds=list(zip(lo, hi)),
                     options=dict(xtol=1e-10, ftol=1e-14))  # fmt: skip
        if r.fun < best[0]:
            best = (float(r.fun), r.x)
    return best


PROBLEMS = [
    AppProblem("pipes_parallel", pipes_parallel, np.array([0.02, 0.02, 0.05]), np.array([0.2, 0.2, 0.95]), None, None,
               3874.178193, "Textbook Blasius [blasius1913aehnlichkeitsgesetz]/laminar friction; fstar by dense grid (121^3) + Powell polish", pipes_parallel_pyomo, "where(Re<Re_c) friction factor, max over two branches"),
    AppProblem("hx_water", hx_water, np.array([0.01, 0.3, 10.0]), np.array([0.05, 5.0, 90.0]), None, None,
               52.543162, "Water properties from IAPWS-IF97 [wagner2000iapws] (iapws 1.5.5 [gomera2024iapws]) at 0.5 MPa, 10-90 C; fstar by dense grid (61^3) + Powell polish", hx_water_pyomo, "4 piecewise-linear property tables (hinge form; jnp.interp is not relaxable), max() Nusselt, where() friction"),
    AppProblem("tariff_blend", tariff_blend, np.zeros(4), np.full(4, 80.0), tariff_blend_ineq, None,
               708.3333333, "Synthetic tariffs; fstar exact by enumerating the 2^4 tariff choices as LPs", tariff_blend_pyomo, "minimum()/maximum() tariffs with demand and quality constraints"),
]

_PRICE_LISTS = (26, 104, 416, 1664)  # 26 = the IEC 60034 ratings; the rest are finer lists over 0.37 .. 315 kW
_FSTAR_MOTOR_LINES = dict.fromkeys(_PRICE_LISTS)
_FSTAR_WATER_MAIN = {26: 173876.835214, 104: 171820.190768, 416: 171703.733532, 1664: 171702.402915}
for _k in _PRICE_LISTS:
    _f, _g = motor_lines(_k)
    PROBLEMS.append(AppProblem(f"motor_lines_{_k}", _f, _M_LO, _M_HI, _g, None, _FSTAR_MOTOR_LINES[_k],
                               "Illustrative costs; fstar = best point of jaxglobal and SCIP (Pyomo twin), evaluated in JAX",
                               motor_lines_pyomo(_k), f"price list of {_k} motor ratings per line (jnp.where); 2 x {_k} binaries in Pyomo"))  # fmt: skip
    _f, _g = water_main(_k)
    PROBLEMS.append(AppProblem(f"water_main_{_k}", _f, _W_LO, _W_HI, _g, None, _FSTAR_WATER_MAIN[_k],
                               "Illustrative costs; fstar = SCIP on the Pyomo twin, proved to relative gap 1e-4",
                               water_main_pyomo(_k), f"price list of {_k} motor ratings (jnp.where), max() over branches; {_k} binaries in Pyomo"))  # fmt: skip

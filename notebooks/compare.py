"""Compare jaxglobal with equation-based global solvers on the same problems.

The equation-based solvers need an algebraic model. We build one from the very
same Python functions by evaluating them on a numpy object array of Pyomo
variables, with ``jnp`` swapped for a shim that maps sin/exp/... to Pyomo.
Functions the modeling layer cannot express (``where``, ``maximum``, ``tanh``)
need a hand reformulation -- which is the point of the comparison.

``jaxglobal.export`` (``to_scip`` / ``to_nl``) now builds these models straight
from the JAX functions, with ``where``/``maximum``/``tanh`` rewritten or
reformulated automatically; see ``notebooks/export_scip.py``. This script keeps
the shim so the stored timings in ``compare_results.json`` stay comparable.

Solvers: SCIP and Couenne (via .nl files), BARON and LindoGlobal (via GAMS).
Wall time is measured for every solver, including model writing / JIT.

Run:  PYTHONPATH=. python notebooks/compare.py   ->  notebooks/results/compare_results.json
"""

import json
import time
import types
import warnings

import numpy as np
import pyomo.environ as pyo

from jaxglobal import minimize
from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS

warnings.filterwarnings("ignore")
TIME_LIMIT = 60
GAP = 1e-6


def _vec(fn):
    return lambda a: np.vectorize(fn, otypes=[object])(a) if isinstance(a, np.ndarray) else fn(a)


class _Unsupported(Exception):
    pass


def _nope(name):
    def f(*a, **k):
        raise _Unsupported(name)

    return f


def _sum(a, axis=None):
    """Sum with Python ``+`` (np.sum trips Pyomo's __array_ufunc__ on object arrays)."""
    a = np.asarray(a, dtype=object)
    if axis is None:
        return sum(a.ravel().tolist())
    a = np.moveaxis(a, axis, 0)
    out = np.empty(a.shape[1:], dtype=object)
    for idx in np.ndindex(out.shape):
        out[idx] = sum(a[(slice(None),) + idx].tolist())
    return out


# jnp shim: enough of numpy's API for the test functions, backed by Pyomo
J = types.SimpleNamespace(
    sin=_vec(pyo.sin), cos=_vec(pyo.cos), exp=_vec(pyo.exp), sqrt=_vec(pyo.sqrt),
    abs=_vec(abs), sum=_sum, arange=np.arange, pi=np.pi, stack=list,
    where=_nope("where"), maximum=_nope("maximum"), tanh=_nope("tanh"),
)  # fmt: skip


def on_pyomo(fn, X):
    """Evaluate a jnp-written function on a Pyomo variable array."""
    g = types.FunctionType(fn.__code__, {**fn.__globals__, "jnp": J}, fn.__name__, fn.__defaults__)
    return g(X)


# --- hand reformulations for functions outside the modeling layer ------------
def _max(u, v):
    return (u + v + abs(u - v)) / 2


def piecewise_manual(X):
    a, b = X[0], X[1]
    smooth = pyo.sin(3 * a) * pyo.cos(2 * b) + 0.1 * (a**2 + b**2)
    kink = _max(a - 0.5, 0.0) + abs(b + 0.3) * 0.5  # where(a>.5, a-.5, 0)
    return smooth + kink + _max(a * b, -0.2)


def tanh_net_manual(X):
    from jaxglobal.problems import _W1, _W2, _b1

    z = X @ _W1 + _b1
    return sum(w * (1 - 2 / (pyo.exp(2 * zi) + 1)) for w, zi in zip(_W2, z))  # tanh via exp


MANUAL = {"piecewise_2d": piecewise_manual, "tanh_net_2d": tanh_net_manual}


def build(p):
    m = pyo.ConcreteModel()
    n = p.lo.size
    m.x = pyo.Var(range(n), bounds=lambda m, i: (p.lo[i], p.hi[i]),
                  initialize=lambda m, i: 0.5 * (p.lo[i] + p.hi[i]))  # same start as jaxglobal
    X = np.empty(n, dtype=object)
    X[:] = [m.x[i] for i in range(n)]
    f = MANUAL[p.name](X) if p.name in MANUAL else on_pyomo(p.f, X)
    m.obj = pyo.Objective(expr=f)
    m.c = pyo.ConstraintList()
    if getattr(p, "ineq", None):
        es = on_pyomo(p.ineq, X)
        for e in es if isinstance(es, list) else [es]:
            m.c.add(e <= 0)
    if getattr(p, "eq", None):
        es = on_pyomo(p.eq, X)
        for e in es if isinstance(es, list) else [es]:
            m.c.add(e == 0)
    return m


def run_pyomo(p, solver):
    try:
        m = build(p)
    except _Unsupported as e:
        return dict(status=f"unsupported:{e}")
    t = time.perf_counter()
    try:
        if solver in ("baron", "lindoglobal"):
            opts = [f"option optcr={GAP}, optca={GAP}, reslim={TIME_LIMIT};"]
            r = pyo.SolverFactory("gams").solve(m, solver=solver, add_options=opts, load_solutions=False)
        elif solver == "scip":
            r = pyo.SolverFactory("scip").solve(
                m, options={"limits/gap": GAP, "limits/absgap": GAP, "limits/time": TIME_LIMIT}, load_solutions=False
            )
        else:  # couenne
            with open("couenne.opt", "w") as fh:
                fh.write(f"allowable_fraction_gap {GAP}\nallowable_gap {GAP}\ntime_limit {TIME_LIMIT}\n")
            r = pyo.SolverFactory("couenne", executable="/Applications/AMPL/couenne").solve(m, load_solutions=False)
    except Exception as e:  # noqa: BLE001
        return dict(status="error", msg=str(e)[:200], time=time.perf_counter() - t)
    dt = time.perf_counter() - t
    tc = str(r.solver.termination_condition)
    fun = None
    if len(r.solution) and tc not in ("noSolution", "licensingProblems", "error"):
        m.solutions.load_from(r)
        fun = pyo.value(m.obj)
    st = getattr(r.solver, "user_time", None)  # GAMS reports solver time; .nl solvers do not
    try:
        st = float(st)
    except (TypeError, ValueError):
        st = None
    return dict(status=tc, fun=fun, time=dt, solver_time=st)


def run_jax(p):
    kw = dict(ineq=p.ineq, eq=p.eq) if hasattr(p, "ineq") else {}
    r = minimize(p.f, p.lo, p.hi, max_iter=5000, **kw)
    return dict(status=r.status, fun=float(r.fun), lb=float(r.lower_bound), time=r.time, nodes=r.nodes)


OUT = "notebooks/results/compare_results.json"


def main():
    import os

    out = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["name"] for r in out}
    for p in list(PROBLEMS) + list(CONSTRAINED_PROBLEMS):
        if p.name in done:
            continue
        row = dict(name=p.name, n=int(p.lo.size), fstar=p.fstar, constrained=hasattr(p, "ineq"),
                   manual=p.name in MANUAL, jaxglobal=run_jax(p))  # fmt: skip
        for s in ("scip", "couenne", "baron", "lindoglobal"):
            row[s] = run_pyomo(p, s)
        out.append(row)
        print(p.name, {k: (v.get("status"), v.get("fun"), round(v.get("time", 0), 2)) for k, v in row.items() if isinstance(v, dict)}, flush=True)
        with open(OUT, "w") as fh:  # save as we go
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

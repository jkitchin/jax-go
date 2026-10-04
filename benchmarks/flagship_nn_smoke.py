"""Standalone smoke test of the NN-surrogate flagship problems.

Usage: python benchmarks/flagship_nn_smoke.py [--limit 60] [--solvers jax,jax_crown,scip,couenne,baron,maingo]
                                              [--formulation reduced|omlt] [name ...]
Default names: the three smallest networks.  Prints one row per (problem, solver).
"""
import argparse
import time

import numpy as np

SCIP = "/opt/homebrew/bin/scip"
COUENNE = "/Applications/AMPL/couenne"
GAMS = "/Library/Frameworks/GAMS.framework/Versions/53/Resources/gams"


def run_jax(p, limit, crown):
    from jaxglobal import minimize

    t = time.perf_counter()
    r = minimize(p.f, p.lo, p.hi, ineq=p.ineq, eq=p.eq, crown=crown, time_limit=limit)
    return dict(status=r.status, ub=r.fun, lb=r.lower_bound, time=time.perf_counter() - t, nodes=r.nodes)


def build_model(p, formulation):
    # none of SCIP (NL), Couenne, BARON (GAMS) accept tanh, so use the exp form 1 - 2/(exp(2z)+1)
    from jaxglobal.applications import nn
    return (nn.omlt_model if formulation == "omlt" else nn.reduced_model)(p.name, act="exp")


def run_pyomo(p, limit, solver, formulation):
    import pyomo.environ as pyo

    m = build_model(p, formulation)
    t = time.perf_counter()
    if solver == "scip":
        opt = pyo.SolverFactory("scip", executable=SCIP)
        opt.options["limits/time"] = limit
        opt.options["limits/gap"] = 1e-6
        opt.options["limits/absgap"] = 1e-6
        res = opt.solve(m, tee=False, load_solutions=False)
    elif solver == "couenne":
        opt = pyo.SolverFactory("couenne", executable=COUENNE)
        opt.options["bonmin.time_limit"] = limit
        opt.options["bonmin.allowable_gap"] = 1e-6
        opt.options["bonmin.allowable_fraction_gap"] = 1e-6
        res = opt.solve(m, tee=False, load_solutions=False)
    elif solver == "baron":
        opt = pyo.SolverFactory("gams", executable=GAMS)
        res = opt.solve(m, solver="baron", tee=False, load_solutions=False,
                        add_options=[f"option reslim={limit};", "option optca=1e-6;", "option optcr=1e-6;"])
    else:
        raise ValueError(solver)
    dt = time.perf_counter() - t
    pr = res.problem
    ub = lb = None
    try:
        ub, lb = float(pr.upper_bound), float(pr.lower_bound)
    except Exception:
        pass
    if lb is not None and not np.isfinite(lb):
        lb = None  # the NL interface returned no dual bound
    msg = str(getattr(res.solver, "message", "") or "").strip().replace("\n", " ")
    try:  # objective of the loaded incumbent, if any
        if len(res.solution):
            m.solutions.load_from(res)
            ub = pyo.value(m.obj)
    except Exception:
        pass
    return dict(status=f"{res.solver.termination_condition}" + (f" [{msg[:20]}]" if solver == "scip" else ""), ub=ub, lb=lb, time=dt, nodes=None)


def run_maingo(p, limit, formulation):
    from jaxglobal.applications import maingo_adapter

    return maingo_adapter.solve(p, limit)


def fmt(v):
    return "-" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--limit", type=float, default=60)
    ap.add_argument("--solvers", default="jax,jax_crown,scip,couenne,baron")
    ap.add_argument("--formulation", default="reduced", choices=["reduced", "omlt"])
    a = ap.parse_args()
    from jaxglobal.applications import nn

    names = a.names or ["peaks_1x5", "peaks_1x10", "peaks_1x20"]
    print(f"{'problem':12s} {'solver':10s} {'status':28s} {'ub':>12s} {'lb':>12s} {'time':>7s} {'nodes':>7s}")
    for p in nn.PROBLEMS:
        if p.name not in names:
            continue
        for s in a.solvers.split(","):
            try:
                if s == "jax":
                    r = run_jax(p, a.limit, False)
                elif s == "jax_crown":
                    r = run_jax(p, a.limit, True)
                elif s == "maingo":
                    r = run_maingo(p, a.limit, a.formulation)
                else:
                    r = run_pyomo(p, a.limit, s, a.formulation)
            except Exception as e:  # noqa: BLE001
                r = dict(status=f"ERROR {type(e).__name__}: {str(e)[:60]}", ub=None, lb=None, time=None, nodes=None)
            print(f"{p.name:12s} {s:10s} {r['status'][:28]:28s} {fmt(r['ub']):>12s} {fmt(r['lb']):>12s} "
                  f"{fmt(r['time']):>7s} {fmt(r['nodes']):>7s}", flush=True)

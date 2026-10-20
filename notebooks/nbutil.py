"""Shared helpers for the jaxglobal-vs-Pyomo notebooks.

Defines the lines-of-code counter, uniform solver wrappers (SCIP, Couenne and
BARON through Pyomo; jaxglobal), and the result-table builder. Every solver
uses relative gap 1e-4 and a 60 s limit.
"""

import inspect
import os
import re
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import pyomo.environ as pyo

warnings.filterwarnings("ignore")
import logging

logging.getLogger("pyomo").setLevel(logging.CRITICAL)  # solver failures are reported in the result table instead
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for jaxglobal

GAP = 1e-4
TIME_LIMIT = 60
COUENNE = shutil.which("couenne") or "/Applications/AMPL/couenne"


def loc(*objs):
    """Count lines of model code: source lines minus blank lines, comments and docstrings."""
    n = 0
    for o in objs:
        src = inspect.getsource(o)
        src = re.sub(r'("""|\'\'\')(.*?)\1', "", src, flags=re.S)  # drop docstrings
        n += sum(1 for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    return n


def model_size(m):
    """(#variables, #constraints) of a Pyomo model: unfixed Var objects and active Constraint objects."""
    from pyomo.environ import Constraint, Var

    nv = sum(1 for v in m.component_data_objects(Var) if not v.fixed)
    nc = sum(1 for _ in m.component_data_objects(Constraint, active=True))
    return nv, nc


_MP = {}


def _scip_mp():
    """True when the scip on PATH is the AMPL MP driver (option names like lim:time), not the classic SCIP binary."""
    if "mp" not in _MP:
        import subprocess

        exe = shutil.which("scip")
        try:
            out = subprocess.run([exe, "-="], capture_output=True, text=True, timeout=30).stdout if exe else ""
        except (OSError, subprocess.SubprocessError):
            out = ""
        _MP["mp"] = "lim:time" in out
    return _MP["mp"]


def solve_pyomo(m, solver="scip", objective=None, gap=GAP, tlim=TIME_LIMIT):
    """Solve a Pyomo model; return dict(obj, bound, time, status). Wall time includes writing the model."""
    obj = objective or next(m.component_data_objects(pyo.Objective, active=True))
    if solver == "baron" and not shutil.which("gams"):
        return dict(solver=solver, obj=np.nan, bound=np.nan, time=np.nan, status="not run: GAMS/BARON not installed")
    t = time.perf_counter()
    try:
        if solver == "baron":
            opts = [f"option optcr={gap}, optca=1e-6, reslim={tlim};"]
            r = pyo.SolverFactory("gams").solve(m, solver="baron", add_options=opts, load_solutions=False)
        elif solver == "scip" and _scip_mp():
            # AMPL MP driver (SCIP >= 10 from ampl_module_scip): it ignores scip.set and takes MP option names
            # from the scip_options environment variable; mip:bestbound returns the dual bound as a suffix
            if not hasattr(m, "bestbound"):
                m.bestbound = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            env = os.environ.get("scip_options")
            os.environ["scip_options"] = f"lim:time={tlim} lim:gap={gap} lim:absgap=1e-6 mip:bestbound=1"
            try:
                r = pyo.SolverFactory("scip").solve(m, load_solutions=False)
            finally:
                os.environ.pop("scip_options") if env is None else os.environ.update(scip_options=env)
        elif solver == "scip":
            r = pyo.SolverFactory("scip").solve(
                m, options={"limits/gap": gap, "limits/absgap": 1e-6, "limits/time": tlim}, load_solutions=False
            )
        elif solver == "couenne":
            with tempfile.TemporaryDirectory() as d:
                cwd = os.getcwd()
                os.chdir(d)
                try:
                    with open("couenne.opt", "w") as fh:
                        fh.write(f"allowable_fraction_gap {gap}\nallowable_gap 1e-6\ntime_limit {tlim}\n")
                    r = pyo.SolverFactory("couenne", executable=COUENNE).solve(m, load_solutions=False)
                finally:
                    os.chdir(cwd)
        else:
            raise ValueError(solver)
    except Exception as e:  # noqa: BLE001
        return dict(solver=solver, obj=np.nan, bound=np.nan, time=time.perf_counter() - t, status="error: " + str(e)[:60])
    dt = time.perf_counter() - t
    tc = str(r.solver.termination_condition)
    fun = lb = np.nan
    if solver == "scip" and _scip_mp():
        # Pyomo does not parse the MP driver's messages ("optimal solution", "gap limit", "time limit") and marks the
        # result unknown, which load_from refuses; classify the message here and mark a returned point as feasible
        msg = str(r.solver.message)
        tc = ("optimal" if "optimal solution" in msg or "gap limit" in msg else "infeasible" if "infeasible" in msg
              else "maxTimeLimit" if "time limit" in msg else tc)  # fmt: skip
        if len(r.solution) and tc != "infeasible":
            from pyomo.opt import SolutionStatus, SolverStatus

            r.solver.status = SolverStatus.ok
            r.solution(0).status = SolutionStatus.feasible
            bb = [v.get("bestbound") for v in r.solution(0).objective.values()]
            lb = float(bb[0]) if bb and bb[0] is not None else np.nan
    if len(r.solution) and tc not in ("noSolution", "licensingProblems", "error", "infeasible", "unknown"):
        try:
            m.solutions.load_from(r)
            fun = pyo.value(obj)
        except ValueError as e:
            return dict(solver=solver, obj=np.nan, bound=np.nan, time=dt, status="error: " + str(e)[:60])
    if solver == "scip" and _scip_mp():
        return dict(solver=solver, obj=fun, bound=lb if np.isfinite(lb) else np.nan, time=dt, status=tc)
    try:
        if solver == "scip":  # SCIP reports through the .nl solver message: gap limit == converged
            tc = "optimal" if "gap limit" in str(r.solver.message) or tc == "optimal" else tc
            lb = float(r.solver.dual_bound)
        else:
            lb = float(r.problem.lower_bound)
        if not np.isfinite(lb):
            lb = np.nan
    except (TypeError, ValueError, AttributeError):
        pass
    return dict(solver=solver, obj=fun, bound=lb, time=dt, status=tc)


def solve_jax(f, lo, hi, solver="jaxglobal", tlim=TIME_LIMIT, **kw):
    """Run jaxglobal.minimize with the same gap and time limit; return dict(obj, bound, time, status)."""
    from jaxglobal import minimize

    t = time.perf_counter()
    r = minimize(f, lo, hi, rel_tol=GAP, abs_tol=1e-6, time_limit=tlim, **kw)
    dt = time.perf_counter() - t
    return dict(solver=solver, obj=float(r.fun), bound=float(r.lower_bound), time=dt, status=r.status, x=r.x, nodes=r.nodes)


def table(rows, extra=None):
    """Results as a DataFrame with the common columns."""
    df = pd.DataFrame(rows)
    cols = ["solver", "obj", "bound", "time", "status"] + (extra or [])
    return df[[c for c in cols if c in df]].round({"obj": 6, "bound": 6, "time": 2})


def save_results(nb, case, loc_pyomo, loc_jax, size, rows, note=""):
    """Append one comparison to results/<nb>.json so 07-pyomo-overview can summarize it.

    ``size`` is (#vars, #constraints) of the Pyomo model; ``rows`` are solver result dicts.
    """
    import json

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{nb}.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    clean = [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items() if k not in ("x",)} for r in rows]
    data[case] = dict(loc_pyomo=loc_pyomo, loc_jax=loc_jax, pyomo_vars=size[0], pyomo_cons=size[1], rows=clean, note=note)
    json.dump(data, open(path, "w"), indent=1)

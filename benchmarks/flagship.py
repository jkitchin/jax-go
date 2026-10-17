"""Flagship benchmark: code-defined models, jaxglobal against equation-based global solvers.

Problems come from ``jaxglobal.applications`` (classes ``kinetics``, ``logic``, ``nn``).
jaxglobal solves the plain JAX function; SCIP, Couenne and BARON solve the Pyomo
formulation of the *same* model written by hand in each class module (the same RK4
steps, ``gdp``/binary reformulations of ``where``/``max``/``interp``). Pyomo models are
written to ``.nl`` (SCIP, Couenne) or ``.gms`` (BARON through GAMS). The hand-written
twins are kept as the reference formulations; ``jaxglobal.export.to_scip`` / ``to_nl``
can now also export the JAX function itself (``where``/``max`` via binaries or exact
identities), but the benchmark keeps the twins so its results stay comparable.

Every (solver, problem) pair runs in its own subprocess with a wall-clock limit and is
appended to the CSV as soon as it finishes, so an interrupted run resumes. All solvers
get relative gap 1e-4, absolute gap 1e-6 and the time limit.

Verdict, against the best known optimum ``fstar`` of that formulation:
  correct     proved optimal and the primal value equals fstar (within 2e-4 relative)
  wrong       claims optimal with a worse primal value, or reports a dual bound above fstar,
              or reports a primal value below fstar (fstar itself is then suspect)
  unsolved    time limit, error, unsupported or no model
  (unknown    no fstar available: the status is reported but not judged)

Usage:
  python benchmarks/flagship.py --solvers jaxglobal,scip,couenne,baron --classes kinetics,logic,nn \
      --only gasoil_h0.025 --time-limit 60 --out results/flagship_smoke.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
import pathlib
import shutil
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import run as R  # noqa: E402  (helpers of the MINLPLib runner: subprocess wrapper, log parsers)

FIELDS = ["solver", "cls", "name", "status", "verdict", "time", "primal", "dual", "nodes", "fstar",
          "loc_jax", "loc_pyomo", "message"]  # fmt: skip
TAG = R.TAG
SCIP = shutil.which("scip") or "/opt/homebrew/bin/scip"
GAMS = shutil.which("gams") or "/Library/Frameworks/GAMS.framework/Versions/53/Resources/gams"
REL_TOL_VERDICT = 2e-4


# ---------------------------------------------------------------- problems and lines of code
def load(classes, only=None):
    from jaxglobal.applications import CLASSES, suite

    out = []
    for c in classes:
        assert c in CLASSES, f"unknown class {c}"
        for p in suite([c]):
            if not only or p.name in only:
                out.append((c, p))
    return out


def loc(fn):
    """Source lines of ``fn`` without blank lines, comment-only lines and docstrings."""
    if fn is None:
        return 0
    src = textwrap.dedent(inspect.getsource(fn))
    drop = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node, clean=False):
            d = node.body[0]
            drop.update(range(d.lineno, d.end_lineno + 1))
    return sum(1 for i, ln in enumerate(src.splitlines(), 1) if ln.strip() and not ln.strip().startswith("#") and i not in drop)


def loc_jax(p):
    return sum(loc(g) for g in (p.f, p.ineq, p.eq) if g is not None)


def verdict(p_fstar, status, primal, dual):
    if p_fstar is None:
        return "unknown"
    if status != "optimal":
        return "unsolved" if status in ("time_limit", "error", "unsupported", "nomodel") else "wrong"
    tol = REL_TOL_VERDICT * max(abs(p_fstar), 1e-3) + 1e-9
    if primal != primal or abs(primal - p_fstar) > tol:
        return "wrong"
    if dual == dual and dual > p_fstar + tol:
        return "wrong"
    return "correct"


# ---------------------------------------------------------------- workers (run in a child process)
def worker(solver, cls, name, tl):
    p = next(q for c, q in load([cls], {name}))
    if solver == "jaxglobal":
        return worker_jaxglobal(p, tl)
    return worker_pyomo(solver, p, tl)


def worker_jaxglobal(p, tl):
    import jax

    jax.config.update("jax_enable_x64", True)
    from jaxglobal import minimize

    t0 = time.perf_counter()
    r = minimize(p.f, p.lo, p.hi, ineq=p.ineq, eq=p.eq, abs_tol=R.ABS_GAP, rel_tol=R.REL_GAP, feas_tol=R.FEAS_TOL,
                 time_limit=tl)  # fmt: skip
    status = {"optimal": "optimal", "infeasible": "infeasible", "time_limit": "time_limit"}.get(r.status, "error")
    return dict(status=status, time=time.perf_counter() - t0, primal=R._num(r.fun), dual=R._num(r.lower_bound),
                nodes=r.nodes, message=r.status)  # fmt: skip


def worker_pyomo(solver, p, tl):
    if p.pyomo is None:
        return dict(status="nomodel", message="no pyomo formulation")
    t0 = time.perf_counter()
    m = p.pyomo()
    build = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as tmp:
        if solver in ("scip", "couenne"):
            nl = pathlib.Path(tmp, "m.nl")
            m.write(str(nl), format="nl", io_options=dict(symbolic_solver_labels=False))
            r = run_scip(nl, tl, tmp) if solver == "scip" else run_couenne(nl, tl, tmp)
        elif solver == "baron":
            gms = pathlib.Path(tmp, "m.gms")
            m.write(str(gms), format="gams", io_options=dict(solver="baron"))
            r = run_baron(gms, tl, tmp)
        else:
            raise ValueError(solver)
    r["message"] = (r.get("message") or "") + f" [pyomo build {build:.1f}s]"
    return r


def run_scip(nl, tl, tmp):
    cmds = (f"set limits time {tl} set limits gap {R.REL_GAP} set limits absgap {R.ABS_GAP} "
            f"set numerics feastol {R.FEAS_TOL} read {nl} opt display statistics quit")  # fmt: skip
    out = R._run([SCIP, "-c", cmds], tl, cwd=tmp)
    if isinstance(out, dict):
        return out
    st = R._grab(r"SCIP Status\s*:\s*(.*)", out) or ""
    status = ("optimal" if "optimal solution found" in st or "gap limit" in st else "infeasible" if "infeasible" in st
              else "time_limit" if "time limit" in st else "error")  # fmt: skip
    return dict(status=status, time=R._float(R._grab(r"Solving Time \(sec\)\s*:\s*(\S+)", out)),
                primal=R._num(R._float(R._grab(r"Primal Bound\s*:\s*(\S+)", out))),
                dual=R._num(R._float(R._grab(r"Dual Bound\s*:\s*(\S+)", out))),
                nodes=R._float(R._grab(r"Solving Nodes\s*:\s*(\d+)", out)), message=st.strip() or out.strip()[-120:])  # fmt: skip


def run_couenne(nl, tl, tmp):
    pathlib.Path(tmp, "couenne.opt").write_text(
        f"time_limit {tl}\nallowable_fraction_gap {R.REL_GAP}\nallowable_gap {R.ABS_GAP}\nfeas_tolerance {R.FEAS_TOL}\n")
    t0 = time.perf_counter()
    out = R._run([R.COUENNE, str(nl), "-AMPL"], tl, cwd=tmp)
    wall = time.perf_counter() - t0
    if isinstance(out, dict):
        return out
    lb, ub = R._float(R._grab(r"Lower bound:\s*(\S+)", out)), R._float(R._grab(r"Upper bound:\s*(\S+)", out))
    t = R._float(R._grab(r"Total solve time:\s*([\d.e+-]+)s", out))
    t = wall if t != t else t
    if ub == ub and lb == lb and ub - lb <= max(R.ABS_GAP, R.REL_GAP * abs(ub)) + 1e-12:
        status = "optimal"
    elif "infeasible" in out.lower() and ub != ub:
        status = "infeasible"
    elif ub != ub and lb != lb and t < tl * 0.99:
        status = "error"
    else:
        status = "time_limit"
    return dict(status=status, time=t, primal=R._num(ub), dual=R._num(lb),
                nodes=R._float(R._grab(r"Branch-and-bound nodes:\s*(\d+)", out)),
                message="" if status != "error" else out.strip()[-160:])  # fmt: skip


def run_baron(gms, tl, tmp):
    trc = pathlib.Path(tmp, "t.trc")
    cmd = [GAMS, str(gms), "solver=baron", f"reslim={tl}", f"optcr={R.REL_GAP}", f"optca={R.ABS_GAP}",
           f"trace={trc}", "traceopt=3", "lo=3", f"curdir={tmp}", "solvelink=5"]  # fmt: skip
    out = R._run(cmd, tl, cwd=tmp)
    if isinstance(out, dict):
        return out
    lst = R._read(pathlib.Path(tmp, "m.lst"))
    bad = R._grab(r"\*\*\* (Cannot handle .*|.*license.*)", out + lst) or R._grab(r"(\*\*\* Status: .*)", out)
    if not trc.exists():
        return dict(status="error", message=(bad or out.strip()[-200:]))
    import csv as _csv

    rows = [ln for ln in trc.read_text().splitlines() if ln and not ln.startswith("*")]
    cols = ["file", "type", "solver", "nlp", "mip", "juliantoday", "dir", "eq", "var", "dvar", "nz", "nlnz", "dnz",
            "mstat", "sstat", "obj", "objest", "res", "iter", "dom", "nodes", "user1"]  # fmt: skip
    r = dict(zip(cols, next(_csv.reader(rows[-1:]))))
    mstat, sstat = int(float(r["mstat"])), int(float(r["sstat"]))
    status = ("optimal" if mstat in (1, 8) and sstat == 1 else "infeasible" if mstat in (4, 19)
              else "time_limit" if sstat == 3 else "error")  # fmt: skip
    return dict(status=status, time=R._float(r["res"]), primal=R._float(r["obj"]), dual=R._float(r["objest"]),
                nodes=R._float(r["nodes"]), message=f"model {mstat} solver {sstat}" + (f" {bad}" if bad else ""))  # fmt: skip


# ---------------------------------------------------------------- driver
def solve(solver, cls, p, tl):
    t0 = time.perf_counter()
    cmd = [sys.executable, __file__, "--worker", f"{solver}:{cls}:{p.name}", "--time-limit", str(tl)]
    out = R._run(cmd, tl, cwd=str(HERE))
    try:
        r = out if isinstance(out, dict) else json.loads(next(ln for ln in out.splitlines() if ln.startswith(TAG))[len(TAG):])
    except Exception:
        r = dict(status="error", message=out.strip()[-200:])
    r.setdefault("time", time.perf_counter() - t0)
    f = p.fstar
    primal, dual = R._float(r.get("primal")), R._float(r.get("dual"))
    row = {k: r.get(k, "") for k in FIELDS}
    row.update(solver=solver, cls=cls, name=p.name, fstar="" if f is None else f, loc_jax=loc_jax(p),
               loc_pyomo=loc(p.pyomo), verdict=verdict(f, r.get("status"), primal, dual))  # fmt: skip
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solvers", default="jaxglobal,scip,couenne,baron")
    ap.add_argument("--classes", default="kinetics,logic,nn")
    ap.add_argument("--only", default="", help="comma-separated problem names")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--jobs", type=int, default=1, help="solves run concurrently (timings degrade when >1)")
    ap.add_argument("--out", default="results/flagship.csv")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        s, c, n = a.worker.split(":")
        print(TAG + json.dumps(worker(s, c, n, a.time_limit)))
        return

    out = pathlib.Path(a.out)
    out = out if out.is_absolute() else HERE / out
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["solver"], r["name"]) for r in csv.DictReader(open(out))}
    else:
        with open(out, "w", newline="") as fh:
            csv.DictWriter(fh, FIELDS).writeheader()
    probs = load([c for c in a.classes.split(",") if c], set(a.only.split(",")) - {""} or None)
    todo = [(s, c, p) for s in a.solvers.split(",") for c, p in probs if (s, p.name) not in done]
    print(f"{len(todo)} runs ({len(done)} already in {out.name}), time limit {a.time_limit:g}s")

    def go(job):
        r = solve(*job, a.time_limit)
        with open(out, "a", newline="") as fh:
            csv.DictWriter(fh, FIELDS).writerow(r)
        print(f"{r['solver']:10s} {r['name']:20s} {r['status']:11s} {r['verdict']:9s} {float(r['time'] or 0):8.2f}s "
              f"primal {r['primal']} dual {r['dual']} fstar {r['fstar']}", flush=True)  # fmt: skip

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, todo))


if __name__ == "__main__":
    main()

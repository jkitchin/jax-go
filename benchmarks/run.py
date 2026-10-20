"""Run global solvers on the MINLPLib continuous set and record the results.

Every (solver, instance) pair runs in its own subprocess with a wall-clock
limit, and its result is appended to ``benchmarks/results/<tag>.csv`` as soon
as it finishes, so an interrupted run resumes where it stopped.

All solvers get the same termination criteria: relative gap 1e-4, absolute
gap 1e-6, feasibility tolerance 1e-6, and the time limit.

Solvers:
  jaxglobal  this package, on the model read by jaxglobal.nl
  scip       SCIP command line, reading the .nl file (pyscipopt's .nl reader if no classic scip binary)
  couenne    Couenne (AMPL build), reading the .nl file
  baron      BARON through GAMS, on the MINLPLib .gms file

Usage:
  python benchmarks/run.py --solvers jaxglobal,scip,couenne,baron --time-limit 60
  python benchmarks/run.py --max-vars 10 --jobs 4 --tag small
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

_T_START = time.time()  # the worker's start (see _drain)
HERE = pathlib.Path(__file__).resolve().parent
LIB = HERE / "minlplib"
REL_GAP, ABS_GAP, FEAS_TOL = 1e-4, 1e-6, 1e-6
EXTRA = {"safe": True} if os.environ.get("JAXGLOBAL_SAFE") == "1" else {}  # check_wrong.py --safe
EXTRA.update(json.loads(os.environ.get("JAXGLOBAL_OPTS", "{}")))  # extra minimize options, e.g. '{"branch_var": "reliability"}'
# "load" (1-min load average when the run started) was added for the SCIP-gap re-baseline; CSVs
# written before it have no such column and are appended to with their own header
FIELDS = ["solver", "name", "status", "time", "primal", "dual", "nodes", "message", "load"]
TAG = "JAXGLOBAL-RESULT "  # marks the worker's result line among solver log output
COUENNE = shutil.which("couenne") or "/Applications/AMPL/couenne"


# ---------------------------------------------------------------- solvers
# Each returns a dict with status in {optimal, time_limit, infeasible,
# unsupported, error}, time (s), and primal / dual bounds in the model's own
# objective sense (NaN when unknown).


def run_jaxglobal(name, tl):
    """Runs in a child Python process (see worker()), so a crash or hang is contained."""
    cmd = [sys.executable, __file__, "--worker", name, "--time-limit", str(tl)]
    out = _run(cmd, tl)
    if isinstance(out, dict):
        return out
    try:
        return json.loads(next(ln for ln in out.splitlines() if ln.startswith(TAG))[len(TAG):])
    except Exception:
        return dict(status="error", message=out.strip()[-200:])


def worker(name, tl):
    import jax

    jax.config.update("jax_enable_x64", True)

    sys.path.insert(0, str(HERE.parent))
    from jaxglobal import minimize
    from jaxglobal.nl import NLError, read_nl

    try:
        M = read_nl(LIB / "nl" / f"{name}.nl", integers="mask")
    except NLError as e:
        return dict(status="unsupported", message=str(e))
    t0 = time.perf_counter()
    r = minimize(M.f, M.lo, M.hi, ineq=M.ineq, eq=M.eq, integer=M.integer, abs_tol=ABS_GAP, rel_tol=REL_GAP,
                 feas_tol=FEAS_TOL, time_limit=tl, **EXTRA)  # fmt: skip
    if r.status == "unbounded_domain":
        return dict(status="unsupported", message="unbounded variables: " + r.message)
    p, d = M.sense * r.fun, M.sense * r.lower_bound  # back to the model's objective sense
    status = {"optimal": "optimal", "infeasible": "infeasible", "time_limit": "time_limit"}.get(r.status, "error")
    return dict(status=status, time=time.perf_counter() - t0, primal=_num(p), dual=_num(d),
                nodes=r.nodes, message=r.status)  # fmt: skip


def _drain(tl):
    """Let the worker's background kernel builds finish (see main()) for at most the runner's
    timeout less a margin, or ``$JAXGLOBAL_DRAIN`` s (0: exit at once)."""
    budget = tl * 1.5 + 60 - (time.time() - _T_START) - 20  # _run's timeout, less what has run and a margin
    if "JAXGLOBAL_DRAIN" in os.environ:
        budget = min(budget, float(os.environ["JAXGLOBAL_DRAIN"]))
    kc = sys.modules.get("jaxglobal.kcache")
    if budget > 0 and kc is not None and kc.cache_root() is not None:  # (nothing to keep without a cache)
        kc.wait_for_builds(budget)


_SCIP_PY = """
import sys, pyscipopt
m = pyscipopt.Model()
m.hideOutput()
m.readProblem(sys.argv[1])
for k, v in (("limits/time", float(sys.argv[2])), ("limits/gap", float(sys.argv[3])), ("limits/absgap", float(sys.argv[4])),
             ("numerics/feastol", float(sys.argv[5]))):
    m.setParam(k, v)
m.optimize()
print("SCIP Status        :", m.getStatus())
print("Solving Time (sec) :", m.getSolvingTime())
print("Primal Bound       :", m.getPrimalbound())
print("Dual Bound         :", m.getDualbound())
print("Solving Nodes      :", m.getNNodes())
print("SCIP Version       :", m.version(), "pyscipopt", pyscipopt.__version__)
"""


def _classic_scip():
    """The classic SCIP binary on PATH (``scip -c``), not the AMPL MP driver, which takes no ``-c``."""
    exe = shutil.which("scip")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-c", "quit"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return exe if "SCIP version" in out else None


def run_scip_pyscipopt(name, tl):
    """SCIP through pyscipopt's .nl reader in a child process, with the same limits and tolerances."""
    nl = str(LIB / "nl" / f"{name}.nl")
    out = _run([sys.executable, "-c", _SCIP_PY, nl, str(tl), str(REL_GAP), str(ABS_GAP), str(FEAS_TOL)], tl)
    if isinstance(out, dict):
        return out
    st = _grab(r"SCIP Status\s*:\s*(.*)", out) or ""
    status = {"optimal": "optimal", "gaplimit": "optimal", "infeasible": "infeasible", "timelimit": "time_limit"}.get(st.strip(), "error")
    return dict(status=status, time=_float(_grab(r"Solving Time \(sec\)\s*:\s*(\S+)", out)),
                primal=_num(_float(_grab(r"Primal Bound\s*:\s*(\S+)", out))),
                dual=_num(_float(_grab(r"Dual Bound\s*:\s*(\S+)", out))),
                nodes=_float(_grab(r"Solving Nodes\s*:\s*(\d+)", out)),
                message=f"{st.strip()} ({(_grab(r'SCIP Version\s*:\s*(.*)', out) or '').strip()})" if st else out.strip()[-160:])  # fmt: skip


def run_scip(name, tl):
    if _classic_scip() is None:
        return run_scip_pyscipopt(name, tl)
    with tempfile.TemporaryDirectory() as tmp:
        cmds = (f"set limits time {tl} set limits gap {REL_GAP} set limits absgap {ABS_GAP} "
                f"set numerics feastol {FEAS_TOL} read {LIB / 'nl' / name}.nl opt "
                f"display statistics quit")  # fmt: skip
        out = _run(["scip", "-c", cmds], tl, cwd=tmp)
    if isinstance(out, dict):
        return out
    st = _grab(r"SCIP Status\s*:\s*(.*)", out) or ""
    status = ("optimal" if "optimal solution found" in st or "gap limit" in st else "infeasible" if "infeasible" in st
              else "time_limit" if "time limit" in st else "error")  # fmt: skip
    return dict(status=status, time=_float(_grab(r"Solving Time \(sec\)\s*:\s*(\S+)", out)),
                primal=_float(_grab(r"Primal Bound\s*:\s*(\S+)", out)),
                dual=_float(_grab(r"Dual Bound\s*:\s*(\S+)", out)),
                nodes=_float(_grab(r"Solving Nodes\s*:\s*(\d+)", out)), message=st.strip())  # fmt: skip


def run_couenne(name, tl, sense):
    with tempfile.TemporaryDirectory() as tmp:
        pathlib.Path(tmp, "couenne.opt").write_text(
            f"time_limit {tl}\nallowable_fraction_gap {REL_GAP}\nallowable_gap {ABS_GAP}\nfeas_tolerance {FEAS_TOL}\n"
        )
        t0 = time.perf_counter()
        out = _run([COUENNE, str(LIB / "nl" / f"{name}.nl"), "-AMPL"], tl, cwd=tmp)
        wall = time.perf_counter() - t0
    if isinstance(out, dict):
        return out
    # Couenne prints bounds of the minimization form (a max objective is negated)
    lb, ub = _float(_grab(r"Lower bound:\s*(\S+)", out)), _float(_grab(r"Upper bound:\s*(\S+)", out))
    t = _float(_grab(r"Total solve time:\s*([\d.e+-]+)s", out))
    t = wall if t != t else t
    if "infeasible" in out.lower() and ub != ub:
        status = "infeasible"
    elif ub == ub and lb == lb and ub - lb <= max(ABS_GAP, REL_GAP * abs(ub)) + 1e-12:
        status = "optimal"
    elif t >= tl * 0.99:
        status = "time_limit"
    else:
        status = "error" if ub != ub and lb != lb else "time_limit"
    return dict(status=status, time=t, primal=_num(sense * ub), dual=_num(sense * lb),
                nodes=_float(_grab(r"Branch-and-bound nodes:\s*(\d+)", out)), message="")  # fmt: skip


def run_gams(name, tl, solver="baron"):
    with tempfile.TemporaryDirectory() as tmp:
        trc = pathlib.Path(tmp, "t.trc")
        cmd = ["gams", str(LIB / "gms" / f"{name}.gms"), f"solver={solver}", f"reslim={tl}",
               f"optcr={REL_GAP}", f"optca={ABS_GAP}", f"trace={trc}", "traceopt=3", "lo=3",
               f"curdir={tmp}", "solvelink=5"]  # fmt: skip
        out = _run(cmd, tl, cwd=tmp)
        if isinstance(out, dict):
            return out
        bad = _grab(r"\*\*\* (Cannot handle .*)", out + _read(pathlib.Path(tmp, f"{name}.lst")))
        if bad:
            return dict(status="unsupported", message=bad)
        if not trc.exists():
            return dict(status="error", message=out.strip()[-200:])
        rows = [ln for ln in trc.read_text().splitlines() if ln and not ln.startswith("*")]
    # trace columns (traceopt=3): see GAMS "Solve trace" documentation
    f = next(csv.reader(rows[-1:]))
    cols = ["file", "type", "solver", "nlp", "mip", "juliantoday", "dir", "eq", "var", "dvar", "nz",
            "nlnz", "dnz", "mstat", "sstat", "obj", "objest", "res", "iter", "dom", "nodes", "user1"]  # fmt: skip
    r = dict(zip(cols, f))
    mstat, sstat = int(float(r["mstat"])), int(float(r["sstat"]))
    ub, lb = _float(r["obj"]), _float(r["objest"])
    closed = ub == ub and lb == lb and abs(ub - lb) <= max(ABS_GAP, REL_GAP * abs(ub)) + 1e-12
    # model status 1 optimal; 2 locally optimal and 8 integer solution are reported when a solve
    # ends at the gap limit (8 for models with integers), so count them solved when the gap is closed
    status = ("optimal" if sstat == 1 and (mstat == 1 or (mstat in (2, 8) and closed)) else "infeasible" if mstat in (4, 19)
              else "time_limit" if sstat == 3 else "error")  # fmt: skip
    return dict(status=status, time=_float(r["res"]), primal=_float(r["obj"]), dual=_float(r["objest"]),
                nodes=_float(r["nodes"]), message=f"model {mstat} solver {sstat}")  # fmt: skip


# ---------------------------------------------------------------- helpers


def _run(cmd, tl, cwd=None):
    """stdout+stderr as text, or a result dict if the process hung or failed to start."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=tl * 1.5 + 60, cwd=cwd,
                           env={**os.environ, "PYTHONWARNINGS": "ignore"})  # fmt: skip
    except subprocess.TimeoutExpired:
        return dict(status="time_limit", time=tl, message="killed")
    except OSError as e:
        return dict(status="error", message=str(e))
    return p.stdout + p.stderr


def _read(path):
    return path.read_text(errors="replace") if path.exists() else ""


def _grab(pat, text):
    m = re.search(pat, text)
    return m.group(1) if m else None


def _float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _num(v):
    v = float(v)
    return v if abs(v) < 1e19 else float("nan")


def instances(max_vars, names=None):
    from minlplib.fetch import references

    rows = list(references().values())
    rows = [r for r in rows if int(r["nvars"]) <= max_vars]
    return [r for r in rows if r["name"] in names] if names else rows


def solve(solver, inst, tl):
    name, sense = inst["name"], (-1 if inst["objsense"] == "max" else 1)
    load = f"{os.getloadavg()[0]:.2f}"
    t0 = time.perf_counter()
    try:
        if solver == "jaxglobal":
            r = run_jaxglobal(name, tl)
        elif solver == "scip":
            r = run_scip(name, tl)
        elif solver == "couenne":
            r = run_couenne(name, tl, sense)
        elif solver in ("baron", "antigone", "lindoglobal", "scip-gams"):
            r = run_gams(name, tl, solver.replace("-gams", ""))
        else:
            raise ValueError(solver)
    except Exception as e:  # a crashing adapter must not stop the run
        r = dict(status="error", message=repr(e)[:200])
    r.setdefault("time", time.perf_counter() - t0)
    return {k: r.get(k, "") for k in FIELDS} | dict(solver=solver, name=name, load=load)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solvers", default="jaxglobal,scip,couenne,baron")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--max-vars", type=int, default=50)
    ap.add_argument("--names", default="", help="comma-separated instance names")
    ap.add_argument("--jobs", type=int, default=4, help="instances run concurrently")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        print(TAG + json.dumps(worker(a.worker, a.time_limit)), flush=True)
        sys.stdout.flush()
        # A kernel still being traced or compiled when minimize returned at its time limit reaches
        # the disk caches only once it finishes: without this a solve that needs more than the time
        # limit to compile never gets warm, however often it is rerun. Wait for those builds (the
        # result is already recorded), but exit well before the runner's timeout (_run), which
        # would lose the result.
        try:
            _drain(a.time_limit)
        finally:
            os._exit(0)
        return

    out = HERE / "results" / f"{a.tag}.csv"
    out.parent.mkdir(exist_ok=True)
    done, fields = set(), FIELDS
    if out.exists():
        with open(out) as fh:
            rd = csv.DictReader(fh)
            done = {(r["solver"], r["name"]) for r in rd}
            fields = rd.fieldnames or FIELDS  # an older CSV keeps its columns (no "load")
    else:
        with open(out, "w", newline="") as fh:
            csv.DictWriter(fh, FIELDS).writeheader()
    insts = instances(a.max_vars, set(a.names.split(",")) if a.names else None)
    todo = [(s, i) for s in a.solvers.split(",") for i in insts if (s, i["name"]) not in done]
    print(f"{len(todo)} runs ({len(done)} already in {out.name}), time limit {a.time_limit:g}s, {a.jobs} jobs")

    def go(job):
        r = solve(job[0], job[1], a.time_limit)
        with open(out, "a", newline="") as fh:
            csv.DictWriter(fh, fields, extrasaction="ignore").writerow(r)
        print(f"{r['solver']:10s} {r['name']:28s} {r['status']:11s} {float(r['time'] or 0):8.2f}s "
              f"primal {r['primal']} dual {r['dual']}", flush=True)  # fmt: skip

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, todo))


if __name__ == "__main__":
    main()

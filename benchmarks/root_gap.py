"""One root-bound measurement (docs/ANALYSIS-root-bounds.md): python benchmarks/root_gap.py <scip|jaxglobal> <nl path> -> a JSON line."""
import sys, json, time, io, contextlib
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
solver, path = sys.argv[1], sys.argv[2]
out = {}
if solver == "jaxglobal":
    import numpy as np
    import jaxglobal
    from jaxglobal.nl import read_nl
    from jaxglobal import minimize
    M = read_nl(path, integers="mask")
    t = time.perf_counter()
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        r = minimize(M.f, M.lo, M.hi, ineq=M.ineq, eq=M.eq, integer=M.integer, abs_tol=1e-6, rel_tol=1e-4,
                     feas_tol=1e-6, max_iter=0, time_limit=180)
    out = dict(root_lb=float(M.sense * r.lower_bound), ub=float(M.sense * r.fun), status=r.status, nodes=r.nodes,
               time=time.perf_counter() - t, sense=int(M.sense))
else:
    import pyscipopt
    m = pyscipopt.Model(); m.hideOutput(); m.readProblem(path)
    m.setParam("limits/time", 60.0); m.setParam("limits/gap", 1e-4); m.setParam("limits/absgap", 1e-6)
    m.setParam("numerics/feastol", 1e-6)
    t = time.perf_counter(); m.optimize()
    out = dict(root_lb=m.getDualboundRoot(), dual=m.getDualbound(), primal=m.getPrimalbound(), status=m.getStatus(),
               time=time.perf_counter() - t, nodes=m.getNNodes())
print("RESULT " + json.dumps(out), flush=True)
import os; os._exit(0)

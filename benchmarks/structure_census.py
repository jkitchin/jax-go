"""Which node bounder each benchmark model gets: lifted quadratic/polynomial LP, convexity certificates.

For every instance of the 2026-10-20 re-run (global50, start50, int50, held out): derive finite
bounds if needed, then ask ``quadratic.quadratic_form`` / ``polynomial_form`` whether minimize would
use the lifted LP, and ``lp.pool_parts`` how many parts are certified convex on the root box.
Writes one CSV row per instance with the warm and cold re-run outcomes.

Usage:  python benchmarks/structure_census.py results/structure_census.csv
"""
import sys, csv, logging, time, signal; logging.disable(logging.INFO)
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/user/jax-go"); sys.path.insert(0, "/home/user/jax-go/benchmarks")
from jaxglobal.nl import read_nl
from jaxglobal import quadratic, lp
from jaxglobal.domain import bound_domain
import report
from minlplib.fetch import references
R = references()
class TO(Exception): pass
def alarm(*_): raise TO()
signal.signal(signal.SIGALRM, alarm)
out = csv.writer(open(sys.argv[1], "w")); out.writerow(["set", "name", "n", "lifted", "convex_parts", "warm_solved", "cold_solved", "note"])
for s in ["global50", "start50", "int50", "heldout"]:
    warm = {r["name"]: report.classify(r, R[r["name"]]) for r in csv.DictReader(open(f"/home/user/jax-go/benchmarks/results/rerun_{s}_warm.csv"))}
    cold = {r["name"]: report.classify(r, R[r["name"]]) for r in csv.DictReader(open(f"/home/user/jax-go/benchmarks/results/rerun_{s}_cold.csv"))}
    for name in warm:
        note = ""; lifted = convex = ""
        try:
            signal.alarm(60)
            M = read_nl(f"/home/user/jax-go/benchmarks/minlplib/nl/{name}.nl", integers="mask"); n = len(M.lo)
            integer = tuple(np.flatnonzero(M.integer))
            lo, hi = np.asarray(M.lo, float), np.asarray(M.hi, float)
            if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
                lo, hi, *_ = bound_domain(M.f, M.ineq, M.eq, lo, hi, integer=M.integer)
            Q = None
            if M.ineq is not None or M.eq is not None:
                Q = quadratic.quadratic_form(M.f, M.ineq, M.eq, n) or quadratic.polynomial_form(M.f, M.ineq, M.eq, n, integer=integer, sym_seconds=0.0)
            lifted = int(Q is not None)
            ok = lp.pool_parts(M.f, M.ineq, M.eq, n)
            if ok is not None and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                pmask, exact, cvx = ok(lo, hi, convex=True)
                convex = int(np.asarray(pmask).sum()) if pmask is not None else 0
            else:
                convex = 0
            signal.alarm(0)
        except TO:
            note = "timeout"
        except Exception as e:
            signal.alarm(0); note = repr(e)[:60]
        out.writerow([s, name, n if not note else "", lifted, convex, warm[name], cold.get(name, ""), note])
        sys.stdout.flush()
print("done")

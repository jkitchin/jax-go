"""Root bound prototype: LP and MILP over linear cuts of jaxglobal's McCormick relaxations.

python benchmarks/root_oa_prototype.py <nl path> [rounds] [milp_time]   (analysis prototype, see docs/ANALYSIS-root-bounds.md)

Cuts (all valid on the root box, so for every feasible point, integer or not):
  t >= cv_f(p) + grad cv_f(p).(x - p)
  cv_g(p) + J cv_g(p) (x - p) <= 0
  cv_h(p) + J cv_h(p) (x - p) <= 0,   cc_h(p) + J cc_h(p) (x - p) >= 0
at the box center, 2n axis points, random points, and then at each round's solution (Kelley /
outer approximation). Prints the LP bound and the MILP dual bound (HiGHS) per round, in the
model's sense.
"""

import io
import contextlib
import json
import sys
import time

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

import jaxglobal
from jaxglobal import domain
from jaxglobal.fbbt import make_fbbt
from jaxglobal.local import _vec
from jaxglobal.nl import read_nl
from jaxglobal.relax import relax

path = sys.argv[1]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
milp_time = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
M = read_nl(path, integers="mask")
n = len(M.lo)
imask = np.asarray(M.integer, bool)
lo, hi = np.asarray(M.lo, float), np.asarray(M.hi, float)
with contextlib.redirect_stderr(io.StringIO()):
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        lo, hi, _, _, _ = domain.bound_domain(M.f, M.ineq, M.eq, lo, hi, integer=imask)
    a, b, e = jax.jit(make_fbbt(M.f, M.ineq, M.eq, rounds=30))(lo, hi, np.inf)
lo, hi = np.maximum(lo, np.asarray(a)), np.minimum(hi, np.asarray(b))
lo[imask], hi[imask] = np.ceil(lo[imask] - 1e-9), np.floor(hi[imask] + 1e-9)
assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), "unbounded root box"
g, h = _vec(M.ineq), _vec(M.eq)
L, H = jnp.asarray(lo), jnp.asarray(hi)
rf, rg, rh = relax(M.f), relax(g), relax(h)


def parts(x):
    mf, mg, mh = rf(x, L, H), rg(x, L, H), rh(x, L, H)
    return mf.cv, mg.cv, mh.cv, mh.cc


val = jax.jit(parts)
jac = jax.jit(jax.jacfwd(parts))
flo = float(rf(0.5 * (L + H), L, H).lo)  # interval lower bound of f on the box

rows_A, rows_lo, rows_hi = [], [], []


def add_cuts(P):
    for p in P:
        p = jnp.asarray(p)
        v = [np.atleast_1d(np.asarray(t, float)) for t in val(p)]
        J = [np.atleast_2d(np.asarray(t, float)) for t in jac(p)]
        pn = np.asarray(p)
        # objective: cvf(p) + gf.(x - p) - t <= 0
        if np.isfinite(v[0][0]) and np.all(np.isfinite(J[0])):
            rows_A.append(np.r_[J[0][0], -1.0]); rows_lo.append(-np.inf); rows_hi.append(J[0][0] @ pn - v[0][0])
        for k, (vv, JJ, sense) in enumerate(((v[1], J[1], "le"), (v[2], J[2], "le"), (v[3], J[3], "ge"))):
            for i in range(vv.size):
                if not (np.isfinite(vv[i]) and np.all(np.isfinite(JJ[i]))) or np.max(np.abs(JJ[i]), initial=0) > 1e9:
                    continue
                rhs = JJ[i] @ pn - vv[i]
                if sense == "le":
                    rows_A.append(np.r_[JJ[i], 0.0]); rows_lo.append(-np.inf); rows_hi.append(rhs + 1e-6)
                else:
                    rows_A.append(np.r_[JJ[i], 0.0]); rows_lo.append(rhs - 1e-6); rows_hi.append(np.inf)


c = 0.5 * (lo + hi)
w = hi - lo
P0 = [c] + [np.clip(c + s * 0.5 * w * np.eye(n)[j], lo, hi) for j in range(n) for s in (-1, 1)]
rng = np.random.default_rng(0)
P0 += list(rng.uniform(lo, hi, (2 * n, n)))
t0 = time.perf_counter()
add_cuts(P0)
bounds_lp = Bounds(np.r_[lo, flo], np.r_[hi, np.inf])
cost = np.r_[np.zeros(n), 1.0]
out = dict(instance=path.split("/")[-1], n=n, n_int=int(imask.sum()), rounds=[])
for rd in range(rounds + 1):
    A = np.array(rows_A)
    lp = linprog(cost, A_ub=None, bounds=list(zip(np.r_[lo, flo], np.r_[hi, np.inf])),
                 A_eq=None, method="highs", options={"presolve": True}) if False else None
    cons = LinearConstraint(A, np.array(rows_lo), np.array(rows_hi))
    rl = milp(cost, constraints=cons, integrality=np.zeros(n + 1), bounds=bounds_lp)
    lp_b = rl.fun if rl.x is not None else np.nan
    rm = milp(cost, constraints=cons, integrality=np.r_[imask.astype(int), 0], bounds=bounds_lp,
              options={"time_limit": milp_time, "mip_rel_gap": 1e-6})
    mb = getattr(rm, "mip_dual_bound", None)
    mb = rm.fun if (mb is None and rm.x is not None) else mb
    out["rounds"].append(dict(round=rd, rows=len(rows_A), lp=M.sense * lp_b, milp_dual=None if mb is None else M.sense * float(mb),
                              milp_status=int(rm.status), t=round(time.perf_counter() - t0, 1)))
    print(json.dumps(out["rounds"][-1]), flush=True)
    new = [x for x in (rl.x, rm.x) if x is not None]
    if not new or rd == rounds:
        break
    add_cuts([np.clip(x[:n], lo, hi) for x in new])
print("RESULT " + json.dumps(out), flush=True)
import os  # noqa: E402

os._exit(0)

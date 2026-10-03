"""Run the global solver on every test problem and print a summary table.

Columns:
  root LB    plain interval-arithmetic bound on the whole box (shows the
             problem is not trivially solved by interval bounds)
  multistart fraction of 20 random L-BFGS-B starts reaching f* (shows the
             problem is multimodal / nonconvex)
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as sp_minimize

from jaxglobal import interval, minimize
from jaxglobal.problems import PROBLEMS


def multistart_success(p, fstar, n=20, seed=0):
    rng = np.random.default_rng(seed)
    vg = jax.jit(jax.value_and_grad(p.f))
    hits = 0
    for _ in range(n):
        r = sp_minimize(lambda x: tuple(np.asarray(v) for v in vg(x)), rng.uniform(p.lo, p.hi),
                        jac=True, method="L-BFGS-B", bounds=list(zip(p.lo, p.hi)))
        hits += r.fun <= fstar + 1e-4 * max(1, abs(fstar))
    return hits / n


if __name__ == "__main__":
    hdr = (f"{'problem':19s} {'n':>2s} {'f* (lit.)':>11s} {'UB found':>11s} {'LB proven':>11s} "
           f"{'gap':>8s} {'root LB':>10s} {'mstart':>6s} {'nodes':>7s} {'time':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for p in PROBLEMS:
        r = minimize(p.f, p.lo, p.hi, abs_tol=1e-5, rel_tol=1e-6, max_iter=4000)
        root_lb = float(interval(p.f)(jnp.asarray(p.lo), jnp.asarray(p.hi))[0])
        fstar = p.fstar if p.fstar is not None else r.fun
        known = "?" if p.fstar is None else f"{p.fstar:.6f}"
        flag = "" if r.converged else "  NOT CONVERGED"
        print(f"{p.name:19s} {p.lo.size:2d} {known:>11s} {r.fun:11.6f} {r.lower_bound:11.6f} "
              f"{r.gap:8.1e} {root_lb:10.3g} {multistart_success(p, fstar):6.0%} "
              f"{r.nodes:7d} {r.time:5.1f}s{flag}")

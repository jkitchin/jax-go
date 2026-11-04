"""One-node split diagnostic: at the root (after root OBBT), split every nonlinear variable once
(at its midpoint, and at the relaxation point) and bound both children with the problem's own
node kernel; report each split's gain in the weaker child's bound, with and without OBBT on the
children, and where jaxglobal's own root pick (widest relative width) ranks.

Usage:  python benchmarks/split_oracle.py wastewater15m1 2446.43   (name, cutoff = an incumbent value)
"""
import sys, time, logging; logging.disable(logging.INFO)
import numpy as np, jax, jax.numpy as jnp; jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/user/jax-go")
from jaxglobal import bnb
from jaxglobal.nl import read_nl
from jaxglobal.lp import _nonlinear_vars
name, cutoff = sys.argv[1], float(sys.argv[2])
M = read_nl(f"/home/user/jax-go/benchmarks/minlplib/nl/{name}.nl", integers="mask"); n = len(M.lo)
lo, hi = np.asarray(M.lo, float), np.asarray(M.hi, float)
K = bnb._Kernels(M.f, M.ineq, M.eq, n_fw=20, n_kelley=4, fbbt=True, fbbt_rounds=5, dbbt=True, obbt="root", obbt_rounds=2,
                 univariate=True, alphabb=True, multivariate=False, lifted=False, crown=False, feas_tol=1e-6, safe=False,
                 integer=(), quadratic=True, cut_pool=0, exact=False)
print("lifted LP used:", K.lifted_form(n) is not None, flush=True)
bound = jax.jit(K.bound.jitted) if hasattr(K.bound, "jitted") else K.bound
obbt = K.obbt.jitted
c = jnp.asarray(cutoff)
def B(L, H):  # node kernel on a batch: (lb, x, fx, lo', hi', xr)
    return [np.asarray(v) for v in bound(jnp.asarray(L), jnp.asarray(H), c)]
lb, x, fx, rlo, rhi, xr = B(lo[None], hi[None])
print(f"root bound {lb[0]:.6g}", flush=True)
olo, ohi, oe = [np.asarray(v) for v in obbt(jnp.asarray(rlo), jnp.asarray(rhi), c)]
lb, x, fx, rlo, rhi, xr = B(olo, ohi)
L0, H0, lb0, xr0 = rlo[0], rhi[0], lb[0], xr[0]
print(f"root bound after OBBT {lb0:.6g}; widths: max {np.max(H0-L0):.3g}", flush=True)
nl = _nonlinear_vars(M.f, M.ineq, M.eq, n); nlv = nl.any(0) if nl is not None else np.ones(n, bool)
cand = np.flatnonzero(nlv & (H0 - L0 > 1e-9))
def children(j, p):
    Ls, Hs = np.tile(L0, (2, 1)), np.tile(H0, (2, 1))
    Hs[0, j] = p; Ls[1, j] = p
    return Ls, Hs
rows = []
for mode in ("mid", "relax"):
    Ls, Hs = [], []
    for j in cand:
        p = 0.5 * (L0[j] + H0[j]) if mode == "mid" else float(np.clip(xr0[j], L0[j] + 0.05 * (H0[j]-L0[j]), H0[j] - 0.05 * (H0[j]-L0[j])))
        a, b = children(j, p); Ls.append(a); Hs.append(b)
    Ls, Hs = np.concatenate(Ls), np.concatenate(Hs)
    clb = B(Ls, Hs)[0].reshape(-1, 2)
    # with OBBT on the children first
    ol, oh, _ = [np.asarray(v) for v in obbt(jnp.asarray(Ls), jnp.asarray(Hs), c)]
    clb2 = B(ol, oh)[0].reshape(-1, 2)
    for k, j in enumerate(cand):
        rows.append((mode, int(j), H0[j] - L0[j], min(clb[k]) - lb0, min(clb2[k]) - lb0))
rel = (H0 - L0) / np.maximum(hi - lo, 1e-300)
pick = cand[np.argmax(rel[cand])]
print(f"{len(cand)} candidate variables; jaxglobal's root pick (widest relative width): x{pick}")
for mode in ("mid", "relax"):
    R = sorted([r for r in rows if r[0] == mode], key=lambda r: -r[3])
    mine = next(r for r in R if r[1] == pick)
    print(f"[{mode}] best splits (gain in min child bound; with child OBBT): " + ", ".join(f"x{r[1]} {r[3]:.4g} ({r[4]:.4g})" for r in R[:5]))
    print(f"[{mode}] jaxglobal's pick x{pick}: gain {mine[3]:.4g} ({mine[4]:.4g}); rank {R.index(mine)+1} of {len(R)}; median gain {np.median([r[3] for r in R]):.4g}")

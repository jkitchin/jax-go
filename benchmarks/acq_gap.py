"""Where the bound gap of GP acquisition functions comes from, and a Taylor-model prototype.

Diagnostic only (not used by the solver); see docs/PLAN-acquisition-relaxations.md.
On the 15-point Matérn 5/2 GP of tests/test_gp.py, for random boxes of each width:

1. ``decoupling``: for LCB, -EI, -PI and -log EI, the gap (sampled min - node bound) and the
   share of it that remains when the exact (mu, sigma) ranges are combined independently
   (the best any rule on (mu, sigma) alone can reach).
2. ``ranges``: the width of mu's and sigma's enclosure over the true range, for the current
   relaxations and for first-order Taylor models with RKHS remainders,

       mu(x) = mu(c) + g.d + r,                    |r| <= ||mu||_H e(d)
       sigma(x) in [s~(d) - e(d), s~(d) + e(d)],   s~(d) = sd_post(f(c) + grad f(c).d)

   with ``e(d)^2 = ||k_x - k_c - D k_c d||_H^2 = 2 sf2 (1 - phi(s) + 2 s phi'(s) - s phi'(0))``,
   ``s = ||d / l||^2``, nondecreasing in ``s`` (so its box maximum is at the farthest corner).
3. ``nodes``: the node bound of LCB and -EI against the vertex bound from those models
   (min over vertices of the concave mu(c) + g.d - kappa s~(d); max over vertices of the
   convex EI(mu(c) + g.d - R e, s~(d) + e)).

    python benchmarks/acq_gap.py [decoupling|ranges|nodes ...]
"""

import itertools
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
import jaxglobal  # noqa: E402,F401  (enables x64)
from jaxglobal import kernels, make_bounder  # noqa: E402
from jaxglobal.gp import expected_improvement as EI  # noqa: E402
from jaxglobal.gp import gp_mean, gp_std  # noqa: E402
from jaxglobal.gp import probability_of_improvement as PI  # noqa: E402
from jaxglobal.relax import relax  # noqa: E402
from test_gp import trained  # noqa: E402

KER = kernels.KERNELS["matern52"]
KER_D1 = jax.grad(KER)  # phi'(s); the k' primitive (finite at s = 0 for matern52)
Z, LS, A, ALPHA, SF2 = trained(n_train=15)
FMIN = float(np.min(np.sin(3 * Z[:, 0]) + Z[:, 1] ** 2)) - 0.05
R = float(np.sqrt(ALPHA @ (SF2 * np.asarray(KER(jnp.asarray((((Z[:, None] - Z[None]) / LS) ** 2).sum(-1))))) @ ALPHA))
WIDTHS = [0.5, 0.2, 0.05, 0.01, 0.002]
NBOX = 25


def mu(x):
    return gp_mean(x, Z, LS, ALPHA, SF2)


def sd(x):
    return gp_std(x, Z, LS, A, SF2)


def kX(a):
    return SF2 * KER(jnp.sum(((a - Z) / LS) ** 2, -1))


def kpost(a, b):
    return SF2 * KER(jnp.sum(((a - b) / LS) ** 2)) - kX(a) @ A @ kX(b)


P2, P12 = jax.grad(kpost, 1), jax.jacfwd(jax.grad(kpost, 1), 0)


def stilde(dl, c):  # sd_post(f(c) + grad f(c).dl): the sqrt of a convex quadratic in dl
    return jnp.sqrt(jnp.maximum(kpost(c, c) + 2 * dl @ P2(c, c) + dl @ P12(c, c) @ dl, 0.0))


def e_max(lo, hi, c):  # RKHS remainder at the farthest corner
    s = jnp.sum((jnp.maximum(c - lo, hi - c) / LS) ** 2)
    return float(jnp.sqrt(jnp.maximum(2 * SF2 * (1 - KER(s) + 2 * s * KER_D1(s) - s * KER_D1(0.0)), 0.0)))


def boxes(w, rng):
    for _ in range(NBOX):
        c = rng.uniform(-1 + w / 2, 1 - w / 2, 2)
        lo, hi = c - w / 2, c + w / 2
        yield jnp.asarray(c), jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(rng.uniform(lo, hi, (3000, 2)))


mus, sds, gmu = jax.jit(jax.vmap(mu)), jax.jit(jax.vmap(sd)), jax.jit(jax.grad(mu))
ST = jax.jit(jax.vmap(stilde, (0, None)))


def decoupling():
    acq = {
        "LCB": (lambda x: mu(x) - 2 * sd(x), lambda m, s: m - 2 * s),
        "-EI": (lambda x: -EI(mu(x), sd(x), FMIN), lambda m, s: -EI(m, s, FMIN)),
        "-PI": (lambda x: -PI(mu(x), sd(x), FMIN), lambda m, s: -PI(m, s, FMIN)),
        "-logEI": (lambda x: -jnp.log(EI(mu(x), sd(x), FMIN)), lambda m, s: -jnp.log(EI(m, s, FMIN))),
    }
    rng = np.random.default_rng(0)
    for name, (f, g) in acq.items():
        bound = jax.jit(make_bounder(f, univariate=True, alphabb=True))
        fv = jax.jit(jax.vmap(f))
        print(f"\n{name}: width | median gap | share left with exact (mu, sigma) ranges | median rel gap")
        for w in WIDTHS:
            gaps, share, rel = [], [], []
            for _, lo, hi, X in boxes(w, rng):
                tmin, lb = float(fv(X).min()), float(bound(lo, hi)[0])
                M, S = mus(X), sds(X)
                dmin = min(float(g(a, b)) for a in (M.min(), M.max()) for b in (S.min(), S.max()))  # g monotone
                if np.isfinite(lb) and np.isfinite(tmin):
                    gaps.append(tmin - lb)
                    share.append(np.clip((tmin - dmin) / max(tmin - lb, 1e-300), 0, 1))
                    rel.append((tmin - lb) / (abs(tmin) + 1e-12))
            print(f"  {w:6.3f} | {np.median(gaps):9.2e} | {np.median(share):4.2f} | {np.median(rel):8.2e}")


def ranges():
    rm, rs = relax(mu), relax(sd)
    rng = np.random.default_rng(1)
    print(f"||mu||_H = {R:.3g}, |alpha|_1 = {np.abs(ALPHA).sum():.3g}")
    print(" width | mu enclosure / true range: McCormick  Taylor | sigma: gp_std  Taylor | e_max")
    for w in WIDTHS:
        r = [[], [], [], [], []]
        for c, lo, hi, X in boxes(w, rng):
            em = e_max(lo, hi, c)
            M, S = mus(X), sds(X)
            tm, ts = float(M.max() - M.min()), float(S.max() - S.min())
            m, s = rm(c, lo, hi), rs(c, lo, hi)
            g = gmu(c)
            St = ST(X - c, c)
            assert float(jnp.max(jnp.abs(S - St))) <= em + 1e-9  # the models hold on the samples
            assert float(jnp.max(jnp.abs(M - mu(c) - (X - c) @ g))) <= R * em + 1e-9
            r[0].append(float(m.hi - m.lo) / tm)
            r[1].append(float(2 * jnp.abs(g) @ (hi - c) + 2 * R * em) / tm)
            r[2].append(float(s.hi - s.lo) / ts)
            r[3].append((float(St.max() - St.min()) + 2 * em) / ts)  # sampled range of s~: diagnostic
            r[4].append(em)
        print(f" {w:6.3f} | {np.median(r[0]):9.1f} {np.median(r[1]):9.2f} | {np.median(r[2]):6.1f} {np.median(r[3]):6.2f}"
              f" | {np.median(r[4]):.1e}")


def nodes():
    fl = lambda x: mu(x) - 2 * sd(x)  # noqa: E731
    fe = lambda x: -EI(mu(x), sd(x), FMIN)  # noqa: E731
    bl = jax.jit(make_bounder(fl, univariate=True, alphabb=True))
    be = jax.jit(make_bounder(fe, univariate=True, alphabb=True))
    fvl, fve = jax.jit(jax.vmap(fl)), jax.jit(jax.vmap(fe))
    rng = np.random.default_rng(2)
    print("width | LCB gap: current  vertex | -EI gap: current  vertex   (median of sampled min - bound)")
    for w in WIDTHS[:-1]:
        r = [[], [], [], []]
        for c, lo, hi, X in boxes(w, rng):
            V = jnp.asarray(list(itertools.product(*zip(np.asarray(lo), np.asarray(hi)))))
            em = e_max(lo, hi, c)
            lin, St = mu(c) + (V - c) @ gmu(c), ST(V - c, c)
            vl = float(jnp.min(lin - 2 * St)) - (R + 2) * em
            ve = -float(jnp.max(EI(lin - R * em, St + em, FMIN)))
            tl, te = float(fvl(X).min()), float(fve(X).min())
            assert vl <= tl + 1e-9 and ve <= te + 1e-9
            r[0].append(tl - float(bl(lo, hi)[0]))
            r[1].append(tl - vl)
            r[2].append(te - float(be(lo, hi)[0]))
            r[3].append(te - ve)
        print(f"{w:5.2f} | {np.median(r[0]):9.2e} {np.median(r[1]):9.2e} | {np.median(r[2]):9.2e} {np.median(r[3]):9.2e}")


if __name__ == "__main__":
    for part in sys.argv[1:] or ["decoupling", "ranges", "nodes"]:
        print(f"\n=== {part} ===")
        {"decoupling": decoupling, "ranges": ranges, "nodes": nodes}[part]()

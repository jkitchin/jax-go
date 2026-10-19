"""Like terms ``a_k * v * log(v)`` of a sum merged into one ``mc_xlogx`` (``relax.LogTerms``, MINLPLib ex6_2_6).

ex6_2_6 (a Gibbs energy of mixing) writes ``x log x`` for each component in several products with
coefficients ``9.6 + 18.3 + 4.1 - 31.0 = 1``. Relaxed term by term, the concave ``-31 x log x`` gets
the secant, whose gap ``31 w / e`` on ``[0, w]`` the search could not close near ``x = 0`` (bound
-1.3e-5 vs the optimum -2.6e-6 after 10k nodes). The merged sum is ``1 * x log x``: convex, exact.
"""

import sys

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize
from jaxglobal.relax import MC, Ctx, LogTerms, eval_jaxpr_mc, relax
from jaxglobal.univariate import make_univariate

R = sys.modules["jaxglobal.relax"]
L = jnp.log


def gibbs(x):
    """ex6_2_6's pattern: scaled logs times the variable, from several sources, plus x * log(sum)."""
    x0, x1, x2 = x[0], x[1], x[2]
    s1 = 3.9235 * x0 + 6.0909 * x1 + 0.92 * x2
    s2 = 3.664 * x0 + 5.168 * x1 + 1.4 * x2
    return (L(s1) * (26.9071667605344 * x0 + 41.7710875549227 * x1 + 6.30931398488382 * x2)
            + 0.668686155614739 * x0 - 1.14374230885457 * x1 + 2.8906196099828 * x2
            + 9.58716676053442 * L(x0) * x0 + 16.9310875549227 * L(x1) * x1 + 0.309313984883821 * L(x2) * x2
            - 9.58716676053442 * L(s1) * x0 - 16.9310875549227 * L(s1) * x1 - 0.309313984883821 * L(s1) * x2
            + 18.32 * L(x0) * x0 + 25.84 * L(x1) * x1 + 7.0 * L(x2) * x2
            - 18.32 * L(s2) * x0 - 25.84 * L(s2) * x1 - 7.0 * L(s2) * x2
            + 4.0643 * L(x0) * x0 + 5.7409 * L(x1) * x1 + 1.6741 * L(x2) * x2
            - 31.9714667605344 * L(x0) * x0 - 48.5119875549227 * L(x1) * x1 - 8.98341398488382 * L(x2) * x2)  # fmt: skip


def forms(x):
    """Every recognized form: ``neg``, ``sub``, ``div`` by a constant, ``(b v) * (c log v)``, a net
    coefficient of exactly 0, a negative net coefficient, and a non-atom remainder."""
    x0, x1 = x[0], x[1]
    return (-(2.5 * L(x0)) * x0 + (x0 * 3.0) * (L(x0) / 4.0) - (x0 * L(x0)) * 0.1
            + 2.0 * (L(x1) * x1) - x1 * (2.0 * L(x1))
            - 3.0 * x1 * L(x1) + jnp.sin(x0) - (x1 * L(x0)) / 3.0 + 1.5)  # fmt: skip


def _mp(fn_name, x):
    """``fn`` at ``x`` in 60-digit arithmetic (the exact value the safe bounds must enclose)."""
    mpmath.mp.dps = 60
    lg = mpmath.log
    v = [mpmath.mpf(float(t)) for t in x]
    m = lambda c: mpmath.mpf(c)  # noqa: E731
    if fn_name == "gibbs":
        x0, x1, x2 = v
        s1 = m(3.9235) * x0 + m(6.0909) * x1 + m(0.92) * x2
        s2 = m(3.664) * x0 + m(5.168) * x1 + m(1.4) * x2
        return (lg(s1) * (m(26.9071667605344) * x0 + m(41.7710875549227) * x1 + m(6.30931398488382) * x2)
                + m(0.668686155614739) * x0 - m(1.14374230885457) * x1 + m(2.8906196099828) * x2
                + (m(9.58716676053442) + m(18.32) + m(4.0643) - m(31.9714667605344)) * lg(x0) * x0
                + (m(16.9310875549227) + m(25.84) + m(5.7409) - m(48.5119875549227)) * lg(x1) * x1
                + (m(0.309313984883821) + m(7.0) + m(1.6741) - m(8.98341398488382)) * lg(x2) * x2
                - (m(9.58716676053442) * x0 + m(16.9310875549227) * x1 + m(0.309313984883821) * x2) * lg(s1)
                - (m(18.32) * x0 + m(25.84) * x1 + m(7.0) * x2) * lg(s2))  # fmt: skip
    x0, x1 = v[0], v[1]
    return (-m(2.5) * lg(x0) * x0 + m(3.0) * x0 * lg(x0) / m(4.0) - x0 * lg(x0) * m(0.1)
            + m(2.0) * lg(x1) * x1 - x1 * m(2.0) * lg(x1) - m(3.0) * x1 * lg(x1)
            + mpmath.sin(x0) - x1 * lg(x0) / m(3.0) + m(1.5))  # fmt: skip


FNS = {"gibbs": (gibbs, 3), "forms": (forms, 2)}


def _boxes(rng, k, n):
    """Ordinary boxes, boxes at (or padded just below) 0, tiny boxes near 0, huge ranges and degenerate ones."""
    kinds = []
    lo = rng.uniform(0, 1, (k, n))
    kinds.append((lo, lo + 10.0 ** rng.uniform(-4, 0, (k, n))))
    lo = np.where(rng.uniform(size=(k, n)) < 0.5, 0.0, -1e-10)
    kinds.append((lo, 10.0 ** rng.uniform(-6, 0, (k, n))))
    lo = 10.0 ** rng.uniform(-14, -6, (k, n))
    kinds.append((lo, lo * 10.0 ** rng.uniform(0, 4, (k, n))))
    lo = 10.0 ** rng.uniform(-9, -3, (k, n))
    kinds.append((lo, 10.0 ** rng.uniform(3, 6, (k, n))))
    lo = 10.0 ** rng.uniform(-8, 1, (k, n))
    kinds.append((lo, lo.copy()))
    lo = np.concatenate([a for a, _ in kinds])
    hi = np.concatenate([b for _, b in kinds])
    mix = rng.permutation(lo.shape[0])  # mix the kinds across coordinates too
    j = rng.uniform(size=lo.shape) < 0.3
    return np.where(j, lo[mix], lo), np.where(j, hi[mix], hi)


def _points(rng, lo, hi, m):
    """``m`` points per box inside its log domain (x > 0), corners included."""
    a = np.maximum(lo, 1e-300)
    t = rng.uniform(size=(lo.shape[0], m, lo.shape[1]))
    t[:, 0], t[:, 1] = 0.0, 1.0
    return a[:, None] + t * (hi - a)[:, None]


def _relaxers(fn, safe):
    prep, rw = make_univariate(fn, safe=safe, interpret=False)
    return [relax(fn, safe=safe), lambda x, lo, hi: rw(x, lo, hi, prep(lo, hi))]


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("name", ["gibbs", "forms"])
def test_valid_relaxations_and_cuts(name, safe):
    """cv <= f <= cc and lo <= f <= hi at random points of edge-case boxes, and the cut of cv (cc) at
    one point stays below (above) f at the others, traced and through the tape relaxer. In safe mode
    against the exact value (60 digits) at the worst points, with no tolerance."""
    fn, n = FNS[name]
    rng = np.random.default_rng(0)
    lo, hi = _boxes(rng, 60, n)
    Z = _points(rng, lo, hi, 12)
    P = Z[:, 2]
    fz = np.asarray(jax.vmap(jax.vmap(fn))(Z))
    assert np.all(np.isfinite(fz))
    # plain mode is not rounding-safe: allow ulps of the terms' magnitude (~1e8 on boxes up to 1e5)
    S = 100 * np.sum(1 + hi * (1 + np.abs(np.log(hi))), axis=1)[:, None]
    tol = 1e-9 * (1 + np.abs(fz)) + 1e-14 * S
    for rf in _relaxers(fn, safe):
        m = jax.jit(jax.vmap(rf))(P, lo, hi)
        gcv = jax.jit(jax.vmap(jax.grad(lambda x, a, b: rf(x, a, b).cv)))(P, lo, hi)
        gcc = jax.jit(jax.vmap(jax.grad(lambda x, a, b: rf(x, a, b).cc)))(P, lo, hi)
        mz = jax.jit(jax.vmap(jax.vmap(rf, in_axes=(0, None, None))))(Z, lo, hi)
        # NaN (no bound) only where a remaining product such as x1 * log(x0) meets 0 * inf; checked
        # against the term-by-term relaxation in test_no_new_nans
        cv, cc = (
            np.nan_to_num(np.asarray(mz.cv), nan=-np.inf),
            np.nan_to_num(np.asarray(mz.cc), nan=np.inf),
        )
        mlo, mhi = (
            np.nan_to_num(np.asarray(m.lo), nan=-np.inf)[:, None],
            np.nan_to_num(np.asarray(m.hi), nan=np.inf)[:, None],
        )
        assert np.mean(np.isfinite(cv)) > 0.5
        d = Z - P[:, None, :]
        cut_lo = np.asarray(m.cv)[:, None] + np.einsum(
            "kj,kij->ki", np.nan_to_num(np.asarray(gcv)), d
        )
        cut_hi = np.asarray(m.cc)[:, None] + np.einsum(
            "kj,kij->ki", np.nan_to_num(np.asarray(gcc)), d
        )
        fin = np.all(np.isfinite(np.asarray(gcv)), 1)[:, None] & np.isfinite(cut_lo)
        fin_hi = np.all(np.isfinite(np.asarray(gcc)), 1)[:, None] & np.isfinite(cut_hi)
        if not safe:
            assert np.all(cv <= fz + tol) and np.all(fz <= cc + tol)
            assert np.all(mlo <= fz + tol) and np.all(fz <= mhi + tol)
            assert np.all(
                np.where(fin, cut_lo, -np.inf) <= fz + 1e-7 * (1 + np.abs(fz)) + tol
            )
            assert np.all(
                np.where(fin_hi, cut_hi, np.inf) >= fz - 1e-7 * (1 + np.abs(fz)) - tol
            )
            continue
        # safe: compare with the exact value where the float64 margin is smallest
        lower = np.minimum(np.maximum(cv, mlo), np.inf)
        upper = np.minimum(cc, mhi)
        for side, b in (("lo", lower), ("hi", upper)):
            gap = (fz - b) if side == "lo" else (b - fz)
            idx = np.argsort(gap / (1 + np.abs(fz)), axis=None)[:40]
            for i in idx:
                k, j = np.unravel_index(i, gap.shape)
                ex = _mp(name, Z[k, j])
                assert (b[k, j] <= ex) if side == "lo" else (ex <= b[k, j]), (
                    side,
                    Z[k, j],
                    lo[k],
                    hi[k],
                )


@pytest.mark.parametrize("name", ["gibbs", "forms"])
def test_no_new_nans(name, monkeypatch):
    """Boxes with a NaN component are a subset of those of the term-by-term relaxation (where the
    product rule's 0 * inf on ``c log(v) * v`` gave NaN, the merged sum gives a bound)."""
    fn, n = FNS[name]
    rng = np.random.default_rng(0)
    lo, hi = _boxes(rng, 60, n)
    P = _points(rng, lo, hi, 3)[:, 2]
    nan = lambda m: np.any([np.isnan(np.asarray(c)) for c in m], axis=0)  # noqa: E731
    new = [nan(jax.jit(jax.vmap(rf))(P, lo, hi)) for rf in _relaxers(fn, False)]
    monkeypatch.setattr(
        LogTerms, "observe", lambda self, name, ins, invals, outvar, out, eqn: out
    )
    old = [nan(jax.jit(jax.vmap(rf))(P, lo, hi)) for rf in _relaxers(fn, False)]
    for a, b in zip(new, old):
        assert not np.any(a & ~b) and a.sum() < b.sum()


def test_tighter_than_term_by_term(monkeypatch):
    """On gibbs, cv and lo are never looser than term by term and much tighter on boxes of width
    0.01 near a corner (where the concave x log x terms are: about 2x at the median)."""
    rng = np.random.default_rng(1)
    c = rng.uniform(0, 0.05, (200, 3))
    lo, hi = np.maximum(c, 1e-6), c + 0.01
    P = rng.uniform(lo, hi)
    new = jax.vmap(relax(gibbs))(P, lo, hi)
    monkeypatch.setattr(
        LogTerms, "observe", lambda self, name, ins, invals, outvar, out, eqn: out
    )
    old = jax.vmap(relax(gibbs))(P, lo, hi)
    tol = 1e-12 * (1 + np.abs(np.asarray(old.cv)))
    assert np.all(np.asarray(new.cv) >= np.asarray(old.cv) - tol)
    assert np.all(np.asarray(new.lo) >= np.asarray(old.lo) - tol)
    f = np.asarray(jax.vmap(gibbs)(P))
    gap_new, gap_old = f - np.asarray(new.cv), f - np.asarray(old.cv)
    assert np.median(gap_old / np.maximum(gap_new, 1e-300)) > 1.5


def test_net_coefficient_is_exact():
    """``2 x log x - 3 x log x + 2 x log x`` is ``x log x``: the relaxation is the convex function itself."""
    f = lambda x: 2.0 * L(x[0]) * x[0] - 3.0 * x[0] * L(x[0]) + (x[0] * L(x[0])) * 2.0  # noqa: E731
    lo, hi = jnp.array([1e-6]), jnp.array([1.0])
    for p in (1e-6, 0.01, 0.3, 1.0):
        m = relax(f)(jnp.array([p]), lo, hi)
        assert np.isclose(
            float(m.cv), p * np.log(p), rtol=1e-12, atol=1e-14
        )  # (term by term: ulps of 7 x log x)
        assert np.isclose(float(m.lo), -np.exp(-1.0), rtol=1e-12)


def test_not_applied_in_interval_mode_or_without_atoms():
    """Interval-only interpretation and sums without ``v log v`` products are unchanged."""
    lt = LogTerms(set(), Ctx(interval_only=True))
    out = MC(*(jnp.asarray(v) for v in (0.0, 1.0, 0.0, 1.0)))
    assert lt.observe("add", (1, 2), [out, out], 3, out, None) is out
    f = lambda x: 2.0 * L(x[0]) * x[1] + x[0] * L(x[1]) - 3.0 * x[1]  # noqa: E731
    closed = jax.make_jaxpr(f)(jnp.ones(2))
    seen = []
    orig = LogTerms._relax
    try:
        LogTerms._relax = lambda self, r: seen.append(r) or orig(self, r)
        eval_jaxpr_mc(
            closed.jaxpr,
            closed.consts,
            [MC(jnp.full(2, 0.5), jnp.full(2, 2.0), jnp.ones(2), jnp.ones(2))],
        )
    finally:
        LogTerms._relax = orig
    assert seen == []


def test_unbounded_and_nonpositive_boxes(monkeypatch):
    """A box with ``v <= 0`` (log undefined) or an infinite upper end: the interval stays [-inf, inf]
    (no NaN) and the result is NaN only where the term-by-term relaxation already is."""
    boxes = (
        ([-1.0, 0.1, 0.1], [-0.5, 1.0, 1.0]),
        ([0.1, 0.1, 0.1], [np.inf, 1.0, 1.0]),
    )

    def run():
        rf = jax.jit(relax(gibbs, safe=True))
        out = []
        for lo, hi in boxes:
            lo, hi = jnp.array(lo), jnp.array(hi)
            out.append(
                rf(
                    jnp.clip(jnp.array([0.5, 0.5, 0.5]), lo, jnp.minimum(hi, 1.0)),
                    lo,
                    hi,
                )
            )
        return out

    new = run()
    monkeypatch.setattr(
        LogTerms, "observe", lambda self, name, ins, invals, outvar, out, eqn: out
    )
    for a, b in zip(new, run()):
        assert float(a.lo) == -np.inf and float(a.hi) == np.inf
        assert all(not np.isnan(float(u)) or np.isnan(float(v)) for u, v in zip(a, b))


@pytest.mark.parametrize("safe", [False, True])
def test_gibbs_problem_solves(safe):
    """min gibbs on the simplex (x_i >= 1e-6): optimal, with a bound below the best value found by
    a multistart local search (slow list)."""
    from scipy.optimize import minimize as sp_min

    lo, hi = np.full(3, 1e-6), np.ones(3)
    eq = lambda x: jnp.array([x[0] + x[1] + x[2] - 1.0])  # noqa: E731
    fj, gj = jax.jit(gibbs), jax.jit(jax.grad(gibbs))
    best = np.inf
    for s in np.random.default_rng(0).dirichlet([0.5] * 3, 60):
        r = sp_min(
            lambda x: float(fj(x)),
            np.clip(s, 1e-6, 1),
            jac=lambda x: np.asarray(gj(x)),
            bounds=[(1e-6, 1)] * 3,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
            method="SLSQP",
            options={"ftol": 1e-15},
        )
        if r.success and abs(r.x.sum() - 1) < 1e-9:
            best = min(best, r.fun)
    r = minimize(
        gibbs, lo, hi, eq=eq, abs_tol=1e-6, rel_tol=1e-4, time_limit=300, safe=safe
    )
    assert r.status == "optimal"
    assert r.lower_bound <= best + 1e-9 and r.fun <= best + 1e-6

"""Products of powers ``c * prod_i b_i ** a_i`` relaxed in log space (``relax.Monomials``, MINLPLib launch).

A posynomial term is convex in ``log x``. Factor-by-factor McCormick products lose that, which left
launch's root gap at ~2% (2214 vs 2257.8 after OBBT). The log-space relaxation of each whole
product is intersected with the product rule's.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize
from jaxglobal.relax import MC, Ctx, Monomials, mc_monomial, relax
from jaxglobal.univariate import make_univariate

R = sys.modules["jaxglobal.relax"]
A = (1.2781, -0.1959, 2.4242, 0.38745, -0.9904)  # launch's first objective term


def term(x):
    return 5.0 * x[0] ** A[0] * x[1] ** A[1] * x[2] ** A[2] * x[3] ** A[3] * x[4] ** A[4]


def mixed(x):
    """Repeated bases (x0 ** 0.7 * x0 ** 0.5), an expression base, a division and a constant factor."""
    return 3.0 * x[0] ** 0.7 * (0.001 * x[1]) ** -0.4 * x[0] ** 0.5 / x[2] * x[2] ** 0.25


def _boxes(rng, k, n):
    lo = 10.0 ** rng.uniform(-2, 1, (k, n))
    hi = lo * 10.0 ** rng.uniform(0.01, 2, (k, n))
    return lo, hi


def _relaxers(fn, safe):
    prep, rw = make_univariate(fn, safe=safe, interpret=False)
    return [relax(fn, safe=safe), lambda x, lo, hi: rw(x, lo, hi, prep(lo, hi))]


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("fn", [term, mixed])
def test_valid_relaxations_and_cuts(fn, safe):
    """cv <= f <= cc and lo <= f <= hi at random points of random positive boxes, and the cut of cv
    (cc) at one point stays below (above) f at the others, traced and through the tape relaxer."""
    n = 5 if fn is term else 3
    rng = np.random.default_rng(0)
    lo, hi = _boxes(rng, 300, n)
    P = rng.uniform(lo, hi)
    Z = rng.uniform(lo[:, None, :], hi[:, None, :], (300, 40, n))
    fz = np.asarray(jax.vmap(jax.vmap(fn))(Z))
    tol = 1e-9 * (1 + np.abs(fz))
    for rf in _relaxers(fn, safe):
        one = jax.jit(jax.vmap(rf))
        m = one(P, lo, hi)
        gcv = jax.jit(jax.vmap(jax.grad(lambda x, l, h: rf(x, l, h).cv)))(P, lo, hi)
        gcc = jax.jit(jax.vmap(jax.grad(lambda x, l, h: rf(x, l, h).cc)))(P, lo, hi)
        mz = jax.jit(jax.vmap(jax.vmap(rf, in_axes=(0, None, None))))(Z, lo, hi)
        for c in m:
            assert np.all(np.isfinite(np.asarray(c)))
        assert np.all(np.asarray(mz.cv) <= fz + tol) and np.all(fz <= np.asarray(mz.cc) + tol)
        assert np.all(np.asarray(m.lo)[:, None] <= fz + tol) and np.all(fz <= np.asarray(m.hi)[:, None] + tol)
        d = Z - P[:, None, :]
        cut_lo = np.asarray(m.cv)[:, None] + np.einsum("kj,kij->ki", np.asarray(gcv), d)
        cut_hi = np.asarray(m.cc)[:, None] + np.einsum("kj,kij->ki", np.asarray(gcc), d)
        assert np.all(cut_lo <= fz + tol) and np.all(cut_hi >= fz - tol)


def test_tighter_than_the_product_rule(monkeypatch):
    """On launch-like boxes the convex side is tighter than factor-by-factor McCormick (and never looser)."""
    rng = np.random.default_rng(1)
    lo, hi = _boxes(rng, 200, 5)
    P = rng.uniform(lo, hi)
    new = jax.vmap(relax(term))(P, lo, hi)
    monkeypatch.setattr(Monomials, "observe", lambda self, name, ins, invals, outvar, out: out)
    old = jax.vmap(relax(term))(P, lo, hi)
    tol = 1e-12 * (1 + np.abs(np.asarray(old.cv)))
    assert np.all(np.asarray(new.cv) >= np.asarray(old.cv) - tol)
    assert np.all(np.asarray(new.cc) <= np.asarray(old.cc) + 1e-12 * (1 + np.abs(np.asarray(old.cc))))
    assert np.mean(np.asarray(new.cv) > np.asarray(old.cv) * (1 + 1e-6)) > 0.5


def test_falls_back_where_a_base_is_not_positive_and_bounded(monkeypatch):
    """A box where a base reaches 0 or is unbounded above keeps the product rule's result exactly,
    values and gradients (no NaN from the log-space branch that is not taken)."""

    def g(x):
        return x[0] ** 1.5 * x[1] ** 0.5 * x[2]

    lo = jnp.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    hi = jnp.array([[2.0, 3.0, 4.0], [2.0, jnp.inf, 4.0]])
    P = jnp.array([[1.0, 2.0, 2.0], [1.5, 2.0, 2.0]])

    def run():
        m = jax.vmap(relax(g))(P, lo, hi)
        gr = jax.vmap(jax.grad(lambda x, l, h: relax(g)(x, l, h).cv))(P, lo, hi)
        return [np.asarray(c) for c in m] + [np.asarray(gr)]

    new = run()
    assert not np.any(np.isnan(new[-1][0]))
    ok, _ = mc_monomial(1.0, [(MC(lo[:, k], hi[:, k], P[:, k], P[:, k]), a) for k, a in enumerate((1.5, 0.5, 1.0))])
    assert not np.any(np.asarray(ok))
    monkeypatch.setattr(Monomials, "observe", lambda self, name, ins, invals, outvar, out: out)
    for a, b in zip(new, run()):
        assert np.array_equal(a, b, equal_nan=True)


def test_not_applied_in_interval_mode_or_to_plain_products():
    """FBBT's interval sweeps are untouched, and products without a power (``x * y``) keep McCormick."""
    m = Monomials(set(), Ctx(interval_only=True))
    out = MC(*(jnp.asarray(v) for v in (0.0, 1.0, 0.0, 1.0)))
    assert m.observe("pow", (0, None), [out, 2.5], 1, out) is out

    def h(x):
        return x[0] * x[1] * x[2]

    lo, hi, p = jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 5.0, 4.0]), jnp.array([1.5, 3.0, 3.5])
    calls = []
    orig = R.mc_monomial

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    R.mc_monomial = spy
    try:
        relax(h)(p, lo, hi)
        assert not calls
        relax(term)(jnp.full(5, 2.0), jnp.ones(5), jnp.full(5, 3.0))
        assert len(calls) == 1  # once, at the whole product, not at every partial product
    finally:
        R.mc_monomial = orig


def test_safe_covers_values_exp_flushes_to_zero():
    """A monomial below the normal range: exp of its log-space bounds flushes to 0, so safe mode adds an
    absolute margin on top of the relative one (intersected with the product rule, hi stayed at 0)."""

    def f(x):
        return x[0] ** 2.5 * x[1] ** 0.5

    lo, hi = jnp.array([1e-124, 1.0]), jnp.array([1.0000001e-124, 1.0])
    out = relax(f, safe=True)(lo, lo, hi)
    assert float(out.hi) >= 1e-310 and float(out.cc) >= 1e-310
    assert float(out.lo) <= 1e-310 and float(out.cv) <= 1e-310


@pytest.mark.parametrize("safe", [False, True])
def test_coefficients_that_overflow_or_scale_below_normal(safe):
    """A constant product that overflows (1e300 * 1e300, 1 / 1e-310) leaves the term to the product rule;
    a small coefficient gets the absolute safe margin after scaling, not before."""
    cases = [(lambda x: x[0] ** 2.5 * x[1] * 1e300 * 1e300, [1e-80, 1e-100], [2e-80, 2e-100])]
    if safe:  # (plain mode's division rule alone already gives inf for / 1e-310)
        cases.append((lambda x: x[0] ** 2.5 * x[1] / 1e-310, [1e-100, 1e-50], [2e-100, 2e-50]))
        cases.append((lambda x: 1e-23 * x[0] ** 2.5 * x[1] ** 0.5, [1e-120, 1.0], [1.0000001e-120, 1.0]))
    for f, lo, hi in cases:
        lo, hi = jnp.array(lo), jnp.array(hi)
        v = float(f(np.asarray(lo)))
        out = relax(f, safe=safe)(lo, lo, hi)
        assert not np.isnan([float(out.lo), float(out.hi), float(out.cv), float(out.cc)]).any()
        assert float(out.lo) <= v <= float(out.hi) and float(out.cv) <= v <= float(out.cc)


@pytest.mark.parametrize("safe", [False, True])
def test_small_posynomial_problem(safe):
    """min 5 x^1.2781 y^-0.1959 z^2.4242 / w^0.9904 + 0.2 w s.t. x * y >= 4, z + w >= 3 on a positive box:
    the bound reaches the optimum found by a dense local search (no wrong answer, plain and safe)."""

    def f(x):
        return 5.0 * x[0] ** 1.2781 * x[1] ** -0.1959 * x[2] ** 2.4242 * x[3] ** -0.9904 + 0.2 * x[3]

    ineq = lambda x: jnp.array([4.0 - x[0] * x[1], 3.0 - x[2] - x[3]])  # noqa: E731
    lo, hi = np.array([0.5, 0.5, 0.2, 0.2]), np.array([4.0, 8.0, 3.0, 3.0])
    r = minimize(f, lo, hi, ineq=ineq, time_limit=60, safe=safe, rel_tol=1e-6)
    assert r.status == "optimal"
    from scipy.optimize import minimize as smin

    best = np.inf
    rng = np.random.default_rng(0)
    for x0 in rng.uniform(lo, hi, (40, 4)):
        s = smin(lambda x: float(f(x)), x0, bounds=list(zip(lo, hi)), method="SLSQP",
                 constraints=[{"type": "ineq", "fun": lambda x: -np.asarray(ineq(x))}])
        if s.success and np.all(np.asarray(ineq(s.x)) <= 1e-7):
            best = min(best, s.fun)
    assert r.lower_bound <= best + 1e-6 * (1 + abs(best))
    assert abs(r.fun - best) <= 1e-4 * (1 + abs(best))

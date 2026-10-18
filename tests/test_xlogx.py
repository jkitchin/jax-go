"""``x * log(x)`` (entropy terms) and ``log`` on boxes that touch 0 (MINLPLib ex8_5_3 / ex8_5_4).

FBBT derives ``x >= exp(-inf) = 0`` for the operand of a log and pads it outward to ``x >= -1e-10``.
On such a box the product rule for ``x * log(x)`` and the log rule were unbounded, so the node bound
was -inf. Now the padding stops at log's domain (``fbbt.DOMAIN_FLOOR``), ``x * log(x)`` is relaxed as
one convex function (``relax.mc_xlogx``) and ``log`` keeps a finite overestimator on such a box.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize
from jaxglobal.fbbt import make_fbbt
from jaxglobal.relax import MC, Ctx, mc_log, mc_xlogx, relax
from jaxglobal.univariate import make_univariate


def g(z):
    z = np.asarray(z, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z > 0, z * np.log(np.where(z > 0, z, 1.0)), 0.0)


def random_boxes(rng, k):
    """Boxes with lower ends at -1e-10, exactly 0, tiny, and ordinary, up to wide upper ends."""
    lo = np.concatenate([np.full(k, -1e-10), np.zeros(k), 10.0 ** rng.uniform(-14, -3, k), rng.uniform(0, 2, k)])
    hi = lo + 10.0 ** rng.uniform(-6, 1.5, lo.size)
    return lo, hi


def _mc(lo, hi, p):
    return MC(jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(p), jnp.asarray(p))


def test_envelope_valid_on_random_points():
    """cv <= g <= cc and lo <= g <= hi at points of [max(lo, 0), hi]; the tangent cut of cv taken at
    any point of the box (also one below 0) stays below g on the whole domain part of the box."""
    rng = np.random.default_rng(0)
    lo, hi = random_boxes(rng, 50)
    P = rng.uniform(lo, hi)  # may be < 0 where lo = -1e-10: still a point the LP can visit
    cvf = jax.jit(jax.vmap(lambda l, h, p: mc_xlogx(_mc(l, h, p))))
    dcv = jax.jit(jax.vmap(jax.grad(lambda p, l, h: mc_xlogx(_mc(l, h, p)).cv), in_axes=(0, 0, 0)))
    m = cvf(lo, hi, P)
    d = np.asarray(dcv(P, lo, hi))
    for c in m:
        assert np.all(np.isfinite(np.asarray(c)))
    assert np.all(np.isfinite(d))
    Z = rng.uniform(np.maximum(lo, 0)[:, None], hi[:, None], (lo.size, 200))
    Z[:, 0], Z[:, 1] = np.maximum(lo, 0), hi
    gz = g(Z)
    tol = 1e-12 * (1 + np.abs(gz))
    assert np.all(np.asarray(m.lo)[:, None] <= gz + tol) and np.all(gz <= np.asarray(m.hi)[:, None] + tol)
    # relaxations evaluated at the domain points
    mz = jax.jit(jax.vmap(jax.vmap(lambda l, h, p: mc_xlogx(_mc(l, h, p)), in_axes=(None, None, 0))))(lo, hi, Z)
    assert np.all(np.asarray(mz.cv) <= gz + tol) and np.all(gz <= np.asarray(mz.cc) + tol)
    cut = np.asarray(m.cv)[:, None] + d[:, None] * (Z - P[:, None])
    assert np.all(cut <= gz + 1e-9 * (1 + np.abs(gz)))


def test_envelope_is_tight_inside_the_domain():
    """Away from 0 the convex side is g itself and the concave side is the secant."""
    lo, hi = np.array([0.1, 0.0, 0.5]), np.array([2.0, 1.0, 0.9])
    p = np.array([0.3, 0.4, 0.7])
    m = mc_xlogx(_mc(lo, hi, p))
    assert np.allclose(np.asarray(m.cv), g(p), rtol=1e-12, atol=1e-12)
    sec = g(lo) + (g(hi) - g(lo)) / (hi - lo) * (p - lo)
    assert np.allclose(np.asarray(m.cc), sec, rtol=1e-12, atol=1e-12)
    assert np.isclose(float(m.lo[1]), -np.exp(-1.0))


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("lo0", [-1e-10, 0.0])
def test_bound_finite_at_padded_lower_end(lo0, safe):
    """``x log x + y log y`` on a box whose lower ends are -1e-10 or 0: finite and valid relaxations,
    traced (``relax``) and through the univariate relaxer, straight-line and interpreted."""

    def f(x):
        return x[0] * jnp.log(x[0]) + jnp.log(x[1]) * x[1] - jnp.log(x[0] + 0.5 - x[1])

    lo, hi = jnp.array([lo0, lo0]), jnp.array([1.0, 1.0])
    pts = jnp.array([[lo0, lo0], [0.5, 0.25], [1.0, 1.0], [1e-9, 0.3]])
    fx = np.array([g(max(p[0], 0)) + g(max(p[1], 0)) - np.log(max(p[0], 0) + 0.5 - max(p[1], 0))
                   if p[0] + 0.5 - p[1] > 0 else np.inf for p in np.asarray(pts)])  # fmt: skip
    outs = [jax.vmap(lambda x: relax(f, safe=safe)(x, lo, hi))(pts)]
    for interpret in (False, True):
        prep, rw = make_univariate(f, safe=safe, interpret=interpret)
        outs.append(jax.vmap(lambda x, prep=prep, rw=rw: rw(x, lo, hi, prep(lo, hi)))(pts))
    for m in outs:
        cv = np.asarray(m.cv)
        assert np.all(np.isfinite(cv)) and np.all(np.isfinite(np.asarray(m.lo)))
        assert np.all(cv <= fx + 1e-8)
    # interpreted and straight-line evaluations of the same tape agree
    assert np.allclose(np.asarray(outs[1].cv), np.asarray(outs[2].cv), rtol=1e-12, atol=1e-12)


def test_log_on_box_touching_zero():
    """``log`` on [lo <= 0, hi]: interval [-inf, log hi], a finite concave cc above log on x > 0."""
    rng = np.random.default_rng(1)
    lo = np.concatenate([np.full(20, -1e-10), np.zeros(20), -rng.uniform(0, 1, 20)])
    hi = 10.0 ** rng.uniform(-3, 2, lo.size)
    P = rng.uniform(lo, hi)
    m = jax.vmap(lambda l, h, p: mc_log(_mc(l, h, p)))(lo, hi, P)
    dcc = jax.vmap(jax.grad(lambda p, l, h: mc_log(_mc(l, h, p)).cc))(P, lo, hi)
    assert np.all(np.isfinite(np.asarray(m.cc))) and np.all(np.isfinite(np.asarray(dcc)))
    assert np.allclose(np.asarray(m.hi), np.log(hi))
    Z = rng.uniform(0, 1, (lo.size, 100)) * hi[:, None] + 1e-300
    cut = np.asarray(m.cc)[:, None] + np.asarray(dcc)[:, None] * (Z - P[:, None])
    assert np.all(cut >= np.log(Z) - 1e-9 * (1 + np.abs(np.log(Z))))
    # positive boxes keep the old rule
    m2 = mc_log(_mc(np.array([0.5]), np.array([2.0]), np.array([1.0])))
    assert np.isclose(float(m2.cc[0]), 0.0) and np.isfinite(float(m2.cv[0]))


@pytest.mark.parametrize("interpret", [False, True])
def test_fbbt_padding_stops_at_log_domain(interpret):
    """From ``x + y = 1`` and ``x log x + y log y <= 0`` FBBT bounds x, y in [0, 1]: the lower ends
    stay at 0 instead of the padded -1e-10."""

    def f(x):
        return x[0] * jnp.log(x[0]) + x[1] * jnp.log(x[1])

    eq = lambda x: jnp.array([x[0] + x[1] - 1.0])  # noqa: E731
    fb = make_fbbt(f, None, eq, interpret=interpret)
    lo, hi, empty = fb(jnp.array([-5.0, -5.0]), jnp.array([5.0, 5.0]))
    assert not bool(empty)
    assert np.all(np.asarray(lo) >= 0.0) and np.all(np.asarray(lo) <= 1e-12)


def test_interval_only_mode():
    """The interval rule (FBBT forward sweeps, alphaBB's interval Hessian) is the same interval."""
    from jaxglobal.relax import apply_xlogx

    class E:  # what apply_xlogx reads from the equation (only in safe mode)
        pass

    x = _mc(np.array([-1e-10]), np.array([3.0]), np.array([1.0]))
    a = apply_xlogx(Ctx(interval_only=True), E, x)
    b = mc_xlogx(x)
    assert np.allclose(np.asarray(a.lo), np.asarray(b.lo)) and np.allclose(np.asarray(a.hi), np.asarray(b.hi))
    assert np.isclose(float(a.lo[0]), -np.exp(-1.0)) and np.isclose(float(a.hi[0]), 3 * np.log(3))


@pytest.mark.parametrize("lo0", [-1e-10, 0.0])
def test_entropy_problem_solves_with_finite_bound(lo0):
    """min x log x + y log y + 0.3 x s.t. x + y = 1: the bound was -inf from the root on."""
    f = lambda x: x[0] * jnp.log(x[0]) + x[1] * jnp.log(x[1]) + 0.3 * x[0]  # noqa: E731
    eq = lambda x: jnp.array([x[0] + x[1] - 1.0])  # noqa: E731
    r = minimize(f, np.array([lo0, lo0]), np.array([1.0, 1.0]), eq=eq, time_limit=30)
    xs = 1.0 / (1.0 + np.exp(0.3))
    fstar = g(xs) + g(1 - xs) + 0.3 * xs
    assert r.status == "optimal"
    assert np.isfinite(r.lower_bound) and r.lower_bound <= fstar + 1e-9
    assert abs(r.fun - fstar) < 1e-5


# --- regressions from the review of PR #41 ---------------------------------------------------------


@pytest.mark.parametrize("lo", [1e-6, 0.1, 0.125, 0.3, 1.0 / np.e, 0.5, 2.0])
def test_cv_slope_at_lower_end_is_exact(lo):
    """At ``x == lo`` (``d == lo``) the cv gradient is ``g'(lo) = 1 + log(lo)``, not half of it: a
    ``maximum(z, d)`` guard tied there and JAX split its derivative 50/50, which gave invalid cuts
    (lo = 0.1: the cut at 0.15 was above ``g``)."""
    hi = 3.0
    cv = lambda p: mc_xlogx(_mc(lo, hi, p)).cv  # noqa: E731
    s = float(jax.grad(cv)(jnp.asarray(lo)))
    if lo < np.exp(-1.0):  # where tighten's max(cv, lo) does not touch cv
        assert np.isclose(s, 1.0 + np.log(lo), rtol=1e-12, atol=1e-12)
    Z = np.linspace(lo, hi, 2001)
    assert np.all(float(cv(jnp.asarray(lo))) + s * (Z - lo) <= g(Z) + 1e-12 * (1 + np.abs(g(Z))))


@pytest.mark.parametrize("safe", [False, True])
def test_relax_cut_at_lower_end_valid(safe):
    """The same through ``relax`` of the user-level function (plain and safe mode)."""
    lo, hi = 0.1, 1.0
    f = lambda x: x[0] * jnp.log(x[0])  # noqa: E731
    r = lambda x: relax(f, safe=safe)(x, jnp.array([lo]), jnp.array([hi])).cv  # noqa: E731
    p = jnp.array([lo])
    s = float(jax.grad(r)(p)[0])
    Z = np.linspace(lo, hi, 2001)
    assert np.all(float(r(p)) + s * (Z - lo) <= g(Z) + 1e-12)


def test_log_cc_slope_at_delta_is_exact():
    """mc_log's cc on a box touching 0 at exactly ``z == d = LOG_DELTA * hi``: slope ``1/d``."""
    from jaxglobal.relax import LOG_DELTA

    hi = 2.0
    d = LOG_DELTA * hi
    cc = lambda p: mc_log(_mc(0.0, hi, p)).cc  # noqa: E731
    s = float(jax.grad(cc)(jnp.asarray(d)))
    assert np.isclose(s, 1.0 / d, rtol=1e-12)
    Z = np.concatenate([np.linspace(1e-12, 4 * d, 500), np.linspace(d, hi, 500)])
    assert np.all(float(cc(jnp.asarray(d))) + s * (Z - d) >= np.log(Z) - 1e-12 * (1 + np.abs(np.log(Z))))


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("univariate", [False, True])
def test_frank_wolfe_box_bound_valid(safe, univariate):
    """make_bounder (Frank-Wolfe on cv, steps exactly onto vertices) on ``x log x + a x`` over
    [0.125, 1], minimum -0.18 at 0.18: the bound was -0.1706 (above the minimum)."""
    from jaxglobal.bnb import make_bounder

    xs = 0.18
    a = -1 - np.log(xs)
    f = lambda x: x[0] * jnp.log(x[0]) + a * x[0]  # noqa: E731
    lb = float(make_bounder(f, safe=safe, univariate=univariate)(jnp.array([0.125]), jnp.array([1.0]))[0])
    assert lb <= -xs + 1e-12


@pytest.mark.parametrize("safe", [False, True])
def test_separable_entropy_plus_nonconvex_solves_correctly(safe):
    """sum_i (x_i log x_i + a x_i) on [0.125, 1]^3 plus a two-well y-term: reported "optimal" with a
    lower bound above the true minimum -0.543001 before."""
    xs, n = 0.18, 3
    a = -1 - np.log(xs)
    f = lambda x: jnp.sum(x[:n] * jnp.log(x[:n]) + a * x[:n]) + (x[n] ** 2 - 1) ** 2 + 0.003 * x[n]  # noqa: E731
    lo, hi = np.array([0.125] * n + [-1.5]), np.array([1.0] * n + [2.0])
    ys = np.linspace(-1.5, 2, 200001)
    true = -n * xs + np.min((ys**2 - 1) ** 2 + 0.003 * ys)
    r = minimize(f, lo, hi, time_limit=20, local_solve=False, safe=safe)
    assert r.lower_bound <= true + 1e-6
    assert r.status == "optimal" and abs(r.fun - true) < 1e-4


def test_secant_with_overflowing_hi_is_inf_not_nan():
    """``hi * log(hi)`` overflows for hi ~ 1e306: cc is +inf instead of NaN."""
    m = mc_xlogx(_mc(3.0, 1e306, 3.0))
    assert not any(np.isnan(float(c)) for c in m)
    assert float(m.cc) == np.inf and np.isclose(float(m.cv), 3 * np.log(3))

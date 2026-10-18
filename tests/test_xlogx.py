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

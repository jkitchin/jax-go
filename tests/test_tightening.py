"""Bound tightening must never remove a point that could be optimal.

Every test samples points of a box, keeps those that are feasible (within the
solver's feasibility tolerance) with ``f <= cutoff``, and checks that the
tightened box still contains all of them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import make_bounder, minimize
from jaxglobal.fbbt import make_fbbt
from jaxglobal.lp import make_constrained_bounder, make_obbt
from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS

FEAS = 1e-6


def sample_boxes(lo, hi, rng, n):
    yield lo, hi
    for _ in range(n):
        a, b = np.sort(rng.uniform(lo, hi, (2, lo.size)), axis=0)
        yield a, b


def keep_mask(p, X, cutoff, feas=FEAS):
    F = np.asarray(jax.vmap(p.f)(jnp.asarray(X)))
    ok = F <= cutoff
    if getattr(p, "ineq", None) is not None:
        G = np.asarray(jax.vmap(lambda x: jnp.atleast_1d(p.ineq(x)))(jnp.asarray(X)))
        ok &= np.all(G <= feas, axis=1)
    if getattr(p, "eq", None) is not None:
        H = np.asarray(jax.vmap(lambda x: jnp.atleast_1d(p.eq(x)))(jnp.asarray(X)))
        ok &= np.all(np.abs(H) <= feas, axis=1)
    return ok, F


def assert_contains(X, lo, hi, empty):
    if len(X) == 0:
        return
    assert not empty, "box with feasible points reported empty"
    lo, hi = np.asarray(lo), np.asarray(hi)
    tol = 1e-9 * (1 + np.abs(X))
    assert np.all(X >= lo - tol) and np.all(X <= hi + tol)


def cutoffs(F):
    return [np.inf, float(np.quantile(F, 0.5)), float(np.quantile(F, 0.05))]


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.name)
def test_fbbt_sound_unconstrained(p):
    rng = np.random.default_rng(0)
    fb = jax.jit(make_fbbt(p.f))
    for a, b in sample_boxes(p.lo, p.hi, rng, 6):
        X = rng.uniform(a, b, (3000, a.size))
        _, F = keep_mask(p, X, np.inf)
        for c in cutoffs(F):
            lo, hi, e = fb(jnp.asarray(a), jnp.asarray(b), c)
            assert np.all(np.asarray(lo) >= a - 1e-12) and np.all(np.asarray(hi) <= b + 1e-12)
            assert_contains(X[F <= c], lo, hi, bool(e))


@pytest.mark.parametrize("p", CONSTRAINED_PROBLEMS, ids=lambda p: p.name)
def test_fbbt_sound_constrained(p):
    """Inequality-feasible samples, and the known minimizer for equality problems."""
    rng = np.random.default_rng(1)
    fb = jax.jit(make_fbbt(p.f, p.ineq, p.eq, feas_tol=FEAS))
    for a, b in sample_boxes(p.lo, p.hi, rng, 6):
        X = rng.uniform(a, b, (3000, a.size))
        _, F = keep_mask(p, X, np.inf)
        for c in cutoffs(F):
            ok, _ = keep_mask(p, X, c)
            lo, hi, e = fb(jnp.asarray(a), jnp.asarray(b), c)
            assert_contains(X[ok], lo, hi, bool(e))
    if p.xstar is not None:
        for _ in range(6):
            w = (p.hi - p.lo) * rng.uniform(0.01, 0.5, p.lo.size)
            a, b = np.maximum(p.lo, p.xstar - w), np.minimum(p.hi, p.xstar + w)
            lo, hi, e = fb(jnp.asarray(a), jnp.asarray(b), p.fstar + 1e-6 * (1 + abs(p.fstar)))
            assert_contains(p.xstar[None], lo, hi, bool(e))


# One function per backward rule: output constrained to [zlo, zhi] on a box.
RULE_CASES = [
    ("add", lambda x: x[0] + x[1], [-2, -2], [2, 2], (3.0, 4.0)),
    ("sub", lambda x: x[0] - x[1], [-2, -2], [2, 2], (1.0, 3.0)),
    ("mul", lambda x: x[0] * x[1], [0.5, -3], [2, 3], (1.0, 2.0)),
    ("div", lambda x: x[0] / x[1], [-2, 0.5], [2, 2], (1.0, 4.0)),
    ("square", lambda x: (lambda y: y * y)(x[0]) + x[1], [-3, 0], [3, 0.1], (-1.0, 1.0)),
    ("ipow_even", lambda x: x[0] ** 4, [-3], [3], (1.0, 16.0)),
    ("ipow_odd", lambda x: x[0] ** 3, [-3], [3], (-1.0, 8.0)),
    ("ipow_neg", lambda x: x[0] ** -2, [0.1], [3], (1.0, 4.0)),
    ("sqrt", lambda x: jnp.sqrt(x[0]), [0], [9], (1.0, 2.0)),
    ("rsqrt", lambda x: jax.lax.rsqrt(x[0]), [0.1], [9], (0.5, 1.0)),
    ("pow", lambda x: x[0] ** 1.5, [0], [9], (1.0, 8.0)),
    ("exp", lambda x: jnp.exp(x[0]), [-3], [3], (0.5, 2.0)),
    ("expm1", lambda x: jnp.expm1(x[0]), [-3], [3], (0.0, 1.0)),
    ("log", lambda x: jnp.log(x[0]), [0.1], [10], (0.0, 1.0)),
    ("log1p", lambda x: jnp.log1p(x[0]), [0], [10], (0.0, 1.0)),
    ("tanh", lambda x: jnp.tanh(x[0]), [-30], [30], (-0.5, 0.5)),
    ("logistic", lambda x: jax.nn.sigmoid(x[0]), [-30], [30], (0.2, 0.7)),
    ("atan", lambda x: jnp.arctan(x[0]), [-30], [30], (0.0, 1.0)),
    ("erf", lambda x: jax.scipy.special.erf(x[0]), [-5], [5], (-0.3, 0.9)),
    ("sinh", lambda x: jnp.sinh(x[0]), [-5], [5], (-1.0, 2.0)),
    ("asinh", lambda x: jnp.arcsinh(x[0]), [-5], [5], (-1.0, 1.0)),
    ("abs", lambda x: jnp.abs(x[0] - 1), [-3], [3], (0.5, 1.0)),
    ("max", lambda x: jnp.maximum(x[0], x[1]), [-3, -3], [3, -1], (0.0, 1.0)),
    ("min", lambda x: jnp.minimum(x[0], x[1]), [-3, 1], [3, 3], (0.0, 0.5)),
    ("reduce_sum", lambda x: jnp.sum(x**2), [-2, -2, -2], [2, 2, 2], (0.0, 1.0)),
    ("reduce_max", lambda x: jnp.max(x), [-2, -2], [2, 2], (-2.0, 0.0)),
    ("dot", lambda x: jnp.array([[1.0, 2.0], [3.0, -1.0]]) @ x, [-2, -2], [2, 2], (0.0, 1.0)),
    ("where", lambda x: jnp.where(x[1] > 0, x[0] ** 2, x[0]), [-2, 0.5], [2, 2], (0.0, 1.0)),
    ("concat", lambda x: jnp.concatenate([x[:1] * 2, x[1:] + 1]), [-2, -2], [2, 2], (0.0, 1.0)),
    ("relu", lambda x: jax.nn.relu(x[0]) - x[1], [-2, -1], [3, 1], (2.5, 3.0)),
]


@pytest.mark.parametrize("case", RULE_CASES, ids=[c[0] for c in RULE_CASES])
def test_backward_rules(case):
    """Each backward rule keeps every point with g(x) in Z, and tightens."""
    name, g, lo, hi, (zlo, zhi) = case
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    # encode zlo <= g <= zhi as  g - zhi <= 0  and  zlo - g <= 0
    fb = jax.jit(make_fbbt(lambda x: 0.0 * x[0], ineq=lambda x: jnp.concatenate(
        [jnp.atleast_1d(g(x)) - zhi, zlo - jnp.atleast_1d(g(x))]), feas_tol=0.0))  # fmt: skip
    a, b, e = fb(jnp.asarray(lo), jnp.asarray(hi), np.inf)
    rng = np.random.default_rng(3)
    X = rng.uniform(lo, hi, (20000, lo.size))
    G = np.asarray(jax.vmap(lambda x: jnp.atleast_1d(g(x)))(jnp.asarray(X)))
    keep = np.all((G >= zlo) & (G <= zhi), axis=1)
    assert keep.any()
    assert_contains(X[keep], a, b, bool(e))
    assert np.any(np.asarray(a) > lo + 1e-6) or np.any(np.asarray(b) < hi - 1e-6), "no tightening"


def test_fbbt_detects_infeasible():
    fb = make_fbbt(lambda x: x[0], ineq=lambda x: 1.5 - x[0] ** 2 - x[1] ** 2)
    _, _, e = fb(jnp.array([-1.0, -0.5]), jnp.array([1.0, 0.5]))
    assert bool(e)


def test_fbbt_saturation_is_safe():
    """tanh saturates to exactly 1.0 in floating point: no false infeasibility."""
    fb = make_fbbt(lambda x: x[0], eq=lambda x: jnp.tanh(x[0]) - 1.0, feas_tol=0.0)
    lo, hi, e = fb(jnp.array([20.0]), jnp.array([30.0]))
    assert not bool(e) and float(lo[0]) <= 20.0 and float(hi[0]) >= 30.0


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.name)
def test_dbbt_sound_unconstrained(p):
    rng = np.random.default_rng(4)
    bound = jax.jit(make_bounder(p.f))
    for a, b in sample_boxes(p.lo, p.hi, rng, 5):
        X = rng.uniform(a, b, (3000, a.size))
        _, F = keep_mask(p, X, np.inf)
        for c in cutoffs(F)[1:]:
            _, _, _, lo, hi = bound(jnp.asarray(a), jnp.asarray(b), c)
            assert_contains(X[F <= c], lo, hi, False)


@pytest.mark.parametrize("p", CONSTRAINED_PROBLEMS, ids=lambda p: p.name)
def test_dbbt_and_obbt_sound_constrained(p):
    rng = np.random.default_rng(5)
    bound = jax.jit(make_constrained_bounder(p.f, p.ineq, p.eq))
    obbt = jax.jit(make_obbt(p.f, p.ineq, p.eq))
    pts = []
    for a, b in sample_boxes(p.lo, p.hi, rng, 4):
        X = rng.uniform(a, b, (3000, a.size))
        ok, F = keep_mask(p, X, np.inf, feas=0.0)
        pts.append((a, b, X, ok, F))
    if p.xstar is not None:
        for _ in range(4):
            w = (p.hi - p.lo) * rng.uniform(0.01, 0.5, p.lo.size)
            a, b = np.maximum(p.lo, p.xstar - w), np.minimum(p.hi, p.xstar + w)
            pts.append((a, b, p.xstar[None], np.array([True]), np.array([p.fstar])))
    for a, b, X, ok, F in pts:
        for c in [np.inf, (p.fstar + 1e-6 * (1 + abs(p.fstar)))] + ([float(np.quantile(F[ok], 0.5))] if ok.sum() > 4 else []):
            keep = X[ok & (F <= c)]
            _, _, inf, lo, hi = bound(jnp.asarray(a), jnp.asarray(b), c)
            assert_contains(keep, lo, hi, bool(inf))
            lo, hi, e = obbt(jnp.asarray(a), jnp.asarray(b), c)
            assert_contains(keep, lo, hi, bool(e))


@pytest.mark.parametrize("name", ["G06", "haverly"])
def test_tightening_reduces_nodes(name):
    p = next(q for q in CONSTRAINED_PROBLEMS if q.name == name)
    off = minimize(p.f, p.lo, p.hi, ineq=p.ineq, eq=p.eq, fbbt=False, dbbt=False, obbt=False)
    on = minimize(p.f, p.lo, p.hi, ineq=p.ineq, eq=p.eq)
    assert on.status == off.status == "optimal"
    assert abs(on.fun - off.fun) <= 1e-4 * (1 + abs(p.fstar))
    assert on.nodes * 2 <= off.nodes, (on.nodes, off.nodes)

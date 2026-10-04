"""Phase-2 relaxations: univariate hull envelopes and the lifted LP bound."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize
from jaxglobal.bnb import make_bounder
from jaxglobal.fbbt import make_tape
from jaxglobal.lifted import make_lifted_bounder
from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS
from jaxglobal.univariate import find_blocks, relax_univariate


def boxes(lo, hi, rng, n):
    yield lo, hi
    for i in range(n):
        a, b = np.sort(rng.uniform(lo, hi, (2, lo.size)), axis=0)
        if i % 2:  # small boxes
            c = rng.uniform(lo, hi)
            w = (hi - lo) * 10.0 ** rng.uniform(-3, -1)
            a, b = np.maximum(lo, c - w), np.minimum(hi, c + w)
        yield a, b


FUNCS = [(p.name, p.f, p.lo, p.hi) for p in PROBLEMS] + [
    (f"{p.name}_{k}", fn, p.lo, p.hi)
    for p in CONSTRAINED_PROBLEMS
    for k, fn in (("ineq", p.ineq), ("eq", p.eq))
    if fn is not None
] + [
    ("poly6", lambda x: x[0] ** 6 - 3 * x[0] ** 4 + x[0] ** 3 + 2 * x[0], np.array([-2.0]), np.array([2.0])),
    ("oscill", lambda x: jnp.sin(5 * x[0]) * jnp.exp(-0.1 * x[0] ** 2) + 0.2 * x[0], np.array([-6.0]), np.array([6.0])),
    ("sep", lambda x: jnp.sum(jnp.cos(3 * x) * x**2 - x**3), -2 * np.ones(3), 2 * np.ones(3)),
]  # fmt: skip


@pytest.mark.parametrize("case", FUNCS, ids=[c[0] for c in FUNCS])
def test_univariate_relaxation_sound(case):
    """lo <= f <= hi and cv <= f <= cc with the hull envelopes."""
    name, f, lo0, hi0 = case
    rf = relax_univariate(lambda x: jnp.atleast_1d(f(x)))
    prep = jax.jit(rf.prepare)
    ev = jax.jit(jax.vmap(lambda x, lo, hi, d: rf(x, lo, hi, d), in_axes=(0, None, None, None)))
    fv = jax.jit(jax.vmap(lambda x: jnp.atleast_1d(f(x))))
    rng = np.random.default_rng(0)
    for lo, hi in boxes(lo0, hi0, rng, 6):
        X = rng.uniform(lo, hi, (2000, lo.size))
        d = prep(jnp.asarray(lo), jnp.asarray(hi))
        m, F = ev(jnp.asarray(X), jnp.asarray(lo), jnp.asarray(hi), d), np.asarray(fv(jnp.asarray(X)))
        ok = np.isfinite(F)
        s = 1e-9 * (1 + np.abs(F[ok]).max(initial=0))
        assert np.all(np.asarray(m.lo)[ok] <= F[ok] + s) and np.all(F[ok] <= np.asarray(m.hi)[ok] + s)
        assert np.all(np.asarray(m.cv)[ok] <= F[ok] + s) and np.all(F[ok] <= np.asarray(m.cc)[ok] + s)


@pytest.mark.parametrize("case", FUNCS[-3:], ids=[c[0] for c in FUNCS[-3:]])
def test_univariate_cuts_valid(case):
    """Subgradient cuts of the hull-tightened cv underestimate f (cv is convex)."""
    name, f, lo, hi = case
    lo, hi = jnp.asarray(lo), jnp.asarray(hi)
    rf = relax_univariate(f)
    d = rf.prepare(lo, hi)
    vg = jax.jit(jax.value_and_grad(lambda x: rf(x, lo, hi, d).cv))
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.uniform(lo, hi, (500, lo.size)))
    F = np.asarray(jax.vmap(f)(X))
    for x0 in rng.uniform(lo, hi, (10, lo.size)):
        v, g = vg(jnp.asarray(x0))
        cut = np.asarray(v + (X - x0) @ g)
        assert np.all(cut <= F + 1e-8 * (1 + np.abs(F)))


def test_hull_is_tight_on_polynomials():
    """The hull bound is nearly the global minimum of a separable polynomial."""
    st = next(p for p in PROBLEMS if p.name == "styblinski_tang_3d")
    lo, hi = jnp.asarray(st.lo), jnp.asarray(st.hi)
    plain = float(make_bounder(st.f)(lo, hi)[0])
    hull = float(make_bounder(st.f, univariate=True)(lo, hi)[0])
    assert plain < -500
    assert st.fstar - 2.0 <= hull <= st.fstar + 1e-9


def test_blocks_skip_exact_and_nonsmooth():
    def f(x):
        return jnp.sum((x - 1.0) ** 2) + jnp.abs(x[0]) * x[0] ** 3

    tape = make_tape(f, jnp.zeros(2))
    with jax.ensure_compile_time_eval():
        assert find_blocks(tape, 2) == []  # squares are exact; |x| is a kink


@pytest.mark.parametrize("p", [p for p in CONSTRAINED_PROBLEMS if p.xstar is not None], ids=lambda p: p.name)
def test_lifted_bound_below_optimum(p):
    bound = jax.jit(make_lifted_bounder(p.f, p.ineq, p.eq, tol=1e-6))
    rng = np.random.default_rng(2)
    for _ in range(4):
        w = (p.hi - p.lo) * rng.uniform(0.01, 0.5, p.lo.size)
        a, b = np.maximum(p.lo, p.xstar - w), np.minimum(p.hi, p.xstar + w)
        lb, _, inf, rlo, rhi = bound(jnp.asarray(a), jnp.asarray(b), p.fstar + 1e-6 * (1 + abs(p.fstar)))
        assert not bool(inf)
        assert float(lb) <= p.fstar + 1e-5 * (1 + abs(p.fstar))
        assert np.all(p.xstar >= np.asarray(rlo) - 1e-7) and np.all(p.xstar <= np.asarray(rhi) + 1e-7)


@pytest.mark.parametrize("p", PROBLEMS[:8], ids=lambda p: p.name)
def test_lifted_bound_unconstrained(p):
    bound = jax.jit(make_lifted_bounder(p.f))
    rng = np.random.default_rng(3)
    fv = jax.jit(jax.vmap(p.f))
    for a, b in boxes(p.lo, p.hi, rng, 4):
        lb = float(bound(jnp.asarray(a), jnp.asarray(b))[0])
        fmin = float(np.min(fv(jnp.asarray(rng.uniform(a, b, (2000, a.size))))))
        assert lb <= fmin + 1e-8 * (1 + abs(fmin))


@pytest.mark.parametrize("name", ["styblinski_tang_3d", "six_hump_camel", "shubert"])
def test_univariate_reduces_nodes(name):
    p = next(q for q in PROBLEMS if q.name == name)
    off = minimize(p.f, p.lo, p.hi, abs_tol=1e-5, rel_tol=1e-6, univariate=False)
    on = minimize(p.f, p.lo, p.hi, abs_tol=1e-5, rel_tol=1e-6, univariate=True)
    assert on.status == off.status == "optimal"
    assert abs(on.fun - p.fstar) <= 1e-4 * (1 + abs(p.fstar))
    assert on.lower_bound <= p.fstar + 1e-5 * (1 + abs(p.fstar))
    assert on.nodes * 3 <= off.nodes, (on.nodes, off.nodes)


SMOOTH = [c for c in FUNCS if c[0] not in ("holder_table", "eggholder", "piecewise_2d")]


@pytest.mark.parametrize("case", SMOOTH, ids=[c[0] for c in SMOOTH])
def test_alphabb_relaxation_sound(case):
    """McCormick + univariate + scaled-Gershgorin alphaBB stays valid, on small boxes too."""
    from jaxglobal.univariate import make_relaxer

    name, f, lo0, hi0 = case
    r = make_relaxer(lambda x: jnp.atleast_1d(f(x)), univariate=True, alphabb=True)
    prep = jax.jit(r.prepare)
    ev = jax.jit(jax.vmap(lambda x, lo, hi, d: r(x, lo, hi, d), in_axes=(0, None, None, None)))
    fv = jax.jit(jax.vmap(lambda x: jnp.atleast_1d(f(x))))
    rng = np.random.default_rng(4)
    for lo, hi in boxes(lo0, hi0, rng, 8):
        X = rng.uniform(lo, hi, (2000, lo.size))
        d = prep(jnp.asarray(lo), jnp.asarray(hi))
        m, F = ev(jnp.asarray(X), jnp.asarray(lo), jnp.asarray(hi), d), np.asarray(fv(jnp.asarray(X)))
        ok = np.isfinite(F)
        s = 1e-9 * (1 + np.abs(F[ok]).max(initial=0))
        assert np.all(np.asarray(m.cv)[ok] <= F[ok] + s) and np.all(F[ok] <= np.asarray(m.cc)[ok] + s)


def test_alphabb_tightens_small_boxes():
    gp = next(p for p in PROBLEMS if p.name == "goldstein_price")
    b0 = jax.jit(make_bounder(gp.f, univariate=True))
    b1 = jax.jit(make_bounder(gp.f, univariate=True, alphabb=True))
    rng = np.random.default_rng(0)
    g0 = g1 = 0.0
    for _ in range(6):
        c = rng.uniform(gp.lo, gp.hi)
        lo, hi = np.maximum(gp.lo, c - 0.08), np.minimum(gp.hi, c + 0.08)
        fmin = float(np.min(jax.vmap(gp.f)(jnp.asarray(rng.uniform(lo, hi, (3000, 2))))))
        l0, l1 = float(b0(lo, hi)[0]), float(b1(lo, hi)[0])
        assert l0 <= l1 <= fmin + 1e-9
        g0, g1 = g0 + fmin - l0, g1 + fmin - l1
    assert g1 < 0.6 * g0


def test_alphabb_skips_nonsmooth():
    from jaxglobal.alphabb import make_alphabb

    prep, _ = make_alphabb(lambda x: jnp.atleast_1d(jnp.abs(x[0]) + x[1] ** 2))
    assert prep(jnp.zeros(2), jnp.ones(2)) is None


def _rand_quadratic(seed, n=4):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Q = 0.5 * (A + A.T)
    b = rng.normal(size=n)
    return lambda x: x @ (jnp.asarray(Q) @ x) + jnp.asarray(b) @ x


def _eigen_check(seed):
    """Indefinite rotated quadratic: valid, convex, never looser; returns the mean gain."""
    from jaxglobal.alphabb import make_alphabb

    q = _rand_quadratic(seed)
    prep, apply = make_alphabb(lambda x: jnp.atleast_1d(q(x)))
    rng = np.random.default_rng(seed)
    lo, hi = jnp.asarray(-rng.uniform(0.5, 2, 4)), jnp.asarray(rng.uniform(0.5, 2, 4))
    a, b, eig = prep(lo, hi)
    assert bool(eig[0][0])  # detected as quadratic
    X = jnp.asarray(rng.uniform(lo, hi, (3000, 4)))
    cv, cc = jax.vmap(lambda x: apply(x, lo, hi, (a, b, eig)))(X)
    F = jax.vmap(q)(X)
    assert np.all(np.asarray(cv[:, 0]) <= np.asarray(F) + 1e-9) and np.all(np.asarray(F) <= np.asarray(cc[:, 0]) + 1e-9)
    # convexity of cv: midpoint inequality on random pairs
    Y = jnp.asarray(rng.uniform(lo, hi, (3000, 4)))
    cy, _ = jax.vmap(lambda x: apply(x, lo, hi, (a, b, eig)))(Y)
    cm, _ = jax.vmap(lambda x: apply(x, lo, hi, (a, b, eig)))(0.5 * (X + Y))
    assert np.all(np.asarray(cm) <= 0.5 * (np.asarray(cv) + np.asarray(cy)) + 1e-9)
    # the eigen part never loosens (it is combined by max) and usually tightens
    off = (jnp.zeros_like(eig[0]),) + tuple(eig[1:])
    cv0, _ = jax.vmap(lambda x: apply(x, lo, hi, (a, b, off)))(X)
    assert np.all(np.asarray(cv) >= np.asarray(cv0))
    return float(jnp.mean(cv[:, 0]) - jnp.mean(cv0[:, 0]))


@pytest.mark.parametrize("seed", range(5))
def test_eigen_relaxation_sound(seed):
    _eigen_check(seed)


def test_eigen_relaxation_sometimes_tighter():
    gains = [_eigen_check(seed) for seed in range(5, 15)]
    assert sum(g > 1e-6 for g in gains) >= 2, gains

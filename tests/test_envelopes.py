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

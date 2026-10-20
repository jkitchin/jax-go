"""Loops in models: ``lax.scan`` (and ``lax.fori_loop`` with static bounds, which lowers to it) is
unrolled by the bounding interpreters; loops and branches they cannot bound raise
NotImplementedError. A scan used to be relaxed as if its body ran once (unsound)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize, relax
from jaxglobal.fbbt import _forward, make_tape
from jaxglobal.univariate import make_relaxer


def f_fori(x):
    return jax.lax.fori_loop(0, 4, lambda i, c: 0.5 * c + x[0] ** 2 * x[1], 0.0)


def f_python(x):
    c = 0.0
    for _ in range(4):
        c = 0.5 * c + x[0] ** 2 * x[1]
    return c


def f_scan(x):
    """Cumulative products and sums over the elements of x (scanned operand, stacked outputs)."""

    def body(c, xi):
        c = c * jnp.sin(xi) + xi**2
        return c, c * xi

    c, ys = jax.lax.scan(body, 1.0, x)
    _, zs = jax.lax.scan(lambda c, y: (c + y, c - y), 0.0, ys, reverse=True)
    return c + jnp.sum(ys) - jnp.sum(zs**2)


def f_polyval(x):
    return jnp.polyval(x, x[0]) + jnp.cumsum(x)[-1] * x[1]


def f_nested(x):
    def outer(i, c):
        return jax.lax.fori_loop(0, 3, lambda j, d: d * x[1] + x[0], c) * 0.3
    return jax.lax.fori_loop(0, 2, outer, x[2])


def boxes(n, k, seed, scale=1.5):
    rng = np.random.default_rng(seed)
    for i in range(k):
        a, b = np.sort(rng.uniform(-scale, scale, (2, n)), 0)
        if i % 2:
            w = 10.0 ** rng.uniform(-3, -1)
            a, b = a - w, a + w
        yield a, b


@pytest.mark.parametrize("f,n", [(f_fori, 2), (f_scan, 3), (f_polyval, 3), (f_nested, 3)],
                         ids=["fori", "scan", "polyval", "nested"])  # fmt: skip
def test_relaxation_encloses(f, n):
    """cv <= f <= cc and lo <= f <= hi at random points; the FBBT interval agrees."""
    rng = np.random.default_rng(0)
    rf = jax.jit(jax.vmap(relax(f), in_axes=(0, None, None)))
    fv = jax.jit(jax.vmap(f))
    tape = make_tape(f, jnp.zeros(n))
    assert not any(node.eqn.primitive.name == "scan" for node in tape.nodes)
    for lo, hi in boxes(n, 8, 1):
        X = rng.uniform(lo, hi, (200, n))
        m, F = rf(X, lo, hi), np.asarray(fv(X))
        tol = 1e-9 * (1 + np.abs(F).max())
        assert np.all(m.lo <= F + tol) and np.all(F <= m.hi + tol)
        assert np.all(m.cv <= F + tol) and np.all(F <= m.cc + tol)
        vals, _ = _forward(tape, (jnp.asarray(lo), jnp.asarray(hi)), {})
        iv = vals[tape.outs[0]]
        assert np.all(np.asarray(iv.lo) <= F + tol) and np.all(F <= np.asarray(iv.hi) + tol)


def test_fori_equals_python_loop():
    """The unrolled relaxation is the relaxation of the Python loop."""
    rng = np.random.default_rng(2)
    a, b = relax(f_fori), relax(f_python)
    for lo, hi in boxes(2, 6, 3):
        x = rng.uniform(lo, hi)
        ma, mb = a(x, lo, hi), b(x, lo, hi)
        for u, v in zip(ma, mb):
            np.testing.assert_allclose(u, v, rtol=1e-12, atol=1e-12)
    m = relax(lambda x: jax.lax.fori_loop(0, 4, lambda i, c: 0.5 * c + x[0] ** 2, 0.0))(
        jnp.array([2.0]), jnp.array([2.0]), jnp.array([2.0]))  # fmt: skip
    np.testing.assert_allclose([m.lo, m.hi, m.cv, m.cc], 7.5)  # 1.875 x^2 (was 4 = one iteration)


def test_interpreted_relaxer():
    """The tape-based relaxer (univariate hulls, alphaBB, interpreter) is sound on a scan model."""
    R = make_relaxer(f_scan, univariate=True, alphabb=True, interpret=True)
    k = jax.jit(jax.vmap(lambda x, lo, hi: R(x, lo, hi, R.prepare(lo, hi))))
    rng = np.random.default_rng(4)
    L, H = map(np.array, zip(*boxes(3, 16, 5)))
    X = rng.uniform(L, H)
    m, F = k(X, L, H), np.asarray(jax.vmap(f_scan)(X))
    assert np.all(np.asarray(m.cv) <= F + 1e-9) and np.all(F <= np.asarray(m.cc) + 1e-9)


def test_minimize_fori():
    """-c + 1.875 x^2 cancels the loop: min (x^2 - 1)^2 = 0 at x = +-1 (was 0.512, 'optimal')."""

    def h(x):
        c = jax.lax.fori_loop(0, 4, lambda i, c: 0.5 * c + x[0] ** 2, 0.0)
        return -c + 1.875 * x[0] ** 2 + (x[0] ** 2 - 1.0) ** 2

    r = minimize(h, [-2.0], [2.0], time_limit=60, local_solve=False)
    assert r.status == "optimal"
    assert abs(r.fun) < 1e-5 and abs(abs(r.x[0]) - 1) < 1e-2
    assert r.lower_bound <= r.fun + 1e-9 and r.lower_bound <= 1e-6


def test_while_raises():
    def f(x):
        return jax.lax.while_loop(lambda c: c[0] < 10.0, lambda c: (c[0] + x[0] ** 2 + 1.0, c[1] + 1), (x[0], 0))[0]

    lo, hi = jnp.array([0.0]), jnp.array([1.0])
    with pytest.raises(NotImplementedError, match="while"):
        relax(f)(lo, lo, hi)
    with pytest.raises(NotImplementedError, match="while"):
        make_tape(f, lo)


def test_cond():
    """A cond on a constant index is its branch; on a variable predicate it raises."""

    def g(x):
        return jax.lax.cond(True, lambda y: y[0] ** 2, lambda y: -y[0], x)

    lo, hi = jnp.array([-1.0]), jnp.array([2.0])
    m = relax(g)(jnp.array([0.5]), lo, hi)
    assert float(m.lo) <= 0.0 and float(m.hi) >= 4.0 and float(m.cv) <= 0.25 <= float(m.cc)

    def v(x):
        return jax.lax.cond(x[0] > 0, lambda y: y[0] ** 2, lambda y: -y[0], x)

    with pytest.raises(NotImplementedError, match="cond"):
        relax(v)(jnp.array([0.5]), lo, hi)
    with pytest.raises(NotImplementedError, match="cond"):
        make_tape(v, lo)

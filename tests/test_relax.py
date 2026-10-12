import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import make_bounder, minimize, relax
from jaxglobal.problems import PROBLEMS


def random_boxes(p, rng, n):
    """Full box, small boxes, and random sub-boxes."""
    yield p.lo, p.hi
    for i in range(n):
        a, b = rng.uniform(p.lo, p.hi), rng.uniform(p.lo, p.hi)
        if i % 2:
            w = (p.hi - p.lo) * 10.0 ** rng.uniform(-3, -1)
            yield np.maximum(p.lo, a - w), np.minimum(p.hi, a + w)
        else:
            yield np.minimum(a, b), np.maximum(a, b)


def vertices(lo, hi, rng, n):
    bits = rng.integers(0, 2, size=(n, lo.size))
    return np.where(bits, hi, lo)


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.name)
def test_relaxation_sound(p):
    """lo <= f <= hi and cv <= f <= cc on sampled points of sampled boxes."""
    rng = np.random.default_rng(0)
    rf = jax.jit(jax.vmap(relax(p.f), in_axes=(0, None, None)))
    fv = jax.jit(jax.vmap(p.f))
    for lo, hi in random_boxes(p, rng, 10):
        X = np.concatenate([rng.uniform(lo, hi, (300, lo.size)), vertices(lo, hi, rng, 20)])
        m, F = rf(X, lo, hi), np.asarray(fv(X))
        scale = 1e-9 * (1 + np.abs(F).max())
        assert np.all(m.lo <= F + scale) and np.all(F <= m.hi + scale)
        assert np.all(m.cv <= F + scale) and np.all(F <= m.cc + scale)


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.name)
def test_cuts_valid(p):
    """Affine cuts from jax.grad(cv) underestimate cv -- including at box vertices,
    where max/min ties used to blend in the concave relaxation's gradient."""
    rng = np.random.default_rng(1)
    for lo, hi in random_boxes(p, rng, 6):
        lo, hi = jnp.asarray(lo), jnp.asarray(hi)
        cv = jax.jit(lambda x: relax(p.f)(x, lo, hi).cv)
        vg = jax.jit(jax.value_and_grad(cv))
        X = jnp.asarray(rng.uniform(lo, hi, (400, lo.size)))
        CV = jax.vmap(cv)(X)
        anchors = np.concatenate([vertices(np.asarray(lo), np.asarray(hi), rng, 4),
                                  rng.uniform(lo, hi, (4, lo.size))])
        for x0 in anchors:
            v, g = vg(jnp.asarray(x0))
            if not (np.isfinite(v) and np.all(np.isfinite(g))):
                continue
            cut = v + (X - x0) @ g
            assert np.max(cut - CV) <= 1e-8 * (1 + np.abs(CV).max())


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.name)
def test_node_bound_below_sampled_min(p):
    rng = np.random.default_rng(2)
    bound = jax.jit(make_bounder(p.f))
    fv = jax.jit(jax.vmap(p.f))
    for lo, hi in random_boxes(p, rng, 8):
        lb = float(bound(jnp.asarray(lo), jnp.asarray(hi))[0])
        fmin = float(np.min(fv(jnp.asarray(rng.uniform(lo, hi, (2000, lo.size))))))
        assert lb <= fmin + 1e-9 * (1 + abs(fmin))


@pytest.mark.parametrize("p", PROBLEMS, ids=lambda p: p.name)
def test_global_minimum(p):
    r = minimize(p.f, p.lo, p.hi, abs_tol=1e-5, rel_tol=1e-6)
    assert r.converged
    tol = max(1e-5, 1e-6 * abs(r.fun))
    if p.fstar is not None:
        assert abs(r.fun - p.fstar) <= 2 * tol
    # certificate check against brute force on low-dimensional problems
    if p.lo.size <= 2:
        g = [np.linspace(a, b, 401) for a, b in zip(p.lo, p.hi)]
        G = np.stack(np.meshgrid(*g), -1).reshape(-1, p.lo.size)
        grid_min = float(np.min(jax.vmap(p.f)(jnp.asarray(G))))
        assert r.lower_bound <= grid_min + 1e-9
        assert r.fun <= grid_min + tol


def test_unsupported_primitive_message():
    def f(x):
        return jnp.floor(x[0]) + x[1]

    with pytest.raises(NotImplementedError, match="floor"):
        relax(f)(jnp.zeros(2), -jnp.ones(2), jnp.ones(2))


def test_alphabb_does_not_amplify_input_rounding():
    """Phi((t - mu) / sigma) via erf with tiny sigma: z spans ~1e3, and alphaBB's slope a (hi - lo) / 2
    used to turn a one-ulp error in the quotient's relaxation into ~4e-8 above Phi (default mode)."""
    from jax.scipy.special import erf

    F = lambda v: 0.5 * (1 + erf((0.3 - v[0]) / v[1] / np.sqrt(2)))  # noqa: E731
    R = jax.jit(jax.vmap(relax(F), in_axes=(0, None, None)))
    FV = jax.jit(jax.vmap(F))
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = np.array([rng.uniform(-3, 3), 10 ** rng.uniform(-3, 0.5)])
        lo, hi = a, a + np.array([10 ** rng.uniform(-3, 0.5), 10 ** rng.uniform(-3, 0.5)])
        X = jnp.asarray(np.concatenate([rng.uniform(lo, hi, (300, 2)), np.array([lo, hi, [lo[0], hi[1]], [hi[0], lo[1]]])]))
        m, V = R(X, jnp.asarray(lo), jnp.asarray(hi)), FV(X)
        assert float(jnp.max(m.cv - V)) <= 1e-14 and float(jnp.max(V - m.cc)) <= 1e-14

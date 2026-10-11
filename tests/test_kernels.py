"""GP kernel primitives (jaxglobal/kernels.py): values, derivatives, sound relaxations, tighter solves."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import kernels, make_bounder, minimize, relax

PLAIN = {
    "matern12": lambda d: jnp.exp(-jnp.sqrt(d)),
    "matern32": lambda d: (1 + jnp.sqrt(3 * d)) * jnp.exp(-jnp.sqrt(3 * d)),
    "matern52": lambda d: (1 + jnp.sqrt(5 * d) + 5 * d / 3) * jnp.exp(-jnp.sqrt(5 * d)),
    "se": lambda d: jnp.exp(-d / 2),
}
NAMES = list(kernels.KERNELS)
LO, HI = np.array([-1.0, -1.0]), np.array([1.0, 1.0])


def kernel_sum(k, n=12, seed=0):
    """A GP-mean-like objective: c0 + sum_i w_i k(d_i(x)) with mixed-sign weights."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-1, 1, (n, 2))
    a = 1.0 / rng.uniform(0.3, 1.0, 2) ** 2
    w = rng.normal(0, 1, n)

    def f(x):
        d = jnp.sum(a * (x - Z) ** 2, axis=-1)
        return 0.3 + w @ k(d)

    return f


def boxes(rng, n):
    yield LO, HI
    for i in range(n):
        a, b = rng.uniform(LO, HI), rng.uniform(LO, HI)
        if i % 2:
            w = (HI - LO) * 10.0 ** rng.uniform(-3, -1)
            yield np.maximum(LO, a - w), np.minimum(HI, a + w)
        else:
            yield np.minimum(a, b), np.maximum(a, b)


@pytest.mark.parametrize("name", NAMES)
def test_values_and_derivatives_match_formula(name):
    k, p = kernels.KERNELS[name], PLAIN[name]
    d = jnp.linspace(0.01, 20.0, 200)
    for g, h in ((k, p), (jax.grad(k), jax.grad(p)), (jax.grad(jax.grad(k)), jax.grad(jax.grad(p)))):
        np.testing.assert_allclose(jax.jit(jax.vmap(g))(d), jax.vmap(h)(d), rtol=1e-10, atol=1e-12)
    assert float(k(0.0)) == 1.0


@pytest.mark.parametrize("safe", [False, True], ids=["plain", "safe"])
@pytest.mark.parametrize("name", NAMES)
def test_relaxation_sound(name, safe):
    """lo <= f <= hi and cv <= f <= cc on sampled points, box vertices included."""
    f = kernel_sum(kernels.KERNELS[name])
    rng = np.random.default_rng(1)
    rf = jax.jit(jax.vmap(relax(f, safe=safe), in_axes=(0, None, None)))
    fv = jax.jit(jax.vmap(f))
    for lo, hi in boxes(rng, 10):
        X = np.concatenate([rng.uniform(lo, hi, (400, 2)), np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])])
        m, F = rf(X, lo, hi), np.asarray(fv(X))
        tol = 1e-9 * (1 + np.abs(F).max())
        assert np.all(m.lo <= F + tol) and np.all(F <= m.hi + tol)
        assert np.all(m.cv <= F + tol) and np.all(F <= m.cc + tol)


@pytest.mark.parametrize("name", NAMES)
def test_cuts_valid(name):
    """Affine cuts from jax.grad(cv) underestimate cv, including at box vertices."""
    f = kernel_sum(kernels.KERNELS[name])
    rng = np.random.default_rng(2)
    for lo, hi in boxes(rng, 6):
        lo, hi = jnp.asarray(lo), jnp.asarray(hi)
        cv = jax.jit(lambda x: relax(f)(x, lo, hi).cv)
        vg = jax.jit(jax.value_and_grad(cv))
        X = jnp.asarray(rng.uniform(lo, hi, (400, 2)))
        CV = jax.vmap(cv)(X)
        for x0 in np.concatenate([np.array([lo, hi, [lo[0], hi[1]]]), rng.uniform(lo, hi, (4, 2))]):
            v, g = vg(jnp.asarray(x0))
            if np.isfinite(v) and np.all(np.isfinite(g)):
                assert np.max(v + (X - x0) @ g - CV) <= 1e-8 * (1 + np.abs(CV).max())


@pytest.mark.parametrize("name", NAMES)
def test_node_bound_below_sampled_min(name):
    """The full node bound (McCormick, univariate envelopes, alphaBB) stays below f."""
    f = kernel_sum(kernels.KERNELS[name])
    rng = np.random.default_rng(3)
    bound = jax.jit(make_bounder(f, univariate=True, alphabb=True))
    fv = jax.jit(jax.vmap(f))
    for lo, hi in boxes(rng, 8):
        lb = float(bound(jnp.asarray(lo), jnp.asarray(hi))[0])
        fmin = float(np.min(fv(jnp.asarray(rng.uniform(lo, hi, (4000, 2))))))
        assert lb <= fmin + 1e-9 * (1 + abs(fmin))


def test_primitive_solve_matches_formula_with_fewer_nodes():
    """Same certified optimum as the jnp formula, and the envelopes need fewer nodes."""
    prim = minimize(kernel_sum(kernels.matern52), LO, HI, abs_tol=1e-4, rel_tol=1e-4)
    plain = minimize(kernel_sum(PLAIN["matern52"]), LO, HI, abs_tol=1e-4, rel_tol=1e-4)
    assert prim.status == plain.status == "optimal"
    assert abs(prim.fun - plain.fun) <= 2e-4 * (1 + abs(plain.fun))
    assert prim.lower_bound <= plain.fun + 1e-6 and plain.lower_bound <= prim.fun + 1e-6
    assert prim.nodes < plain.nodes, (prim.nodes, plain.nodes)

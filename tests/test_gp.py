"""GP predictions (jaxglobal/gp.py): gp_std matches the formula, relaxes soundly, and tightens solves."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import kernels, make_bounder, minimize
from jaxglobal.gp import gp_mean, gp_std
from jaxglobal.relax import relax

LO, HI = np.array([-1.0, -1.0]), np.array([1.0, 1.0])


def trained(kernel="matern52", noise=1e-6, n_train=15, seed=0):
    """A small 2-D GP: training inputs, length scales, A = K^-1, alpha, sf2."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-1, 1, (n_train, 2))
    ls, sf2 = np.array([0.5, 0.7]), 1.3
    d = (((Z[:, None] - Z[None]) / ls) ** 2).sum(-1)
    K = sf2 * np.asarray(kernels.KERNELS[kernel](jnp.asarray(d))) + noise * np.eye(n_train)
    A = np.linalg.inv(K)
    y = np.sin(3 * Z[:, 0]) + Z[:, 1] ** 2
    return Z, ls, A, A @ y, sf2


def formula(Z, ls, A, sf2, kernel):
    def std(x):
        k = sf2 * kernels.KERNELS[kernel](jnp.sum(((x - Z) / ls) ** 2, axis=-1))
        return jnp.sqrt(jnp.maximum(sf2 - k @ A @ k, 1e-12))

    return std


def boxes(rng, n):
    yield LO, HI
    for i in range(n):
        a, b = rng.uniform(LO, HI), rng.uniform(LO, HI)
        if i % 2:
            w = (HI - LO) * 10.0 ** rng.uniform(-3, -1)
            yield np.maximum(LO, a - w), np.minimum(HI, a + w)
        else:
            yield np.minimum(a, b), np.maximum(a, b)


@pytest.mark.parametrize("kernel", ["matern52", "se", "matern32"])
def test_matches_formula(kernel):
    Z, ls, A, _, sf2 = trained(kernel)
    f, g = (lambda x: gp_std(x, Z, ls, A, sf2, kernel=kernel)), formula(Z, ls, A, sf2, kernel)
    X = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, (20, 2)))
    # gp_std evaluates sf2 - ||W^T k||^2 (A = W W^T), the formula sf2 - k^T A k: equal up to rounding
    # of the cancellation sf2 - (~sf2) where sigma is small
    np.testing.assert_allclose(jax.jit(jax.vmap(f))(X), jax.vmap(g)(X), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(jax.vmap(jax.grad(f))(X), jax.vmap(jax.grad(g))(X), rtol=1e-7, atol=1e-9)
    if kernel != "matern32":  # matern32's k'' is infinite at a training point; elsewhere fine
        np.testing.assert_allclose(jax.hessian(f)(X[0]), jax.hessian(g)(X[0]), rtol=1e-6, atol=1e-8)


def test_lipschitz_bound_holds():
    """|sigma(x) - sigma(c)| <= sqrt(2 sf2 (1 - kernel(d(x, c)))): the inequality the rule relies on."""
    for noise in (1e-6, 1e-2):
        Z, ls, A, _, sf2 = trained(noise=noise)
        g = jax.vmap(formula(Z, ls, A, sf2, "matern52"))
        rng = np.random.default_rng(2)
        X, C = jnp.asarray(rng.uniform(-1, 1, (5000, 2))), jnp.asarray(rng.uniform(-1, 1, (5000, 2)))
        h = jnp.sqrt(2 * sf2 * (1 - kernels.matern52(jnp.sum(((X - C) / ls) ** 2, axis=1))))
        assert float(jnp.max(jnp.abs(g(X) - g(C)) - h)) <= 1e-9


@pytest.mark.parametrize("safe", [False, True], ids=["plain", "safe"])
@pytest.mark.parametrize("noise", [1e-6, 1e-2])
@pytest.mark.parametrize("kernel", ["matern52", "se", "matern32"])
def test_relaxation_sound(kernel, noise, safe):
    Z, ls, A, _, sf2 = trained(kernel, noise)
    f = lambda x: gp_std(x, Z, ls, A, sf2, kernel=kernel)  # noqa: E731
    g = jax.vmap(formula(Z, ls, A, sf2, kernel))
    rf = jax.jit(jax.vmap(relax(f, safe=safe), in_axes=(0, None, None)))
    rng = np.random.default_rng(3)
    for lo, hi in boxes(rng, 12):
        X = np.concatenate([rng.uniform(lo, hi, (500, 2)), np.array([lo, hi, [lo[0], hi[1]], [hi[0], lo[1]]])])
        m, F = rf(X, lo, hi), np.asarray(g(jnp.asarray(X)))
        tol = 1e-9 * (1 + np.abs(F).max())
        assert np.all(m.lo <= F + tol) and np.all(F <= m.hi + tol)
        assert np.all(m.cv <= F + tol) and np.all(F <= m.cc + tol)


def test_cuts_valid():
    Z, ls, A, alpha, sf2 = trained()
    f = lambda x: gp_mean(x, Z, ls, alpha, sf2) - 2.0 * gp_std(x, Z, ls, A, sf2)  # noqa: E731  (LCB)
    rng = np.random.default_rng(4)
    for lo, hi in boxes(rng, 6):
        lo, hi = jnp.asarray(lo), jnp.asarray(hi)
        cv = jax.jit(lambda x: relax(f)(x, lo, hi).cv)
        vg = jax.jit(jax.value_and_grad(cv))
        X = jnp.asarray(rng.uniform(lo, hi, (400, 2)))
        CV = jax.vmap(cv)(X)
        for x0 in np.concatenate([np.array([lo, hi]), rng.uniform(lo, hi, (4, 2))]):
            v, gr = vg(jnp.asarray(x0))
            if np.isfinite(v) and np.all(np.isfinite(gr)):
                assert np.max(v + (X - x0) @ gr - CV) <= 1e-8 * (1 + np.abs(CV).max())


def test_node_bound_below_sampled_min():
    Z, ls, A, alpha, sf2 = trained()
    lcb = lambda x: gp_mean(x, Z, ls, alpha, sf2) - 2.0 * gp_std(x, Z, ls, A, sf2)  # noqa: E731
    bound = jax.jit(make_bounder(lcb, univariate=True, alphabb=True))
    fv = jax.jit(jax.vmap(lcb))
    rng = np.random.default_rng(5)
    for lo, hi in boxes(rng, 8):
        lb = float(bound(jnp.asarray(lo), jnp.asarray(hi))[0])
        fmin = float(np.min(fv(jnp.asarray(rng.uniform(lo, hi, (4000, 2))))))
        assert lb <= fmin + 1e-9 * (1 + abs(fmin))


def test_chance_constraint_certified_and_tighter():
    """min -mean s.t. mean + 2 std <= c: same optimum as the formula, in fewer nodes."""
    Z, ls, A, alpha, sf2 = trained(noise=1e-4)
    obj = lambda x: -gp_mean(x, Z, ls, alpha, sf2)  # noqa: E731
    c = 0.5  # binds: the unconstrained maximizer of the mean violates it
    con_p = lambda x: gp_mean(x, Z, ls, alpha, sf2) + 2.0 * gp_std(x, Z, ls, A, sf2) - c  # noqa: E731
    std_f = formula(Z, ls, A, sf2, "matern52")
    con_f = lambda x: gp_mean(x, Z, ls, alpha, sf2) + 2.0 * std_f(x) - c  # noqa: E731
    rp = minimize(obj, LO, HI, ineq=con_p, abs_tol=1e-3, rel_tol=1e-3, time_limit=120)
    rf = minimize(obj, LO, HI, ineq=con_f, abs_tol=1e-3, rel_tol=1e-3, time_limit=120)
    assert rp.status == "optimal"
    if rf.status == "optimal":
        assert abs(rp.fun - rf.fun) <= 2e-3 * (1 + abs(rf.fun))
        assert rp.nodes <= rf.nodes, (rp.nodes, rf.nodes)
    assert rp.lower_bound <= rf.fun + 1e-6

"""jaxglobal's pounce callbacks (local._Callbacks) give pounce's dense derivatives, compressed or not."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import local

pytest.importorskip("pounce")
from pounce.jax import from_jax  # noqa: E402


def test_coloring_separates_columns_sharing_a_row():
    rows = np.array([0, 0, 1, 1, 2, 2, 3])
    cols = np.array([0, 1, 1, 2, 3, 4, 4])
    color, k = local._color_columns(rows, cols, 5)
    for r in np.unique(rows):
        cs = cols[rows == r]
        assert len(set(color[cs])) == len(cs)
    assert k <= 3


def _sparse_model(n=12):
    def f(x):
        return jnp.sum((x - 0.3) ** 2 * jnp.arange(1, n + 1)) + x[0] * x[1] + jnp.sin(x[2] * x[3])

    def c(x):  # chain constraints: each row couples two neighbours, plus one dense row
        chain = x[1:] ** 2 + x[:-1] * x[1:] - 1.0
        return jnp.concatenate([chain, jnp.atleast_1d(jnp.sum(jnp.exp(0.1 * x)) - n)])

    return f, c


def _dense_model(n=6):
    def f(x):
        return jnp.sum(x) ** 4  # full Hessian: n colors, so not compressed

    def c(x):
        return jnp.atleast_1d(jnp.prod(1.0 + x**2) - 3.0)  # full Jacobian row

    return f, c


@pytest.mark.parametrize("model,n,compressed", [(_sparse_model, 12, True), (_dense_model, 6, False)],
                         ids=["sparse", "dense"])
def test_callbacks_match_pounce_dense(model, n, compressed):
    f, c = model(n)
    m = c(jnp.zeros(n)).size
    prob = from_jax(f, c, n=n, m=m, lb=np.full(n, -2.0), ub=np.full(n, 2.0),
                    cl=np.full(m, -1e20), cu=np.zeros(m))  # fmt: skip
    ref = prob.problem_obj
    cb = local._Callbacks(f, c, n, m, ref)
    assert (cb._hess_seed is not None) == compressed
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.uniform(-2, 2, n)
        lam, s = rng.normal(size=m), rng.uniform(0.5, 2.0)
        assert cb.objective(x) == pytest.approx(ref.objective(x), rel=1e-14, abs=1e-14)
        np.testing.assert_allclose(cb.gradient(x), ref.gradient(x), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(cb.constraints(x), ref.constraints(x), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(cb.jacobian(x), ref.jacobian(x), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cb.hessian(x, lam, s), ref.hessian(x, lam, s), rtol=1e-12, atol=1e-12)


def test_unconstrained_callbacks():
    n = 8
    f = lambda x: jnp.sum(x**4) + x[0] * x[7]  # noqa: E731
    prob = from_jax(f, n=n, lb=np.full(n, -1.0), ub=np.full(n, 1.0))
    cb = local._Callbacks(f, None, n, 0, prob.problem_obj)
    x = np.linspace(-0.9, 0.9, n)
    np.testing.assert_allclose(cb.hessian(x, np.zeros(0), 1.5), prob.problem_obj.hessian(x, np.zeros(0), 1.5), rtol=1e-12)


def test_local_solve_finds_constrained_minimum():
    f, c = _sparse_model(12)
    lo, hi = np.full(12, -2.0), np.full(12, 2.0)
    solve = local.make_local(f, lambda x: c(x), None, lo, hi, "pounce")
    X = solve(np.random.default_rng(1).uniform(-1, 1, (3, 12)))
    for x in X:
        assert np.max(np.asarray(c(jnp.asarray(x)))) <= 1e-6

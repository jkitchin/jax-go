"""``lp.ipm(..., box=True)``: implicit unit-box rows, early stop, predictor-corrector."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linprog

from jaxglobal import lp

jax.config.update("jax_enable_x64", True)


def _lp(seed, m=12, n=6):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))
    z = rng.uniform(0.2, 0.8, size=n)  # a strictly feasible point in the box
    b = A @ z + rng.uniform(0.05, 1.0, size=m)
    c = rng.normal(size=n)
    return c, A, b


def _cert(c, A, b):
    def cert(y):  # weak duality over the unit box, as the bounders use it
        y = jnp.maximum(y[: A.shape[0]], 0.0)
        return jnp.sum(jnp.minimum(c + A.T @ y, 0.0)) - y @ b

    return cert


@pytest.mark.parametrize("seed", range(6))
def test_box_rows_implicit_same_iterates(seed):
    """Without predictor-corrector, the implicit box gives ipm's result with explicit box rows."""
    c, A, b = (jnp.asarray(v) for v in _lp(seed))
    n = c.size
    Ab = jnp.concatenate([A, jnp.eye(n), -jnp.eye(n)])
    bb = jnp.concatenate([b, jnp.ones(n), jnp.zeros(n)])
    z0, _, v0 = lp.ipm(c, Ab, bb, 40, cert=_cert(c, A, b))
    z1, y1, v1 = lp._ipm_box(c, A, b, 40, 0.2, _cert(c, A, b), 1e-10, jnp.inf, pc=False)
    assert y1.shape == (A.shape[0],)
    np.testing.assert_allclose(z1, z0, atol=1e-7)
    np.testing.assert_allclose(v1, v0, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("pc", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_certificate_valid_and_tight(seed, pc):
    c, A, b = _lp(seed)
    opt = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * c.size, method="highs").fun
    z, y, v = lp._ipm_box(*(jnp.asarray(t) for t in (c, A, b)), 40, 0.2,
                          _cert(*(jnp.asarray(t) for t in (c, A, b))), 1e-10, jnp.inf, pc=pc)  # fmt: skip
    assert float(v) <= opt + 1e-9
    assert float(v) >= opt - 1e-6 * (1 + abs(opt))
    assert float(_cert(c, A, b)(y)) == pytest.approx(float(v))  # y attains the returned certificate


@pytest.mark.parametrize("pc", [False, True])
def test_target_stops_early_with_valid_bound(pc):
    c, A, b = _lp(3)
    opt = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * c.size, method="highs").fun
    args = [jnp.asarray(t) for t in (c, A, b)]
    _, _, v = lp._ipm_box(*args, 40, 0.2, _cert(*args), 1e-10, opt - 0.5, pc=pc)
    assert opt - 0.5 <= float(v) <= opt + 1e-9


@pytest.mark.parametrize("pc", [False, True])
def test_infeasible_lp_bound_large(pc):
    """An empty LP: the duals follow a Farkas ray, so the certificate grows past any cutoff."""
    n = 3
    A = jnp.concatenate([jnp.ones((1, n)), -jnp.ones((1, n))])
    b = jnp.array([0.5, -2.0])  # sum z <= 0.5 and sum z >= 2
    c = jnp.ones(n)
    _, _, v = lp._ipm_box(c, A, b, 40, 0.2, _cert(c, A, b), 1e-10, jnp.inf, pc=pc)
    assert float(v) > 10.0


def test_vmap():
    lps = [_lp(s) for s in range(4)]
    C, A, B = (jnp.asarray(np.stack(t)) for t in zip(*lps))
    V = jax.vmap(lambda c, a, b: lp.ipm(c, a, b, 40, cert=_cert(c, a, b), box=True)[2])(C, A, B)
    for (c, a, b), v in zip(lps, V):
        opt = linprog(c, A_ub=a, b_ub=b, bounds=[(0, 1)] * c.size, method="highs").fun
        assert float(v) <= opt + 1e-9 and float(v) >= opt - 1e-6 * (1 + abs(opt))

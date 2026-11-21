"""The HiGHS node LP of the lifted quadratic bounder (``quadratic.NODE_LP = "highs"``)."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxglobal import quadratic as Qm  # noqa: E402

pytestmark = pytest.mark.skipif(importlib.util.find_spec("highspy") is None, reason="needs highspy")


def _problem():
    # min (x0 - 0.3)^2 + x0 x1 + x1  s.t.  x0 + x1 >= 0.5, x0 x1 <= 0.4
    f = lambda v: (v[0] - 0.3) ** 2 + v[0] * v[1] + v[1]  # noqa: E731
    g = lambda v: jnp.array([0.5 - v[0] - v[1], v[0] * v[1] - 0.4])  # noqa: E731
    return f, g, Qm.quadratic_form(f, g, None, 2)


@pytest.fixture
def highs(monkeypatch):
    monkeypatch.setattr(Qm, "NODE_LP", "highs")


def _grid_min(f, g, lo, hi, k=401):
    a, b = np.meshgrid(np.linspace(lo[0], hi[0], k), np.linspace(lo[1], hi[1], k))
    X = np.stack([a.ravel(), b.ravel()], 1)
    F = np.asarray(jax.vmap(f)(jnp.asarray(X)))
    G = np.asarray(jax.vmap(g)(jnp.asarray(X)))
    ok = np.all(G <= 0, axis=1)
    return F[ok].min() if ok.any() else np.inf


def test_bound_valid_and_matches_ipm(highs, monkeypatch):
    f, g, Q = _problem()
    rng = np.random.default_rng(0)
    for _ in range(8):
        u, w = rng.uniform(-1, 2, 2), rng.uniform(-1, 2, 2)
        lo, hi = np.minimum(u, w), np.maximum(u, w) + 0.1
        lb_h = float(Qm.make_quadratic_bounder(Q, n_kelley=4)(jnp.asarray(lo), jnp.asarray(hi))[0])
        fmin = _grid_min(f, g, lo, hi)
        assert lb_h <= fmin + 1e-6
        monkeypatch.setattr(Qm, "NODE_LP", "ipm")
        lb_i = float(Qm.make_quadratic_bounder(Q, n_kelley=4)(jnp.asarray(lo), jnp.asarray(hi))[0])
        monkeypatch.setattr(Qm, "NODE_LP", "highs")
        if np.isfinite(lb_i):  # (the IPM may stop short: HiGHS's exact LP bound is at least as tight)
            assert lb_h >= lb_i - 1e-6 * (1 + abs(lb_i))


def test_infeasible_box_is_inf(highs):
    f, g, Q = _problem()
    lo, hi = jnp.array([-1.0, -1.0]), jnp.array([-0.6, -0.6])  # x0 + x1 <= -1.2 < 0.5
    lb, _, infeasible, _, _ = jax.jit(Qm.make_quadratic_bounder(Q, n_kelley=2))(lo, hi)
    assert float(lb) == np.inf and bool(infeasible)


def test_vmap_jit_and_grad(highs):
    f, g, Q = _problem()
    bound = Qm.make_quadratic_bounder(Q, n_kelley=4)
    lo, hi = jnp.array([-1.0, -1.0]), jnp.array([1.0, 2.0])
    lb = lambda lo, hi: bound(lo, hi)[0]  # noqa: E731
    one = float(lb(lo, hi))
    batch = np.asarray(jax.jit(jax.vmap(lb))(jnp.stack([lo, lo]), jnp.stack([hi, hi])))
    assert np.allclose(batch, one)
    gr = jax.grad(lb, argnums=(0, 1))(lo, hi)
    fw = jax.jacfwd(lb, argnums=(0, 1))(lo, hi)
    for a, b in zip(gr, fw):
        assert np.all(np.isfinite(np.asarray(a))) and np.allclose(np.asarray(a), np.asarray(b), atol=1e-8)

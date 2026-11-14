"""Eigenvector cuts of the lifted quadratic LP (quadratic._eigen_cuts): validity at sampled
points of random boxes, and the bound they give on a convex quadratic over a wide box."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from jaxglobal.quadratic import _layout, _lp, make_quadratic_bounder, quadratic_form

jax.config.update("jax_enable_x64", True)


def random_qp(n, seed, psd=False):
    """min x'Ax + c.x s.t. x'Bx + d.x <= e: A positive definite (``psd``) or indefinite, B indefinite."""
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, n))
    A = G @ G.T / n if psd else (G + G.T) / 2
    H = rng.normal(size=(n, n))
    B = (H + H.T) / 2
    c, d = rng.normal(size=n) * 3, rng.normal(size=n)

    def quad(M, a, x):  # elementwise, so that the tape scalarizes
        return sum(float(M[i, j]) * x[i] * x[j] for i in range(n) for j in range(n)) + sum(
            float(a[i]) * x[i] for i in range(n))

    f = lambda x: quad(A, c, x)  # noqa: E731
    g = lambda x: jnp.stack([quad(B, d, x) - 5.0])  # noqa: E731
    return f, g, A, c


def lifted(Q, x):
    return np.concatenate([x, x[Q.pairs[:, 0]] * x[Q.pairs[:, 1]]])


def test_eigen_rows_present():
    f, g, _, _ = random_qp(5, 0)
    Q = quadratic_form(f, g, None, 5)
    assert Q.n_eig > 0 and Q.n_sec > 0  # (indefinite rows: both kinds)
    x = np.random.default_rng(1).normal(size=5)
    v = lifted(Q, x)
    for row, col, cw, cx, K in ((Q.eig_row, Q.eig_col, Q.eig_cw, Q.eig_cx, Q.n_eig),
                                (Q.sec_row, Q.sec_col, Q.sec_cw, Q.sec_cx, Q.n_sec)):
        t = np.bincount(row, cx * v[col], K)
        s = np.bincount(row, cw * v[col], K)
        np.testing.assert_allclose(s, t * t, rtol=1e-10, atol=1e-10)  # the identity (v.x)^2 = sum v_i v_j w_ij


def test_kinds_follow_curvature():
    """A convex objective gets tangents only, a concave constraint secants only."""
    n = 4
    f, _, _, _ = random_qp(n, 3, psd=True)
    g = lambda x: jnp.stack([-(x[0] + x[1] + x[2]) ** 2 - (x[1] - x[3]) ** 2 + 1.0])  # noqa: E731
    Q = quadratic_form(f, g, None, n)
    assert Q.n_eig >= n and Q.n_sec >= 1
    assert quadratic_form(f, None, None, n).n_sec == 0


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_rows_hold_at_sampled_points(seed, safe):
    """Every McCormick and eigen row of the LP (secants, fixed tangents, and Kelley tangents set at an arbitrary
    point) holds at (x, x x') for points x sampled in random boxes, corners included."""
    n = 4 + seed % 3
    f, g, _, _ = random_qp(n, seed)
    Q = quadratic_form(f, g, None, n)
    assert Q.n_eig > 0
    rng = np.random.default_rng(10 + seed)
    L = _layout(Q, n, 2, False)
    R, C = L.pattern.rows, L.pattern.cols
    sel = np.zeros(L.pattern.m, bool)
    sel[:4 * len(Q.pairs)] = True  # McCormick
    sel[L.pattern.m - L.eig_types * Q.n_eig - Q.n_sec:] = True  # eigen rows (last: no cutoff row)
    assert L.eig_types == 5 and Q.n_sec > 0
    for _ in range(5):
        c0, w = rng.normal(size=n) * 3, rng.uniform(0.01, 5.0, n)
        lo, hi = c0 - w, c0 + w
        vlo, vhi, vals, b, tangents = _lp(Q, L, jnp.asarray(lo), jnp.asarray(hi), 0.0, safe)
        for k in range(2):  # Kelley rows at points outside the lifted feasible set, too
            vals, b = tangents(vals, b, jnp.asarray(rng.uniform(np.asarray(vlo), np.asarray(vhi))), k)
        vals, b = np.asarray(vals), np.asarray(b)
        pts = [rng.uniform(lo, hi) for _ in range(200)] + [np.where(rng.random(n) < 0.5, lo, hi) for _ in range(50)]
        for x in pts:
            v = lifted(Q, x)
            Av = np.bincount(R, vals * v[C], L.pattern.m)
            live = np.isfinite(b) & sel  # (the problem's own rows need not hold at an infeasible x)
            assert np.all(Av[live] <= b[live] + 1e-9 * (1 + np.abs(b[live]))), (Av - b)[live].max()


@pytest.mark.parametrize("seed", range(3))
def test_bound_valid_and_tighter(seed):
    """On a convex quadratic over a wide box the bound is valid (below sampled values and the
    box minimum) and much tighter than the McCormick-only LP; on indefinite rows it stays valid."""
    n = 6
    f, g, A, c = random_qp(n, seed, psd=True)
    lo, hi = -50.0 * np.ones(n), 50.0 * np.ones(n)
    res = minimize(lambda x: x @ A @ x + c @ x, np.zeros(n), jac=lambda x: 2 * A @ x + c,
                   bounds=list(zip(lo, hi)), method="L-BFGS-B", options=dict(ftol=1e-15, gtol=1e-12))
    xs, fmin = res.x, float(res.fun)  # convex: the box minimum
    Q = quadratic_form(f, None, None, n)  # no constraint: the box minimum is fmin
    lb = float(jax.jit(make_quadratic_bounder(Q))(jnp.asarray(lo), jnp.asarray(hi))[0])
    lb0 = float(jax.jit(make_quadratic_bounder(Q._replace(n_eig=0)))(jnp.asarray(lo), jnp.asarray(hi))[0])
    assert lb0 <= lb <= fmin + 1e-6 * (1 + abs(fmin))
    assert fmin - lb < 0.05 * (fmin - lb0)  # most of the McCormick gap closed
    Q = quadratic_form(f, g, None, n)  # with an indefinite constraint: below every feasible value
    lb = float(jax.jit(make_quadratic_bounder(Q))(jnp.asarray(lo), jnp.asarray(hi))[0])
    rng = np.random.default_rng(seed)
    gj, fj = jax.jit(g), jax.jit(f)
    for x in np.clip(np.concatenate([rng.uniform(lo, hi, (500, n)), xs + rng.normal(size=(500, n))]), lo, hi):
        if float(gj(jnp.asarray(x))[0]) <= 0:
            assert lb <= float(fj(jnp.asarray(x))) + 1e-9

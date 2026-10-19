"""Equality rows h = p / d = 0 cleared of their denominator in the lifted LP (quadratic_form).

A ratio whose denominator interval contains 0 has no finite McCormick relaxation, so the
reduced-space bounder drops every cut of such a row and the bound of a parameter-estimation
model (ex8_4_5) never leaves the trivial f >= 0. The cleared polynomial row p = 0 is linked
to the box by the McCormick facets of its products instead.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal
from jaxglobal.quadratic import _layout, _lp, make_quadratic_bounder, quadratic_form

jax.config.update("jax_enable_x64", True)

# fit y_k = a / (t_k + b) to data d_k; the denominators t_k + b change sign on b in [-1, 1]
T = np.array([0.2, 0.5, 1.0])
D = np.array([0.9, 0.3, 0.35])
LO = np.array([-1.0, -1.0, 0.1, 0.1, 0.1])
HI = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
OPT = 0.0300815991514807  # global minimum (grid search over (a, b) and the search below agree)
X_OPT = np.array([0.22054952, 0.04867215, 0.88690881, 0.40196959, 0.21031313])


def fit_f(x):
    return sum((x[2 + k] - D[k]) ** 2 for k in range(3))


def fit_h(x):
    return jnp.stack([x[0] / (T[k] + x[1]) - x[2 + k] for k in range(3)])


def lifted_point(Q, x):
    v = np.concatenate([x, np.zeros(len(Q.pairs))])
    for p, (a, b) in enumerate(Q.pairs):
        v[x.size + p] = v[a] * v[b]
    return v


def den_value(Q, x, r):
    t = Q.den_row == r
    return np.sum(Q.den_coef[t] * np.prod(x[None, :] ** Q.den_pow[t], axis=1))


def test_rational_equality_is_cleared():
    """Each row of the form is h(x) d(x): the polynomial p with h = p / d, at any point."""
    Q = quadratic_form(fit_f, None, fit_h, 5)
    assert Q is not None and Q.n_h == 3
    assert sorted(set(Q.den_row.tolist())) == [0, 1, 2]
    assert len(Q.rlt_var) == 0  # a cleared row is no linear equality for RLT
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.uniform(LO, HI)
        v = lifted_point(Q, x)
        np.testing.assert_allclose(Q.E[0] @ v + Q.c[0], fit_f(x), rtol=1e-12)
        for r in range(3):
            d = den_value(Q, x, r)
            np.testing.assert_allclose(d, T[r] + x[1], rtol=1e-12)
            np.testing.assert_allclose(Q.E[1 + r] @ v + Q.c[1 + r], float(fit_h(x)[r]) * d, rtol=1e-9, atol=1e-12)


def test_nested_ratios_are_cleared():
    """Sums and quotients of ratios: one common denominator."""
    h = lambda x: jnp.stack([x[0] / (1.0 + x[1]) + x[1] / (2.0 + x[0]) - x[2],  # noqa: E731
                             (x[0] / (1.0 + x[1])) / (x[2] + 3.0) - 0.1])  # fmt: skip
    Q = quadratic_form(lambda x: x[2], None, h, 3, degree=4)
    assert Q is not None
    x = np.array([0.3, -0.4, 0.7])
    v = np.concatenate([x, np.zeros(len(Q.pairs))])
    for p, (a, b) in enumerate(Q.pairs):
        v[3 + p] = v[a] * v[b]
    for r in range(2):
        np.testing.assert_allclose(Q.E[1 + r] @ v + Q.c[1 + r], float(h(x)[r]) * den_value(Q, x, r), atol=1e-12)


def test_ratio_in_objective_or_inequality_has_no_form():
    """Clearing an inequality needs the sign of the denominator, and the objective cannot be
    cleared: no form (the reduced-space bounder is used)."""
    assert quadratic_form(lambda x: x[0] / (1.0 + x[1]), None, lambda x: x[0] - x[1], 2) is None
    assert quadratic_form(lambda x: x[0], lambda x: x[0] / (1.0 + x[1]) - 0.5, None, 2) is None


def test_cleared_row_keeps_tolerance_feasible_points():
    """|h| <= tol only gives |p| = |h d| <= tol |d|: a point with h = tol where |d| = 4 must
    satisfy every LP row."""
    tol = 1e-3
    h = lambda x: jnp.stack([x[0] / x[1] - x[2]])  # noqa: E731
    Q = quadratic_form(lambda x: x[2], None, h, 3)
    lo, hi = np.array([0.0, 2.0, 0.0]), np.array([4.0, 4.0, 1.0])
    x = np.array([(0.5 + tol) * 4.0, 4.0, 0.5])
    assert float(h(x)[0]) == pytest.approx(tol)
    L = _layout(Q, 3, 0, False)
    vlo, vhi, vals, b, _ = _lp(Q, L, jnp.asarray(lo), jnp.asarray(hi), tol, False)
    A = np.zeros((L.pattern.m, vlo.size))
    np.add.at(A, (L.pattern.rows, L.pattern.cols), np.asarray(vals))
    assert np.all(A @ lifted_point(Q, x) <= np.asarray(b) + 1e-12)


@pytest.mark.parametrize("safe", [False, True])
def test_cleared_bound_is_valid_and_not_trivial(safe):
    """With b in [-0.3, 0.3] (the denominator 0.2 + b changes sign) the lifted LP proves more
    than f >= 0, where the reduced-space cuts prove nothing; on boxes around the optimum the
    bound never exceeds the optimum."""
    from jaxglobal.lp import make_constrained_bounder

    Q = quadratic_form(fit_f, None, fit_h, 5)
    bound = jax.jit(make_quadratic_bounder(Q, tol=1e-6, safe=safe))
    lo, hi = LO.copy(), HI.copy()
    lo[1], hi[1] = -0.3, 0.3
    lb0 = float(bound(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
    lr = float(jax.jit(make_constrained_bounder(fit_f, None, fit_h, tol=1e-6))(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
    assert lr <= 0.0 and 1e-3 < lb0 <= OPT
    rng = np.random.default_rng(2)
    w = HI - LO
    for _ in range(6):
        lo = np.maximum(LO, X_OPT - rng.uniform(0, 0.3, 5) * w)
        hi = np.minimum(HI, X_OPT + rng.uniform(0, 0.3, 5) * w)
        lb, _, infeasible, rlo, rhi = bound(jnp.asarray(lo), jnp.asarray(hi), OPT + 1e-6)
        assert not infeasible and float(lb) <= OPT + 1e-9
        assert np.all(np.asarray(rlo) <= X_OPT + 1e-7) and np.all(X_OPT <= np.asarray(rhi) + 1e-7)


def test_minimize_rational_fit():
    """The search proves the global minimum of the fit."""
    r = jaxglobal.minimize(fit_f, LO, HI, eq=fit_h, rel_tol=1e-6, abs_tol=1e-8)
    assert r.status == "optimal"
    assert r.fun == pytest.approx(OPT, abs=1e-7)
    assert r.lower_bound <= OPT + 1e-9

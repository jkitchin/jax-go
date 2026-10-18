"""Lifted LP for quadratic problems (quadratic.py): coefficients, bound, RLT rows, OBBT, search."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linprog

import jaxglobal
from jaxglobal.quadratic import _layout, _lp, make_quadratic_bounder, make_quadratic_obbt, quadratic_form

from test_constrained import HAVERLY_X, haverly1

jax.config.update("jax_enable_x64", True)


def lifted_point(Q, x):
    """(x, w) with every product column at its value."""
    return np.concatenate([x, x[Q.pairs[:, 0]] * x[Q.pairs[:, 1]]])


def test_form_is_exact():
    """Every function is E (x, w) + c with w the products, at any point."""
    f, g, h, lo, hi = haverly1()
    Q = quadratic_form(f, g, h, 8)
    assert Q is not None and len(Q.pairs) == 2  # p Px and p Py (p (Px + Py) expands to the same two)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.uniform(lo, hi)
        vals = np.concatenate([[f(x)], g(x), h(x)])
        np.testing.assert_allclose(Q.E @ lifted_point(Q, x) + Q.c, vals, rtol=1e-12, atol=1e-9)


def test_not_quadratic():
    """A cubic term or an exp: no form, the reduced-space bounder is used."""
    assert quadratic_form(lambda x: x[0] * x[1] * x[2], lambda x: x[0] - 1.0, None, 3) is None
    assert quadratic_form(lambda x: x[0], lambda x: jnp.exp(x[0]) - x[1], None, 2) is None


def test_rlt_rows_hold_at_feasible_point():
    """A linear equality times a variable whose products with the equality's variables are all
    columns gives an RLT row, zero at every feasible point; other variables give none."""
    f = lambda x: x[0] * x[2] - x[1] * x[2] + x[3]  # noqa: E731
    h = lambda x: jnp.stack([x[0] + x[1] - 1.0])  # noqa: E731
    Q = quadratic_form(f, None, h, 4)
    assert Q.rlt_var.tolist() == [2]  # x_2 (x_0 + x_1 - 1) = 0 uses only the columns x_0 x_2, x_1 x_2
    x = np.array([0.3, 0.7, 2.0, 5.0])
    np.testing.assert_allclose(Q.rlt @ lifted_point(Q, x), 0.0, atol=1e-12)


def highs_bound(Q, lo, hi, tol):
    """The lifted LP of the box solved by HiGHS, from the same rows."""
    n = lo.size
    L = _layout(Q, n, 0, False)
    vlo, vhi, vals, b, _ = _lp(Q, L, jnp.asarray(lo), jnp.asarray(hi), tol, False)
    A = sparse.csr_matrix((np.asarray(vals), (L.pattern.rows, L.pattern.cols)), shape=(L.pattern.m, vlo.size))
    res = linprog(Q.E[0], A_ub=A, b_ub=np.asarray(b), bounds=list(zip(np.asarray(vlo), np.asarray(vhi))), method="highs")
    assert res.status == 0
    return res.fun + Q.c[0]


@pytest.mark.parametrize("safe", [False, True])
def test_bound_matches_highs_and_is_valid(safe):
    """On boxes around the optimum the certified bound equals HiGHS' LP value (to the IPM's
    accuracy), never exceeds the optimum, and the box reduction keeps the optimum."""
    f, g, h, lo0, hi0 = haverly1()
    Q = quadratic_form(f, g, h, 8)
    bound = jax.jit(make_quadratic_bounder(Q, tol=1e-6, safe=safe))
    rng = np.random.default_rng(1)
    w = hi0 - lo0
    boxes = [(lo0, hi0)] + [(np.maximum(lo0, HAVERLY_X - rng.uniform(0, 0.5, 8) * w),
                             np.minimum(hi0, HAVERLY_X + rng.uniform(0, 0.5, 8) * w)) for _ in range(6)]  # fmt: skip
    for lo, hi in boxes:
        lb, x, infeasible, rlo, rhi = bound(jnp.asarray(lo), jnp.asarray(hi), -400.0 + 1e-6)
        assert not infeasible
        assert float(lb) <= -400.0 + 1e-6
        assert float(lb) == pytest.approx(highs_bound(Q, lo, hi, 1e-6), abs=1e-5 * 400)
        assert np.all(np.asarray(rlo) <= HAVERLY_X + 1e-9) and np.all(HAVERLY_X <= np.asarray(rhi) + 1e-9)


def test_lifted_bound_is_tighter_than_reduced_space():
    """On the root box the shared product columns give a tighter bound than the reduced-space cuts."""
    from jaxglobal.lp import make_constrained_bounder

    f, g, h, lo, hi = haverly1()
    Q = quadratic_form(f, g, h, 8)
    lq = float(jax.jit(make_quadratic_bounder(Q, tol=1e-6))(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
    lr = float(jax.jit(make_constrained_bounder(f, g, h, tol=1e-6))(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
    assert lq >= lr - 1e-6
    assert lq <= -400.0


def test_infeasible_box_is_pruned():
    """A box with no feasible point gets an infinite (or above any cutoff) bound."""
    f = lambda x: x[0]  # noqa: E731
    h = lambda x: jnp.stack([x[0] * x[1] - 1.0])  # noqa: E731
    Q = quadratic_form(f, None, h, 2)
    lb, *_ = jax.jit(make_quadratic_bounder(Q, tol=1e-6))(jnp.array([0.0, 0.0]), jnp.array([0.5, 0.5]), jnp.inf)
    assert float(lb) > 1e6


def test_obbt_keeps_optimum():
    f, g, h, lo, hi = haverly1()
    Q = quadratic_form(f, g, h, 8)
    nlo, nhi, empty = jax.jit(make_quadratic_obbt(Q, tol=1e-6))(jnp.asarray(lo), jnp.asarray(hi), -400.0 + 1e-6)
    assert not empty
    assert np.all(np.asarray(nlo) <= HAVERLY_X + 1e-9) and np.all(HAVERLY_X <= np.asarray(nhi) + 1e-9)
    assert np.mean((np.asarray(nhi) - np.asarray(nlo)) / (hi - lo)) < 1.0


def test_minimize_haverly_quadratic():
    """The search on the lifted LP proves the optimum, in fewer nodes than the reduced-space LP."""
    f, g, h, lo, hi = haverly1()
    r = jaxglobal.minimize(f, lo, hi, ineq=g, eq=h, rel_tol=1e-4, abs_tol=1e-6, time_limit=120)
    assert r.status == "optimal" and r.verified
    assert r.fun == pytest.approx(-400.0, rel=1e-4)
    assert r.lower_bound <= -400.0 + 1e-6
    r0 = jaxglobal.minimize(f, lo, hi, ineq=g, eq=h, rel_tol=1e-4, abs_tol=1e-6, time_limit=120, quadratic=False)
    assert r0.status == "optimal" and r.nodes <= r0.nodes


def _three_circles():
    """Three unit-diameter circles in a W x H rectangle, minimize W H: v = (x1, y1, ..., W, H, a)."""
    n = 9

    def f(v):
        return v[8]

    def g(v):
        c = [(v[0], v[1]), (v[2], v[3]), (v[4], v[5])]
        d = [1.0 - (c[i][0] - c[j][0]) ** 2 - (c[i][1] - c[j][1]) ** 2 for i in range(3) for j in range(i + 1, 3)]
        box = [v[2 * i + k] + 0.5 - v[6 + k] for i in range(3) for k in range(2)]
        return jnp.stack(d + box)

    def h(v):
        return jnp.atleast_1d(v[6] * v[7] - v[8])

    lo = np.array([0.5] * 6 + [1.0, 1.0, 1.0])
    hi = np.array([3.5] * 6 + [4.0, 4.0, 16.0])
    return f, g, h, lo, hi, n


def test_squares_bounded_below():
    """Kelley tangents on the squares are skipped only where no row needs a square bounded below."""
    from jaxglobal.quadratic import squares_bounded_below

    f, g, h, lo, hi, n = _three_circles()
    assert not squares_bounded_below(quadratic_form(f, g, h, n))  # reverse-convex: only secants bind
    assert squares_bounded_below(quadratic_form(lambda x: x[0] ** 2 + x[1], None, None, 2))  # convex objective
    assert squares_bounded_below(quadratic_form(lambda x: x[1], lambda x: x[0] ** 2 - x[1], None, 2))  # convex row
    assert squares_bounded_below(quadratic_form(lambda x: x[1], None, lambda x: x[0] ** 2 - x[1], 2))  # equality
    assert not squares_bounded_below(quadratic_form(lambda x: -x[0] ** 2 + x[1], None, None, 2))  # concave
    assert not squares_bounded_below(quadratic_form(lambda x: x[0] * x[1], None, None, 2))  # no square


def test_skipped_tangents_do_not_change_the_bound(monkeypatch):
    """On boxes of the circle model the bound without Kelley rounds equals the bound with them."""
    import jaxglobal.quadratic as qd

    f, g, h, lo, hi, n = _three_circles()
    Q = quadratic_form(f, g, h, n)
    rng = np.random.default_rng(3)
    boxes = []
    for _ in range(16):
        c = rng.uniform(lo, hi)
        w = rng.uniform(0.3, 1.0) * (hi - lo)
        boxes.append((np.clip(c - w, lo, hi), np.clip(c + w, lo, hi)))
    skip = make_quadratic_bounder(Q, n_kelley=4, tol=1e-6)
    monkeypatch.setattr(qd, "squares_bounded_below", lambda Q: True)
    full = make_quadratic_bounder(Q, n_kelley=4, tol=1e-6)
    checked = 0
    for blo, bhi in boxes:
        a, b = skip(jnp.asarray(blo), jnp.asarray(bhi))[0], full(jnp.asarray(blo), jnp.asarray(bhi))[0]
        if b > 1e3 or a > 1e3:  # an infeasible box: both certificates diverge and prune it
            assert a > 1e3 and b > 1e3
        else:
            assert abs(a - b) <= 1e-6 * (1 + abs(b))
            checked += 1
    assert checked >= 3


def test_circle_packing_solves():
    """Three unit circles: the optimal rectangle is 1 x 3 (area 3) or the triangle packing area."""
    f, g, h, lo, hi, n = _three_circles()
    r = jaxglobal.minimize(f, lo, hi, ineq=g, eq=h, time_limit=60)
    assert r.status == "optimal"
    assert r.lower_bound <= r.fun + 1e-6
    # stacked triangle: W = 1 + sqrt(3)/2... the solver's answer must be at most the 1 x 3 strip
    assert r.fun <= 3.0 + 1e-6

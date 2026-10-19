"""Sparse lifted LP for polynomials of degree > 2 (quadratic.polynomial_form): every monomial is a
column that is the product of two earlier columns, linked by McCormick facets (or a square's
secant and tangents) over interval column ranges. Tests: the form is exact, every row and column
range holds at every point of the box, the certified bound is below the true minimum and equals
HiGHS on the same rows, degree-2 problems give the quadratic form unchanged, and the search."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linprog

import jaxglobal
from jaxglobal.quadratic import (_layout, _lift, _lp, make_quadratic_bounder, make_quadratic_obbt,
                                 polynomial_form, quadratic_form)  # fmt: skip

from test_constrained import haverly1

jax.config.update("jax_enable_x64", True)


def lifted_point(Q, x):
    """(x, w): each column the product of its two factor columns, in column order."""
    v = list(np.asarray(x, float))
    for a, b in Q.pairs:
        v.append(v[a] * v[b])
    return np.asarray(v)


def poly_fn(terms):
    """A jax function sum_t c_t prod x^e, built from mul and integer_pow (both appear in .nl tapes)."""

    def fn(x):
        out = 0.0
        for c, mono in terms:
            t = c
            for i in sorted(set(mono)):
                e = mono.count(i)
                t = t * (x[i] if e == 1 else x[i] ** e)
            out = out + t
        return out

    return fn


def random_problem(seed, n=5, degree=4, n_terms=8, n_g=2):
    """A random polynomial objective and inequalities ``p_k(x) - b_k <= 0`` that hold at x0."""
    rng = np.random.default_rng(seed)

    def terms():
        out = []
        for _ in range(n_terms):
            d = int(rng.integers(1, degree + 1))
            out.append((float(rng.normal()), tuple(sorted(rng.integers(0, n, d).tolist()))))
        out.append((float(rng.normal()), tuple(sorted(rng.choice(n, degree, replace=degree > n).tolist()))))
        return out

    f = poly_fn(terms())
    gs = [poly_fn(terms()) for _ in range(n_g)]
    lo, hi = rng.uniform(-2.0, 0.5, n), rng.uniform(0.5, 2.0, n)
    x0 = 0.5 * (lo + hi)
    b = np.array([float(g(jnp.asarray(x0))) for g in gs]) + rng.uniform(0.1, 1.0, n_g)

    def g(x):
        return jnp.stack([gk(x) for gk in gs]) - b

    return f, g, lo, hi


def random_boxes(lo, hi, rng, k):
    yield lo, hi
    for _ in range(k):
        a, b = rng.uniform(lo, hi), rng.uniform(lo, hi)
        yield np.minimum(a, b), np.maximum(a, b) + 1e-3 * (hi - lo)


def test_lift_factors():
    """Each column is the product of two earlier ones, intermediate products are shared, and a
    degree-2 monomial is split into its two variables."""
    monos = {(0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (2, 2, 2, 2), (1, 4, 5, 6, 7)}
    made = _lift(monos)
    assert monos <= set(made)
    for m, (a, b) in made.items():
        assert tuple(sorted(a + b)) == m
        for part in (a, b):
            assert len(part) == 1 or part in made
    assert made[(0, 1)] == ((0,), (1,))
    assert made[(0, 1, 2)] == ((0, 1), (2,))  # reuses (0, 1)
    assert made[(2, 2, 2, 2)] == ((2, 2), (2, 2))  # x^4 = (x^2)^2: a square of a column


@pytest.mark.parametrize("seed", range(6))
def test_form_is_exact(seed):
    f, g, lo, hi = random_problem(seed)
    n = lo.size
    assert quadratic_form(f, g, None, n) is None  # degree 4: not quadratic
    Q = polynomial_form(f, g, None, n)
    assert Q is not None and Q.level.max() >= 2
    rng = np.random.default_rng(seed)
    for _ in range(5):
        x = rng.uniform(lo, hi)
        vals = np.concatenate([[f(jnp.asarray(x))], g(jnp.asarray(x))])
        np.testing.assert_allclose(Q.E @ lifted_point(Q, x) + Q.c, vals, rtol=1e-11, atol=1e-9)


def test_degree_limit_and_non_polynomial():
    assert polynomial_form(lambda x: x[0] ** 7, lambda x: x[0] - 1.0, None, 1) is None  # degree 7 > MAX_DEGREE
    assert polynomial_form(lambda x: x[0] * x[1] * x[2], lambda x: x[0] - 1.0, None, 3, degree=2) is None
    assert polynomial_form(lambda x: x[0], lambda x: jnp.exp(x[0]) - x[1], None, 2) is None
    Q = polynomial_form(lambda x: x[0] * x[1] * x[2], lambda x: x[0] - 1.0, None, 3, max_columns=4)
    assert Q is None  # 3 variables + (0, 1) + (0, 1, 2) = 5 columns > 4


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("safe", [False, True])
def test_rows_hold_at_random_points(seed, safe):
    """Every McCormick / square row and every column range holds at (x, exact products) for random
    x in random sub-boxes: the LP contains every point of the box."""
    f, g, lo0, hi0 = random_problem(seed)
    n = lo0.size
    Q = polynomial_form(f, g, None, n)
    L = _layout(Q, n, 0, False)
    rng = np.random.default_rng(100 + seed)
    for lo, hi in random_boxes(lo0, hi0, rng, 8):
        vlo, vhi, vals, b, _ = _lp(Q, L, jnp.asarray(lo), jnp.asarray(hi), 0.0, safe)
        A = sparse.csr_matrix((np.asarray(vals), (L.pattern.rows, L.pattern.cols)), shape=(L.pattern.m, vlo.size))
        b, vlo, vhi = np.asarray(b), np.asarray(vlo), np.asarray(vhi)
        mc = slice(0, 4 * len(Q.pairs))  # the facet rows (the g rows need not hold: x is any point)
        for _ in range(50):
            x = rng.uniform(lo, hi)
            v = lifted_point(Q, x)
            scale = 1e-10 * (1 + np.abs(v))
            assert np.all(vlo <= v + scale) and np.all(v <= vhi + scale)
            r = (A @ v - b)[mc]
            assert np.all(r <= 1e-9 * (1 + np.abs(b[mc]) + abs(A[mc]) @ np.abs(v)))
        for corner in range(4):  # the corners of the box too (facets are tight there)
            x = np.where(np.random.default_rng(corner).random(n) < 0.5, lo, hi)
            v = lifted_point(Q, x)
            assert np.all((A @ v - b)[mc] <= 1e-9 * (1 + np.abs(b[mc]) + abs(A[mc]) @ np.abs(v)))


def highs_bound(Q, lo, hi, tol=0.0):
    n = lo.size
    L = _layout(Q, n, 0, False)
    vlo, vhi, vals, b, _ = _lp(Q, L, jnp.asarray(lo), jnp.asarray(hi), tol, False)
    A = sparse.csr_matrix((np.asarray(vals), (L.pattern.rows, L.pattern.cols)), shape=(L.pattern.m, vlo.size))
    res = linprog(Q.E[0], A_ub=A, b_ub=np.asarray(b), bounds=list(zip(np.asarray(vlo), np.asarray(vhi))), method="highs")
    return res.fun + Q.c[0] if res.status == 0 else (np.inf if res.status == 2 else None)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("safe", [False, True])
def test_bound_below_true_min(seed, safe):
    """On random boxes the certified bound is at most the minimum of f over sampled feasible
    points (an upper estimate of the true minimum) and agrees with HiGHS on the same LP."""
    f, g, lo0, hi0 = random_problem(seed)
    n = lo0.size
    Q = polynomial_form(f, g, None, n)
    bound = jax.jit(make_quadratic_bounder(Q, safe=safe))
    bound0 = jax.jit(make_quadratic_bounder(Q, n_kelley=0, safe=safe))
    fv, gv = jax.jit(jax.vmap(f)), jax.jit(jax.vmap(g))
    rng = np.random.default_rng(200 + seed)
    for lo, hi in random_boxes(lo0, hi0, rng, 6):
        lb, x, infeasible, rlo, rhi = bound(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)
        lb = float(lb)
        X = rng.uniform(lo, hi, (4000, n))
        X = np.concatenate([X, np.where(rng.random((64, n)) < 0.5, lo, hi)])
        feas = np.all(np.asarray(gv(jnp.asarray(X))) <= 0.0, axis=1)
        if feas.any():
            fmin = float(np.min(np.asarray(fv(jnp.asarray(X)))[feas]))
            assert lb <= fmin + 1e-9 * (1 + abs(fmin))
            Xf = X[feas]  # range reduction keeps every feasible point with f <= cutoff (inf here)
            assert np.all(np.asarray(rlo) <= Xf.min(0) + 1e-9) and np.all(Xf.max(0) <= np.asarray(rhi) + 1e-9)
        lb0 = float(bound0(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
        lb0 = min(lb0, 1e30)  # an infeasible LP: any certificate above the cutoff will do
        assert lb >= lb0 - 1e-7 * (1 + abs(lb0))  # Kelley tangents of the squares only add rows
        hb = highs_bound(Q, lo, hi)
        if hb is not None and np.isfinite(hb):  # the same rows (no tangents); HiGHS has its own tolerances
            assert lb0 == pytest.approx(hb, abs=1e-4 * (1 + abs(hb)))


def test_safe_bound_not_above_plain():
    f, g, lo, hi = random_problem(3)
    Q = polynomial_form(f, g, None, lo.size)
    a = float(jax.jit(make_quadratic_bounder(Q))(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
    b = float(jax.jit(make_quadratic_bounder(Q, safe=True))(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
    assert b <= a + 1e-12 * (1 + abs(a))


def _same_form(Q2, Qp):
    for k in ("E", "c", "pairs", "rlt", "rlt_var"):
        np.testing.assert_array_equal(getattr(Q2, k), getattr(Qp, k))
    assert (Q2.n_g, Q2.n_h) == (Qp.n_g, Qp.n_h)


def quadratic_problem(seed):
    """A random quadratic objective and inequalities (``random_problem`` at degree 2) and a linear
    equality through the box centre, whose products with the variables give RLT rows."""
    f, g, lo, hi = random_problem(seed, n=6, degree=2, n_terms=12)
    x0 = 0.5 * (lo + hi)

    def h(x):
        return jnp.stack([x[0] + 2.0 * x[1] - x[2] - (x0[0] + 2.0 * x0[1] - x0[2])])

    def f2(x):  # x_3 times every variable of h: the RLT row x_3 h(x) = 0 has all its columns
        return f(x) + 0.3 * x[3] * (x[0] - x[1] + x[2])

    return f2, g, h, lo, hi


@pytest.mark.parametrize("which", ["haverly", 0, 1, 2])
@pytest.mark.parametrize("safe", [False, True])
def test_degree2_equivalence(which, safe):
    """On a quadratic problem the polynomial form is the quadratic form (same columns, rows and
    RLT rows), and both bounders give the same bound and box."""
    f, g, h, lo, hi = haverly1() if which == "haverly" else quadratic_problem(which)
    n = lo.size
    Q2 = quadratic_form(f, g, h, n)
    Qp = polynomial_form(f, g, h, n)
    assert Q2 is not None and Qp is not None
    _same_form(Q2, Qp)
    assert np.all(Qp.level == 1)
    if which != "haverly":
        assert Q2.rlt_var.size > 0
    b2 = jax.jit(make_quadratic_bounder(Q2, tol=1e-6, safe=safe))
    bp = jax.jit(make_quadratic_bounder(Qp, tol=1e-6, safe=safe))
    for blo, bhi in random_boxes(lo, hi, np.random.default_rng(0), 3):
        r2 = b2(jnp.asarray(blo), jnp.asarray(bhi), jnp.inf)
        rp = bp(jnp.asarray(blo), jnp.asarray(bhi), jnp.inf)
        for u, v in zip(r2, rp):
            np.testing.assert_array_equal(np.asarray(u), np.asarray(v))


def test_obbt_keeps_feasible_points():
    """OBBT over the lifted polynomial LP never cuts off a feasible point with f <= cutoff."""
    f, g, lo, hi = random_problem(4)
    n = lo.size
    Q = polynomial_form(f, g, None, n)
    rng = np.random.default_rng(4)
    X = rng.uniform(lo, hi, (20000, n))
    feas = np.all(np.asarray(jax.vmap(g)(jnp.asarray(X))) <= 0.0, axis=1)
    fx = np.asarray(jax.vmap(f)(jnp.asarray(X)))
    cutoff = float(np.quantile(fx[feas], 0.3))
    keep = X[feas & (fx <= cutoff)]
    nlo, nhi, empty = jax.jit(make_quadratic_obbt(Q))(jnp.asarray(lo), jnp.asarray(hi), cutoff)
    assert not empty
    assert np.all(np.asarray(nlo) <= keep.min(0) + 1e-9) and np.all(keep.max(0) <= np.asarray(nhi) + 1e-9)


def det3(x):
    """Determinant of the 3x3 matrix x[0:9] (row major), as a sum of 6 triple products."""
    m = [[x[3 * i + j] for j in range(3)] for i in range(3)]
    return (m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] + m[0][2] * m[1][0] * m[2][1]
            - m[0][2] * m[1][1] * m[2][0] - m[0][0] * m[1][2] * m[2][1] - m[0][1] * m[1][0] * m[2][2])  # fmt: skip


def test_minimize_binary_determinant(monkeypatch):
    """max det of a 3x3 0/1 matrix (= 2; hadamard_N in miniature): t <= det(B), maximize t, on the
    lifted polynomial LP (the exact variable-elimination shortcut, ``bnb._pseudo_boolean``, off);
    the result is optimal, verified, and its bound is valid."""
    monkeypatch.setattr(jaxglobal.bnb, "_pseudo_boolean", lambda *a, **k: None)
    f = lambda x: -x[9]  # noqa: E731
    g = lambda x: jnp.stack([x[9] - det3(x)])  # noqa: E731
    lo, hi = np.r_[np.zeros(9), -6.0], np.r_[np.ones(9), 6.0]
    assert quadratic_form(f, g, None, 10) is None and polynomial_form(f, g, None, 10) is not None
    r = jaxglobal.minimize(f, lo, hi, ineq=g, integer=np.arange(9), rel_tol=1e-4, abs_tol=1e-6, time_limit=120)
    assert r.status == "optimal" and r.verified
    assert r.fun == pytest.approx(-2.0, abs=1e-5)
    assert r.lower_bound <= -2.0 + 1e-6


def test_minimize_cubic_constrained():
    """A continuous degree-3 problem: min x0 x1 x2 + x0^2 x1 s.t. x0 + x1 + x2 <= 1.5 on
    [-1, 1]^3; the lifted LP's search proves an optimum that sampling cannot beat."""
    f = lambda x: x[0] * x[1] * x[2] + x[0] ** 2 * x[1]  # noqa: E731
    g = lambda x: jnp.stack([x[0] + x[1] + x[2] - 1.5])  # noqa: E731
    lo, hi = -np.ones(3), np.ones(3)
    r = jaxglobal.minimize(f, lo, hi, ineq=g, rel_tol=1e-4, abs_tol=1e-6, time_limit=120)
    assert r.status == "optimal"
    X = np.random.default_rng(0).uniform(lo, hi, (200000, 3))
    X = X[X.sum(1) <= 1.5]
    fmin = float(np.min(X[:, 0] * X[:, 1] * X[:, 2] + X[:, 0] ** 2 * X[:, 1]))
    assert r.lower_bound <= fmin + 1e-9
    assert r.fun <= fmin + 1e-3


def test_minimize_cancelled_monomial():
    """x0 x1^2 - x1^2 x0 cancels to a 0.0 coefficient: its product has no column, and the
    expansion must not index one (it raised KeyError). Optimum 0.11510 as on the reduced path."""
    f = lambda x: x[0] ** 3 + x[0] * x[1] * x[1] - x[1] * x[1] * x[0] + x[1]  # noqa: E731
    g = lambda x: jnp.array([0.5 - x[0] - x[1]])  # noqa: E731
    assert polynomial_form(f, g, None, 2) is not None
    r = jaxglobal.minimize(f, -np.ones(2), np.ones(2), ineq=g, time_limit=60)
    assert r.status == "optimal"
    assert r.fun == pytest.approx(0.1150988, abs=1e-5)
    assert r.lower_bound <= 0.1150988 + 1e-6


def offset_problem(c):
    f = lambda x: (x[0] - c) ** 4 + (x[1] - c - 1.0) ** 2 * (x[0] - c + 2.0) ** 2  # noqa: E731
    g = lambda x: jnp.array([(2 * c + 0.5) - x[0] - x[1]])  # noqa: E731
    return f, g, np.array([c - 1.0, c - 1.0]), np.array([c + 1.0, c + 3.0])


def test_polynomial_form_skips_large_offset():
    """Expanding (x - c)^4 around 0 for large c is a weak, badly conditioned relaxation: the form
    is skipped (reduced-space path) above OFFSET_LIMIT, kept for a small offset."""
    assert polynomial_form(*offset_problem(12345.6789)[:2], None, 2) is None
    assert polynomial_form(*offset_problem(100.0)[:2], None, 2) is None
    assert polynomial_form(*offset_problem(2.0)[:2], None, 2) is not None


@pytest.mark.parametrize("safe", [False, True])
def test_minimize_large_offset_at_root(safe):
    """The shifted polynomial solves at the root (as on main), plain and safe; minimum 0 at
    x = (c, c + 1)."""
    c = 12345.6789
    f, g, lo, hi = offset_problem(c)
    r = jaxglobal.minimize(f, lo, hi, ineq=g, safe=safe, time_limit=60)
    assert r.status == "optimal" and r.nodes == 1
    assert r.fun == pytest.approx(0.0, abs=1e-6)
    assert r.lower_bound <= 1e-9

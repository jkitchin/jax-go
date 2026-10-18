"""Cuts on the lifted polynomial LP of binary problems (quadratic._binary_cuts, _separate), the
objective lattice (_lattice, _snap), rows idle on the box (_unit) and the column cap for degree > 2.

Tests: every cut holds at every binary point; the cuts are off on boxes where a binary is not in
[0, 1]; the bound (cuts, separation and lattice rounding) is below the enumerated minimum on random
fixings, plain and safe; the IPM matches HiGHS on boxes with fixed binaries (it stalled there on
the rows that hold on the whole box); the lattice detection and its box-bound case; the search."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linprog

import jaxglobal
from jaxglobal.quadratic import (MAX_COLUMNS, _layout, _lp, _snap, binary_only, make_quadratic_bounder,
                                 polynomial_form, quadratic_form)  # fmt: skip

from test_lifted_poly import det3, lifted_point, poly_fn

jax.config.update("jax_enable_x64", True)


def binary_terms(seed, n=6, degree=4, n_terms=10, integral=True):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_terms):
        d = int(rng.integers(2, degree + 1))
        c = float(rng.integers(-3, 4)) if integral else float(rng.normal())
        out.append((c or 1.0, tuple(sorted(rng.choice(n, d, replace=False).tolist()))))
    return out


def epigraph(terms, n):
    """min -t s.t. t <= p(x) (x binary, t continuous): the hadamard / autocorr shape."""
    p = poly_fn(terms)

    def f(x):
        return -x[n]

    def g(x):
        return jnp.stack([x[n] - p(x)])

    bound = sum(abs(c) for c, _ in terms) + 1.0
    return f, g, np.r_[np.zeros(n), -bound], np.r_[np.ones(n), bound], p


def binary_points(n):
    return np.array(list(itertools.product([0.0, 1.0], repeat=n)))


@pytest.mark.parametrize("seed", range(5))
def test_cuts_hold_at_binary_points(seed):
    n = 6
    terms = binary_terms(seed, n)
    f, g, lo, hi, _ = epigraph(terms, n)
    Q = polynomial_form(f, g, None, n + 1, integer=np.arange(n))
    assert Q.cut_rhs.size and Q.pool_rhs.size and set(Q.binary.tolist()) <= set(range(n))
    for x in binary_points(n):
        v = lifted_point(Q, np.r_[x, 0.0])
        assert np.all(Q.cuts @ v <= Q.cut_rhs + 1e-12)
        lhs = np.sum(Q.pool_vals * v[Q.pool_cols], axis=1)
        assert np.all(lhs <= Q.pool_rhs + 1e-12)


def test_cuts_off_outside_binary_box():
    n = 6
    f, g, lo, hi, _ = epigraph(binary_terms(0, n), n)
    Q = polynomial_form(f, g, None, n + 1, integer=np.arange(n))
    L = _layout(Q, n + 1, 0, False)
    r0 = 4 * len(Q.pairs) + Q.n_g
    cut = slice(r0, r0 + Q.cut_rhs.size)
    b = np.asarray(_lp(Q, L, jnp.asarray(lo), jnp.asarray(hi), 0.0, False)[3])
    np.testing.assert_array_equal(b[cut], Q.cut_rhs)
    hi2 = hi.copy()
    hi2[Q.binary[0]] = 2.0  # an integer variable that is not binary on this box: the cuts do not hold
    b = np.asarray(_lp(Q, L, jnp.asarray(lo), jnp.asarray(hi2), 0.0, False)[3])
    assert np.all(np.isinf(b[cut]))


def fixings(n, rng, k):
    yield np.zeros(n), np.ones(n)
    for _ in range(k):
        lo, hi = np.zeros(n), np.ones(n)
        fix = rng.choice(n, int(rng.integers(1, n - 1)), replace=False)
        lo[fix] = hi[fix] = rng.integers(0, 2, fix.size)
        yield lo, hi


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("safe", [False, True])
def test_bound_below_enumerated_min(seed, safe):
    """With the cuts, a separation round and the lattice rounding, the bound on random fixings is at
    most the minimum over the box's binary points (epigraph form, both lattice shapes)."""
    n = 7
    terms = binary_terms(seed, n)
    f, g, lo0, hi0, p = epigraph(terms, n)
    Q = polynomial_form(f, g, None, n + 1, integer=np.arange(n))
    assert Q.lattice is not None and Q.lattice[2] == n
    pv = np.array([float(p(jnp.asarray(x))) for x in binary_points(n)])
    X = binary_points(n)
    tol = 1e-6
    bound = jax.jit(make_quadratic_bounder(Q, tol=tol, safe=safe, n_sep=2))
    rng = np.random.default_rng(seed)
    for lo, hi in fixings(n, rng, 10):
        inbox = np.all((X >= lo) & (X <= hi), axis=1)
        for thi in (hi0[n], float(np.max(pv[inbox])) - 0.5):  # also a box bound of t inside the range of p
            blo, bhi = np.r_[lo, lo0[n]], np.r_[hi, thi]
            fm = -float(np.max(np.minimum(pv[inbox], thi)))  # the best t is min(p(x), t_hi)
            lb = float(bound(jnp.asarray(blo), jnp.asarray(bhi), jnp.inf)[0])
            assert lb <= fm + 1e-9, (lo, hi, thi, lb, fm)
    # (a) the objective itself is the polynomial: min p(x) s.t. sum(x) >= 2
    Qa = polynomial_form(p, lambda x: jnp.stack([2.0 - sum(x[i] for i in range(n))]), None, n, integer=np.arange(n))
    assert Qa.lattice is not None and Qa.lattice[2] == -1
    bound = jax.jit(make_quadratic_bounder(Qa, tol=tol, safe=safe))
    for lo, hi in fixings(n, rng, 10):
        inbox = np.all((X >= lo) & (X <= hi), axis=1) & (X.sum(1) >= 2)
        if inbox.any():
            lb = float(bound(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
            assert lb <= float(pv[inbox].min()) + 1e-9


def highs(Q, L, lo, hi, tol):
    vlo, vhi, vals, b, _ = _lp(Q, L, jnp.asarray(lo), jnp.asarray(hi), tol, False)
    A = sparse.csr_matrix((np.asarray(vals), (L.pattern.rows, L.pattern.cols)), shape=(L.pattern.m, vlo.size))
    b = np.asarray(b)
    keep = np.isfinite(b)
    res = linprog(Q.E[0], A_ub=A[keep], b_ub=b[keep], bounds=list(zip(np.asarray(vlo), np.asarray(vhi))),
                  method="highs")  # fmt: skip
    return res.fun + Q.c[0]


@pytest.mark.parametrize("seed", range(3))
def test_ipm_matches_highs_on_fixed_binaries(seed):
    """Fixed binaries make many cut rows hold on the whole box (rhs 0, all columns fixed); left in,
    they leave the IPM no interior and its certificate stalled far below the LP value (hadamard_5
    at 10 fixings: -10813 against -2). Real coefficients (no lattice rounding), no separation."""
    n = 8
    terms = binary_terms(seed, n, n_terms=14, integral=False)
    f, g, lo0, hi0, _ = epigraph(terms, n)
    Q = polynomial_form(f, g, None, n + 1, integer=np.arange(n))
    assert Q.lattice is None
    L = _layout(Q, n + 1, 0, False)
    bound = jax.jit(make_quadratic_bounder(Q, tol=1e-6, n_sep=0))
    rng = np.random.default_rng(seed)
    for lo, hi in fixings(n, rng, 8):
        blo, bhi = np.r_[lo, lo0[n]], np.r_[hi, hi0[n]]
        hb = highs(Q, L, blo, bhi, 1e-6)
        lb = float(bound(jnp.asarray(blo), jnp.asarray(bhi), jnp.inf)[0])
        assert lb <= hb + 1e-7 * (1 + abs(hb))
        assert lb == pytest.approx(hb, abs=1e-4 * (1 + abs(hb)))


def test_lattice_detection():
    f = lambda x: -x[9]  # noqa: E731
    g = lambda x: jnp.stack([x[9] - det3(x)])  # noqa: E731
    Q = polynomial_form(f, g, None, 10, integer=np.arange(9))
    assert Q.lattice[:5] == (1.0, 0.0, 9, -1.0, 1.0)
    assert Q.lattice.vars.tolist() == list(range(9)) and Q.lattice.mag == 6.0
    assert polynomial_form(f, g, None, 10).lattice is None  # no integer variables given
    # t in a second row, or a continuous variable in the polynomial: no lattice
    g2 = lambda x: jnp.stack([x[9] - det3(x), x[9] - 3.0])  # noqa: E731
    assert polynomial_form(f, g2, None, 10, integer=np.arange(9)).lattice is None
    assert polynomial_form(f, g, None, 10, integer=np.arange(8)).lattice is None
    # (a): integer coefficients times products of integers, step = gcd, offset = constant
    fa = lambda x: 4.0 * x[0] * x[1] * x[2] - 6.0 * x[1] * x[2] * x[3] + 1.5  # noqa: E731
    Qa = polynomial_form(fa, lambda x: jnp.stack([x[0] - 1.0]), None, 4, integer=np.arange(4))
    assert Qa.lattice[:3] == (2.0, 1.5, -1)


def test_snap():
    f = lambda x: -x[9]  # noqa: E731
    g = lambda x: jnp.stack([x[9] - det3(x)])  # noqa: E731
    Q = polynomial_form(f, g, None, 10, integer=np.arange(9))
    lo, hi = jnp.asarray(np.r_[np.zeros(9), -6.0]), jnp.asarray(np.r_[np.ones(9), 6.0])
    assert float(_snap(Q, jnp.asarray(-2.3), lo, hi)) == pytest.approx(-2.0, abs=1e-12)
    assert float(_snap(Q, jnp.asarray(-2.3), lo, hi)) <= -2.0
    assert float(_snap(Q, jnp.asarray(-2.0), lo, hi)) == pytest.approx(-2.0, abs=1e-12)
    assert float(_snap(Q, -jnp.inf, lo, hi)) == -np.inf
    # t <= 0.5: f >= -0.5 is the box bound, not a lattice point; the snap stops there
    hi2 = hi.at[9].set(0.5)
    assert float(_snap(Q, jnp.asarray(-0.7), lo, hi2)) <= -0.5
    assert float(_snap(Q, jnp.asarray(-1.7), lo, hi2)) == pytest.approx(-1.0, abs=1e-12)
    # an integer variable of the polynomial with hi = 2: monomials are not 0/1, no rounding
    assert float(_snap(Q, jnp.asarray(-2.3), lo, hi.at[4].set(2.0))) == -2.3
    # the rows hold to tol: t <= det + tol, so f >= -det - tol; the lattice moves down by tol
    assert float(_snap(Q, jnp.asarray(-2.3), lo, hi, tol=1e-6)) <= -2.0 - 1e-6


def test_binary_only_and_column_cap():
    f = lambda x: -x[9]  # noqa: E731
    g = lambda x: jnp.stack([x[9] - det3(x)])  # noqa: E731
    assert binary_only(polynomial_form(f, g, None, 10, integer=np.arange(9)), 10)
    assert not binary_only(polynomial_form(f, g, None, 10), 10)
    g2 = lambda x: jnp.stack([x[9] - det3(x) + x[10]])  # noqa: E731  (a continuous x10)
    assert not binary_only(polynomial_form(f, g2, None, 11, integer=np.arange(9)), 11)
    # degree 3 with more than MAX_COLUMNS columns: kept by polynomial_form, not at degree 2
    n = 16
    tri = [(1.0, t) for t in itertools.combinations(range(n), 3)]
    pair = [(1.0, t) for t in itertools.combinations(range(n + 14), 2)]
    one = lambda x: jnp.stack([x[0] - 1.0])  # noqa: E731
    Q3 = polynomial_form(poly_fn(tri), one, None, n)
    assert Q3 is not None and n + len(Q3.pairs) > MAX_COLUMNS
    assert polynomial_form(poly_fn(pair), one, None, n + 14) is None  # 30 + 435 columns, degree 2
    assert quadratic_form(poly_fn(pair), one, None, n + 14) is None
    # an explicit max_columns still caps a degree-3 problem
    assert polynomial_form(poly_fn(tri), one, None, n, max_columns=MAX_COLUMNS) is not None
    assert polynomial_form(poly_fn(tri), one, None, n, max_columns=100) is None


@pytest.mark.parametrize("safe", [False, True])
def test_minimize_binary_polynomial(safe):
    """The search on an epigraph binary polynomial (FBBT skipped: binary_only) proves the enumerated
    optimum."""
    n = 8
    terms = binary_terms(11, n, n_terms=16)
    f, g, lo, hi, p = epigraph(terms, n)
    best = max(float(p(jnp.asarray(x))) for x in binary_points(n))
    r = jaxglobal.minimize(f, lo, hi, ineq=g, integer=np.arange(n), rel_tol=1e-4, abs_tol=1e-6, safe=safe,
                           time_limit=120)  # fmt: skip
    assert r.status == "optimal" and r.verified
    assert r.fun == pytest.approx(-best, abs=1e-5)
    assert r.lower_bound <= -best + 1e-6

"""Big-M coefficient tightening of the lifted quadratic LP (quadratic.with_bigm / _bigm): the
tightened rows hold at every sampled point of random boxes with the binaries integral, the
LP bound stays below the box minimum, and the tightening raises the bound where it applies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import minimize
from jaxglobal.quadratic import _bigm, make_quadratic_bounder, quadratic_form, with_bigm

jax.config.update("jax_enable_x64", True)


def layout_problem():
    """Two segments [x0, x0 + 1] and [x1, x1 + 2] on [0, 10] that must not overlap, the order chosen
    by y2 (y2 = 1: x0 left of x1) / y3, by big-M rows; x0 must lie in a disk of radius 1 around
    (2, 0) (with x4) when y5 = 1, else around (8, 0) (y6 = 1). Minimize |x0 - x1| (x7 >= both
    differences) plus 0.1 x4^2."""
    M = 12.0

    def f(x):
        return x[7] + 0.1 * x[4] * x[4]

    def g(x):
        return jnp.stack([
            x[0] + 1.0 - x[1] + M * x[2] - M,  # y2 = 1: x0 + 1 <= x1
            x[1] + 2.0 - x[0] + M * x[3] - M,  # y3 = 1: x1 + 2 <= x0
            (x[0] - 2.0) ** 2 + x[4] ** 2 - 1.0 + 100.0 * x[5] - 100.0,
            (x[0] - 8.0) ** 2 + x[4] ** 2 - 1.0 + 100.0 * x[6] - 100.0,
            x[0] - x[1] - x[7],
            x[1] - x[0] - x[7],
        ])

    def h(x):
        return jnp.stack([x[2] + x[3] - 1.0, x[5] + x[6] - 1.0])

    lo = np.array([0, 0, 0, 0, -3, 0, 0, 0], float)
    hi = np.array([10, 10, 1, 1, 3, 1, 1, 20], float)
    return f, g, h, lo, hi, [2, 3, 5, 6]


def lifted(Q, x):
    return np.concatenate([x, x[Q.pairs[:, 0]] * x[Q.pairs[:, 1]]])


def column_box(Q, lo, hi):
    n = lo.size
    vlo, vhi = [lo], [hi]
    for a, b in Q.pairs:
        c = [lo[a] * lo[b], lo[a] * hi[b], hi[a] * lo[b], hi[a] * hi[b]]
        wl = 0.0 if a == b and lo[a] < 0 < hi[a] else min(c)
        vlo.append([wl])
        vhi.append([max(c)])
    return np.concatenate(vlo)[: n + len(Q.pairs)], np.concatenate(vhi)[: n + len(Q.pairs)]


def test_marks_big_m_rows():
    f, g, h, lo, hi, ints = layout_problem()
    Q = with_bigm(quadratic_form(f, g, h, lo.size), lo.size, ints)
    assert sorted(Q.bm_row.tolist()) == [0, 1, 2, 3]
    assert sorted(Q.bm_col.tolist()) == [2, 3, 5, 6]
    assert Q.bm_qvar.shape[1] >= 2  # the disks' squares are kept with their linear terms
    assert with_bigm(quadratic_form(f, g, h, lo.size), lo.size, []).bm_row.size == 0


@pytest.mark.parametrize("seed", range(4))
def test_tightened_rows_hold(seed):
    """At sampled points of random sub-boxes with binary values in {0, 1}: every point that
    satisfies an original row satisfies its tightened row."""
    f, g, h, lo, hi, ints = layout_problem()
    n = lo.size
    Q = with_bigm(quadratic_form(f, g, h, n), n, ints)
    G = Q.E[1:1 + Q.n_g]
    rng = np.random.default_rng(seed)
    tightened = 0
    for _ in range(30):
        a, b = rng.uniform(lo, hi), rng.uniform(lo, hi)
        blo, bhi = np.minimum(a, b), np.maximum(a, b)
        blo[ints], bhi[ints] = 0.0, 1.0
        vlo, vhi = column_box(Q, blo, bhi)
        bg = -Q.c[1:1 + Q.n_g]
        m_new, r_new = (np.asarray(t) for t in _bigm(Q, jnp.asarray(vlo), jnp.asarray(vhi), jnp.asarray(bg)))
        tightened += int(np.sum(np.abs(m_new - Q.bm_coef) > 1e-9))
        for _ in range(200):
            x = rng.uniform(blo, bhi)
            x[ints] = rng.integers(0, 2, len(ints))
            v = lifted(Q, x)
            for k, r in enumerate(Q.bm_row):
                act = G[r] @ v
                if act <= bg[r]:  # the original row holds
                    row = G[r].copy()
                    row[Q.bm_col[k]] = m_new[k]
                    assert row @ v <= r_new[k] + 1e-9 * (1 + abs(r_new[k]))
    assert tightened > 0


@pytest.mark.parametrize("safe", [False, True])
def test_bound_valid_and_tighter(safe):
    """On sub-boxes, the bound with tightening stays below the minimum over the box (by enumerating
    the binaries and a fine grid of x0, x1) and is at least the plain bound."""
    f, g, h, lo, hi, ints = layout_problem()
    n = lo.size
    Q0 = quadratic_form(f, g, h, n)
    Q1 = with_bigm(Q0, n, ints)
    b0 = jax.jit(make_quadratic_bounder(Q0, tol=1e-6, safe=safe))
    b1 = jax.jit(make_quadratic_bounder(Q1, tol=1e-6, safe=safe))
    for x0r, x1r in [((0, 10), (0, 10)), ((1, 3), (0, 10)), ((1, 3), (2, 6)), ((7, 9), (3, 9)), ((1.5, 2.5), (0, 4))]:
        blo, bhi = lo.copy(), hi.copy()
        blo[0], bhi[0] = x0r
        blo[1], bhi[1] = x1r
        # the box minimum (x4 = 0 is optimal): binaries enumerated, x0 and x1 on a grid
        X0, X1 = np.meshgrid(np.linspace(*x0r, 401), np.linspace(*x1r, 401))
        d = np.abs(X0 - X1)
        disk = ((np.abs(X0 - 2.0) <= 1.0) | (np.abs(X0 - 8.0) <= 1.0))
        order = (X0 + 1.0 <= X1 + 1e-12) | (X1 + 2.0 <= X0 + 1e-12)
        ok = disk & order
        best = float(d[ok].min()) if ok.any() else np.inf
        l0, l1 = float(b0(blo, bhi, np.inf)[0]), float(b1(blo, bhi, np.inf)[0])
        assert l1 <= best + 1e-6
        assert l1 >= l0 - 1e-6



@pytest.mark.parametrize("safe", [False, True])
def test_tightening_raises_bound(safe):
    """min 3 y - x, x <= 10 y (y binary) on x in [0, 2]: the LP over the big-M row gives -1.4
    (y = 0.2, x = 2); tightened to x <= 2 y it gives the box minimum 0."""
    f = lambda x: 3.0 * x[1] - x[0]  # noqa: E731
    g = lambda x: jnp.stack([x[0] - 10.0 * x[1], x[0] * x[0] - 100.0])  # noqa: E731  (a product: the lifted LP)
    lo, hi = np.array([0.0, 0.0]), np.array([2.0, 1.0])
    Q0 = quadratic_form(f, g, None, 2)
    Q1 = with_bigm(Q0, 2, [1])
    l0 = float(jax.jit(make_quadratic_bounder(Q0, tol=1e-6, safe=safe))(lo, hi, np.inf)[0])
    l1 = float(jax.jit(make_quadratic_bounder(Q1, tol=1e-6, safe=safe))(lo, hi, np.inf)[0])
    assert abs(l0 + 1.4) < 1e-3
    assert -1e-3 < l1 <= 1e-6


def test_minimize_same_optimum():
    f, g, h, lo, hi, ints = layout_problem()
    r0 = minimize(f, lo, hi, ineq=g, eq=h, integer=ints, bigm=False, time_limit=60)
    r1 = minimize(f, lo, hi, ineq=g, eq=h, integer=ints, bigm=True, time_limit=60)
    assert r0.status == r1.status == "optimal"
    assert abs(r0.fun - r1.fun) <= 1e-4 * (1 + abs(r0.fun))
    assert r1.lower_bound <= r1.fun + 1e-9

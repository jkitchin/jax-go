"""Block elimination of the lifted LP's normal equations (quadratic._blocks, _factor) and the
spread Kelley tangents (KELLEY_SPREAD): same solves, same bounds up to IPM rounding, valid bounds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal.quadratic as qd
from jaxglobal.quadratic import _blocks, _factor, _layout, _pattern, make_quadratic_bounder, quadratic_form

jax.config.update("jax_enable_x64", True)


def _arrow_spd(rng, sizes, ns):
    """A random SPD matrix whose columns split into blocks of ``sizes`` coupled only through
    ``ns`` separator columns, randomly permuted; returns it and its row pattern."""
    N = sum(sizes) + ns
    perm = rng.permutation(N)
    rows, cols, r = [], [], 0
    start = 0
    sep = perm[N - ns:]
    for s in sizes:
        blk = perm[start:start + s]
        start += s
        for _ in range(2 * s):  # rows over a few block columns and a few separator columns
            c = np.concatenate([rng.choice(blk, min(s, 3), replace=False), rng.choice(sep, min(ns, 2), replace=False)])
            rows.append(np.full(c.size, r))
            cols.append(c)
            r += 1
    pat = _pattern(rows, cols, r)
    A = np.zeros((r, N))
    A[pat.rows, pat.cols] = rng.normal(size=pat.rows.size)
    M = A.T @ np.diag(rng.uniform(0.1, 10.0, r)) @ A + np.diag(rng.uniform(1e-3, 1.0, N))
    return M, pat


@pytest.mark.parametrize("sizes,ns", [([5, 7, 3, 6, 6, 4, 8, 5, 5, 7, 6, 4], 6), ([4] * 20, 0), ([9] * 9, 3)])
def test_factor_matches_dense_solve(monkeypatch, sizes, ns):
    monkeypatch.setattr(qd, "BLOCK_MIN_N", 0)
    monkeypatch.setattr(qd, "BLOCK_GAIN", 1.0)
    rng = np.random.default_rng(len(sizes) + ns)
    M, pat = _arrow_spd(rng, sizes, ns)
    N = M.shape[0]
    B = _blocks(pat, N)
    assert B is not None
    # every row's columns lie in one block plus the separator; each column is in exactly one place
    where = np.full(N, -1)
    for k, (b, v) in enumerate(zip(B.blk, B.valid)):
        where[b[v]] = k
    assert np.all(where[B.sep] == -1)
    where[B.sep] = -2
    assert np.all(where != -1)
    for r in range(pat.m):
        ks = set(where[pat.cols[pat.rows == r]]) - {-2}
        assert len(ks) <= 1
    r = rng.normal(size=N)
    z = np.asarray(_factor(jnp.asarray(M), B)(jnp.asarray(r)))
    np.testing.assert_allclose(M @ z, r, rtol=1e-8, atol=1e-8 * np.abs(r).max())
    z0 = np.asarray(_factor(jnp.asarray(M), None)(jnp.asarray(r)))
    np.testing.assert_allclose(z, z0, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("sizes,ns", [([5, 7, 3, 6, 6, 4, 8, 5, 5, 7, 6, 4], 6), ([4] * 20, 0)])
def test_assemble_matches_dense_normal_matrix(monkeypatch, sizes, ns):
    """The normal matrix packed in place (``_assemble``) solves as the dense one, Jacobi-scaled."""
    monkeypatch.setattr(qd, "BLOCK_MIN_N", 0)
    monkeypatch.setattr(qd, "BLOCK_GAIN", 1.0)
    rng = np.random.default_rng(5 + ns)
    _, pat = _arrow_spd(rng, sizes, ns)
    N = int(pat.cols.max()) + 1
    B = _blocks(pat, N)
    assert B is not None and B.dest is not None
    vals = rng.normal(size=pat.rows.size)
    d = rng.uniform(1e-3, 1e3, pat.m)  # y / s
    w = vals[pat.pa] * vals[pat.pb] * d[pat.rows[pat.pa]]
    dg = rng.uniform(1e-6, 1.0, N)
    M = np.zeros((N, N))
    np.add.at(M, (pat.cols[pat.pa], pat.cols[pat.pb]), w)
    M += np.diag(dg)
    dm, solve = qd._assemble(B, jnp.asarray(w), jnp.asarray(dg), N)
    np.testing.assert_allclose(np.asarray(dm), np.sqrt(np.diag(M)), rtol=1e-12)
    r = rng.normal(size=N)
    Ms = M / np.outer(np.asarray(dm), np.asarray(dm))
    np.testing.assert_allclose(Ms @ np.asarray(solve(jnp.asarray(r))), r, rtol=1e-7, atol=1e-7 * np.abs(r).max())


def test_factor_breakdown_is_nan(monkeypatch):
    """An indefinite matrix gives NaN (the IPM stops on a nonfinite step), as the dense Cholesky."""
    monkeypatch.setattr(qd, "BLOCK_MIN_N", 0)
    monkeypatch.setattr(qd, "BLOCK_GAIN", 1.0)
    rng = np.random.default_rng(3)
    M, pat = _arrow_spd(rng, [4] * 10, 2)
    M[0, 0] = -100.0
    z = np.asarray(_factor(jnp.asarray(M), _blocks(pat, M.shape[0]))(jnp.ones(M.shape[0])))
    assert not np.all(np.isfinite(z))


def test_small_or_dense_problems_keep_the_dense_cholesky():
    rng = np.random.default_rng(0)
    M, pat = _arrow_spd(rng, [4] * 10, 2)  # 42 columns < BLOCK_MIN_N
    assert _blocks(pat, M.shape[0]) is None
    rows = [np.full(80, r) for r in range(5)]  # every row holds every column: one block
    cols = [np.arange(80) for _ in range(5)]
    assert _blocks(_pattern(rows, cols, 5), 80) is None


P = np.array([[0.1, 0.2], [0.9, 0.8], [0.15, 0.3], [0.8, 0.95], [0.5, 0.1]])
K = 2


def _kmeans_hull():
    """A small k-means clustering model in the hull form of minlp.org kClus_*_HRc: points P,
    K centers c_k in [0, 1]^2; per (point j, cluster k) a binary i, the disaggregated center
    x (= i c), its complement y (= (1 - i) c), z >= |x|^2 / i (as i z >= |x|^2) and the distance
    d >= z - 2 p.x + |p|^2 i; each point in one cluster; minimize sum d."""
    J = len(P)
    nc = 2 * K
    per = 7  # i, x0, x1, y0, y1, z, d

    def v(x, j, k):  # (scalar entries only: the lifted form reads a scalarized tape)
        b = nc + (j * K + k) * per
        return x[b], (x[b + 1], x[b + 2]), (x[b + 3], x[b + 4]), x[b + 5], x[b + 6]

    def f(x):
        return sum(v(x, j, k)[4] for j in range(J) for k in range(K))

    def ineq(x):
        g = []
        for j in range(J):
            for k in range(K):
                i, xx, yy, z, d = v(x, j, k)
                px = float(P[j, 0]) * xx[0] + float(P[j, 1]) * xx[1]
                g += [xx[0] ** 2 + xx[1] ** 2 - i * z, z - 2 * px + float(P[j] @ P[j]) * i - d]
                g += [xx[0] - i, xx[1] - i, yy[0] - (1 - i), yy[1] - (1 - i)]
        return jnp.stack(g)

    def eq(x):
        h = []
        for j in range(J):
            h.append(sum(v(x, j, k)[0] for k in range(K)) - 1.0)
            for k in range(K):
                i, xx, yy, z, d = v(x, j, k)
                h += [x[2 * k] - xx[0] - yy[0], x[2 * k + 1] - xx[1] - yy[1]]
        return jnp.stack(h)

    n = nc + J * K * per
    lo, hi = np.zeros(n), np.ones(n)
    for j in range(J):
        for k in range(K):
            b = nc + (j * K + k) * per
            hi[b + 5], hi[b + 6] = 2.0, 2.0
    ints = np.array([nc + (j * K + k) * per for j in range(J) for k in range(K)])
    return f, ineq, eq, lo, hi, ints, v


def _feasible_point(rng, lo, v, assign):
    """A feasible point of ``_kmeans_hull`` for the assignment ``assign`` (cluster of each point)."""
    x = np.zeros(lo.size)
    c = rng.uniform(0.0, 1.0, (K, 2))
    x[:2 * K] = c.reshape(-1)
    nc, per = 2 * K, 7
    for j in range(len(P)):
        for k in range(K):
            b = nc + (j * K + k) * per
            i = float(assign[j] == k)
            xx, yy = i * c[k], (1 - i) * c[k]
            z = xx @ xx
            x[b:b + 7] = [i, *xx, *yy, z, z - 2 * P[j] @ xx + (P[j] @ P[j]) * i]
    return x


@pytest.mark.parametrize("spread", [(), (-0.05, 0.05)])
@pytest.mark.parametrize("safe", [False, True])
def test_hull_kmeans_bounds_block_vs_dense(monkeypatch, spread, safe):
    """On the hull k-means model the block solve gives the dense solve's bounds (to IPM rounding)
    on random boxes, and no bound is above a feasible point of its box (with spread tangents too)."""
    monkeypatch.setattr(qd, "BLOCK_MIN_N", 0)
    monkeypatch.setattr(qd, "KELLEY_SPREAD", spread)
    f, ineq, eq, lo0, hi0, ints, v = _kmeans_hull()
    n = lo0.size
    Q = quadratic_form(f, ineq, eq, n)
    assert Q is not None
    assert _layout(Q, n, 2, False).blocks is not None
    rng = np.random.default_rng(11)
    LO, HI, pts = [], [], []
    for b in range(12):
        lo, hi = lo0.copy(), hi0.copy()
        assign = rng.integers(0, K, len(P))
        fixed = rng.choice(len(P), b % len(P), replace=False)
        for j in fixed:
            for k in range(K):
                lo[ints[j * K + k]] = hi[ints[j * K + k]] = float(assign[j] == k)
        LO.append(lo)
        HI.append(hi)
        pts.append([_feasible_point(rng, lo0, v, assign) for _ in range(20)])
    LO, HI = jnp.asarray(np.array(LO)), jnp.asarray(np.array(HI))
    out = {}
    for block in (False, True):
        monkeypatch.setattr(qd, "BLOCK_SOLVE", block)
        for nk in (0, 2):
            fn = jax.jit(jax.vmap(make_quadratic_bounder(Q, n_kelley=nk, tol=1e-6, safe=safe), in_axes=(0, 0, None)))
            out[block, nk] = np.asarray(fn(LO, HI, jnp.inf)[0])
    # one LP: the same value. With Kelley rounds the tangents are taken at the IPM's point inside
    # the optimal face, which the two solves can leave at different places (bounds 0.0122 and
    # 0.0075 on one box here, both valid; with 6 rounds they agree to 1e-4)
    np.testing.assert_allclose(out[True, 0], out[False, 0], rtol=1e-5, atol=1e-5)  # (IPM stopping rules)
    for key, lbs in out.items():
        for lb, X in zip(lbs, pts):
            F = [float(f(jnp.asarray(x))) for x in X]
            assert lb <= min(F) + 1e-9, key
    assert np.all(out[True, 2] >= out[True, 0] - 1e-5)
    assert np.max(out[True, 2]) > 0.01  # the fixed boxes have a positive bound

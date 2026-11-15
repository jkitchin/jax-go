"""Host dual simplex for the lifted node LP (``minimize(node_lp="simplex")``, ``simplex.py``).

Every bound must stay a weak-duality certificate: whatever the simplex returns (optimal duals, a
dual ray, a failure, a basis from another box), no feasible point may lie below it.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import minimize, simplex
from jaxglobal.quadratic import make_quadratic_bounder, quadratic_form

from test_constrained import haverly1
from test_quadratic import _qcqp, highs_bound

jax.config.update("jax_enable_x64", True)

SOLVERS = [s for s in ("highs", "soplex") if simplex.available(s)]
pytestmark = pytest.mark.skipif(not SOLVERS, reason="neither highspy nor pyscipopt is installed")


def _samples(f, g, lo0, hi0, n=20000, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(lo0, hi0, (n, lo0.size))
    G = np.asarray(jax.vmap(g)(jnp.asarray(X)))
    X = X[np.all(G <= 0.0, 1)]
    return X, np.asarray(jax.vmap(f)(jnp.asarray(X)))


def _boxes(lo0, hi0, k, seed=3):
    rng = np.random.default_rng(seed)
    c, w = rng.uniform(lo0, hi0, (k, lo0.size)), rng.uniform(0.2, 1.0, (k, lo0.size)) * (hi0 - lo0) / 2
    return np.clip(c - w, lo0, hi0), np.clip(c + w, lo0, hi0)


def _bounders(Q, safe, n_kelley, store):
    kw = dict(n_kelley=n_kelley, tol=1e-6, safe=safe)
    ipm = jax.jit(jax.vmap(make_quadratic_bounder(Q, **kw), in_axes=(0, 0, None)))
    spx = jax.jit(jax.vmap(make_quadratic_bounder(Q, node_lp="simplex", lp_store=store, **kw), in_axes=(0, 0, None)))
    return ipm, spx


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("safe", [False, True])
def test_simplex_bounds_are_certified(monkeypatch, solver, safe):
    """Nonconvex QCQP, random boxes: the simplex bound (with Kelley rounds) has no sampled
    feasible point below it, and a single LP reaches the LP's value (HiGHS through scipy), which
    the IPM's certificate only approaches from below."""
    monkeypatch.setattr(simplex, "SOLVER", solver)
    f, g, lo0, hi0 = _qcqp()
    Q = quadratic_form(f, g, None, 4)
    X, F = _samples(f, g, lo0, hi0)
    st = simplex.Store()
    ipm4, spx4 = _bounders(Q, safe, 4, st)
    ipm1, spx1 = _bounders(Q, safe, 0, st)
    lo, hi = _boxes(lo0, hi0, 16)
    l4 = np.asarray(spx4(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
    l1 = np.asarray(spx1(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
    i1 = np.asarray(ipm1(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])
    checked = 0
    for j in range(len(lo)):
        inside = np.all((X >= lo[j]) & (X <= hi[j]), 1)
        if inside.any():
            assert l4[j] <= F[inside].min() + 1e-9
            assert l1[j] <= F[inside].min() + 1e-9
            ref = highs_bound(Q, lo[j], hi[j], 1e-6)
            assert l1[j] <= ref + 1e-9 * (1 + abs(ref))
            assert l1[j] == pytest.approx(ref, abs=1e-6 * (1 + abs(ref)))
            assert l1[j] >= i1[j] - 1e-6 * (1 + abs(ref))
            checked += 1
    assert checked >= 6
    assert st.stats[simplex.OPTIMAL] > 0 and st.stats[simplex.FAILED] == 0


def _lp_infeasible(Q, lo, hi, tol):
    """Whether the lifted LP of the box (no tangent rounds) is infeasible, by HiGHS through scipy."""
    from scipy import sparse
    from scipy.optimize import linprog

    from jaxglobal.quadratic import _layout, _lp

    L = _layout(Q, lo.size, 0, False)
    vlo, vhi, vals, b, _ = _lp(Q, L, jnp.asarray(lo), jnp.asarray(hi), tol, False)
    A = sparse.csr_matrix((np.asarray(vals), (L.pattern.rows, L.pattern.cols)), shape=(L.pattern.m, vlo.size))
    res = linprog(Q.E[0], A_ub=A, b_ub=np.asarray(b), bounds=list(zip(np.asarray(vlo), np.asarray(vhi))), method="highs")
    return res.status == 2


@pytest.mark.parametrize("solver", SOLVERS)
def test_infeasible_boxes_prune_by_dual_ray(monkeypatch, solver):
    """Haverly's pooling problem on boxes away from its feasible set: infeasible LPs are pruned
    (bound above the cutoff) by the certificate of the simplex's dual ray, and the root box,
    which holds the optimum, is not."""
    monkeypatch.setattr(simplex, "SOLVER", solver)
    f, g, h, lo0, hi0 = haverly1()
    Q = quadratic_form(f, g, h, 8)
    st = simplex.Store()
    _, spx = _bounders(Q, True, 4, st)
    lo, hi = _boxes(lo0, hi0, 16, seed=5)
    lo, hi = np.concatenate([lo0[None], lo]), np.concatenate([hi0[None], hi])
    cutoff = -400.0 + 1e-6
    lb = np.asarray(spx(jnp.asarray(lo), jnp.asarray(hi), cutoff)[0])
    assert lb[0] <= cutoff
    bad = np.array([_lp_infeasible(Q, lo[j], hi[j], 1e-6) for j in range(len(lo))])
    assert bad.sum() >= 3 and not bad[0]
    assert np.all(lb[bad] >= cutoff)
    assert st.stats[simplex.INFEASIBLE] > 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_failed_solves_fall_back_to_the_ipm(monkeypatch, solver):
    """Every solve failing (or returning a useless ray) gives the IPM path's bounds."""
    monkeypatch.setattr(simplex, "SOLVER", solver)
    f, g, lo0, hi0 = _qcqp()
    Q = quadratic_form(f, g, None, 4)
    lo, hi = _boxes(lo0, hi0, 8)
    ref = np.asarray(jax.jit(jax.vmap(make_quadratic_bounder(Q, tol=1e-6, safe=True), in_axes=(0, 0, None)))(
        jnp.asarray(lo), jnp.asarray(hi), jnp.inf)[0])  # fmt: skip
    cls = simplex._HighsLP if solver == "highs" else simplex._SoplexLP
    for fake in (lambda self: (simplex.FAILED, None, None, 0),
                 lambda self: (simplex.INFEASIBLE, None, np.zeros(10_000), 0)):  # fmt: skip
        monkeypatch.setattr(cls, "solve", fake)
        st = simplex.Store()
        b = jax.jit(jax.vmap(make_quadratic_bounder(Q, tol=1e-6, safe=True, node_lp="simplex", lp_store=st),
                             in_axes=(0, 0, None)))(jnp.asarray(lo), jnp.asarray(hi), jnp.inf)  # fmt: skip
        np.testing.assert_allclose(np.asarray(b[0]), ref, rtol=0, atol=1e-12)
        assert st.stats[simplex.OPTIMAL] == 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_any_start_basis_keeps_bounds_valid(monkeypatch, solver):
    """Bases from other boxes, random ones and malformed ones (``Store.give``) change where the
    simplex starts, not the bound: the single-LP bound still equals the LP's value."""
    monkeypatch.setattr(simplex, "SOLVER", solver)
    f, g, lo0, hi0 = _qcqp()
    Q = quadratic_form(f, g, None, 4)
    st = simplex.Store()
    _, spx = _bounders(Q, False, 0, st)
    lo, hi = _boxes(lo0, hi0, 8)
    L, H = jnp.asarray(lo), jnp.asarray(hi)
    base = np.asarray(spx(L, H, jnp.inf)[0])
    own = st.take(lo, hi)
    assert all(b is not None for b in own)
    rng = np.random.default_rng(0)
    nb = own[0].size
    starts = [np.roll(own, 1), np.array([rng.integers(0, 3, nb).astype(np.int8) for _ in lo] + [None], dtype=object)[:-1],
              np.array([np.zeros(3, np.int8)] * len(lo) + [None], dtype=object)[:-1]]  # fmt: skip
    for s in starts:
        st.give(lo, hi, s)
        got = np.asarray(spx(L, H, jnp.inf)[0])
        st.take(lo, hi)
        for j in range(len(lo)):
            if np.isfinite(base[j]):
                ref = highs_bound(Q, lo[j], hi[j], 1e-6)
                assert got[j] <= ref + 1e-9 * (1 + abs(ref))
                assert got[j] == pytest.approx(base[j], abs=1e-6 * (1 + abs(ref)))
    assert st.stats[simplex.FAILED] == 0


def _mixed():
    f = lambda x: (x[0] - 1.3) ** 2 + x[0] * x[1] - 2.0 * x[1] * x[2] + x[2] ** 2 - 0.4 * x[3] ** 2  # noqa: E731
    g = lambda x: jnp.stack([x[0] ** 2 + x[1] ** 2 - 9.0, x[1] * x[2] - 2.5, x[3] ** 2 + x[2] - 3.0])  # noqa: E731
    return f, g, np.full(4, -3.0), np.full(4, 3.0)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("safe", [False, True])
def test_integer_search_matches_enumeration(monkeypatch, solver, safe):
    """A nonconvex mixed-integer QCQP: the search with the simplex (children starting from their
    parents' bases) finds the optimum of enumerating the integers, its lower bound is below it,
    and it agrees with the IPM search."""
    monkeypatch.setattr(simplex, "SOLVER", solver)
    f, g, lo, hi = _mixed()
    kw = dict(ineq=g, integer=[0, 1], abs_tol=1e-7, rel_tol=1e-7, safe=safe, max_iter=2000)
    ref = np.inf
    for a, b in itertools.product(range(-3, 4), repeat=2):  # integers fixed: continuous global solves
        L, H = lo.copy(), hi.copy()
        L[:2] = H[:2] = (a, b)
        r = minimize(f, L, H, **kw)
        ref = min(ref, r.fun)
    rs = minimize(f, lo, hi, node_lp="simplex", **kw)
    ri = minimize(f, lo, hi, **kw)
    assert rs.status == "optimal" and ri.status == "optimal"
    assert rs.fun == pytest.approx(ref, abs=1e-5)
    assert rs.lower_bound <= ref + 1e-6
    assert rs.fun == pytest.approx(ri.fun, abs=1e-5)
    assert rs.simplex is not None and rs.simplex["optimal"] > 0 and rs.simplex["warm"] > 0
    assert ri.simplex is None


def test_option_errors(monkeypatch):
    f, g, lo, hi = _qcqp()
    with pytest.raises(ValueError, match="node_lp"):
        minimize(f, lo, hi, ineq=g, node_lp="dual", max_iter=1)
    monkeypatch.setattr(simplex, "available", lambda solver=None: None)
    with pytest.raises(ImportError, match="highspy"):
        minimize(f, lo, hi, ineq=g, node_lp="simplex", max_iter=1)

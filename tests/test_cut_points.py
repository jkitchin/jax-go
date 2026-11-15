"""Cut generation of the reduced-space bounder and OBBT (``lp._point_cuts``): cuts only at the
center and the axis points of variables some part is nonlinear in, kept as the flat list of LP
rows, and points mapped in chunks above ``CUT_BYTES``. None of it may change a bound."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import lp


def f(x):  # x3 enters every function linearly
    return x[0] * x[1] + jnp.exp(0.3 * x[0]) + x[2] - 0.5 * x[3]


def ineq(x):
    return jnp.stack([x[0] ** 2 + x[1] ** 2 - 2.0 + 0.1 * x[2], x[2] + x[3] - 1.5, -x[0] * x[1] - x[3]])


def eq(x):
    return jnp.atleast_1d(x[1] * x[2] - 0.2 + 0.0 * x[3] + 0.1 * x[0])


def boxes(k, seed):
    rng = np.random.default_rng(seed)
    A = np.sort(rng.uniform(-2.0, 2.0, (2, k, 4)), 0)
    return jnp.asarray(A[0]), jnp.asarray(A[1])


def run_bound(cutoff=jnp.inf):
    cb = lp.make_constrained_bounder(f, ineq, eq, n_kelley=3, tol=1e-6, univariate=True, alphabb=True)
    L, H = boxes(12, 0)
    return [np.asarray(a) for a in jax.jit(jax.vmap(lambda a, b: cb(a, b, cutoff)))(L, H)]


def run_obbt():
    ob = lp.make_obbt(f, ineq, eq, rounds=2, tol=1e-6, univariate=True, alphabb=True)
    L, H = boxes(6, 1)
    return [np.asarray(a) for a in jax.jit(jax.vmap(lambda a, b: ob(a, b, jnp.asarray(3.0))))(L, H)]


def test_axis_points_skip_linear_variables():
    nl = lp._nonlinear_vars(f, ineq, eq, 4)
    assert nl is not None and not nl[:, 3].any() and nl[:, :3].any(axis=0).all()
    keep = lp._axis_points(nl, 4)
    nlv = np.flatnonzero(nl.any(axis=0))
    assert keep.tolist() == [0, *(1 + nlv).tolist(), *(1 + 4 + nlv).tolist()]


@pytest.mark.parametrize("cutoff", [jnp.inf, 0.5])
def test_bounds_unchanged_by_point_selection_and_chunking(monkeypatch, cutoff):
    ref_keep = lp._axis_points
    monkeypatch.setattr(lp, "_axis_points", lambda nl, n: None)  # every point (the former cut set)
    full = run_bound(cutoff)
    monkeypatch.setattr(lp, "_axis_points", ref_keep)
    sel = run_bound(cutoff)
    monkeypatch.setattr(lp, "CUT_BYTES", 1.0)  # one point per chunk
    chunked = run_bound(cutoff)
    for a, b in zip(full, sel):  # the same LPs, row for row
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    # chunks (lax.map) round differently from one vmap: the bound and the reduced box agree to
    # rounding; the LP point of a degenerate LP may move (not compared)
    for i in (0, 2, 3, 4):
        np.testing.assert_allclose(full[i], chunked[i], rtol=1e-7, atol=1e-7)
    assert np.all(np.isfinite(full[0]) | (full[0] == np.inf))


def test_obbt_unchanged_by_point_selection_and_chunking(monkeypatch):
    ref_keep = lp._axis_points
    monkeypatch.setattr(lp, "_axis_points", lambda nl, n: None)
    full = run_obbt()
    monkeypatch.setattr(lp, "_axis_points", ref_keep)
    monkeypatch.setattr(lp, "CUT_BYTES", 1.0)
    sel = run_obbt()
    np.testing.assert_array_equal(full[2], sel[2])  # the same boxes proven infeasible
    ok = ~full[2]  # (an infeasible box's bounds carry no information)
    assert ok.sum() >= 2
    for a, b in zip(full[:2], sel[:2]):
        np.testing.assert_allclose(a[ok], b[ok], rtol=1e-7, atol=1e-7)

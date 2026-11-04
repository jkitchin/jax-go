"""The shared-kernel lifted LP (``quadratic_generic``) against ``quadratic.make_quadratic_bounder``."""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import quadratic as Qm
from jaxglobal.nl import read_nl
from jaxglobal.quadratic_generic import _EXES, make_generic_quadratic_bounder

DATA = pathlib.Path(__file__).resolve().parent / "data"


def _no_pool(Q):
    """The form without the separation pool and the lattice (left out of the shared kernel)."""
    return Q._replace(pool_cols=np.zeros((0, 1), np.int64), pool_vals=np.zeros((0, 1)), pool_rhs=np.zeros(0),
                      lattice=None)  # fmt: skip


def _boxes(lo, hi, k, integer=(), seed=0):
    rng = np.random.default_rng(seed)
    w = hi - lo
    a = lo + rng.uniform(0, 0.3, (k - 1, lo.size)) * w
    b = lo + rng.uniform(0.4, 1.0, (k - 1, lo.size)) * w
    L, H = np.vstack([lo, a]), np.vstack([hi, np.maximum(a, b)])
    if len(integer):
        ii = list(integer)
        L[:, ii] = np.ceil(L[:, ii])
        H[:, ii] = np.maximum(np.floor(H[:, ii]), L[:, ii])
    return L, H


def _compare(Q, n, L, H):
    ref = jax.vmap(Qm.make_quadratic_bounder(Q, n_kelley=4, tol=1e-6), in_axes=(0, 0, None))
    r = np.asarray(ref(jnp.asarray(L), jnp.asarray(H), jnp.inf)[0])
    g = np.asarray(make_generic_quadratic_bounder(Q, n, tol=1e-6)(L, H, np.inf)[0])
    # an infeasible LP's certificate diverges (Farkas ray): such boxes are pruned by both, at
    # values that depend on the iterates; compare those as "pruned", the others by value
    scale = 1e6 * (1 + abs(r[0]))
    pruned = lambda v: ~np.isfinite(v) | (v > scale)  # noqa: E731
    np.testing.assert_array_equal(pruned(g), pruned(r))
    ok = ~pruned(r)
    assert ok[0]
    # the same LP padded: the interior-point iterates differ, and the certificates agree to the LP's
    # accuracy, which is relative to the scale of its objective (coefficients times column ranges)
    w = np.max(H - L, axis=0)
    span = np.concatenate([w, np.full(Q.E.shape[1] - n, np.max(np.abs(np.concatenate([L, H])), initial=1.0) ** 2)])
    obj_scale = float(np.linalg.norm(Q.E[0] * span))
    np.testing.assert_allclose(g[ok], r[ok], rtol=1e-6, atol=1e-7 * (1.0 + obj_scale))


@pytest.mark.parametrize("name", ["clay0203m", "smallinvDAXr3b010-011", "kall_circles_c6a"])
def test_same_bounds_as_make_quadratic_bounder(name):
    M = read_nl(DATA / f"{name}.nl", integers="mask")
    n = len(M.lo)
    integer = tuple(np.flatnonzero(M.integer))
    Q = Qm.quadratic_form(M.f, M.ineq, M.eq, n) or Qm.polynomial_form(M.f, M.ineq, M.eq, n, integer=integer,
                                                                        sym_seconds=0.0)  # fmt: skip
    if Q is None:
        pytest.skip("no lifted form")
    lo, hi = np.maximum(np.asarray(M.lo, float), -1e3), np.minimum(np.asarray(M.hi, float), 1e3)
    _compare(_no_pool(Q), n, *_boxes(lo, hi, 6, integer))


def test_polynomial_levels_and_validity():
    """A degree-3 polynomial (products of products) with an equality: bounds match the reference
    and never exceed the true minimum (dense sample of the feasible set)."""
    f = lambda x: x[0] * x[1] * x[2] - x[0] ** 2 + 0.5 * x[1]  # noqa: E731
    g = lambda x: jnp.array([x[0] + x[1] + x[2] - 2.0])  # noqa: E731
    n = 3
    Q = Qm.polynomial_form(f, g, None, n, sym_seconds=0.0)
    assert Q is not None and Q.level.max() >= 2
    lo, hi = -np.ones(3), np.array([2.0, 2.0, 2.0])
    _compare(_no_pool(Q), n, *_boxes(lo, hi, 5))
    lb = float(np.asarray(make_generic_quadratic_bounder(_no_pool(Q), n)(lo[None], hi[None], np.inf)[0])[0])
    X = np.random.default_rng(0).uniform(lo, hi, (200000, 3))
    feas = np.asarray(jax.vmap(g)(jnp.asarray(X)))[:, 0] <= 0
    assert lb <= float(np.min(np.asarray(jax.vmap(f)(jnp.asarray(X[feas]))))) + 1e-9


def test_one_executable_per_bucket():
    f1 = lambda x: x[0] * x[1] + x[2] ** 2  # noqa: E731
    f2 = lambda x: 3.0 * x[0] * x[2] - x[1] ** 2 + x[0]  # noqa: E731
    g = lambda x: jnp.array([x[0] + x[1] - 1.0])  # noqa: E731
    lo, hi = -np.ones(3), 2 * np.ones(3)
    a = make_generic_quadratic_bounder(Qm.quadratic_form(f1, g, None, 3), 3)
    b = make_generic_quadratic_bounder(Qm.quadratic_form(f2, g, None, 3), 3)
    a(lo[None], hi[None], np.inf)
    before = len(_EXES)
    b(lo[None], hi[None], np.inf)
    assert len(_EXES) == before

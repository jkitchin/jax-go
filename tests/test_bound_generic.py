"""The shared-kernel McCormick bounder (``bound_generic``) against ``lp.make_constrained_bounder``."""

import functools
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import bound_generic as BG
from jaxglobal import lp, relax_interp
from jaxglobal.nl import read_nl

DATA = pathlib.Path(__file__).resolve().parent / "data"


@pytest.fixture
def mccormick_reference(monkeypatch):
    """``make_constrained_bounder`` with the same relaxation: interpreted McCormick, no
    convexity certificates (univariate hulls and alphaBB are off by default)."""
    monkeypatch.setattr(relax_interp, "RULE", "rows")
    monkeypatch.setattr(relax_interp, "MIN_ROWS", 0)
    monkeypatch.setattr(lp, "_parts_fn", functools.partial(lp._parts_fn, convex=False))
    return lp.make_constrained_bounder


def _boxes(lo, hi, k, seed=0):
    lo, hi = np.maximum(lo, -1e3), np.minimum(hi, 1e3)
    rng = np.random.default_rng(seed)
    w = hi - lo
    a = lo + rng.uniform(0, 0.3, (k - 1, lo.size)) * w
    b = lo + rng.uniform(0.4, 1.0, (k - 1, lo.size)) * w
    return np.vstack([lo, a]), np.vstack([hi, np.maximum(a, b)])


@pytest.mark.parametrize("name", ["nvs01", "m3"])
def test_same_bounds_as_make_constrained_bounder(name, mccormick_reference):
    M = read_nl(DATA / f"{name}.nl", integers="mask")
    n = len(M.lo)
    gb = BG.make_generic_bounder(M.f, M.ineq, M.eq, n=n, tol=1e-6)
    assert gb is not None
    ref = jax.vmap(mccormick_reference(M.f, M.ineq, M.eq, tol=1e-6), in_axes=(0, 0, None))
    L, H = _boxes(np.asarray(M.lo, float), np.asarray(M.hi, float), 6)
    r = [np.asarray(v) for v in ref(jnp.asarray(L), jnp.asarray(H), jnp.inf)]
    g = [np.asarray(v) for v in gb(L, H, np.inf)]
    np.testing.assert_array_equal(g[2], r[2])  # the same boxes found infeasible
    # an infeasible LP drives the duals along a Farkas ray: both certificates blow up (the box is
    # pruned), to values that depend on the iterates; compare those as "pruned", the rest by value
    big = lambda v: ~np.isfinite(v) | (v > 1e100)  # noqa: E731
    np.testing.assert_array_equal(big(g[0]), big(r[0]))
    fin = ~big(r[0])
    # the same cuts (exact: test_same_relaxation_as_the_interpreter) and LP; the padded variables
    # and dead rows move the IPM's iterates, so the certificates agree to LP accuracy
    np.testing.assert_allclose(g[0][fin], r[0][fin], rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("name", ["nvs01", "m3", "clay0203m"])
def test_same_relaxation_as_the_interpreter(name, mccormick_reference):
    """The parts [cv_f, cv_g..., cv_h..., -cc_h...] and their subgradients at points of a box are
    those of ``lp._parts_fn`` (interpreted McCormick): the cuts the LP is built from."""
    M = read_nl(DATA / f"{name}.nl", integers="mask")
    n = len(M.lo)
    gb = BG.make_generic_bounder(M.f, M.ineq, M.eq, n=n)
    _, parts, _ = lp._parts_fn(M.f, M.ineq, M.eq, False, convex=False)
    lo, hi = np.maximum(np.asarray(M.lo, float), -1e3), np.minimum(np.asarray(M.hi, float), 1e3)
    N, d = gb.g.N, gb.data
    Lp = jnp.concatenate([jnp.asarray(lo), jnp.zeros(N - n)])
    Hp = jnp.concatenate([jnp.asarray(hi), jnp.zeros(N - n)])
    rng = np.random.default_rng(1)
    K = int(np.sum(np.asarray(d.part_live)))
    for x in lo + rng.uniform(0, 1, (4, n)) * (hi - lo):
        xp = jnp.concatenate([jnp.asarray(x), jnp.zeros(N - n)])
        V, G = BG._sweep(xp, Lp, Hp, d, False, gb.g.tier)
        sign = np.asarray(d.part_sign)[:K]
        Vp, Gp = np.asarray(V)[np.asarray(d.part_slot)[:K]], np.asarray(G)[np.asarray(d.part_slot)[:K]]
        v = np.where(sign > 0, Vp[:, 2], -Vp[:, 3])
        J = np.where((sign > 0)[:, None], Gp[:, 0], -Gp[:, 1])[:, :n]
        rv = np.asarray(parts(jnp.asarray(x), jnp.asarray(lo), jnp.asarray(hi), (None, None, None)))
        rJ = np.asarray(jax.jacfwd(lambda z: parts(z, jnp.asarray(lo), jnp.asarray(hi), (None, None, None)))(jnp.asarray(x)))
        np.testing.assert_allclose(v, rv, rtol=1e-12, atol=1e-12)  # sums may be bracketed differently
        np.testing.assert_allclose(J, rJ, rtol=1e-12, atol=1e-12)


def test_bound_is_valid():
    """The certificate is a lower bound of f on every feasible point (dense sample)."""
    f = lambda x: x[0] * x[1] - jnp.exp(x[0]) + x[1] ** 2  # noqa: E731
    g = lambda x: jnp.array([x[0] + x[1] - 1.5, x[0] * x[0] - x[1] - 0.5])  # noqa: E731
    lo, hi = np.array([-1.0, -1.0]), np.array([2.0, 2.0])
    gb = BG.make_generic_bounder(f, g, None, n=2, tol=0.0)
    lb = float(np.asarray(gb(lo[None], hi[None], np.inf)[0])[0])
    X = np.stack(np.meshgrid(np.linspace(-1, 2, 301), np.linspace(-1, 2, 301)), -1).reshape(-1, 2)
    feas = np.all(np.asarray(jax.vmap(g)(jnp.asarray(X))) <= 0, axis=1)
    assert lb <= float(np.min(np.asarray(jax.vmap(f)(jnp.asarray(X[feas]))))) + 1e-9


def test_one_executable_per_bucket():
    f1 = lambda x: x[0] * x[1] + x[2] ** 2  # noqa: E731
    f2 = lambda x: (x[0] - x[2]) * 3.0 + x[1] ** 2 - x[0] * x[2]  # noqa: E731
    lo, hi = -np.ones(3), 2 * np.ones(3)
    a, b = BG.make_generic_bounder(f1, None, None, n=3), BG.make_generic_bounder(f2, None, None, n=3)
    a(lo[None], hi[None], np.inf)
    before = len(BG._EXES)
    b(lo[None], hi[None], np.inf)
    assert len(BG._EXES) == before

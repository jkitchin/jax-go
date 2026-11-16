"""The reduced-space node bound with one shared cut pass (``lp.SHARED_CUTS``) and with the affine parts
linearized at the center only (``lp.SPLIT_AFFINE``) gives the bounds of the earlier layout (first LP,
then Kelley rounds with their own cut evaluation, every part at every point)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jaxglobal.lp as LP  # noqa: E402
from test_constrained import haverly1  # noqa: E402


def _bounder(f, g, h, shared, chunk, monkeypatch, L, H, safe=False, split=False):
    """The compiled bounder for boxes shaped like ``L`` (compiled here: the switches are read
    while tracing)."""
    monkeypatch.setattr(LP, "SHARED_CUTS", shared)
    monkeypatch.setattr(LP, "SPLIT_AFFINE", split)
    monkeypatch.setattr(LP, "SHARED_CHUNK", chunk)
    cb = LP.make_constrained_bounder(f, g, h, tol=1e-6, univariate=True, alphabb=True, safe=safe)
    fn = jax.jit(jax.vmap(cb, in_axes=(0, 0, None))).lower(jnp.asarray(L), jnp.asarray(H), 0.0).compile()
    monkeypatch.undo()
    return lambda L, H, cut: fn(jnp.asarray(L), jnp.asarray(H), float(cut))


def _boxes(lo, hi, k, seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(lo, hi, (k, lo.size))
    b = rng.uniform(lo, hi, (k, lo.size))
    L, H = np.minimum(a, b), np.maximum(a, b)
    L[0], H[0] = lo, hi
    return L, H


def _mixed():
    def f(x):
        return x[0] ** 2 + jnp.exp(0.3 * x[1]) * x[2] - x[0] * x[1] + jnp.sqrt(1.0 + x[3] ** 2)

    def g(x):
        return jnp.stack([x[0] * x[2] - 1.5, x[1] ** 2 + x[3] ** 2 - 4.0, x[0] + x[1] + x[2] - 3.0])

    def h(x):
        return jnp.stack([x[0] * x[3] - 0.5 * x[2]])

    return f, g, h, -np.ones(4), 2.0 * np.ones(4)


@pytest.mark.parametrize("model", ["haverly", "mixed"])
@pytest.mark.parametrize("chunk,shared,split", [(1, True, False), (3, True, False), (32, True, False),
                                                (3, False, True), (3, True, True)])
def test_shared_matches_separate(model, chunk, shared, split, monkeypatch):
    f, g, h, lo, hi = haverly1() if model == "haverly" else _mixed()
    L, H = _boxes(lo, hi, 6, 1)
    old = _bounder(f, g, h, False, chunk, monkeypatch, L, H)
    new = _bounder(f, g, h, shared, chunk, monkeypatch, L, H, split=split)
    for cut in (np.inf, -350.0 if model == "haverly" else 2.0):
        a, b = old(L, H, cut), new(L, H, cut)
        lb_a, lb_b = np.asarray(a[0]), np.asarray(b[0])
        # pruned boxes (bound past the cutoff, or an LP-infeasible box's huge certificate) agree as such
        cap = min(cut, 1e10)
        live_a, live_b = lb_a < cap, lb_b < cap
        assert np.array_equal(live_a, live_b), (lb_a, lb_b)
        assert np.array_equal(np.isfinite(lb_a[live_a]), np.isfinite(lb_b[live_a]))
        fin = live_a & np.isfinite(lb_a)
        assert np.allclose(lb_a[fin], lb_b[fin], rtol=1e-6, atol=1e-6), (lb_a, lb_b)
        assert np.array_equal(np.asarray(a[2]), np.asarray(b[2]))  # infeasible
        for k in (3, 4):  # reduced boxes of the live boxes (range reduction amplifies IPM rounding)
            assert np.allclose(np.asarray(a[k])[live_a], np.asarray(b[k])[live_a], rtol=0, atol=1e-3)


def test_shared_valid_at_optimum_safe(monkeypatch):
    """Safe mode: the bound of every box holding Haverly's optimum stays below -400."""
    from test_constrained import HAVERLY_X

    f, g, h, lo, hi = haverly1()
    rng = np.random.default_rng(5)
    L = np.maximum(lo, HAVERLY_X - rng.uniform(0, 50, (8, 8)) * (hi - lo) / 300)
    H = np.minimum(hi, HAVERLY_X + rng.uniform(0, 50, (8, 8)) * (hi - lo) / 300)
    new = _bounder(f, g, h, True, 4, monkeypatch, L, H, safe=True, split=True)
    lb = np.asarray(new(L, H, np.inf)[0])
    assert np.all(lb <= -400.0 + 1e-6)


def test_auto_rule(monkeypatch):
    """``"auto"`` shares the pass only on tapes of at least ``SHARED_MIN_EQNS`` equations."""
    f, g, h, lo, hi = haverly1()
    monkeypatch.setattr(LP, "SHARED_CUTS", "auto")
    eqns = LP._tape_eqns(f, g, h, 8)
    assert eqns > 0
    monkeypatch.setattr(LP, "SHARED_MIN_EQNS", eqns)
    assert LP._shared_cuts(f, g, h)[0](8)
    monkeypatch.setattr(LP, "SHARED_MIN_EQNS", eqns + 1)
    assert not LP._shared_cuts(f, g, h)[0](8)


def test_split_auto_rule():
    """``"auto"`` splits when at least half of the parts are affine."""
    from jaxglobal.lp import _nonlinear_vars

    f, g, h, _, _ = haverly1()
    nl = _nonlinear_vars(f, g, h, 8)
    assert (~nl.any(axis=1)).sum() >= nl.any(axis=1).sum()  # haverly: 7 affine of 11 parts
    f, g, h, _, _ = _mixed()
    nl = _nonlinear_vars(f, g, h, 4)
    assert (~nl.any(axis=1)).sum() < nl.any(axis=1).sum()


@pytest.mark.parametrize("model", ["haverly", "mixed"])
@pytest.mark.parametrize("own_cuts", [True, False])
def test_obbt_split_matches_unsplit(model, own_cuts, monkeypatch):
    """OBBT with the affine parts linearized at the center only (``SPLIT_AFFINE``) tightens the
    boxes as the unsplit layout does: the same LP rows, in another order."""
    f, g, h, lo, hi = haverly1() if model == "haverly" else _mixed()
    L, H = _boxes(lo, hi, 4, 2)
    out = {}
    for split in (False, True):
        monkeypatch.setattr(LP, "SPLIT_AFFINE", split)
        ob = LP.make_obbt(f, g, h, tol=1e-6, univariate=True, alphabb=True, own_cuts=own_cuts)
        fn = jax.jit(jax.vmap(ob, in_axes=(0, 0, None))).lower(jnp.asarray(L), jnp.asarray(H), 0.0).compile()
        out[split] = [fn(jnp.asarray(L), jnp.asarray(H), float(cut)) for cut in (np.inf, -350.0 if model == "haverly" else 2.0)]
    monkeypatch.undo()
    for a, b in zip(out[False], out[True]):
        assert np.array_equal(np.asarray(a[2]), np.asarray(b[2]))  # infeasible
        ok = ~np.asarray(a[2])
        for k in (0, 1):
            assert np.allclose(np.asarray(a[k])[ok], np.asarray(b[k])[ok], rtol=0, atol=1e-4 * (hi - lo).max())


@pytest.mark.parametrize("model", ["haverly", "mixed"])
def test_affine_forms(model):
    """``_affine_forms`` gives each affine part as ``A x + b`` (the function's value everywhere) and
    NaN rows for the nonlinear parts."""
    f, g, h, lo, hi = haverly1() if model == "haverly" else _mixed()
    n = lo.size
    A, b = LP._affine_forms(f, g, h, n)
    nl = LP._nonlinear_vars(f, g, h, n)
    aff = ~nl.any(axis=1)
    assert aff.any()
    assert np.all(np.isnan(b[~aff])) and np.all(np.isfinite(A[aff]))

    def parts(x):
        out = [jnp.atleast_1d(f(x)), jnp.atleast_1d(g(x))]
        if h is not None:
            out += [jnp.atleast_1d(h(x)), -jnp.atleast_1d(h(x))]
        return np.concatenate([np.asarray(o) for o in out])

    rng = np.random.default_rng(0)
    for x in rng.uniform(lo, hi, (5, n)):
        v = parts(jnp.asarray(x))
        assert np.allclose(A[aff] @ x + b[aff], v[aff], rtol=1e-12, atol=1e-9)


def test_affine_exact_matches_relaxation_pass(monkeypatch):
    """With ``AFFINE_EXACT`` the split bounder gives the bounds of the split bounder that linearizes
    the affine parts by a forward-mode pass of the relaxation."""
    f, g, h, lo, hi = haverly1()
    L, H = _boxes(lo, hi, 6, 3)
    monkeypatch.setattr(LP, "AFFINE_EXACT", False)
    old = _bounder(f, g, h, True, 4, monkeypatch, L, H, split=True)
    new = _bounder(f, g, h, True, 4, monkeypatch, L, H, split=True)  # (monkeypatch.undo restored AFFINE_EXACT)
    for cut in (np.inf, -350.0):
        a, b = old(L, H, cut), new(L, H, cut)
        lb_a, lb_b = np.asarray(a[0]), np.asarray(b[0])
        fin = np.isfinite(lb_a) & (lb_a < min(cut, 1e10))
        assert np.allclose(lb_a[fin], lb_b[fin], rtol=1e-7, atol=1e-7), (lb_a, lb_b)

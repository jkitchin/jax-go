"""A NaN node lower bound carries no information: it is treated as -inf, never as a bound.

A bounder can return NaN for a box (e.g. a relaxation evaluated outside the domain of log or
sqrt). Such a bound must not prune the box, rank it, close the gap, or reach the result's
``lower_bound`` as NaN."""

import jax.numpy as jnp
import numpy as np

from jaxglobal import bnb, minimize

XSTAR = np.array([0.3, -0.2])


def _nan_bounds(monkeypatch, when):
    """Make the bound kernel return lb = NaN (and no incumbent candidate) for every box
    ``[lo, hi]`` with ``when(lo, hi)`` true; returns a counter of the boxes so poisoned."""
    orig, n = bnb._Precompiled.__call__, {"nan": 0}

    def call(self, *args, **kw):
        out = orig(self, *args, **kw)
        if not self.name.startswith("bound"):  # "bound" or "bound_pool"
            return out
        blo, bhi = np.asarray(args[0]), np.asarray(args[1])
        bad = np.array([when(lo, hi) for lo, hi in zip(blo, bhi)])
        n["nan"] += int(bad.sum())
        lb, x, fx, *rest = out
        return (jnp.where(bad, jnp.nan, lb), x, jnp.where(bad, jnp.inf, fx), *rest)

    monkeypatch.setattr(bnb._Precompiled, "__call__", call)
    return n


def _contains_xstar(lo, hi):
    return bool(np.all((lo <= XSTAR) & (XSTAR <= hi)))


def test_nan_bound_never_reaches_the_result(monkeypatch):
    """Every box holding the minimizer has a NaN bound: the search cannot close the gap, and the
    result (here at max_iter, with the last iteration's children still open) has
    ``lower_bound == -inf``, not NaN, nor a NaN anywhere in the history."""
    f = lambda x: (x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2 + 0.00123  # noqa: E731
    n = _nan_bounds(monkeypatch, _contains_xstar)
    r = minimize(f, -np.ones(2), np.ones(2), max_iter=4, local_solve=False, obbt=False)
    assert n["nan"] > 0
    assert r.status == "max_iter" and not r.converged
    assert r.lower_bound == -np.inf
    assert not np.isnan([h[2] for h in r.history]).any()
    assert all(h[2] == -np.inf for h in r.history)  # the NaN box is open (not pruned) throughout


def test_nan_bound_does_not_prune(monkeypatch):
    """Large boxes holding the minimizer have NaN bounds and offer no incumbent. Had a NaN bound
    pruned them (``NaN < ub`` is False) the minimizer's region would be lost and the search would
    end at a worse point; treated as -inf they are split until their children get real bounds."""
    f = lambda x: (x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2 + 0.00456  # noqa: E731
    n = _nan_bounds(monkeypatch, lambda lo, hi: _contains_xstar(lo, hi) and np.max(hi - lo) > 0.3)
    r = minimize(f, -np.ones(2), np.ones(2), local_solve=False, obbt=False, abs_tol=1e-6, rel_tol=1e-6)
    assert n["nan"] > 0
    assert r.status == "optimal" and r.converged
    assert abs(r.fun - 0.00456) < 1e-5
    assert np.isfinite(r.lower_bound) and r.lower_bound <= r.fun + 1e-12
    assert r.fun - r.lower_bound <= 1e-5
    assert not np.isnan([h[2] for h in r.history]).any()

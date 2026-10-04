"""Fast checks of the synthetic generators (jaxglobal/synthetic.py)."""

import numpy as np
import pytest

from jaxglobal import minimize, synthetic as syn

TINY = [
    lambda: syn.univariate_sum(3, seed=1),
    lambda: syn.qcqp(3, 2, seed=1),
    lambda: syn.pooling(1, 2, 1, seed=1),
    lambda: syn.nn_surrogate(2, hidden=4, layers=1, seed=1),
]


@pytest.mark.parametrize("make", TINY)
def test_deterministic(make):
    a, b = make(), make()
    x = 0.5 * (a.lo + a.hi) + 0.1
    assert a.name == b.name and np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi)
    assert float(a.f(x)) == float(b.f(x))


def test_seed_changes_problem():
    x = np.full(5, 0.3)
    assert float(syn.univariate_sum(5, 0).f(x)) != float(syn.univariate_sum(5, 1).f(x))


def test_shapes():
    p = syn.qcqp(7, 3)
    assert p.lo.shape == (7,) and p.ineq(p.xstar).shape == (3,) and np.shape(p.f(p.xstar)) == ()
    p = syn.pooling(2, 3, 2)
    n = 3 * 2 + 2 + 2 * 2
    assert p.lo.shape == (n,) and p.ineq(p.xstar).shape == (3 + 2 + 2 + 2,) and p.eq(p.xstar).shape == (4,)
    assert np.shape(syn.nn_surrogate(4).f(np.zeros(4))) == ()


def test_feasible_points():
    for p in (syn.qcqp(10, 5, seed=3), syn.pooling(2, 3, 2, seed=3)):
        assert np.all(p.lo <= p.xstar) and np.all(p.xstar <= p.hi)
        assert np.max(p.ineq(p.xstar)) <= 0
        if p.eq is not None:
            assert np.max(np.abs(p.eq(p.xstar))) < 1e-12


def test_univariate_sum_fstar():
    p = syn.univariate_sum(3, seed=2)
    r = minimize(p.f, p.lo, p.hi)
    assert r.status == "optimal" and abs(r.fun - p.fstar) < 1e-5 and r.lower_bound <= p.fstar + 1e-6


@pytest.mark.parametrize("make", TINY[1:])
def test_tiny_solves(make):
    p = make()
    r = minimize(p.f, p.lo, p.hi, ineq=getattr(p, "ineq", None), eq=getattr(p, "eq", None), time_limit=120)
    assert r.status == "optimal" and r.gap <= 1e-4 + 1e-6 * abs(r.fun)

"""Scheduling rules of ``minimize`` under a time limit (docs/ANALYSIS-sched.md): the outer-approximation
heuristic's cuts from the cut pool's kernel, the search's local-solve share, the root dive and plain-kernel
gates. They only change when incumbent-only work runs, so the answers must not change."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal.bnb as bnb
import jaxglobal.milpbound as milpbound
from jaxglobal import minimize


def _convex_bigm():
    # min exp(x) + exp(z) + 0.1 (y0 + y1) s.t. x <= y0, z <= y1, x + z >= 1, y binary. Optimum
    # 2 e^0.5 + 0.2 at y = (1, 1); y = (1, 0) gives e + 1 + 0.1. exp is not quadratic, so the
    # reduced-space LP gets a cut pool (its convex parts certified)
    f = lambda v: jnp.exp(v[0]) + jnp.exp(v[1]) + 0.1 * (v[2] + v[3])
    ineq = lambda v: jnp.array([v[0] - v[2], v[1] - v[3], 1.0 - v[0] - v[1]])
    return f, ineq, np.zeros(4), np.ones(4)


OPT = 2 * np.exp(0.5) + 0.2


def test_oa_from_the_pool_kernel(monkeypatch):
    made = []
    MB = milpbound.MilpBound

    class Spy(MB):
        def __init__(self, *a, **k):
            made.append(k.get("tangents") is not None)
            super().__init__(*a, **k)

    monkeypatch.setattr(milpbound, "MilpBound", Spy)
    f, ineq, lo, hi = _convex_bigm()
    a = minimize(f, lo, hi, ineq=ineq, integer=[2, 3], time_limit=60.0)
    assert a.status == "optimal" and abs(a.fun - OPT) < 1e-5 and a.lower_bound <= a.fun + 1e-6
    from jaxglobal import kcache

    kcache.wait_for_builds(60)  # (the heuristic's setup runs on the kernel tracer thread)
    assert made and made[0]  # its cuts from the pool's tangent kernel
    monkeypatch.setattr(bnb, "_OA_POOL", False)
    b = minimize(f, lo, hi, ineq=ineq, integer=[2, 3], time_limit=60.0)
    assert b.status == "optimal" and abs(b.fun - a.fun) < 1e-5


@pytest.mark.parametrize("share", [(0.0, 0.0), None])
def test_search_local_share_same_answer(monkeypatch, share):
    monkeypatch.setattr(bnb, "_SEARCH_LOCAL_SHARE", share)
    monkeypatch.setattr(bnb, "_POOL_NLP_SEARCH", share is not None)
    f, ineq, lo, hi = _convex_bigm()
    r = minimize(f, lo, hi, ineq=ineq, integer=[2, 3], time_limit=60.0)
    assert r.status == "optimal" and abs(r.fun - OPT) < 1e-5 and r.verified


def test_gates_off_same_answer(monkeypatch):
    for name, val in (("_ROOT_DIVE_NOWAIT", False), ("_OBBT_HELD", False), ("_PLAIN_FIT", None),
                      ("_OA_EARLY", False), ("_OA_ROUNDS", 8)):
        monkeypatch.setattr(bnb, name, val)
    f, ineq, lo, hi = _convex_bigm()
    r = minimize(f, lo, hi, ineq=ineq, integer=[2, 3], time_limit=60.0)
    assert r.status == "optimal" and abs(r.fun - OPT) < 1e-5


def test_gate_is_open():
    g = bnb._Gate()
    assert not g.is_open
    fut = g.submit(lambda: 1)
    assert not fut.done()
    g.open()
    assert g.is_open and fut.result(timeout=30) == 1

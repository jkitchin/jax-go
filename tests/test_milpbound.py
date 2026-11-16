"""The integer-aware root bound (jaxglobal/milpbound.py)."""

import jax.numpy as jnp
import numpy as np

from jaxglobal import minimize
from jaxglobal.milpbound import MilpBound


def _bigm():
    # min (x - 0.3)^2 + (z - 0.3)^2 + 0.1 (y0 + y1) s.t. x <= y0, z <= y1, x + z >= 1, y binary.
    # LP relaxation: y = (x, z) fractional; with y integral both binaries must be 1 or one variable is 0.
    f = lambda v: (v[0] - 0.3) ** 2 + (v[1] - 0.3) ** 2 + 0.1 * (v[2] + v[3])
    ineq = lambda v: jnp.array([v[0] - v[2], v[1] - v[3], 1.0 - v[0] - v[1]])
    lo, hi = np.zeros(4), np.ones(4)
    imask = np.array([False, False, True, True])
    return f, ineq, lo, hi, imask


def test_bound_is_valid_and_beats_the_lp():
    f, ineq, lo, hi, imask = _bigm()
    r = minimize(f, lo, hi, ineq=ineq, integer=[2, 3], milp_root=False)
    B = MilpBound(f, ineq, None, 4)
    b_int = B.bound(lo, hi, imask, budget=20.0, rounds=10)
    assert b_int is not None and b_int <= r.fun + 1e-6
    # the LP relaxation (y fractional = x, z) has optimum 2 (0.2)^2 + 0.1 = 0.18 < 0.2 + 0.08
    assert b_int > 0.18 + 1e-3


def test_infeasible_box():
    f, ineq, lo, hi, imask = _bigm()
    B = MilpBound(f, ineq, None, 4)
    hi2 = hi.copy()
    hi2[2:] = 0.0  # both binaries 0: x = z = 0 violates x + z >= 1
    assert B.bound(lo, hi2, imask, budget=10.0) == np.inf


def test_no_integers_no_bound():
    f, ineq, lo, hi, _ = _bigm()
    assert MilpBound(f, ineq, None, 4).bound(lo, hi, np.zeros(4, bool)) is None


def test_minimize_same_answer():
    f, ineq, lo, hi, _ = _bigm()
    a = minimize(f, lo, hi, ineq=ineq, integer=[2, 3], milp_root=True)
    b = minimize(f, lo, hi, ineq=ineq, integer=[2, 3], milp_root=False)
    assert a.status == b.status == "optimal"
    assert abs(a.fun - b.fun) < 1e-5 and a.lower_bound <= a.fun + 1e-6


def _exact_fixed(f, ineq, lo, hi, imask):
    """A fixed-integer completion for ``propose``: SLSQP over the continuous variables."""
    from scipy.optimize import minimize as sp_min

    tried = []

    def complete(X):
        Y, F = [], []
        for x in np.atleast_2d(X):
            tried.append(tuple(x[imask]))
            bnds = [(v, v) if m else (a, b) for v, m, a, b in zip(x, imask, lo, hi)]
            r = sp_min(lambda z: float(f(jnp.asarray(z))), np.clip(x, lo, hi), bounds=bnds, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda z: -np.asarray(ineq(jnp.asarray(z)))}])
            ok = np.all(np.asarray(ineq(jnp.asarray(r.x))) <= 1e-6)
            Y.append(r.x)
            F.append(float(f(jnp.asarray(r.x))) if ok else np.inf)
        complete.ub = min([complete.ub] + F)
        return np.array(Y), np.array(F), complete.ub

    complete.ub = np.inf
    return complete, tried


def test_propose_finds_the_optimal_assignment():
    # optimum 0.28 at y = (1, 1), x = z = 0.5; y = (1, 0) or (0, 1) gives 0.68, y = (0, 0) is infeasible
    f, ineq, lo, hi, imask = _bigm()
    complete, tried = _exact_fixed(f, ineq, lo, hi, imask)
    st = MilpBound(f, ineq, None, 4).propose(lo, hi, imask, complete, budget=20.0, rounds=6)
    assert st["rounds"] >= 1 and abs(complete.ub - 0.28) < 1e-5
    assert len(tried) == len(set(tried))  # no-good cuts: no binary assignment proposed twice
    assert (0.0, 0.0) not in tried  # the cuts already exclude the infeasible assignment


def test_propose_respects_the_cutoff():
    # every cut set bounds f below by 0.08 on x + z >= 1: with incumbent 0 the first MILP is infeasible
    f, ineq, lo, hi, imask = _bigm()
    complete, tried = _exact_fixed(f, ineq, lo, hi, imask)
    st = MilpBound(f, ineq, None, 4).propose(lo, hi, imask, complete, cutoff=0.0, budget=10.0)
    assert not tried and st["done"]


def test_minimize_same_answer_without_oa(monkeypatch):
    import jaxglobal.bnb as bnb

    f, ineq, lo, hi, _ = _bigm()
    a = minimize(f, lo, hi, ineq=ineq, integer=[2, 3])
    monkeypatch.setattr(bnb, "_OA", False)
    b = minimize(f, lo, hi, ineq=ineq, integer=[2, 3])
    assert a.status == b.status == "optimal" and a.verified
    assert abs(a.fun - b.fun) < 1e-5 and a.lower_bound <= a.fun + 1e-6

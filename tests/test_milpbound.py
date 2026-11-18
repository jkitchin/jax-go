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


def _pool_tangents(f, ineq, eq=None):
    """``MilpBound``'s ``tangents`` from ``lp.make_pool_cuts`` (as ``minimize`` does with its cut pool's kernel)."""
    import jax

    from jaxglobal.lp import make_pool_cuts

    cuts = jax.jit(jax.vmap(make_pool_cuts(f, ineq, eq), in_axes=(0, None, None)))
    return lambda P, lo, hi: tuple(np.asarray(z) for z in cuts(jnp.asarray(P), jnp.asarray(lo), jnp.asarray(hi)))


def test_tangent_cuts_are_valid_and_match_the_affine_rows():
    # the cuts from the pool's tangent kernel: every cut is below its part on the box, and the
    # affine rows' cuts are the rows themselves (the own cut program's too)
    f, ineq, lo, hi, imask = _bigm()
    own, tan = MilpBound(f, ineq, None, 4), MilpBound(f, ineq, None, 4, tangents=_pool_tangents(f, ineq))
    rng = np.random.default_rng(0)
    P = rng.uniform(lo, hi, (5, 4))
    (vf, vg, _, _), (Jf, Jg, _, _) = tan.cut_values(P, lo, hi)
    (vf0, vg0, _, _), (Jf0, Jg0, _, _) = own.cut_values(P, lo, hi)
    np.testing.assert_allclose(vg, vg0, atol=1e-12)
    np.testing.assert_allclose(Jg, Jg0, atol=1e-12)
    np.testing.assert_allclose(vf, vf0, atol=1e-9)  # (f is convex: both are its tangent planes)
    X = rng.uniform(lo, hi, (200, 4))
    fx = np.array([float(f(jnp.asarray(x))) for x in X])
    for k in range(len(P)):  # each objective cut below f on the box
        cut = vf[k, 0] + (X - P[k]) @ Jf[k, 0]
        assert np.all(cut <= fx + 1e-9)


def test_propose_with_tangent_cuts():
    f, ineq, lo, hi, imask = _bigm()
    complete, tried = _exact_fixed(f, ineq, lo, hi, imask)
    st = MilpBound(f, ineq, None, 4, tangents=_pool_tangents(f, ineq)).propose(lo, hi, imask, complete, budget=20.0,
                                                                               rounds=6)
    assert st["rounds"] >= 1 and abs(complete.ub - 0.28) < 1e-5
    assert len(tried) == len(set(tried))


def test_compile_options_same_cuts(monkeypatch):
    # the cut program compiled with ``COMPILE_OPTIONS`` (LLVM -O0) and with XLA's defaults
    import jaxglobal.milpbound as mb

    f, ineq, lo, hi, _ = _bigm()
    P = np.random.default_rng(1).uniform(lo, hi, (3, 4))
    a = MilpBound(f, ineq, None, 4)
    assert a.ready() and a._exe is not None
    monkeypatch.setattr(mb, "COMPILE_OPTIONS", {})
    b = MilpBound(f, ineq, None, 4)
    assert b.ready()
    import jax

    for x, y in zip(jax.tree.leaves(a.cut_values(P, lo, hi)), jax.tree.leaves(b.cut_values(P, lo, hi))):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-9, atol=1e-12)

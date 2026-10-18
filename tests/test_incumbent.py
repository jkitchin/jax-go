"""Incumbent quality: near-zero objectives and integer neighbourhood polish (SCIP-gap plan, item C)."""

import importlib.util
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize
from jaxglobal.local import MAX_OBJ_SCALE, NEAR_ZERO, near_zero_options
from jaxglobal.nl import read_nl

DATA = pathlib.Path(__file__).resolve().parent / "data"
needs_pounce = pytest.mark.skipif(importlib.util.find_spec("pounce") is None, reason="pounce not installed")


def test_near_zero_options():
    assert near_zero_options(np.inf) is None and near_zero_options(np.nan) is None
    assert near_zero_options(NEAR_ZERO) is None and near_zero_options(-3.0) is None
    o = near_zero_options(7.1e-7, abs_tol=1e-6)  # below abs_tol: scale 1 / abs_tol
    assert o == {"obj_scaling_factor": 1e6, "mu_init": 1e-6}
    assert near_zero_options(-1e-3)["obj_scaling_factor"] == pytest.approx(1e3)
    assert near_zero_options(0.0, abs_tol=1e-9)["obj_scaling_factor"] == MAX_OBJ_SCALE


def _solve(name, **kw):
    M = read_nl(DATA / f"{name}.nl", integers="mask")
    return minimize(M.f, M.lo, M.hi, ineq=M.ineq, eq=M.eq, integer=M.integer, abs_tol=1e-6, rel_tol=1e-4,
                    feas_tol=1e-6, **kw)  # fmt: skip


@needs_pounce
def test_near_zero_objective_incumbent():
    """ex6_2_6: optimum -2.6025e-6 at a bound; unscaled pounce solves all stop at the interior point +7.1e-7."""
    r = _solve("ex6_2_6", max_iter=1, time_limit=600)
    assert r.verified and r.fun <= -2.5e-6


@needs_pounce
def test_integer_neighbourhood_polish():
    """smallinvDAXr3b010-011: the root incumbent was 0.4004; one +-1 / swap round reaches the optimum."""
    r = _solve("smallinvDAXr3b010-011", max_iter=1, time_limit=600)
    assert r.verified and abs(r.fun - 0.398797498) <= 1e-4 * 0.398797498


@needs_pounce
@pytest.mark.parametrize("name, fstar", [("m3", 37.8), ("clay0203m", 41573.26)])
def test_free_slack_instances_solve(name, fstar):
    r = _solve(name, time_limit=600)
    assert r.status == "optimal" and r.verified
    assert abs(r.fun - fstar) <= 1e-4 * abs(fstar)


def test_polish_respects_constraints():
    """Integer neighbours that improve f but break a constraint are never accepted.

    min -(x0 + x1) - y  s.t.  x0 + x1 + y <= 3.5,  x0 - x1 <= 1,  x integer in [0, 5], y in [0, 1]:
    every +1 move from an optimal point is infeasible; the optimum is -3.5."""
    f = lambda v: -(v[0] + v[1]) - v[2]  # noqa: E731
    g = lambda v: jnp.stack([v[0] + v[1] + v[2] - 3.5, v[0] - v[1] - 1.0])  # noqa: E731
    r = minimize(f, [0, 0, 0], [5, 5, 1], ineq=g, integer=[0, 1], max_iter=500)
    assert r.verified and r.status == "optimal"
    assert abs(r.fun + 3.5) <= 1e-6
    assert np.all(np.asarray(g(jnp.asarray(r.x))) <= 1e-6) and np.all(r.x[:2] == np.round(r.x[:2]))


def _st_z():
    """MINLPLib st_z (optimum 0 at x = 0) on a finite box. The relaxation minimizer sits at
    x3 = -feas_tol, violating x1 + x2 - x3 <= 0 by exactly feas_tol, with f = -2e-6 < 0."""

    def f(x):
        return -x[0] ** 2 - x[1] ** 2 - x[2] ** 2 + 2 * x[2]

    def ineq(x):
        return jnp.stack([x[0] + x[1] - x[2], -x[0] + x[1] - x[2], 12 * x[0] + 5 * x[1] + 12 * x[2] - 22.8,
                          12 * x[0] + 12 * x[1] + 7 * x[2] - 17.1, -6 * x[0] + x[1] + x[2] - 1.9])  # fmt: skip

    lo, hi = np.array([-100.0, 0.0, -100.0]), np.array([100.0, 100.0, 100.0])
    return minimize(f, lo, hi, ineq=ineq, abs_tol=1e-6, rel_tol=1e-4, feas_tol=1e-6)


@needs_pounce
def test_incumbent_repaired_to_strict_feasibility():
    r = _st_z()
    assert r.status == "optimal" and r.verified
    assert r.max_violation <= 1e-9  # repaired, not the feas_tol-violating relaxation point
    assert r.fun >= -1e-8  # no objective value below the true optimum 0
    assert r.lower_bound <= r.fun
    assert "repaired" in r.message


@needs_pounce
def test_incumbent_repair_does_not_walk_away():
    """The feas_tol-feasible minimizer (-5, 0) (f = -5) is strictly feasible only at y = 1 (f = 5),
    while x ~ 0, y = 0 is strictly feasible with f ~ 0. A repair to (-5, 1) would be reported
    "optimal" with f = 5 against a lower bound of -5: the relaxed incumbent is kept instead."""
    f = lambda x: x[0] + 10 * x[1]  # noqa: E731
    g = lambda x: jnp.atleast_1d(0.9e-6 * (1 - x[1]) - x[0] * (x[0] + 5) ** 2)  # noqa: E731
    r = minimize(f, np.array([-10.0, 0.0]), np.array([10.0, 2.0]), ineq=g, abs_tol=1e-6, rel_tol=1e-6, feas_tol=1e-6)
    assert r.verified
    if r.status == "optimal":
        assert r.fun - r.lower_bound <= 1e-3 * (1 + abs(r.fun))


@needs_pounce
def test_local_solution_feasible_after_clipping():
    """pounce must not relax the variable bounds (Ipopt's default ``bound_relax_factor`` 1e-8).

    The optimum has x0 at its upper bound with an equality of slope 1e4 in x0: a solution
    relaxed by 1e-8 past the bound and clipped back violates the equality by about 1e-4,
    so ``screen`` rejects it (ex5_4_3 found its optimum at the root and threw it away
    for 50 s this way)."""
    from jaxglobal.local import make_local

    lo, hi = np.array([0.0, 0.0]), np.array([1.0, 2e4])
    f = lambda x: -x[0] - 1e-6 * x[1]  # noqa: E731
    eq = lambda x: jnp.atleast_1d(x[1] - 1e4 * x[0])  # noqa: E731
    solve = make_local(f, None, eq, lo, hi, "pounce")
    X = np.clip(np.asarray(solve(np.array([[0.3, 100.0], [0.9, 1e4]]))), lo, hi)
    assert np.all(np.abs(X[:, 1] - 1e4 * X[:, 0]) <= 1e-6)
    assert np.allclose(X[:, 0], 1.0, atol=1e-6)


@needs_pounce
def test_polish_batches_fit_the_budget(monkeypatch):
    """Polish batches of local solves are sized to the time left: the first holds at most
    ``_POLISH_FIRST`` neighbours, and each solve's wall time is capped at its share of the
    time left (batch: 64 mostly infeasible neighbours took 40 s against a 2 s budget)."""
    from jaxglobal import bnb

    calls = []
    orig = bnb.make_local

    def recording(*a, **k):
        solve = orig(*a, **k)

        def wrapped(X0, LO=None, HI=None, options=None):
            if options and "max_wall_time" in options:
                calls.append((len(X0), options["max_wall_time"]))
            return solve(X0, LO, HI, options=options)

        return wrapped

    monkeypatch.setattr(bnb, "make_local", recording)
    bnb.clear_cache()
    f = lambda v: -(v[0] + v[1]) - v[2]  # noqa: E731
    g = lambda v: jnp.stack([v[0] + v[1] + v[2] - 3.5, v[0] - v[1] - 1.0])  # noqa: E731
    r = minimize(f, [0, 0, 0], [5, 5, 1], ineq=g, integer=[0, 1], max_iter=500)
    assert r.status == "optimal" and abs(r.fun + 3.5) <= 1e-6
    assert calls and calls[0][0] <= bnb._POLISH_FIRST
    assert all(0 < m <= bnb._POLISH_CHUNK and 0 < cap <= bnb._POLISH_MIN + 1.0 for m, cap in calls)

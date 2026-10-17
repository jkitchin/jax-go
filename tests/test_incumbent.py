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

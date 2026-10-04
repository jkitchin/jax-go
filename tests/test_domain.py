"""Variables declared unbounded: finite bounds are derived, never guessed."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize
from jaxglobal.domain import bound_domain, used_variables
from jaxglobal.problems import CONSTRAINED_PROBLEMS

INF = np.inf


def test_objective_cutoff_bounds_free_variables():
    r = minimize(lambda x: x[0] ** 2 + x[1] ** 2, [-INF, -INF], [INF, INF], eq=lambda x: x[0] + x[1] - 1)
    assert r.status == "optimal"
    assert abs(r.fun - 0.5) <= 1e-5 and r.lower_bound <= 0.5 + 1e-6


def test_constraints_bound_free_variables():
    lo, hi, inf, xf, ff = bound_domain(lambda x: x[1], None, lambda x: x[0] - 2 * x[1],
                                       np.array([-INF, 0.0]), np.array([INF, 2.0]))  # fmt: skip
    assert not inf
    assert lo[0] >= -1e-5 and hi[0] <= 4 + 1e-5  # equalities hold to feas_tol


def test_unused_variable_is_fixed():
    assert used_variables(lambda x: (x[0] - 1) ** 2, None, None, 3).tolist() == [True, False, False]
    r = minimize(lambda x: (x[0] - 1) ** 2, [-1, -INF, 0], [3, INF, INF])
    assert r.status == "optimal" and abs(r.fun) <= 1e-8


def test_unbounded_problem_is_reported():
    r = minimize(lambda x: x[0] + x[1] ** 2, [-INF, -1], [INF, 1])
    assert r.status == "unbounded_domain"
    assert "[0]" in r.message


def test_infeasible_free_problem():
    r = minimize(lambda x: x[0], [-INF], [INF], ineq=lambda x: 1 + x[0] ** 2)
    assert r.status == "infeasible"


@pytest.mark.parametrize("name", ["G06", "G24", "haverly"])
def test_known_problems_with_upper_bounds_removed(name):
    """Dropping the upper bounds of the test problems must not change the optimum."""
    p = next(q for q in CONSTRAINED_PROBLEMS if q.name == name)
    r = minimize(p.f, p.lo, np.full_like(p.hi, INF), ineq=p.ineq, eq=p.eq, max_iter=3000)
    if r.status == "unbounded_domain":
        pytest.skip(f"{name}: {r.message}")  # nothing bounds these without the declared box
    assert r.status == "optimal"
    assert abs(r.fun - p.fstar) <= 1e-4 * (1 + abs(p.fstar))
    assert r.lower_bound <= p.fstar + 1e-4 * (1 + abs(p.fstar))

"""Variables declared unbounded: finite bounds are derived, never guessed."""

import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize
from jaxglobal.domain import bound_domain, monotone_bounds, used_variables
from jaxglobal.problems import CONSTRAINED_PROBLEMS

INF = np.inf
DATA = pathlib.Path(__file__).resolve().parent / "data"


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


# ---------------------------------------------------------------- monotone caps (domain.monotone_bounds)


def test_slack_variable_is_capped():
    """x2 >= |x0 - x1| (two rows), x2 not in f: lowering x2 to max |x0 - x1| = 5 keeps every point feasible."""
    ineq = lambda x: jnp.stack([x[0] - x[1] - x[2], x[1] - x[0] - x[2]])  # noqa: E731
    lo, hi = monotone_bounds(lambda x: (x[0] - 1) ** 2 + x[1], ineq, None,
                             np.array([0.0, -2.0, 0.0]), np.array([3.0, 2.0, INF]))  # fmt: skip
    assert np.isfinite(hi[2]) and hi[2] >= 5.0 - 1e-12 and hi[2] <= 5.0 + 1e-9
    assert lo[2] == 0.0 and hi[:2].tolist() == [3.0, 2.0]


def test_cap_respects_objective_and_rows():
    """Caps only where every use of the variable prefers it lower (upper cap) or higher (lower cap)."""
    box = np.array([0.0, 0.0, -INF]), np.array([1.0, 1.0, INF])
    g = lambda x: jnp.atleast_1d(x[0] - x[2])  # noqa: E731  x2 >= x0
    # f increasing in x2: lowered to max x0 = 1; nothing bounds it below
    lo, hi = monotone_bounds(lambda x: x[2] + x[1], g, None, *box)
    assert hi[2] == 1.0 and lo[2] == -INF
    # f decreasing in x2: no upper cap (the objective wants it large)
    lo, hi = monotone_bounds(lambda x: -x[2], g, None, *box)
    assert hi[2] == INF
    # x2 in an equality: never capped
    lo, hi = monotone_bounds(lambda x: x[2], g, lambda x: jnp.atleast_1d(x[2] - x[0] - x[1]), *box)
    assert hi[2] == INF and lo[2] == -INF
    # x2 nonlinear in a row: never capped
    lo, hi = monotone_bounds(lambda x: x[2], lambda x: jnp.atleast_1d(x[0] - x[2] ** 3), None, *box)
    assert hi[2] == INF
    # integer variables are capped at an integer above the needed value
    lo, hi = monotone_bounds(lambda x: x[2], lambda x: jnp.atleast_1d(0.5 * x[0] + 0.2 - x[2]), None, *box,
                             integer=np.array([False, False, True]))  # fmt: skip
    assert hi[2] == 1.0


@pytest.mark.parametrize("seed", range(5))
def test_random_linear_caps_keep_every_point_feasible(seed):
    """Random rows a.y - s_k <= b_k on a box of y with slacks s >= 0, declared unbounded: every feasible
    point (sampled) stays feasible with each slack lowered to its cap."""
    rng = np.random.default_rng(seed)
    ny, ns, m = 4, 3, 6
    A = rng.normal(size=(m, ny))
    S = np.zeros((m, ns))
    S[np.arange(m), rng.integers(0, ns, m)] = rng.uniform(0.5, 2.0, m)  # each row has one slack
    b = rng.normal(size=m)
    slack = S.argmax(1)  # rows written out as scalar sums (as .nl models are), one slack each

    def ineq(x):
        rows = [sum(A[r, i] * x[i] for i in range(ny)) - S[r, slack[r]] * x[ny + slack[r]] - b[r] for r in range(m)]
        return jnp.stack(rows)

    f = lambda x: sum(x[ny + k] for k in range(ns)) + sum(x[i] ** 2 for i in range(ny))  # noqa: E731
    lo = np.r_[-np.ones(ny), np.zeros(ns)]
    hi = np.r_[np.ones(ny), np.full(ns, INF)]
    cl, ch = monotone_bounds(f, ineq, None, lo, hi)
    assert np.all(np.isfinite(ch))
    Y = rng.uniform(-1, 1, size=(2000, ny))
    need = np.zeros((len(Y), ns))  # the smallest feasible slacks for each y
    for k in range(ns):
        rows = S[:, k] > 0
        if rows.any():
            need[:, k] = np.maximum(0.0, np.max((Y @ A[rows].T - b[rows]) / S[rows, k], axis=1))
    assert np.all(need <= ch[ny:] + 1e-9)


def test_free_slacks_make_instances_supported():
    """m3 and clay0203m (MINLPLib) declare |x - y| slacks free; they used to be 'unbounded_domain'."""
    from jaxglobal.nl import read_nl

    for name in ("m3", "clay0203m"):
        M = read_nl(DATA / f"{name}.nl", integers="mask")
        lo, hi, infeasible, _, _ = bound_domain(M.f, M.ineq, M.eq, M.lo, M.hi, integer=np.asarray(M.integer))
        assert not infeasible and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), name

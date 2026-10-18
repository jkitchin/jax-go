"""Variables declared unbounded: finite bounds are derived, never guessed."""

import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize
from jaxglobal.domain import LinearRelaxation, bound_domain, monotone_bounds, used_variables
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


# ---------------------------------------------------------------- LP bounds (domain.LinearRelaxation)


def _recycle():
    """x = (1 - t)(20 + x + y), y <= x / 7, t in [0.2, 0.3]: x <= 0.8 * 20 / (1 - 0.8 * 8 / 7) = 186.67.

    The recycle loop of a staged design (MINLPLib ``launch``): FBBT from x, y in [0, inf)
    cannot bound x (every sweep sees an infinite sum), a linear relaxation can.
    Variables (x, y, t, s) with s = 20 + x + y."""
    eq = lambda v: jnp.stack([v[0] - (1 - v[2]) * v[3], v[3] - 20 - v[0] - v[1]])  # noqa: E731
    ineq = lambda v: jnp.atleast_1d(7 * v[1] - v[0])  # noqa: E731
    return eq, ineq, np.array([0.0, 0.0, 0.2, -INF]), np.array([INF, INF, 0.3, INF])


def test_lp_bounds_recycle_loop():
    eq, ineq, lo, hi = _recycle()
    f = lambda v: -v[0]  # noqa: E731
    from jaxglobal.fbbt import make_fbbt

    a, b, _ = make_fbbt(f, ineq, eq, rounds=30)(jnp.asarray(lo), jnp.asarray(hi), INF)
    assert not np.isfinite(b[0])  # FBBT alone cannot
    R = LinearRelaxation(f, ineq, eq, 4)
    assert R.ok
    l2, h2 = R.bounds(np.asarray(a), np.asarray(b))
    xmax = 0.8 * 20 / (1 - 0.8 * 8 / 7)
    assert np.isfinite(h2[0]) and xmax - 1e-6 <= h2[0] <= xmax * 1.01
    assert np.all(np.isfinite(l2)) and np.all(np.isfinite(h2))
    r = minimize(f, lo, hi, ineq=ineq, eq=eq)
    assert r.status == "optimal" and abs(r.fun + xmax) <= 1e-3 * xmax


def test_lp_bounds_keep_feasible_points():
    """Every bound from the LP holds at sampled feasible points (here: the whole curve t -> x(t))."""
    eq, ineq, lo, hi = _recycle()
    R = LinearRelaxation(lambda v: v[0], ineq, eq, 4)
    l2, h2 = R.bounds(lo, hi, tighten_all=True)
    for t in np.linspace(0.2, 0.3, 11):
        for frac in np.linspace(0.0, 1.0, 5):  # y = frac * x / 7, x = (1 - t) 20 / (1 - (1 - t)(1 + frac / 7))
            x = (1 - t) * 20 / (1 - (1 - t) * (1 + frac / 7))
            v = np.array([x, frac * x / 7, t, 20 + x + frac * x / 7])
            assert np.all(v >= l2 - 1e-9) and np.all(v <= h2 + 1e-9), (t, frac)


def test_lp_bounds_objective_cutoff_with_log_terms():
    """f = x0 log x0 + x1 - log(x1 - x0) grows like x1 - log x1: the cutoff f <= f(x*) bounds x1.

    FBBT cannot (x1 and log(x1 - x0) are two unbounded terms of one sum, and the
    interval of x0 log x0 on [0, 1] is unbounded below); the LP has the tangents of
    -log and of the convex x log x."""
    f = lambda x: x[0] * jnp.log(x[0]) + x[1] - jnp.log(x[1] - x[0])  # noqa: E731
    lo, hi = np.array([0.0, -INF]), np.array([1.0, INF])
    lo2, hi2, infeasible, xf, ff = bound_domain(f, None, None, lo, hi)
    assert not infeasible and np.all(np.isfinite(lo2)) and np.all(np.isfinite(hi2))
    # optimum: x0 = 1/e... any point within the cutoff must stay: check a fine grid of good points
    X0, X1 = np.meshgrid(np.linspace(1e-6, 1, 200), np.linspace(1e-3, 20, 400))
    F = X0 * np.log(X0) + X1 - np.log(np.maximum(X1 - X0, 1e-300))
    keep = (X1 > X0) & (F <= ff)
    assert np.all(X1[keep] <= hi2[1] + 1e-9) and np.all(X1[keep] >= lo2[1] - 1e-9)


def test_lp_bounds_leave_truly_free_variables_alone():
    """A variable nothing bounds stays free: no artificial box."""
    f = lambda x: x[1] ** 2  # noqa: E731
    R = LinearRelaxation(f, lambda x: jnp.atleast_1d(x[0] - x[1]), None, 2)
    l2, h2 = R.bounds(np.array([-INF, -1.0]), np.array([INF, 1.0]), cutoff=1.0)
    assert l2[0] == -INF and h2[0] <= 1.0 + 1e-2  # x0 <= x1 <= 1, x0 unbounded below


def _same(a, b):
    """Exact equality of nested results (NaN equal to NaN)."""
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


_JIT_CASES = [
    ("sqrt", lambda x: jnp.sqrt(x[0]) + x[1], lambda x: jnp.atleast_1d(x[0] * x[1] - 1), None, [0, -INF], [INF, INF]),
    ("div", lambda x: x[0] / x[1] + x[2], lambda x: jnp.array([x[2] - x[0], x[1] - 3]), None,
     [-INF, 1, -INF], [INF, INF, INF]),
    ("const_row", lambda x: x[0], lambda x: jnp.array([1.0 + 0 * x[0], x[0] - x[1]]), None, [-INF, -1], [INF, 1]),
    ("abs_max", lambda x: jnp.abs(x[0]) + jnp.maximum(x[1], x[2]), lambda x: jnp.atleast_1d(x[0] + x[1] + x[2] - 1),
     lambda x: jnp.atleast_1d(x[0] - 2 * x[2]), [-INF, -INF, 0], [INF, INF, INF]),
    ("nan_lo", lambda x: x[0] + x[1], lambda x: jnp.atleast_1d(x[0] * x[1] - 1), None, [0, 0], [INF, INF]),
] + [(p.name, p.f, p.ineq, p.eq, p.lo, np.where(np.arange(len(p.hi)) % 2 == 0, INF, p.hi))
     for p in CONSTRAINED_PROBLEMS if p.name in ("G06", "haverly", "reactor")]  # fmt: skip


@pytest.mark.parametrize("case", _JIT_CASES, ids=[c[0] for c in _JIT_CASES])
def test_jitted_sweeps_match_eager(case, monkeypatch):
    """The structural sweeps give the same results jitted (small tapes) and eager (large ones)."""
    from jaxglobal import domain as D

    _, f, g, h, lo, hi = case
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    boxes = [(lo, hi), (np.where(np.isfinite(lo), lo, -5), np.where(np.isfinite(hi), hi, 5)), (0 * lo, 0 * hi)]
    out = {}
    for mode, limit in (("jit", 10**9), ("eager", -1)):
        monkeypatch.setattr(D, "_JIT_MAX_NODES", limit)
        with np.errstate(all="ignore"):
            rows = [D._row_upper(f, g, h, L, H) for L, H in boxes]
        s = D._row_structure(f, g, h, lo.size)
        mb = D.monotone_bounds(f, g, h, lo.copy(), hi.copy(), struct=s) if s is not None else None
        out[mode] = (D.used_variables(f, g, h, lo.size), rows, mb, D.bound_domain(f, g, h, lo, hi)[:3])
    assert _same(out["jit"], out["eager"])

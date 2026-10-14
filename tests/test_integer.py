"""Integer variables in minimize (docs/ROADMAP.md, Integers: plan, stage 2)."""

import csv
import itertools
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
NL = ROOT / "benchmarks" / "minlplib" / "nl"


def test_pure_integer_quadratic():
    """The continuous optimum (2.6, -1.3) is not integral; the integer one is (3, -1)."""
    r = minimize(lambda x: (x[0] - 2.6) ** 2 + (x[1] + 1.3) ** 2, [-5.0, -5.0], [5.0, 5.0], integer=[0, 1])
    assert r.status == "optimal" and r.verified
    np.testing.assert_array_equal(r.x, [3.0, -1.0])
    assert abs(r.fun - 0.25) < 1e-9


def test_mixed_integer_constrained():
    """min (x - 0.4)^2 + (y - 1.6)^2 + z^2  s.t.  x + y + z >= 2.5,  x, y integer, z continuous.
    Enumeration: x = 0 or 1, y = 2: z = 0.5 (x = 0) or 0 (x = 1)."""
    f = lambda v: (v[0] - 0.4) ** 2 + (v[1] - 1.6) ** 2 + v[2] ** 2  # noqa: E731
    g = lambda v: jnp.atleast_1d(2.5 - v[0] - v[1] - v[2])  # noqa: E731
    r = minimize(f, [-3.0, -3.0, -3.0], [3.0, 3.0, 3.0], ineq=g, integer=np.array([True, True, False]))
    assert r.status == "optimal" and r.verified
    best = min(
        (x - 0.4) ** 2 + (y - 1.6) ** 2 + max(2.5 - x - y, 0.0) ** 2 for x in range(-3, 4) for y in range(-3, 4) if 2.5 - x - y <= 3
    )
    assert abs(r.fun - best) < 1e-6
    assert r.x[0] == round(r.x[0]) and r.x[1] == round(r.x[1])


def test_linear_integers_are_branched():
    """Integers that enter only linearly must still be split (linear continuous variables never are):
    min -x - 2y  s.t.  3x + 4y <= 12.5,  x^2 + y^2 <= 9,  x, y integer >= 0."""
    f = lambda v: -v[0] - 2.0 * v[1]  # noqa: E731
    g = lambda v: jnp.stack([3 * v[0] + 4 * v[1] - 12.5, v[0] ** 2 + v[1] ** 2 - 9.0])  # noqa: E731
    r = minimize(f, [0.0, 0.0], [5.0, 5.0], ineq=g, integer=[0, 1])
    pts = [(x, y) for x in range(6) for y in range(6) if 3 * x + 4 * y <= 12.5 and x * x + y * y <= 9]
    assert r.status == "optimal"
    assert abs(r.fun - min(-x - 2 * y for x, y in pts)) < 1e-9


def test_infeasible_by_integrality():
    """0.2 <= x <= 0.8 holds no integer: infeasible, from the bounds or from a constraint."""
    r = minimize(lambda x: x[0] ** 2 + x[1] ** 2, [0.2, -1.0], [0.8, 1.0], integer=[0])
    assert r.status == "infeasible"
    g = lambda x: jnp.stack([0.2 - x[0], x[0] - 0.8])  # noqa: E731
    r = minimize(lambda x: x[0] ** 2 + x[1] ** 2, [-2.0, -1.0], [2.0, 1.0], ineq=g, integer=[0])
    assert r.status == "infeasible"


def test_unverified_when_not_integral():
    """The returned point is checked for integrality like the constraints."""
    from jaxglobal.bnb import _verify

    f = lambda x: x[0] ** 2  # noqa: E731
    _, ok, why = _verify(f, None, None, np.array([0.5]), 0.25, np.array([-1.0]), np.array([1.0]), 1e-6,
                         integer=np.array([True]))  # fmt: skip
    assert not ok and "integ" in why


@pytest.mark.parametrize("seed", range(4))
def test_matches_enumeration(seed):
    """Random small mixed problems against enumerating every integer assignment and solving the
    continuous subproblem globally (the integers fixed by their bounds)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    c = rng.uniform(-1.5, 1.5, 3)
    w = rng.uniform(0.5, 2.0)

    def f(x):
        return jnp.sum((A @ x - c) ** 2) + w * jnp.sin(2.0 * x[2]) + 0.3 * x[0] * x[1]

    def g(x):
        return jnp.atleast_1d(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 6.0)

    lo, hi = np.array([-2.0, -2.0, -2.0]), np.array([2.0, 2.0, 2.0])
    r = minimize(f, lo, hi, ineq=g, integer=[0, 1], abs_tol=1e-7, rel_tol=1e-7)
    best = np.inf
    for i, j in itertools.product(range(-2, 3), repeat=2):
        if i * i + j * j > 6.0:
            continue
        s = minimize(f, [i, j, -2.0], [i, j, 2.0], ineq=g, abs_tol=1e-7, rel_tol=1e-7)
        if s.status == "optimal":
            best = min(best, s.fun)
    assert r.status == "optimal" and r.verified
    assert abs(r.fun - best) <= 1e-5 * (1 + abs(best)), (r.fun, best)


def _ref(name):
    p = ROOT / "benchmarks" / "minlplib" / "instances_int.csv"
    rows = {r["name"]: r for r in csv.DictReader(open(p))} if p.exists() else {}
    return rows.get(name)


TINY = ["ex1221", "ex1222", "ex1226", "nvs03", "nvs04", "nvs06", "st_e13", "st_e36", "prob03", "gbd", "st_miqp3"]


@pytest.mark.parametrize("name", TINY)
def test_minlplib_tiny(name):
    """Tiny MINLPLib instances (at most 5 variables) against their reference optima."""
    path = NL / f"{name}.nl"
    ref = _ref(name)
    if not path.exists() or ref is None:
        pytest.skip("integer MINLPLib instances not downloaded (benchmarks/minlplib/fetch.py --integer)")
    from jaxglobal.nl import read_nl

    M = read_nl(path, integers="mask")
    r = minimize(M.f, M.lo, M.hi, ineq=M.ineq, eq=M.eq, integer=M.integer, abs_tol=1e-6, rel_tol=1e-6, time_limit=120)
    assert r.status == "optimal" and r.verified, r
    want = float(ref["primalbound"])
    assert abs(M.sense * r.fun - want) <= 1e-4 * (1 + abs(want)), (M.sense * r.fun, want)


def test_nvs01_data():
    """nvs01 from tests/data (always available): integers inside nonlinear constraints."""
    from jaxglobal.nl import read_nl

    M = read_nl(pathlib.Path(__file__).parent / "data" / "nvs01.nl", integers="mask")
    r = minimize(M.f, M.lo, M.hi, ineq=M.ineq, eq=M.eq, integer=M.integer, time_limit=120)
    assert r.status == "optimal" and r.verified
    assert abs(r.fun - 12.4696688215682) < 1e-4 * 13
    assert np.all(r.x[M.integer] == np.round(r.x[M.integer]))



def test_unbounded_domain_cutoff_is_integer_feasible():
    """With an unbounded variable, bounds are derived by FBBT against the value of a feasible
    point. That point must be integral: here the relaxation reaches y = 0 (x = 0.5), which as a
    cutoff would remove every integer point (y >= 0.25); gear4 (MINLPLib) was reported
    infeasible this way."""
    g = lambda v: jnp.atleast_1d((v[0] - 0.5) ** 2 - v[1])  # noqa: E731
    r = minimize(lambda v: v[1], [-3.0, 0.0], [3.0, np.inf], ineq=g, integer=[0])
    assert r.status == "optimal" and r.verified
    assert abs(r.fun - 0.25) < 1e-6

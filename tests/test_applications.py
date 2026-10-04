"""Smoke tests for jaxglobal.applications (interface and two tiny solves)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal
from jaxglobal.applications import AppProblem, kinetics, logic, suite


def test_interface():
    probs = suite(["kinetics", "logic"])
    assert [p.name for p in probs]
    names = [p.name for p in probs]
    assert len(set(names)) == len(names)
    for p in probs:
        assert isinstance(p, AppProblem)
        assert callable(p.f) and len(p.lo) == len(p.hi) and np.all(p.lo <= p.hi)
        x = jnp.asarray(0.5 * (p.lo + p.hi))
        assert np.isfinite(float(p.f(x)))
        assert p.fstar is not None and p.source and callable(p.pyomo)
    assert suite(["no_such_class"]) == []  # a missing class module is skipped


def test_kinetics_rk4_close_to_accurate_ode():
    from scipy.integrate import solve_ivp

    theta = np.array([11.85, 8.35, 0.995])
    p = kinetics.make_gasoil(0.1)
    rk = float(p.f(jnp.asarray(theta)))
    s = solve_ivp(lambda t, u: [-(theta[0] + theta[2]) * u[0] ** 2, theta[0] * u[0] ** 2 - theta[1] * u[1]],
                  (0, kinetics.GASOIL_TAU[-1]), [1.0, 0.0], t_eval=kinetics.GASOIL_TAU, rtol=1e-10, atol=1e-12)  # fmt: skip
    ref = float(((s.y.T - kinetics.GASOIL_Z) ** 2).sum())
    assert abs(rk - ref) < 1e-3 * max(ref, 1e-3)


def test_kinetics_pyomo_builds():
    # compiling the 20-step RK4 chain inside jaxglobal takes minutes, so only the model structure is checked here
    pytest.importorskip("pyomo")
    m = kinetics.PROBLEMS[0].pyomo()
    assert len(list(m.component_data_objects(__import__("pyomo.environ", fromlist=["x"]).Constraint))) > 40
    g = jax.grad(kinetics.PROBLEMS[0].f)(jnp.array([11.0, 8.0, 1.0]))
    assert np.all(np.isfinite(np.asarray(g)))


def test_logic_tariff_exact():
    p = next(q for q in logic.PROBLEMS if q.name == "tariff_blend")
    assert abs(logic._tariff_exact() - p.fstar) < 1e-5
    r = jaxglobal.minimize(p.f, p.lo, p.hi, ineq=p.ineq, rel_tol=1e-3, abs_tol=1e-3, time_limit=120)
    assert r.fun == pytest.approx(p.fstar, rel=2e-3)
    assert r.lower_bound <= p.fstar * (1 + 1e-6)


def test_logic_friction_switch_continuous():
    p = next(q for q in logic.PROBLEMS if q.name == "pipes_parallel")
    x = jnp.array([0.05, 0.04, 0.5])
    assert np.isfinite(float(p.f(x)))
    f = lambda re: np.where(re < logic.RE_C, 64 / re, 0.3164 * re**-0.25)  # noqa: E731
    assert f(logic.RE_C * (1 - 1e-9)) == pytest.approx(f(logic.RE_C * (1 + 1e-9)), rel=1e-6)

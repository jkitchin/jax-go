"""Smoke tests for jaxglobal.applications.nn (weights, forward pass, Pyomo builders)."""
import numpy as np
import pyomo.environ as pyo
import pytest

from jaxglobal.applications import nn

NAMES = ["peaks_1x5", "peaks_2x20"]


def test_problems_load():
    assert len(nn.PROBLEMS) == len(nn.GRID)
    for p in nn.PROBLEMS:
        assert p.lo.shape == (2,) and np.all(p.hi == 3.0) and "NOT the paper" in p.source


@pytest.mark.parametrize("name", NAMES)
def test_f_matches_numpy(name):
    p = next(q for q in nn.PROBLEMS if q.name == name)
    Ws, bs = nn.load_weights(name)
    X = np.random.default_rng(0).uniform(-3, 3, size=(5, 2))
    for x in X:
        assert float(p.f(x)) == pytest.approx(float(nn.forward_np(Ws, bs, x)), abs=1e-5)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("act", ["tanh", "exp"])
def test_pyomo_reduced_matches_f(name, act):
    p = next(q for q in nn.PROBLEMS if q.name == name)
    m = nn.reduced_model(name, act=act)
    for x in np.random.default_rng(1).uniform(-3, 3, size=(3, 2)):
        for i in range(2):
            m.x[i].value = float(x[i])
        assert pyo.value(m.obj) == pytest.approx(float(p.f(x)), abs=1e-5)
    assert pyo.value(p.pyomo().obj) is not None


def test_pyomo_omlt_matches_f():
    pytest.importorskip("omlt")
    name = "peaks_1x5"
    p = next(q for q in nn.PROBLEMS if q.name == name)
    m = nn.omlt_model(name)
    x = np.array([0.5, -1.0])
    ipopt = pyo.SolverFactory("ipopt")
    if not ipopt.available(False):
        pytest.skip("ipopt not available to propagate full-space variables")
    for i in range(2):
        m.nn.inputs[i].fix(float(x[i]))
    ipopt.solve(m)
    assert pyo.value(m.obj) == pytest.approx(float(p.f(x)), abs=1e-5)

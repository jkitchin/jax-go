"""Tests for jaxglobal.export: SCIP models and .nl files built from JAX functions.

The SCIP solves use small time limits; every check evaluates the original JAX
function at SCIP's solution, so a wrong translation shows up as a mismatch.
"""

import os
import re
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import export
from jaxglobal.export import ExportError, ExportSizeWarning
from jaxglobal.nl import read_nl
from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS

ps = pytest.importorskip("pyscipopt")

PROB = {p.name: p for p in list(PROBLEMS) + list(CONSTRAINED_PROBLEMS)}


def _solve(model, time=3.0):
    model.hideOutput()
    model.setParam("limits/time", time)
    model.optimize()
    assert model.getNSols() > 0, f"SCIP found no solution (status {model.getStatus()})"
    return model.getStatus(), export.solution(model), model.getObjVal()


def _check(f, x, obj, ineq=None, eq=None, integer=None, tol=1e-5):
    """The JAX functions at SCIP's x agree with SCIP's objective, and x is feasible."""
    fx = float(f(jnp.asarray(x)))
    assert abs(fx - obj) <= tol * (1 + abs(obj)), (fx, obj)
    if ineq is not None:
        assert np.all(np.asarray(ineq(jnp.asarray(x))) <= tol * 10), np.asarray(ineq(jnp.asarray(x)))
    if eq is not None:
        assert np.all(np.abs(np.asarray(eq(jnp.asarray(x)))) <= tol * 10), np.asarray(eq(jnp.asarray(x)))
    if integer is not None:
        xi = x[np.asarray(integer)]
        assert np.allclose(xi, np.round(xi), atol=1e-6)
    return fx


def _grid_min(f, lo, hi, n=401):
    g = np.meshgrid(*[np.linspace(a, b, n) for a, b in zip(lo, hi)], indexing="ij")
    X = np.stack([c.ravel() for c in g], -1)
    return float(np.min(jax.vmap(f)(jnp.asarray(X))))


# ---------------------------------------------------------------------------
# SCIP: library problems
# ---------------------------------------------------------------------------

SCIP_CASES = ["six_hump_camel", "mccormick", "goldstein_price", "hartmann3", "G06", "G11", "G24", "haverly",
              "reactor", "piecewise_2d", "tanh_net_2d"]  # fmt: skip


@pytest.mark.parametrize("name", SCIP_CASES)
def test_scip_library(name):
    p = PROB[name]
    ineq, eq = getattr(p, "ineq", None), getattr(p, "eq", None)
    m = export.to_scip(p.f, p.lo, p.hi, ineq=ineq, eq=eq)
    assert set(m.data.report.approximated) <= {"lt", "le", "gt", "ge"}  # only ties of comparisons
    status, x, obj = _solve(m)
    assert np.all(x >= np.asarray(p.lo) - 1e-6) and np.all(x <= np.asarray(p.hi) + 1e-6)
    _check(p.f, x, obj, ineq, eq)
    if status == "optimal" and p.fstar is not None:
        assert abs(obj - p.fstar) <= 1e-3 * (1 + abs(p.fstar)), (obj, p.fstar)


# ---------------------------------------------------------------------------
# SCIP: custom problems
# ---------------------------------------------------------------------------


def test_scip_integer():
    def f(x):
        return (x[0] - 1.4) ** 2 + (x[1] - 2.6) ** 2

    def g(x):
        return jnp.array([x[0] + x[1] - 4.0])

    for integer in ([1], np.array([False, True])):
        m = export.to_scip(f, [0.0, 0.0], [3.0, 5.0], ineq=g, integer=integer)
        assert m.data.x[1].vtype() == "INTEGER" and m.data.x[0].vtype() == "CONTINUOUS"
        status, x, obj = _solve(m)
        _check(f, x, obj, ineq=g, integer=[1])
        assert status == "optimal"
        np.testing.assert_allclose(x, [1.0, 3.0], atol=1e-5)
        assert abs(obj - 0.32) < 1e-6


def test_scip_equality():
    def f(x):
        return x[0] ** 2 + 2 * x[1] ** 2

    def h(x):
        return jnp.array([x[0] + x[1] - 1.0])

    m = export.to_scip(f, [-2.0, -2.0], [2.0, 2.0], eq=h)
    status, x, obj = _solve(m)
    _check(f, x, obj, eq=h)
    np.testing.assert_allclose(x, [2 / 3, 1 / 3], atol=1e-4)
    assert abs(obj - 2 / 3) < 1e-5


def _nonsmooth(x):  # continuous across its switching surfaces, so ties do not matter
    a = jnp.where(x[0] > 0.2, (x[0] - 0.2) ** 2, 0.5 * (0.2 - x[0]))
    b = jnp.maximum(x[1], 0.5) + jax.nn.relu(x[0] - x[1]) + jnp.clip(x[1] - 1.0, -0.25, 0.25)
    return a + b + jnp.minimum(x[0] ** 2, 0.3)


@pytest.mark.parametrize("minmax", ["abs", "bigm"])
def test_scip_max_where(minmax):
    lo, hi = [-1.0, -1.0], [2.0, 2.0]
    m = export.to_scip(_nonsmooth, lo, hi, minmax=minmax)
    r = m.data.report
    assert "select" in r.reformulated and "gt" in r.reformulated
    assert ("max" in r.rewritten) == (minmax == "abs") and ("max" in r.reformulated) == (minmax == "bigm")
    status, x, obj = _solve(m)
    _check(_nonsmooth, x, obj)
    if status == "optimal":
        assert obj <= _grid_min(_nonsmooth, lo, hi) + 1e-6


def test_scip_tanh_sigmoid_softplus():
    def f(x):
        return jnp.tanh(2 * x[0]) * x[1] + jax.nn.sigmoid(x[1]) + jax.nn.softplus(x[0] - x[1]) + 0.1 * x @ x

    lo, hi = [-2.0, -2.0], [2.0, 2.0]
    m = export.to_scip(f, lo, hi)
    r = m.data.report
    assert {"tanh", "logistic", "log1p"} <= set(r.rewritten)
    status, x, obj = _solve(m)
    _check(f, x, obj)
    if status == "optimal":
        assert abs(obj - _grid_min(f, lo, hi)) < 1e-3


def test_scip_scan_and_fori():
    def f(x):
        c, ys = jax.lax.scan(lambda c, xi: (0.5 * c + xi**2 - xi, c), 0.0, x)
        s = jax.lax.fori_loop(0, 3, lambda i, a: a + jnp.sin(x[i]) * x[(i + 1) % 3], 0.0)
        return c + jnp.sum(ys) + 0.1 * s

    lo, hi = -np.ones(3), 2 * np.ones(3)
    m = export.to_scip(f, lo, hi)
    assert any("scan" in s for s in m.data.report.notes)
    status, x, obj = _solve(m)
    _check(f, x, obj)


def test_scip_logsumexp_and_reductions():
    w = jnp.array([[1.0, -0.5], [0.3, 2.0], [-1.0, 0.7]])

    def f(x):
        z = w @ x
        return jax.nn.logsumexp(z) + jnp.max(x) - jnp.mean(x) + jnp.prod(x + 3.0) / 10.0

    m = export.to_scip(f, [-1.0, -1.0], [1.0, 1.0])
    assert m.data.report.assumptions  # isfinite of the max taken as True
    status, x, obj = _solve(m)
    _check(f, x, obj)


def test_scip_sign_and_pow():
    def f(x):
        return jnp.sign(x[0]) * 0.1 + x[1] ** x[0] + jnp.abs(x[0] - 0.3)

    m = export.to_scip(f, [-1.0, 0.5], [1.0, 2.0])
    r = m.data.report
    assert "sign" in r.reformulated and "pow(x, y)" in r.rewritten
    status, x, obj = _solve(m)
    fx = float(f(jnp.asarray(x)))
    # at x0 == 0 the export lets sign take any of -1, 0, 1
    assert abs(fx - obj) <= 1e-5 or abs(x[0]) < 1e-6


def test_scip_shared_subexpressions_stay_small():
    def f(x):  # each step uses c twice: a tree would double every step
        return jax.lax.fori_loop(0, 30, lambda i, c: jnp.sin(c) * c + x[0], x[1])

    m = export.to_scip(f, [-1.0, -1.0], [1.0, 1.0], share_min=5)
    assert "shared subexpression" in m.data.report.reformulated
    status, x, obj = _solve(m, time=5.0)
    _check(f, x, obj, tol=1e-4)


def test_scip_linear_objective_no_epigraph():
    m = export.to_scip(lambda x: 2 * x[0] - x[1], [0.0, 0.0], [1.0, 1.0], ineq=lambda x: x[0] ** 2 + x[1] ** 2 - 1)
    assert not m.data.report.epigraph
    status, x, obj = _solve(m)
    assert abs(obj - (-1.0)) < 1e-6


# ---------------------------------------------------------------------------
# The DAG evaluates like JAX
# ---------------------------------------------------------------------------


def test_graph_matches_jax():
    fns = [p for p in PROB.values()] + [None]
    rng = np.random.default_rng(1)
    for p in fns:
        if p is None:
            f, lo, hi, ineq, eq = _nonsmooth, np.array([-1.0, -1.0]), np.array([2.0, 2.0]), None, None
        else:
            f, lo, hi = p.f, np.asarray(p.lo, float), np.asarray(p.hi, float)
            ineq, eq = getattr(p, "ineq", None), getattr(p, "eq", None)
        prob = export._build(f, lo, hi, ineq, eq)
        X = rng.uniform(lo, hi, size=(7, lo.size))
        vals = export._evaluate(prob.g, [prob.obj] + [c[0] for c in prob.cons], X)
        np.testing.assert_allclose(vals[0] * np.ones(len(X)), jax.vmap(f)(jnp.asarray(X)), rtol=1e-10, atol=1e-10)
        ref = [jax.vmap(lambda x, fn=fn: jnp.atleast_1d(fn(x)))(jnp.asarray(X)) for fn in (ineq, eq) if fn is not None]
        if ref:
            ref = np.concatenate([np.asarray(r) for r in ref], axis=1)
            got = np.stack([v * np.ones(len(X)) for v in vals[1:]], axis=1)
            np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# .nl round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PROB))
def test_nl_roundtrip(name, tmp_path):
    p = PROB[name]
    ineq, eq = getattr(p, "ineq", None), getattr(p, "eq", None)
    path = tmp_path / f"{name}.nl"
    r = export.to_nl(p.f, p.lo, p.hi, path, ineq=ineq, eq=eq)
    m = read_nl(str(path))
    order = r.var_order
    np.testing.assert_array_equal(np.asarray(m.lo), np.asarray(p.lo, float)[order])
    np.testing.assert_array_equal(np.asarray(m.hi), np.asarray(p.hi, float)[order])
    X = np.random.default_rng(2).uniform(p.lo, p.hi, size=(10, len(p.lo)))
    for x in X:
        xn = jnp.asarray(x[order])
        np.testing.assert_allclose(float(m.f(xn)), float(p.f(jnp.asarray(x))), rtol=1e-10, atol=1e-10)
        # read_nl writes g(x) <= 0 as body - hi; equalities as body - value
        if ineq is not None:
            np.testing.assert_allclose(np.asarray(m.ineq(xn)), np.atleast_1d(ineq(jnp.asarray(x))), atol=1e-10)
        if eq is not None:
            np.testing.assert_allclose(np.asarray(m.eq(xn)), np.atleast_1d(eq(jnp.asarray(x))), atol=1e-10)


def test_nl_roundtrip_nonsmooth_and_integer(tmp_path):
    def f(x):
        return _nonsmooth(x[:2]) + jnp.tanh(x[2]) + jax.nn.sigmoid(x[0]) + jnp.sign(x[1]) + jnp.arctan(x[2])

    def g(x):
        return jnp.array([x[0] + x[2] - 2.0, jnp.where(x[1] < 0.5, x[1], -x[2])])

    lo, hi = np.array([-1.0, -1.0, 0.0]), np.array([2.0, 2.0, 3.0])
    path = tmp_path / "ns.nl"
    r = export.to_nl(f, lo, hi, path, ineq=g, integer=[2])
    np.testing.assert_array_equal(r.var_order, [0, 1, 2])
    m = read_nl(str(path), integers="mask")
    np.testing.assert_array_equal(np.asarray(m.integer), [False, False, True])
    X = np.random.default_rng(3).uniform(lo, hi, size=(20, 3))
    X[:, 2] = np.round(X[:, 2])
    for x in X:
        np.testing.assert_allclose(float(m.f(jnp.asarray(x))), float(f(jnp.asarray(x))), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(m.ineq(jnp.asarray(x))), np.asarray(g(jnp.asarray(x))), atol=1e-10)


def test_nl_reformulate_has_binaries(tmp_path):
    path = tmp_path / "ref.nl"
    r = export.to_nl(_nonsmooth, [-1.0, -1.0], [2.0, 2.0], path, nonsmooth="reformulate")
    assert r.n_binary >= 1 and "select" in r.reformulated
    m = read_nl(str(path), integers="mask")
    assert int(np.sum(m.integer)) == r.n_binary
    assert len(m.lo) == 2 + r.n_aux
    text = path.read_text()
    assert "o35" not in text  # no 'if' left


def _scip_nl(path, time=10):
    out = subprocess.run(["scip", "-c", f"set limits time {time} read {path} opt display statistics quit"],
                         capture_output=True, text=True, timeout=120).stdout  # fmt: skip
    mobj = re.search(r"Primal Bound\s*:\s*([-+0-9.eE]+)", out)
    return float(mobj.group(1)) if mobj else None


@pytest.mark.skipif(shutil.which("scip") is None, reason="no scip binary")
@pytest.mark.parametrize("name", ["G06", "mccormick", "haverly"])
def test_nl_solved_by_scip_binary(name, tmp_path):
    p = PROB[name]
    path = tmp_path / f"{name}.nl"
    export.to_nl(p.f, p.lo, p.hi, path, ineq=getattr(p, "ineq", None), eq=getattr(p, "eq", None))
    obj = _scip_nl(path)
    assert obj is not None and abs(obj - p.fstar) <= 1e-3 * (1 + abs(p.fstar)), obj


@pytest.mark.skipif(shutil.which("couenne") is None, reason="no couenne binary")
def test_nl_solved_by_couenne(tmp_path):
    p = PROB["G06"]
    path = tmp_path / "g06.nl"
    export.to_nl(p.f, p.lo, p.hi, path, ineq=p.ineq)
    out = subprocess.run(["couenne", str(path), "-AMPL"], capture_output=True, text=True, timeout=120,
                         cwd=tmp_path).stdout  # fmt: skip
    mobj = re.search(r"Upper bound:\s*([-+0-9.eE]+)", out)
    assert mobj is not None and abs(float(mobj.group(1)) - p.fstar) <= 1e-3 * abs(p.fstar), out[-500:]


# ---------------------------------------------------------------------------
# Refusals and warnings
# ---------------------------------------------------------------------------


def test_refuse_erf():
    with pytest.raises(ExportError, match="erf"):
        export.to_scip(lambda x: jax.scipy.special.erf(x[0]) + x[1] ** 2, [-1.0, -1.0], [1.0, 1.0])
    with pytest.raises(ExportError, match="erf"):
        export.to_nl(lambda x: jax.scipy.special.erf(x[0]), [-1.0], [1.0], os.devnull)
    r = export.inspect(lambda x: jax.scipy.special.erf(x[0]), [-1.0], [1.0])
    assert any("erf" in s for s in r.refused)


def test_refuse_data_dependent_while():
    def f(x):
        return jax.lax.while_loop(lambda c: c < 10.0, lambda c: c * 2.0 + x[0], x[1])

    with pytest.raises(ExportError, match="while_loop"):
        export.to_scip(f, [1.0, 1.0], [2.0, 2.0])


def test_static_while_is_unrolled():
    def f(x):
        return jax.lax.while_loop(lambda c: c[0] < 4, lambda c: (c[0] + 1, c[1] * x[0] + 1.0), (0, x[1]))[1]

    r = export.inspect(f, [0.0, 0.0], [1.0, 1.0])
    assert not r.refused and any("while" in s for s in r.notes)


def test_refuse_dynamic_index():
    table = jnp.array([1.0, 4.0, 2.0, 3.0])

    def f(x):
        return table[jnp.floor(x[0]).astype(jnp.int32)] + x[1]

    with pytest.raises(ExportError, match="convert_element_type|variable-dependent indices"):
        export.to_scip(f, [0.0, 0.0], [3.0, 1.0])


def test_refuse_cond_on_variable():
    def f(x):
        return jax.lax.cond(x[0] > 0, lambda: x[1], lambda: -x[1])

    with pytest.raises(ExportError, match="cond"):
        export.to_scip(f, [-1.0, -1.0], [1.0, 1.0])


def test_refuse_where_without_bounds():
    def f(x):
        return jnp.where(x[0] > 0, x[0] ** 2, x[1])

    with pytest.raises(ExportError, match="finite bound"):
        export.to_scip(f, [-1.0, -np.inf], [1.0, np.inf])
    # the .nl target writes 'if' natively and needs no bounds
    r = export.inspect(f, [-1.0, -np.inf], [1.0, np.inf], target="nl")
    assert not r.refused


def test_refuse_atan_for_scip_only():
    def f(x):
        return jnp.arctan(x[0])

    with pytest.raises(ExportError, match="atan"):
        export.to_scip(f, [-1.0], [1.0])
    assert not export.inspect(f, [-1.0], [1.0], target="nl").refused


def test_infeasible_constant_constraint():
    with pytest.raises(ExportError, match="infeasible"):
        export.to_scip(lambda x: x[0] ** 2, [-1.0], [1.0], ineq=lambda x: jnp.array([1.0 + 0 * x[0]]))


def test_size_warning():
    with pytest.warns(ExportSizeWarning):
        export.to_scip(lambda x: jnp.sum(jnp.sin(x) ** 2), -np.ones(4), np.ones(4), max_size=3)


def test_report_str():
    r = export.inspect(_nonsmooth, [-1.0, -1.0], [2.0, 2.0])
    s = str(r)
    assert "reformulated: select" in s and "rewritten: max" in s


# ---------------------------------------------------------------------------
# Regressions (PR #22 review)
# ---------------------------------------------------------------------------


def _sum2(x):  # 2-term sums: a 2-element dot product and a 2-term big-M row
    return jnp.dot(x[:2], x[1:]) + jnp.sin(x[0])


def test_nl_two_term_sum_is_not_a_sumlist(tmp_path):
    path = tmp_path / "sum2.nl"
    export.to_nl(_sum2, [-1.0] * 3, [1.0] * 3, path)
    lines = path.read_text().split("\n")
    # ASL's sumlist o54 needs 3+ operands
    assert all(int(lines[k + 1]) >= 3 for k, ln in enumerate(lines) if ln == "o54")
    m = read_nl(str(path))
    x = jnp.array([0.3, -0.7, 0.2])
    assert abs(float(m.f(x)) - float(_sum2(x))) < 1e-12


@pytest.mark.skipif(shutil.which("scip") is None, reason="no scip binary")
def test_nl_two_term_sum_scip_binary(tmp_path):
    path = tmp_path / "sum2.nl"
    export.to_nl(_sum2, [-1.0] * 3, [1.0] * 3, path)
    obj = _scip_nl(path)
    assert obj is not None and abs(obj - _grid_min(_sum2, [-1.0] * 3, [1.0] * 3, n=41)) < 1e-3, obj


def _couenne(path):
    out = subprocess.run(["couenne", str(path), "-AMPL"], capture_output=True, text=True, timeout=120,
                         cwd=path.parent)  # fmt: skip
    mobj = re.search(r"Upper bound:\s*([-+0-9.eE]+)", out.stdout)
    return float(mobj.group(1)) if mobj else None


@pytest.mark.skipif(shutil.which("couenne") is None, reason="no couenne binary")
def test_nl_two_term_sum_couenne(tmp_path):
    path = tmp_path / "sum2.nl"
    export.to_nl(_sum2, [-1.0] * 3, [1.0] * 3, path)
    obj = _couenne(path)
    assert obj is not None and abs(obj - _grid_min(_sum2, [-1.0] * 3, [1.0] * 3, n=41)) < 1e-3, obj


def _portable_f(x):
    return (jnp.tanh(x[0]) + jnp.maximum(x[1], 0.3) + jnp.where(x[2] > 0, x[2], -0.5 * x[2])
            + jnp.sum(x[:2] * x[1:]))  # fmt: skip


def test_nl_portable_operators(tmp_path):
    path = tmp_path / "portable.nl"
    r = export.to_nl(_portable_f, [-1.0] * 3, [1.0] * 3, path, portable=True)
    lines = path.read_text().split("\n")
    ops = {ln for ln in lines if ln.startswith("o")}
    allowed = {f"o{export.NL_OPS[k]}" for k in ("add", "sub", "mul", "div", "pow", "neg", "abs", "sqrt", "log",
                                                 "exp", "sin", "cos")} | {"o54"}  # fmt: skip
    assert ops <= allowed, ops - allowed
    assert not any(ln.startswith("V") for ln in lines)
    assert "tanh" in r.rewritten and "select" in r.reformulated
    with pytest.raises(ExportError, match="atan"):
        export.to_nl(lambda x: jnp.arctan(x[0]), [-1.0], [1.0], tmp_path / "a.nl", portable=True)


@pytest.mark.skipif(shutil.which("scip") is None or shutil.which("couenne") is None, reason="no scip/couenne")
def test_nl_portable_binaries(tmp_path):
    path = tmp_path / "portable.nl"
    export.to_nl(_portable_f, [-1.0] * 3, [1.0] * 3, path, portable=True)
    fmin = _grid_min(_portable_f, [-1.0] * 3, [1.0] * 3, n=41)  # minimum at a vertex
    assert abs(_scip_nl(path) - fmin) < 1e-4
    assert abs(_couenne(path) - fmin) < 1e-4


def _where_sqrt(x):
    return jnp.where(x[0] > 0, jnp.sqrt(x[0]) + 1.0, x[0])  # min -1 at x = -1


def test_where_branch_domain_refused(tmp_path):
    for f in (_where_sqrt, lambda x: jnp.where(x[0] > 0, x[0] ** 1.5, x[0])):
        with pytest.raises(ExportError, match="undefined on part of the box"):
            export.to_scip(f, [-1.0], [1.0])
        with pytest.raises(ExportError, match="undefined on part of the box"):
            export.to_nl(f, [-1.0], [1.0], tmp_path / "w.nl", nonsmooth="reformulate")
    with pytest.raises(ExportError, match="comparison"):
        export.to_scip(lambda x: jnp.where(jnp.log(x[0]) > 0, 1.0, x[0]), [-1.0], [1.0])
    # native 'if' evaluates one branch only: exact
    assert not export.inspect(_where_sqrt, [-1.0], [1.0], target="nl").refused
    # inside the domain, or made total, it exports
    _, x, obj = _solve(export.to_scip(_where_sqrt, [0.5], [1.0]))
    _check(_where_sqrt, x, obj)

    def guarded(x):
        return jnp.where(x[0] > 0, jnp.sqrt(jnp.maximum(x[0], 0.0)) + 1.0, x[0])

    _, x, obj = _solve(export.to_scip(guarded, [-1.0], [1.0]))
    assert abs(obj + 1.0) < 1e-6
    _check(guarded, x, obj)


def test_bigm_limits():
    def f(x):
        return jnp.where(jnp.exp(x[0]) > 2.0, -jnp.exp(x[0]) * 1e-20, x[0])

    with pytest.warns(export.ExportBigMWarning):
        m = export.to_scip(f, [-30.0], [30.0])
    assert any("big-M" in s for s in m.data.report.notes)
    for h in (50.0, 60.0):
        with pytest.raises(ExportError, match="big-M constant"):
            export.to_scip(f, [-h], [h])


def test_division_by_constant_zero(tmp_path):
    with pytest.raises(ExportError, match="division by the constant 0"):
        export.to_scip(lambda x: x[0] / 0.0, [-1.0], [1.0])
    with pytest.raises(ExportError, match="division by the constant 0"):
        export.to_nl(lambda x: x[0] / 0.0, [-1.0], [1.0], tmp_path / "d.nl")


def test_ties_reported_and_strict_eps():
    def f(x):  # x > 0: -0.5, x == 0: 0.5, x < 0: 0
        return jnp.where(x[0] > 0, -1.0, 0.0) + jnp.where(x[0] >= 0, 0.5, 0.0)

    m = export.to_scip(f, [-1.0], [1.0])
    assert {"gt", "ge"} <= set(m.data.report.approximated)
    _, x, obj = _solve(m)
    assert obj <= -0.5 + 1e-9  # a relaxation: may be -1 at the tie x = 0, which JAX never gives
    m = export.to_scip(f, [-1.0], [1.0], strict_eps=1e-4)
    assert "restricted" in m.data.report.approximated["gt"][1]
    _, x, obj = _solve(m)
    _check(f, x, obj)
    assert abs(obj + 0.5) < 1e-6


def test_sign_strict_eps():
    def f(x):
        return -(jnp.sign(x[0]) ** 2)  # -1 except f(0) = 0

    m = export.to_scip(f, [-1.0], [1.0], strict_eps=1e-4)
    assert "sign" in m.data.report.approximated
    _, x, obj = _solve(m)
    _check(f, x, obj)


def test_cumulative_ops():
    def f(x):
        return jnp.cumsum(x)[-1] ** 2 + jnp.cumprod(x)[1] + jax.lax.cummax(x)[-1]

    _, x, obj = _solve(export.to_scip(f, [-1.0] * 3, [1.0] * 3))
    _check(f, x, obj)


def test_precision_casts_reported():
    def f(x):
        return jax.lax.reduce_precision(x[0], exponent_bits=5, mantissa_bits=10) + x.astype(jnp.float32)[1]

    r = export.inspect(f, [-1.0] * 2, [1.0] * 2)
    if jnp.zeros(1).dtype == jnp.float64:  # with x64 the cast is a real f64 -> f32 conversion
        assert any("float32" in k for k in r.approximated)
    assert "reduce_precision" in r.approximated


def test_zero_times_partial_keeps_domain():
    r = export.inspect(lambda x: 0.0 * jnp.sqrt(x[0]) + x[1], [-1.0] * 2, [1.0] * 2)
    assert r.exact["sqrt"] == 1  # not folded to 0 (JAX: 0 * nan = nan)
    r = export.inspect(lambda x: 0.0 * x[0] + x[1], [-1.0] * 2, [1.0] * 2)
    assert r.exact["mul"] == 0


def test_product_expansion_is_lifted():
    m = export.to_scip(lambda x: jnp.prod(1.0 + x), [0.0] * 20, [1.0] * 20)
    assert "product operand" in m.data.report.reformulated
    _, x, obj = _solve(m)
    assert abs(obj - 1.0) < 1e-6


def test_asinh_large_negative():
    with pytest.warns(export.ExportBigMWarning):  # sign(x) log(|x| + sqrt(x^2 + 1)), |x| up to 1e8
        m = export.to_scip(lambda x: jnp.arcsinh(x[0]), [-1e8], [1.0])
    _, x, obj = _solve(m)
    assert abs(obj - np.arcsinh(-1e8)) < 1e-6


# exp overflow: SCIP treats values >= 1e20 as invalid evaluations and cuts those points off


@pytest.mark.parametrize(
    "f, lo, hi, fstar",
    [
        (lambda x: jnp.tanh(x[0]) * x[1], [25.0, 1.0], [30.0, 2.0], 1.0),
        (lambda x: jax.nn.sigmoid(x[0]) - 0.001 * x[0], [-60.0], [-50.0], 0.05),
        (lambda x: jax.nn.sigmoid(-x[0] * x[1]) + 0.001 * x[0], [20.0, 20.0], [30.0, 30.0], 0.02),
        (lambda x: jnp.sin(jnp.tanh(x[0] ** 2)) + 0.001 * x[0], [20.0], [30.0], np.sin(1.0) + 0.02),
    ],
)
def test_tanh_sigmoid_no_exp_overflow(f, lo, hi, fstar):
    _, x, obj = _solve(export.to_scip(f, lo, hi))
    _check(f, x, obj)
    assert abs(obj - fstar) < 1e-6, obj


def test_tanh_sigmoid_straddling_forms(tmp_path):
    """Over a box that straddles 0 with large |x| the overflow-free forms are used; they equal JAX everywhere."""

    def f(x):
        return jnp.tanh(x[0]) + jax.nn.sigmoid(x[1]) + jnp.tanh(-2.0 * x[1])

    lo, hi = [-400.0, -400.0], [400.0, 400.0]
    path = tmp_path / "s.nl"
    r = export.to_nl(f, lo, hi, path, portable=True)
    assert "overflow" in r.rewritten["tanh"][1] and "overflow" in r.rewritten["logistic"][1]
    m = read_nl(str(path))
    for x in ([-400.0, 400.0], [-30.0, 25.0], [-1e-3, 2e-3], [0.0, 0.0], [0.7, -0.4], [60.0, -60.0], [400.0, -50.0]):
        x = jnp.array(x)
        assert abs(float(m.f(x)) - float(f(x))) < 1e-14, x


def test_exp_overflow_refused():
    with pytest.raises(ExportError, match="SCIP's infinity"):
        export.to_scip(lambda x: jnp.exp(x[0]), [0.0], [60.0])
    with pytest.raises(ExportError, match="SCIP's infinity"):
        export.to_scip(lambda x: jnp.expm1(x[0]) * x[1], [0.0, 0.0], [50.0, 1.0])
    _, x, obj = _solve(export.to_scip(lambda x: jnp.exp(x[0]), [0.0], [40.0]))
    assert abs(obj - 1.0) < 1e-6


def test_asinh_acosh_large_arguments(tmp_path):
    _, x, obj = _solve(export.to_scip(lambda x: jnp.arcsinh(x[0]), [1e11], [1e12]))  # x^2 would reach 1e24
    assert abs(obj - np.arcsinh(1e11)) < 1e-6
    r = export.inspect(lambda x: jnp.arcsinh(x[0]) + jnp.arccosh(x[1]), [-1e12, 1.0], [-1.0, 1e12])
    assert not r.refused
    assert "-(log(-x)" in r.rewritten["asinh"][1] and "x^-2" in r.rewritten["acosh"][1]
    path = tmp_path / "a.nl"
    export.to_nl(lambda x: jnp.arcsinh(x[0]) + jnp.arccosh(x[1]), [-1e12, 1.0], [-1.0, 1e12], path, portable=True)
    m = read_nl(str(path))
    for x in ([-1e12, 1e12], [-3.0, 1.5], [-1.0, 1.0]):
        x = jnp.array(x)
        assert abs(float(m.f(x)) - float(jnp.arcsinh(x[0]) + jnp.arccosh(x[1]))) < 1e-12


def test_asinh_label_and_inspect_target():
    r = export.inspect(lambda x: jnp.arcsinh(x[0]), [-1e5], [1.0])
    assert r.rewritten["asinh"][1].startswith("sign(x)")
    r = export.inspect(lambda x: jnp.arcsinh(x[0]), [-1e5], [-10.0])
    assert "-(log(-x)" in r.rewritten["asinh"][1]
    with pytest.raises(ValueError, match="target"):
        export.inspect(lambda x: x[0], [0.0], [1.0], target="baron")


@pytest.mark.parametrize(
    "f",
    [
        lambda x: jnp.log(1 + x @ x),
        lambda x: jnp.log1p(x @ x),
        lambda x: jnp.log(1 + x[0] * x[0]),
        lambda x: jnp.log(1.5 + x[0] * x[1] - x[0] * x[1]),
        lambda x: jnp.sqrt(1 + x @ x) + (1 + x[0] * x[0]) ** 1.5,
        lambda x: 1.0 / (x[0] - 0.5) ** 2 + x[1],  # a pole (division by an interval containing 0) is not an overflow
    ],
)
def test_overflow_check_ignores_poles_and_domain_edges(tmp_path, f):
    """Interval bounds of x*x lose the dependency, so log's operand can look negative; that is not an overflow."""
    lo, hi = [-2.0] * 3, [2.0] * 3
    for target, kw in (("scip", {}), ("nl", {}), ("nl", {"portable": True}), ("nl", {"nonsmooth": "reformulate"})):
        assert not export.inspect(f, lo, hi, target=target, **kw).refused
    export.to_scip(f, lo, hi)
    export.to_nl(f, lo, hi, tmp_path / "l.nl")


def test_division_overflow_refused():
    # the divisor stays above 0 but gets within 1e-30 of it: -x / (x^2 + 1e-30) reaches -5e14 near x = 1e-15
    with pytest.raises(ExportError, match="'div' reaches"):
        export.to_scip(lambda x: -x[0] / (x[0] * x[0] + 1e-30), [0.0], [1.0])
    _, x, obj = _solve(export.to_scip(lambda x: -1e-12 / (x[0] * x[0] + 1e-24), [0.0], [1.0]))  # 1e12: fine
    assert abs(obj + 1e12) < 1e-3


def test_variable_power_overflow_refused(tmp_path):
    """x ** y with a finite bound >= 1e20 overflows in the solver (Couenne returned -3e-12 for a true -1e5)."""

    def f(x):
        return -1e-25 * x[0] ** x[1]  # x0 ** x1 reaches 1e30

    lo, hi = [1.0, 0.0], [1e3, 10.0]
    for kw in ({}, {"portable": True}, {"nonsmooth": "reformulate"}):
        with pytest.raises(ExportError, match="'pow' reaches"):
            export.to_nl(f, lo, hi, tmp_path / "p.nl", **kw)
    with pytest.raises(ExportError, match="reaches"):
        export.to_scip(f, lo, hi)
    # controls: 1e18 exports; an infinite bound (x can be 0, a domain edge) is not an overflow
    path = tmp_path / "c.nl"
    export.to_nl(lambda x: -1e-13 * x[0] ** x[1], lo, [1e3, 6.0], path, portable=True)
    m = read_nl(str(path))
    assert abs(float(m.f(jnp.array([1e3, 6.0]))) + 1e5) < 1e-6
    assert not export.inspect(lambda x: x[0] ** x[1], [0.0, -1.0], [2.0, 1.0], target="nl").refused

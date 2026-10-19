"""FBBT must only apply domain rules to values that reach the outputs.

* ``jnp.log(x)[0]`` computes ``log`` of every element of ``x`` but uses only element 0: log's
  domain must not narrow ``x[1]`` (``x[1] >= 0``) nor report a box with ``x[1] < 0`` empty.
* ``jnp.sqrt(x) > 0.5`` is defined (False) where ``sqrt(x)`` is NaN: sqrt's domain must not
  narrow ``x`` either.
* CROWN's interval pass leaves an empty untaken ``jnp.where`` arm unbounded, as FBBT does.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize
from jaxglobal.crown import _plan_of, make_crown
from jaxglobal.fbbt import guarded_slots, joint_tape, make_fbbt, value_live

W = jnp.where
DEAD = {
    "unused element": lambda x: (x[1] + 0.5) ** 2 + jnp.log(x)[0],
    "unused element, decided where": lambda x: (x[1] + 0.5) ** 2 + W(x[0] > 0.5, jnp.log(x)[0], 0.0),
    "unused element, sqrt": lambda x: (x[1] + 0.5) ** 2 + jnp.sum(jnp.sqrt(x)[:1]),
    "unused element, gather": lambda x: (x[1] + 0.5) ** 2 + jnp.sum(jnp.log(x)[jnp.array([0, 0])]),
    "unused element, broadcast": lambda x: (x[1] + 0.5) ** 2 + jnp.sum(jnp.log(jnp.broadcast_to(x, (3, 2)))[:, 0]),
    "unused element, cumsum": lambda x: (x[1] + 0.5) ** 2 + jnp.cumsum(jnp.log(x))[0],
    "overwritten element": lambda x: (x[1] + 0.5) ** 2
    + jnp.sum(lax.dynamic_update_slice(jnp.log(x), jnp.zeros(1), (1,))),
}
modes = [pytest.param(False, id="plain"), pytest.param(True, id="safe")]
interps = [pytest.param(False, id="traced"), pytest.param(True, id="interp")]


@pytest.mark.parametrize("safe", modes)
@pytest.mark.parametrize("interpret", interps)
@pytest.mark.parametrize("name", list(DEAD))
def test_unused_elements_do_not_narrow(name, interpret, safe):
    f = DEAD[name]
    fb = make_fbbt(f, None, None, interpret=interpret, safe=safe)
    lo, hi, e = fb(jnp.array([0.6, -1.0]), jnp.array([1.0, 1.0]))
    assert not bool(e)
    assert float(lo[1]) <= -1.0 + 1e-9  # x1 is free of log's domain
    lo, hi, e = fb(jnp.array([0.6, -1.0]), jnp.array([1.0, -0.2]))
    assert not bool(e)  # optimum at x1 = -0.5
    assert float(lo[1]) <= -1.0 + 1e-9 and float(hi[1]) >= -0.2 - 1e-9
    if "where" not in name and "overwritten" not in name and "cumsum" not in name:  # the used element still gets the domain: x0 > 0
        lo, hi, e = fb(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))
        assert not bool(e) and float(lo[0]) >= -1e-9


@pytest.mark.parametrize("safe", modes)
def test_unused_element_minimize(safe):
    r = minimize(DEAD["unused element"], np.array([0.6, -1.0]), np.array([1.0, -0.2]), time_limit=30, safe=safe)
    assert r.status == "optimal"
    assert abs(r.fun - np.log(0.6)) < 1e-6 and abs(r.x[1] + 0.5) < 1e-3


NAN_PRED = {
    "sqrt > c": lambda x: (x[0] + 0.5) ** 2 + W(jnp.sqrt(x[0]) > 0.5, 1.0, 0.0) + 0.0 * x[1],
    "log < c": lambda x: (x[0] + 0.5) ** 2 + W(jnp.log(x[0]) < -0.5, 1.0, 0.0) + 0.0 * x[1],
    "log1p(x)*x >= c": lambda x: (x[0] + 0.5) ** 2 + W(jnp.log1p(x[0]) * x[1] >= 0.1, 1.0, 0.0),
    # one ``sqrt`` slot (CSE) both compared and taken as an arm
    "where(sqrt > c, sqrt, 0)": lambda x: (x[0] + 0.5) ** 2 + W(jnp.sqrt(x[0]) > 0.5, jnp.sqrt(x[0]), 0.0)
    + 0.0 * x[1],
    "where(log < c, log, 0)": lambda x: (x[0] + 0.5) ** 2 + W(jnp.log(x[0]) < -0.5, jnp.log(x[0]), 0.0)
    + 0.0 * x[1],
    "nan_to_num(sqrt)": lambda x: (x[0] + 0.5) ** 2 + jnp.nan_to_num(jnp.sqrt(x[0])) + 0.0 * x[1],
    "int cast of sqrt": lambda x: (x[0] + 0.5) ** 2 + jnp.sqrt(x[0]).astype(jnp.int32) + 0.0 * x[1],
    "sqrt ** 0": lambda x: (x[0] + 0.5) ** 2 + jnp.sqrt(x[0]) ** 0 + 0.0 * x[1],
}


@pytest.mark.parametrize("safe", modes)
@pytest.mark.parametrize("interpret", interps)
@pytest.mark.parametrize("name", list(NAN_PRED))
def test_domain_inside_comparison_does_not_cut(name, interpret, safe):
    f = NAN_PRED[name]
    fb = make_fbbt(f, None, None, interpret=interpret, safe=safe)
    for lo0, hi0 in [((-1.0, -1.0), (3.0, 3.0)), ((-2.0, -1.0), (-1.1, 1.0))]:
        lo, hi, e = fb(jnp.array(lo0), jnp.array(hi0))
        assert not bool(e)
        np.testing.assert_array_equal(np.asarray(lo), lo0)
        np.testing.assert_array_equal(np.asarray(hi), hi0)


@pytest.mark.parametrize("safe", modes)
def test_domain_inside_comparison_minimize(safe):
    # f(x) = -1 + 0.1 x0 at x0 < 0 (log(x0) is NaN, the comparison False): minimum -1.2 at x0 = -2
    f = lambda x: W(jnp.log(x[0]) < -0.5, 5.0 + x[0], -1.0 + 0.1 * x[0]) + x[1] ** 2  # noqa: E731
    r = minimize(f, np.array([-2.0, -1.0]), np.array([-0.5, 1.0]), time_limit=30, safe=safe)
    assert r.status == "optimal" and abs(r.fun + 1.2) < 1e-6


@pytest.mark.parametrize("safe", modes)
def test_compared_arm_minimize(safe):
    # the compared ``log(x0)`` is also an arm: it must not make x0 < 0 infeasible. (The lower
    # bound need not close: the relaxation does not decide ``log(x0) < c`` False where x0 < 0.)
    f = lambda x: W(jnp.log(x[0]) < -0.5, 5.0 + jnp.log(x[0]), -1.0 + 0.1 * x[0]) + x[1] ** 2  # noqa: E731
    # a node cap, not a time limit: on a slow runner 5 s can pass before the first incumbent
    r = minimize(f, np.array([-2.0, -1.0]), np.array([-0.5, 1.0]), max_iter=20, safe=safe)
    assert r.status in ("optimal", "max_iter") and abs(r.fun + 1.2) < 1e-6


@pytest.mark.parametrize("safe", modes)
@pytest.mark.parametrize("interpret", interps)
def test_compared_arm_keeps_tight_bounds(interpret, safe):
    # in the domain, the taken arm still narrows: (x1+0.5)^2 + sqrt(x0) <= 0.9 on x0 in [0.3, 1]
    f = lambda x: (x[1] + 0.5) ** 2 + W(jnp.sqrt(x[0]) > 0.5, jnp.sqrt(x[0]), 0.0)  # noqa: E731
    fb = make_fbbt(f, None, None, interpret=interpret, safe=safe)
    lo, hi, e = fb(jnp.array([0.3, -1.0]), jnp.array([1.0, 1.0]), 0.9)
    assert not bool(e)
    np.testing.assert_allclose(np.asarray(hi), [0.81, np.sqrt(0.9 - np.sqrt(0.3)) - 0.5], atol=1e-6)


@pytest.mark.parametrize("safe", modes)
@pytest.mark.parametrize("interpret", interps)
def test_integer_cast_does_not_narrow(interpret, safe):
    # int(x0) <= 0 holds on x0 < 1: the cast is not the identity backward
    fb = make_fbbt(lambda x: x[0] ** 2 + x[1] ** 2, lambda x: jnp.atleast_1d(x[0].astype(jnp.int32)).astype(float),
                   None, interpret=interpret, safe=safe)  # fmt: skip
    lo, hi, e = fb(jnp.array([-2.0, -1.0]), jnp.array([2.0, 1.0]))
    assert not bool(e) and float(hi[0]) >= 1.0 - 1e-9


@pytest.mark.parametrize("name", ["cumsum", "dot"])
def test_fully_used_operations_still_narrow(name):
    g = {"cumsum": lambda v: jnp.sum(jnp.cumsum(v)), "dot": lambda v: jnp.sum(jnp.array([[1.0, 2.0], [3.0, 0.0]]) @ v)}
    fb = make_fbbt(lambda x: g[name](jnp.log(x)), None, None)
    lo, hi, e = fb(jnp.array([-1.0, 0.1]), jnp.array([1.0, 1.0]))
    assert not bool(e) and float(lo[0]) >= -1e-9


def test_value_live_masks():
    x = jnp.zeros(2)
    tape = joint_tape(DEAD["unused element"], None, None, x)
    live = value_live(tape)
    logs = [n.outs[0] for n in tape.nodes if n.eqn.primitive.name == "log"]
    assert live[logs[0]].tolist() == [True, False]
    assert logs[0] in guarded_slots(tape)
    tape = joint_tape(NAN_PRED["sqrt > c"], None, None, x)
    (s,) = [n.outs[0] for n in tape.nodes if n.eqn.primitive.name == "sqrt"]
    assert not value_live(tape).get(s, np.zeros((), bool)).any()  # reaches f only through ``>``
    assert s in guarded_slots(tape)
    # a tape without where, unused elements or comparisons has no guarded slot
    tape = joint_tape(lambda x: jnp.sum(jnp.log(x)) + x[0] * x[1], None, None, x)
    assert not guarded_slots(tape)


@pytest.mark.parametrize("safe", modes)
def test_crown_forward_uses_guarded_slots(safe):
    # CROWN handles a where with a constant predicate and unused elements
    def f(x):
        return (x[0] + 0.5) ** 2 + W(jnp.array(False), jnp.log(x[0]), 0.0) + jnp.log(x)[1]

    x = jnp.zeros(2)
    tape, plan, guarded = _plan_of(f)(x)
    assert plan is not None
    assert guarded == guarded_slots(tape) and guarded
    bound = make_crown(f, safe=safe)
    lo, hi = np.array([-1.0, 0.5]), np.array([-0.2, 2.0])
    lb = float(bound(jnp.asarray(lo), jnp.asarray(hi)))
    X = np.random.default_rng(0).uniform(lo, hi, (20000, 2))
    F = np.asarray(jax.vmap(f)(jnp.asarray(X)))
    assert np.isfinite(lb) and lb <= F.min() + 1e-9

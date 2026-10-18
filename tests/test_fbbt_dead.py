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
    if "where" not in name:  # the used element still gets the domain: x0 > 0
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
}


@pytest.mark.parametrize("safe", modes)
@pytest.mark.parametrize("name", list(NAN_PRED))
def test_domain_inside_comparison_does_not_cut(name, safe):
    f = NAN_PRED[name]
    fb = make_fbbt(f, None, None, interpret=False, safe=safe)
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

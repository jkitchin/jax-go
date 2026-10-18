"""FBBT through ``jnp.where`` (``select_n``) must only narrow an arm's operands where that arm is taken.

``jnp.where(x > 0.5, jnp.log(x), 0.0)`` evaluates ``log(x)`` everywhere but uses it only where
``x > 0.5``. Neither log's domain (``x > 0``) nor an empty interval of the arm at ``x < 0`` may
cut off the points ``x < 0``, where the other arm is taken.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import minimize
from jaxglobal.fbbt import guarded_slots, joint_tape, make_fbbt
from jaxglobal.fbbt_interp import make_propagate

ARMS = {
    "log": lambda x: jnp.where(x > 0.5, jnp.log(x), 0.0),
    "sqrt": lambda x: jnp.where(x > 0, jnp.sqrt(x), 0.0),
    "log1p": lambda x: jnp.where(x > 1.5, jnp.log1p(x - 1.0), 0.0),
    "rsqrt": lambda x: jnp.where(x > 0.5, jax.lax.rsqrt(x), 0.0),
    "pow": lambda x: jnp.where(x > 0, x**1.5, 0.0),
    "xlogx": lambda x: jnp.where(x > 0.5, x * jnp.log(x), 0.0),
    "double-where log": lambda x: jnp.where(x > 0.5, jnp.log(jnp.where(x > 0.5, x, 1.0)), 0.0),
    "double-where sqrt": lambda x: jnp.where(x > 0, jnp.sqrt(jnp.where(x > 0, x, 1.0)), 0.0),
}


def objective(arm):
    # minimum 0 at x = -0.5, where the guarded arm is not taken
    return lambda x: (x[0] + 0.5) ** 2 + arm(x[0]) + 0.0 * x[1]


@pytest.mark.parametrize("safe", [False, True], ids=["plain", "safe"])
@pytest.mark.parametrize("interpret", [False, True], ids=["traced", "interp"])
@pytest.mark.parametrize("name", list(ARMS))
def test_where_arm_does_not_cut_untaken_points(name, interpret, safe):
    f = objective(ARMS[name])
    fb = make_fbbt(f, None, None, interpret=interpret, safe=safe)
    lo, hi = np.array([-1.0, -1.0]), np.array([3.0, 3.0])
    X = np.random.default_rng(0).uniform(lo, hi, (20000, 2))
    F = np.asarray(jax.vmap(f)(jnp.asarray(X)))
    for cutoff in (np.inf, 1.0, 0.3, 0.01):
        a, b, e = fb(jnp.asarray(lo), jnp.asarray(hi), cutoff)
        keep = X[F <= cutoff]
        assert keep.size and not bool(e), (cutoff, bool(e))
        assert np.all(keep >= np.asarray(a) - 1e-9) and np.all(keep <= np.asarray(b) + 1e-9), (cutoff, a, b)
    # a box entirely outside the arm's domain is feasible: the other arm is taken there
    a, b, e = fb(jnp.array([-2.0, -1.0]), jnp.array([-1.0, 1.0]))
    assert not bool(e) and np.allclose(a, [-2.0, -1.0]) and np.allclose(b, [-1.0, 1.0])


@pytest.mark.parametrize("safe", [False, True], ids=["plain", "safe"])
def test_where_decided_predicate_still_narrows(safe):
    """Where the predicate is decided on the box, the taken arm narrows as before (domain included)."""
    f = lambda x: jnp.where(x[1] > 0, jnp.log(x[0]), 0.0)  # noqa: E731
    fb = make_fbbt(f, None, None, interpret=False, safe=safe)
    a, _, e = fb(jnp.array([-1.0, 0.5]), jnp.array([3.0, 2.0]))
    assert not bool(e) and a[0] >= 0.0
    # and its interval: log(x0) <= -1 on the taken arm gives x0 <= e^-1
    _, b, e = fb(jnp.array([-1.0, 0.5]), jnp.array([3.0, 2.0]), -1.0)
    assert not bool(e) and b[0] <= np.exp(-1.0) + 1e-6
    # element by element: only x1 (where x1 > 0.5 on the box) is narrowed by log(x1) <= 0.2
    fb = make_fbbt(lambda x: 0.0 * x[0], lambda x: jnp.where(x > 0.5, jnp.log(x), 0.0) - 0.2, None, interpret=False,
                   safe=safe)  # fmt: skip
    a, b, e = fb(jnp.array([-1.0, 0.6, -1.0]), jnp.array([3.0, 3.0, 0.4]))
    assert not bool(e) and np.allclose(a, [-1.0, 0.6, -1.0]) and np.allclose(b, [3.0, np.exp(0.2), 0.4], atol=1e-6)
    # an arm empty on the box when it is the one taken: infeasible
    _, _, e = make_fbbt(lambda x: x[0] + jnp.where(x[1] > 0, jnp.sqrt(x[0]), 0.0), None, None, interpret=False,
                        safe=safe)(jnp.array([-2.0, 0.5]), jnp.array([-1.0, 2.0]))  # fmt: skip
    assert bool(e)


@pytest.mark.parametrize("safe", [False, True], ids=["plain", "safe"])
def test_double_where_stays_tight(safe):
    """The double-where idiom narrows (finitely) once the box decides the predicate."""
    g = ARMS["double-where log"]
    fb = make_fbbt(lambda x: 0.0 * x[0], lambda x: jnp.atleast_1d(0.5 - g(x[0])), None, interpret=False, safe=safe)
    a, b, e = fb(jnp.array([0.6]), jnp.array([3.0]))
    assert not bool(e) and abs(float(a[0]) - np.exp(0.5)) < 1e-5 and float(b[0]) == 3.0
    r = minimize(objective(g), np.array([-1.0, 0.0]), np.array([3.0, 0.0]), time_limit=30, safe=safe)
    assert r.status == "optimal" and np.isfinite(r.lower_bound) and abs(r.fun) < 1e-6 and r.lower_bound <= 1e-6


@pytest.mark.parametrize("safe", [False, True], ids=["plain", "safe"])
def test_where_sqrt_optimum_below_domain(safe):
    """End to end: the optimum x = -2 lies where the sqrt arm is not taken (main returned -0.446)."""
    f = lambda x: x[0] + jnp.where(x[0] > 0, jnp.sqrt(x[0]), 0.0)  # noqa: E731
    r = minimize(f, np.array([-2.0]), np.array([2.0]), time_limit=30, safe=safe)
    assert r.status == "optimal" and r.lower_bound <= -2.0 + 1e-6 and abs(r.fun + 2.0) < 1e-6


def test_guarded_slots_and_interpreter():
    """Only slots reached through a ``select_n`` case are guarded; the interpreter declines such tapes,
    so they always take the traced path."""
    x = jnp.zeros(2)
    tape = joint_tape(lambda x: jnp.log(x[1]) + jnp.where(x[0] > 0.5, jnp.log(x[0]), 0.0), None, None, x)
    names = {s: n.eqn.primitive.name for n in tape.nodes for s in n.outs}
    g = guarded_slots(tape)
    # log(x[0]) is guarded; x[0] itself also feeds the predicate and log(x[1]) the sum, so they are not
    assert [names[s] for s in g].count("log") == 1 and not any(names[s] == "squeeze" for s in g)
    assert make_propagate(tape, None, None, 2, 1e-6, 5) is None
    assert not guarded_slots(joint_tape(lambda x: jnp.log(x[0]) + x[1], None, None, x))

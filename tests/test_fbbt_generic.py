"""The shared FBBT kernel (``fbbt_generic``): same bounds as the row interpreter, one executable per bucket."""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal import fbbt_generic as G
from jaxglobal.fbbt import joint_tape
from jaxglobal.fbbt_interp import make_propagate
from jaxglobal.nl import read_nl

DATA = pathlib.Path(__file__).resolve().parent / "data"


def _boxes(lo, hi, k, seed=0):
    """The box itself and ``k - 1`` random sub-boxes of it (infinite bounds clipped to +-1e3)."""
    lo, hi = np.maximum(lo, -1e3), np.minimum(hi, 1e3)
    rng = np.random.default_rng(seed)
    w = hi - lo
    a = lo + rng.uniform(0, 0.3, (k - 1, lo.size)) * w
    b = lo + rng.uniform(0.4, 1.0, (k - 1, lo.size)) * w
    return np.vstack([lo, a]), np.vstack([hi, np.maximum(a, b)])


def _compare(f, ineq, eq, lo, hi, integer=(), cutoff=np.inf, safe=False, rounds=5):
    n = lo.size
    g = G.make_generic_fbbt(f, ineq, eq, n=n, rounds=rounds, feas_tol=1e-6, safe=safe, integer=integer)
    assert g is not None
    with jax.ensure_compile_time_eval():
        tape = joint_tape(f, ineq, eq, np.zeros(n))
    ref = make_propagate(tape, ineq, eq, n, 1e-6, rounds, safe, scheduled=False, integer=integer)
    L, H = _boxes(lo, hi, 8)
    rl, rh, re = (np.asarray(v) for v in jax.vmap(ref, in_axes=(0, 0, None))(jnp.asarray(L), jnp.asarray(H), cutoff))
    gl, gh, ge = (np.asarray(v) for v in g(L, H, cutoff))
    np.testing.assert_array_equal(ge, re)
    ok = ~re
    assert ok.any(), "no box survived: the test checks nothing"
    np.testing.assert_array_equal(gl[ok], rl[ok])  # the same rules in the same order: equal, not close
    np.testing.assert_array_equal(gh[ok], rh[ok])
    return g


@pytest.mark.parametrize("name", ["nvs01", "m3", "clay0203m", "ex6_2_6"])
def test_same_bounds_as_the_row_interpreter(name):
    M = read_nl(DATA / f"{name}.nl", integers="mask")
    integer = tuple(np.flatnonzero(M.integer))
    _compare(M.f, M.ineq, M.eq, np.asarray(M.lo, float), np.asarray(M.hi, float), integer=integer)


def test_cutoff_safe_and_the_full_table():
    """A code-defined model with an objective cutoff, in safe mode, with opcodes outside ``CORE``."""
    f = lambda x: jnp.sin(3 * x[0]) + jnp.tanh(x[1]) * x[0] + x[1] ** 2  # noqa: E731
    g = lambda x: jnp.array([x[0] + x[1] - 1.0, jnp.exp(x[0]) - 3.0])  # noqa: E731
    lo, hi = np.array([-2.0, -2.0]), np.array([2.0, 2.0])
    for safe in (False, True):
        gen = _compare(f, g, None, lo, hi, cutoff=0.5, safe=safe)
        assert gen.g.tier == "full"


def test_one_executable_per_bucket():
    """Two different problems of the same size bucket use one compiled kernel."""
    f1 = lambda x: x[0] * x[1] + x[2] ** 2  # noqa: E731
    f2 = lambda x: (x[0] - x[2]) * 3.0 + x[1] ** 2 - x[0] * x[2]  # noqa: E731
    lo, hi = -np.ones(3), 2 * np.ones(3)
    a = G.make_generic_fbbt(f1, None, None, n=3)
    b = G.make_generic_fbbt(f2, None, None, n=3)
    assert a.g.tier == b.g.tier == "core"
    a(lo[None], hi[None], 1.0)
    before = len(G._EXES)
    b(lo[None], hi[None], 1.0)
    assert len(G._EXES) == before  # no new executable for the second problem


def test_unsupported_opcode_falls_back():
    """An exponent outside ``POWERS`` has no opcode: the caller keeps its own kernel."""
    assert G.make_generic_fbbt(lambda x: x[0] ** 11 + x[1], None, None, n=2) is None


def test_stops_at_an_exact_fixpoint():
    """More rounds than needed change nothing (the loop stops when a round moves no bound)."""
    f = lambda x: x[0] * x[1]  # noqa: E731
    g = lambda x: jnp.array([x[0] + x[1] - 1.0])  # noqa: E731
    lo, hi = np.array([0.0, 0.0]), np.array([4.0, 4.0])
    r5 = G.make_generic_fbbt(f, g, None, n=2, rounds=5)(lo[None], hi[None], 0.1)
    r50 = G.make_generic_fbbt(f, g, None, n=2, rounds=50)(lo[None], hi[None], 0.1)
    for a, b in zip(r5, r50):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

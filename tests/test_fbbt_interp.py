"""The FBBT tape interpreter (jaxglobal/fbbt_interp.py) against fbbt.propagate."""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal.fbbt import joint_tape, make_fbbt
from jaxglobal.fbbt_interp import make_propagate, scalarize

NL = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "minlplib" / "nl"


def scalar_model():
    """A scalar, .nl-like model: products, sums, powers, exp, sqrt, a constraint and an equality."""

    def f(x):
        a, b, c = x[0], x[1], x[2]
        return a * b + jnp.exp(0.3 * c) - 2.0 * a**2 + jnp.sqrt(1.0 + b * b) - c * c

    def g(x):
        return jnp.stack([x[0] + x[1] * x[2] - 1.5, x[0] * x[0] - 2.0 * x[1] - 0.5])

    def h(x):
        return jnp.atleast_1d(x[0] - x[1] + 0.25 * x[2] - 0.1)

    return f, g, h


@pytest.fixture(params=["rows", "steps"])
def sweep(request, monkeypatch):
    """Run a test with the row-by-row and the scheduled (step-by-step) sweeps."""
    import jaxglobal.fbbt_interp as FI

    monkeypatch.setattr(FI, "SCHEDULE_MIN_ROWS", 10**9 if request.param == "rows" else 0)
    return request.param


@pytest.mark.parametrize("cutoff", [np.inf, 0.5, -1.0])
def test_never_looser_than_propagate(cutoff, sweep):
    """At least as tight as propagate (structural equations are aliases instead of separate,
    padded nodes, so updates reach every use at once): every box propagate proves empty is
    empty, and no bound is looser beyond rounding. Soundness is checked separately below."""
    f, g, h = scalar_model()
    lo0, hi0 = -2.0 * np.ones(3), 2.0 * np.ones(3)
    ref = jax.jit(jax.vmap(make_fbbt(f, g, h, rounds=5, interpret=False), in_axes=(0, 0, None)))
    new = jax.jit(jax.vmap(make_fbbt(f, g, h, rounds=5, interpret=True), in_axes=(0, 0, None)))
    rng = np.random.default_rng(0)
    A = np.sort(rng.uniform(lo0, hi0, (2, 32, 3)), 0)
    BL, BH = jnp.asarray(A[0]).at[0].set(lo0), jnp.asarray(A[1]).at[0].set(hi0)
    r, q = ref(BL, BH, cutoff), new(BL, BH, cutoff)
    re, qe = np.asarray(r[2]), np.asarray(q[2])
    assert np.all(qe[re])  # every box propagate proves empty
    ok = ~qe
    sc = 1 + np.maximum(np.abs(np.asarray(r[0])), np.abs(np.asarray(r[1])))
    assert np.all((np.asarray(q[0] - r[0]) / sc)[ok] >= -1e-12)
    assert np.all((np.asarray(r[1] - q[1]) / sc)[ok] >= -1e-12)


def test_sound_on_sampled_feasible_points(sweep):
    """Every sampled point with f <= cutoff and g <= 0 lies in the tightened box."""
    f, g, _ = scalar_model()
    fb = jax.jit(make_fbbt(f, g, None, rounds=5, interpret=True))
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (200000, 3))
    F = np.asarray(jax.vmap(f)(jnp.asarray(X)))
    G = np.asarray(jax.vmap(g)(jnp.asarray(X)))
    boxes = [(-2.0 * np.ones(3), 2.0 * np.ones(3))] + [tuple(np.sort(rng.uniform(-2, 2, (2, 3)), 0)) for _ in range(20)]
    for cutoff in (np.inf, 0.0, -1.0):
        for blo, bhi in boxes:
            keep = (F <= cutoff) & np.all(G <= 0, axis=1) & np.all((X >= blo) & (X <= bhi), axis=1)
            lo, hi, empty = fb(jnp.asarray(blo), jnp.asarray(bhi), cutoff)
            if keep.any():
                assert not bool(empty)
                assert np.all(X[keep] >= np.asarray(lo) - 1e-9) and np.all(X[keep] <= np.asarray(hi) + 1e-9)


def test_vectorized_tape_keeps_the_compiled_path():
    """auto mode only interprets mostly scalar tapes."""
    from jaxglobal.fbbt_interp import use_interpreter

    W = jnp.asarray(np.random.default_rng(2).normal(size=(50, 4)))
    tape = joint_tape(lambda x: jnp.sum(jnp.tanh(W @ x) ** 2), None, None, jnp.zeros(4))
    assert not use_interpreter(tape, 4)


@pytest.mark.skipif(not (NL / "st_e37.nl").exists(), reason="MINLPLib instances not downloaded")
def test_nl_model_scalarizes():
    from jaxglobal.nl import read_nl

    M = read_nl(NL / "st_e37.nl")
    tape = joint_tape(M.f, M.ineq, M.eq, jnp.zeros(len(M.lo)))
    prog = scalarize(tape, len(M.lo))
    assert prog is not None and len(prog.op) > 100 and len(prog.opcodes) < 20
    assert make_propagate(tape, M.ineq, M.eq, len(M.lo), 1e-6, 5) is not None


def test_interpreter_is_used_inside_jit_and_vmap(monkeypatch):
    """Scalarizing evaluates structural equations on concrete arrays; inside an enclosing trace
    JAX stages them (omnistaging) unless forced eager, and fbbt silently fell back to the compiled
    path in exactly the kernels it was meant for."""
    import jaxglobal.fbbt_interp as FI

    calls = []
    make = FI.make_propagate

    def spy(*a, **k):
        run = make(*a, **k)
        return None if run is None else (lambda *args: calls.append(1) or run(*args))

    monkeypatch.setattr(FI, "make_propagate", spy)
    f, g, h = scalar_model()
    B = jnp.zeros((4, 3))
    for interpret in (False, True):
        jax.make_jaxpr(jax.jit(jax.vmap(make_fbbt(f, g, h, interpret=interpret), in_axes=(0, 0, None))))(B, B + 1, 0.0)
        assert bool(calls) == interpret

def test_second_trace_does_not_reuse_first_trace_constants(sweep):
    """The interpreter is built lazily inside the first trace and cached for later ones (other
    batch shapes): its constants must be concrete, not tracers of the first trace. Before the
    fix the ahead-of-time compile of a second shape took them as extra inputs ("compiled for
    N inputs but called with 3") and jit raised UnexpectedTracerError."""
    f, g, h = scalar_model()
    fb = jax.jit(jax.vmap(make_fbbt(f, g, h, rounds=3, interpret=True), in_axes=(0, 0, None)))
    one = jax.jit(make_fbbt(f, g, h, rounds=3, interpret=True))
    lo0, hi0 = -2.0 * np.ones(3), 2.0 * np.ones(3)
    ref = np.asarray(one(jnp.asarray(lo0), jnp.asarray(hi0), 0.5)[0])
    for b in (4, 2):
        L, H = jnp.asarray(np.tile(lo0, (b, 1))), jnp.asarray(np.tile(hi0, (b, 1)))
        arg = jax.ShapeDtypeStruct((b, 3), jnp.float64)
        exe = fb.trace(arg, arg, jax.ShapeDtypeStruct((), jnp.float64)).lower().compile()
        out = exe(L, H, 0.5)
        np.testing.assert_allclose(np.asarray(out[0]), np.tile(ref, (b, 1)))
        np.testing.assert_allclose(np.asarray(fb(L, H, 0.5)[0]), np.tile(ref, (b, 1)))

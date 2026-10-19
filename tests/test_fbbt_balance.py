"""fbbt_interp.balance_sums: long add chains (sums read from .nl files) rebuilt as balanced trees."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
import jaxglobal.fbbt_interp as FI
from jaxglobal.fbbt import joint_tape

N = 12
rng0 = np.random.default_rng(3)
C = rng0.uniform(0.1, 1.0, (N, N))
P = rng0.uniform(1.0, 5.0, N)


def chain_model():
    """A scalar portfolio-like model: a quadratic summed term by term into one long chain (as the
    .nl reader writes it), a partial sum that is reused, and a linear budget row."""

    def f(x):
        return x[N]

    def g(x):
        s = 0.0
        for i in range(N):
            for j in range(i, N):
                s = s + C[i, j] * x[i] * x[j]
        part = x[0] + x[1]  # reused below: must stay its own slot
        budget = 0.0
        for i in range(N):
            budget = budget + P[i] * x[i]
        return jnp.stack([s - x[N], 20.0 - budget, budget - 24.0, part * x[2] + part - 3.0])

    return f, g


def program():
    f, g = chain_model()
    tape = joint_tape(f, g, None, jnp.zeros(N + 1))
    return tape, FI._live_rows(FI.scalarize(tape, N + 1))


def _check_topological(prog):
    known = set(range(prog.n)) | set(np.flatnonzero(~np.isnan(prog.const_val)).tolist())
    for a, b, o in zip(prog.a, prog.b, prog.out):
        assert int(a) in known and int(b) in known
        known.add(int(o))
    for o in prog.outs:
        assert set(np.ravel(o).tolist()) <= known


def test_balance_keeps_rows_and_outputs_and_cuts_depth():
    _, prog = program()
    bal = FI.balance_sums(prog)
    assert len(bal.op) == len(prog.op)
    assert sorted(bal.out.tolist()) == sorted(prog.out.tolist())  # same slots, reused
    _check_topological(bal)
    s0, s1 = FI.schedule(prog).steps, FI.schedule(bal).steps
    assert s0 >= N * (N + 1) // 2  # the 78-term chain
    assert s1 <= 20
    assert FI.schedule(FI.balance_sums(bal)).steps <= s1  # a second pass gains nothing


def test_balance_evaluates_the_same_functions():
    """Point evaluation of the rebuilt program equals the chain's (up to rounding)."""
    tape, prog = program()
    bal = FI.balance_sums(prog)
    rng = np.random.default_rng(0)

    def evaluate(p, x):
        v = np.where(np.isnan(p.const_val), 0.0, p.const_val)
        v[: p.n] = x
        names = [oc[0] for oc in p.opcodes]
        for op, a, b, o in zip(p.op, p.a, p.b, p.out):
            nm = names[op]
            if nm in ("add", "add_any"):
                v[o] = v[a] + v[b]
            elif nm == "sub":
                v[o] = v[a] - v[b]
            elif nm == "mul":
                v[o] = v[a] * v[b]
            elif nm == "neg":
                v[o] = -v[a]
            elif nm == "integer_pow":
                v[o] = v[a] ** p.opcodes[op][-1].params["y"]
            else:
                raise AssertionError(nm)
        return np.concatenate([v[np.ravel(o)] for o in p.outs])

    for _ in range(5):
        x = rng.uniform(0, 3, N + 1)
        np.testing.assert_allclose(evaluate(bal, x), evaluate(prog, x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("cutoff", [np.inf, 1.5])
def test_balanced_fbbt_matches_the_chain(cutoff, monkeypatch):
    """The scheduled FBBT sweep gives the same boxes (and empty verdicts) with and without the
    rebuild: interval sums do not depend on the bracketing."""
    f, g = chain_model()
    tape, _ = program()
    integer = tuple(range(N))
    monkeypatch.setattr(FI, "SCHEDULE_MIN_ROWS", 0)
    new = FI.make_propagate(tape, g, None, N + 1, 1e-6, 5, integer=integer)
    monkeypatch.setattr(FI, "balance_sums", lambda p: p)
    old = FI.make_propagate(tape, g, None, N + 1, 1e-6, 5, integer=integer)
    rng = np.random.default_rng(1)
    B = 24
    LO = np.zeros((B, N + 1))
    HI = np.concatenate([rng.integers(1, 12, (B, N)).astype(float), np.full((B, 1), 50.0)], axis=1)
    LO[:, N] = -5.0
    run = lambda fn: [np.asarray(v) for v in jax.jit(jax.vmap(fn, in_axes=(0, 0, None)))(LO, HI, cutoff)]  # noqa: E731
    a, b = run(new), run(old)
    np.testing.assert_array_equal(a[2], b[2])
    ok = ~a[2]
    assert ok.any()
    sc = 1.0 + np.abs(b[0]) + np.abs(b[1])
    assert np.all((np.abs(a[0] - b[0]) / sc)[ok] <= 1e-9)
    assert np.all((np.abs(a[1] - b[1]) / sc)[ok] <= 1e-9)

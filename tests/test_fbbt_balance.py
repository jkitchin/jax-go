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


def slot_values(p, x):
    """Point values of all of ``p``'s slots at ``x``."""
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
    return v


def evaluate(p, x):
    v = slot_values(p, x)
    return np.concatenate([v[np.ravel(o)] for o in p.outs])


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
    for _ in range(5):
        x = rng.uniform(0, 3, N + 1)
        np.testing.assert_allclose(evaluate(bal, x), evaluate(prog, x), rtol=1e-12, atol=1e-12)


def test_balance_drops_stale_ids():
    """``ids`` entries naming a slot the rebuild moved are dropped; the others still name slots
    with the same values."""
    _, prog = program()
    bal = FI.balance_sums(prog)
    assert set(bal.ids) < set(prog.ids)
    x = np.random.default_rng(2).uniform(0, 3, N + 1)
    v0, v1 = slot_values(prog, x), slot_values(bal, x)
    for t, v in bal.ids.items():
        np.testing.assert_allclose(v1[np.asarray(v)], v0[np.asarray(prog.ids[t])], rtol=1e-12, atol=1e-12)


def _propagators(tape, g, h, n, integer=(), monkeypatch=None):
    """The scheduled FBBT sweep of the chain and of the balanced program, the pick forced."""
    monkeypatch.setattr(FI, "SCHEDULE_MIN_ROWS", 0)
    runs = {}
    for tag in ("chain", "bal"):
        pick = (lambda p, *_: (p, FI.schedule(p))) if tag == "chain" else \
            (lambda p, *_: (lambda q: (q, FI.schedule(q)))(FI.balance_sums(p)))  # fmt: skip
        monkeypatch.setattr(FI, "_pick_program", pick)
        runs[tag] = FI.make_propagate(tape, g, h, n, 1e-6, 5, integer=integer)
    assert FI.schedule(runs["bal"].program).steps < FI.schedule(runs["chain"].program).steps
    return runs["bal"], runs["chain"]


def _same_boxes(new, old, LO, HI, cutoff, both=False):
    """Equal empty verdicts (with ``both``: some empty, some not) and equal boxes (up to rounding) where not empty."""
    run = lambda fn: [np.asarray(v) for v in jax.jit(jax.vmap(fn, in_axes=(0, 0, None)))(LO, HI, cutoff)]  # noqa: E731
    a, b = run(new), run(old)
    np.testing.assert_array_equal(a[2], b[2])
    ok = ~a[2]
    assert ok.any() and (a[2].any() or not both)
    for u, v in zip(a[:2], b[:2]):
        u, v = u[ok], v[ok]
        np.testing.assert_array_equal(np.isinf(u), np.isinf(v))
        fin = np.isfinite(v)
        assert np.all(u[~fin] == v[~fin])
        assert np.all(np.abs(u - v)[fin] <= 1e-9 * (1.0 + np.abs(v[fin])))


@pytest.mark.parametrize("cutoff", [np.inf, 1.5])
def test_balanced_fbbt_matches_the_chain(cutoff, monkeypatch):
    """The scheduled FBBT sweep gives the same boxes (and empty verdicts) with and without the
    rebuild: interval sums do not depend on the bracketing."""
    f, g = chain_model()
    tape, _ = program()
    new, old = _propagators(tape, g, None, N + 1, tuple(range(N)), monkeypatch)
    rng = np.random.default_rng(1)
    B = 24
    LO = np.zeros((B, N + 1))
    HI = np.concatenate([rng.integers(1, 12, (B, N)).astype(float), np.full((B, 1), 50.0)], axis=1)
    LO[:, N] = -5.0
    _same_boxes(new, old, LO, HI, cutoff)


def random_model(rng, n=6):
    """Random sums of random terms, each summed into one chain; partial sums taken mid-chain are
    read again by later terms and rows (fan-out out of a chain); inequality and equality rows."""
    kinds = rng.integers(0, 4, 400)
    idx = rng.integers(0, n, (400, 2))
    coef = rng.uniform(-2.0, 2.0, 400)
    lens = rng.integers(12, 30, 5)
    rhs = rng.uniform(-1.0, 3.0, 5)

    def sums(x):
        out, parts, k = [], [], 0
        for L in lens:
            s = 0.0
            for _ in range(L):
                i, j = idx[k]
                kind = kinds[k] if parts else min(kinds[k], 2)
                term = [x[i], x[i] * x[j], x[i] ** 2, None][kind]
                if term is None:  # a partial sum read again, inside another sum
                    term = parts[j % len(parts)] * x[i]
                s = s + coef[k] * term
                if k % 7 == 3:
                    parts.append(s)
                k += 1
            out.append(s)
        return out, parts

    def g(x):
        out, parts = sums(x)
        return jnp.stack([out[0] - rhs[0], out[1] - rhs[1], -out[2] - rhs[2], parts[1] * parts[2] - rhs[3]])

    def h(x):
        out, _ = sums(x)
        return jnp.stack([out[3] - rhs[4], out[4] - out[0]])

    return (lambda x: x[0]), g, h


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_balanced_fbbt_matches_the_chain_randomized(seed, monkeypatch):
    """Random sums with fan-out out of the chains, equality rows: the balanced program's FBBT
    boxes and empty verdicts are the chain's."""
    n = 6
    rng = np.random.default_rng(seed)
    f, g, h = random_model(rng, n)
    tape = joint_tape(f, g, h, jnp.zeros(n))
    new, old = _propagators(tape, g, h, n, (), monkeypatch)
    B = 32
    c = rng.uniform(-2.0, 2.0, (B, n))
    w = rng.uniform(0.05, 2.0, (B, n)) * rng.choice([0.1, 1.0], (B, 1))
    _same_boxes(new, old, c - w, c + w, np.inf, both=True)

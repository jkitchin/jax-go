"""The interpreter thresholds (A3): op-count estimate of the straight-line kernel and the decision rules."""

import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
import jaxglobal.fbbt_interp as FI
import jaxglobal.relax_interp as RI
from jaxglobal.fbbt import joint_tape, make_tape

NL = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "minlplib" / "nl"


def chain(n, terms):
    """A scalar model of ``terms`` shifted products x_i * x_j + 0.5, summed: about 3 rows per term."""

    def f(x):
        out = x[0] * x[1] + 0.5
        for k in range(1, terms):
            out = out + (x[k % n] * x[(k + 1) % n] + 0.5)
        return out

    return f


def test_tape_size_counts_scalar_rows():
    n = 4
    tape = make_tape(chain(n, 5), jnp.zeros(n))
    rows, eqns = FI.tape_size(tape)
    assert rows == eqns and 10 <= rows <= 15
    assert len(FI.scalarize(tape, n).op) == rows


def test_vectorized_rows_count_elements():
    W = jnp.ones((7, 3))
    tape = make_tape(lambda x: jnp.tanh(W @ x), jnp.zeros(3))
    rows, eqns = FI.tape_size(tape)
    assert rows > eqns  # the tanh of 7 elements is 7 rows in one equation


def test_cut_points():
    assert RI.cut_points(3, 4) == 2 * 3 + 1 + 4
    assert RI.cut_points(10, 0) == 21
    assert RI.cut_points(5) == 11 + RI.N_KELLEY


def test_kernel_ops_formula():
    n = 4
    tape = make_tape(chain(n, 5), jnp.zeros(n))
    rows, _ = FI.tape_size(tape)
    assert RI.kernel_ops(tape, n) == rows * RI.ops_per_row(n)
    assert RI.kernel_ops(tape, n, n_kelley=0) == rows * RI.ops_per_row(n, 0)
    assert RI.ops_per_row(n, 0) < RI.ops_per_row(n)


@pytest.mark.parametrize("n", [3, 8, 30])
def test_kernel_ops_grows_with_rows_and_n(n):
    small = RI.kernel_ops(make_tape(chain(n, 5), jnp.zeros(n)), n)
    big = RI.kernel_ops(make_tape(chain(n, 50), jnp.zeros(n)), n)
    assert big > small > 0
    wider = RI.kernel_ops(make_tape(chain(n + 10, 5), jnp.zeros(n + 10)), n + 10)
    assert wider > small


def test_ops_rule_threshold(monkeypatch):
    n = 6
    tape = make_tape(chain(n, 20), jnp.zeros(n))
    ops = RI.kernel_ops(tape, n)
    rows, _ = FI.tape_size(tape)
    monkeypatch.setattr(RI, "RULE", "ops")
    monkeypatch.setattr(RI, "MIN_ROWS", rows + 1)
    monkeypatch.setattr(RI, "MIN_OPS", ops)
    assert RI.use_relax_interpreter(tape, n)
    monkeypatch.setattr(RI, "MIN_OPS", ops + 1)
    assert not RI.use_relax_interpreter(tape, n)
    monkeypatch.setattr(RI, "MIN_ROWS", rows)  # long tapes are interpreted whatever their width
    assert RI.use_relax_interpreter(tape, n)


def test_rows_rule_is_the_former_threshold(monkeypatch):
    """``RULE = "rows"`` is ``use_interpreter(tape, n, min_rows=MIN_ROWS)``, as before A3."""
    monkeypatch.setattr(RI, "RULE", "rows")
    for terms, n in [(5, 4), (40, 4), (300, 10), (400, 30)]:
        tape = make_tape(chain(n, terms), jnp.zeros(n))
        for m in (0, 100, 600, 10**9):
            monkeypatch.setattr(RI, "MIN_ROWS", m)
            assert RI.use_relax_interpreter(tape, n) == FI.use_interpreter(tape, n, min_rows=m)


def test_vectorized_tape_is_never_interpreted(monkeypatch):
    W = jnp.asarray(np.random.default_rng(2).normal(size=(50, 4)))
    tape = make_tape(lambda x: jnp.sum(jnp.tanh(W @ x) ** 2), jnp.zeros(4))
    for rule in ("ops", "rows"):
        monkeypatch.setattr(RI, "RULE", rule)
        monkeypatch.setattr(RI, "MIN_OPS", 0)
        monkeypatch.setattr(RI, "MIN_ROWS", 0)
        assert not RI.use_relax_interpreter(tape, 4)
        assert not FI.use_interpreter(tape, 4)


def test_fbbt_rule(monkeypatch):
    n = 4
    tape = joint_tape(chain(n, 30), None, None, jnp.zeros(n))
    rows, _ = FI.tape_size(tape)
    assert FI.fbbt_ops(rows, 5) == rows * 11
    monkeypatch.setattr(FI, "FBBT_MIN_OPS", FI.fbbt_ops(rows))
    assert FI.use_interpreter(tape, n)
    monkeypatch.setattr(FI, "FBBT_MIN_OPS", FI.fbbt_ops(rows) + 1)
    assert not FI.use_interpreter(tape, n)
    monkeypatch.setattr(FI, "RULE", "rows")
    monkeypatch.setattr(FI, "MIN_ROWS", rows)
    assert FI.use_interpreter(tape, n)
    monkeypatch.setattr(FI, "MIN_ROWS", rows + 1)
    assert not FI.use_interpreter(tape, n)


def test_fbbt_default_matches_former_rule():
    """At minimize's default 5 rounds the FBBT op rule chooses as ``rows >= 50`` did."""
    for terms in (5, 20, 21, 25, 26, 60):
        tape = joint_tape(chain(4, terms), None, None, jnp.zeros(4))
        rows, _ = FI.tape_size(tape)
        assert FI.use_interpreter(tape, 4) == (rows >= 50)


def test_explicit_min_rows_keeps_its_meaning():
    """``min_rows=0`` (lp, bnb, structure) only asks whether the tape is mostly scalar."""
    tape = make_tape(chain(4, 2), jnp.zeros(4))
    assert FI.use_interpreter(tape, 4, min_rows=0)
    assert not FI.use_interpreter(tape, 4, min_rows=10**6)


def _nl_tapes(name):
    from jaxglobal.lp import _as_vec
    from jaxglobal.nl import read_nl

    M = read_nl(NL / f"{name}.nl", integers="mask")
    n = int(np.asarray(M.lo).size)
    return n, {k: make_tape(_as_vec(fn), np.zeros(n)) for k, fn in (("f", M.f), ("g", M.ineq), ("h", M.eq))
               if fn is not None}  # fmt: skip


def test_default_is_the_former_rule():
    """Until the A3 ablation calibrates MIN_OPS, the default relaxer rule is ``rows >= 600``."""
    assert RI.RULE == "rows" and RI.MIN_ROWS == 600
    for terms, n in [(5, 4), (199, 4), (200, 4), (250, 40)]:
        tape = make_tape(chain(n, terms), jnp.zeros(n))
        assert RI.use_relax_interpreter(tape, n) == FI.use_interpreter(tape, n, min_rows=600)


@pytest.mark.skipif(not (NL / "hadamard_5.nl").exists(), reason="MINLPLib instances not downloaded")
def test_ops_rule_on_instances(monkeypatch):
    """The targets of the opt-in op-count rule (docs/ROADMAP.md, A3)."""
    monkeypatch.setattr(RI, "RULE", "ops")
    n, tapes = _nl_tapes("hadamard_5")
    assert RI.use_relax_interpreter(tapes["g"], n)  # 503 rows at n = 26: interpreted
    n, tapes = _nl_tapes("ex6_2_6")
    assert not RI.use_relax_interpreter(tapes["f"], n)  # the interpreter halves its node rate
    n, tapes = _nl_tapes("kriging_peaks-red030")
    assert RI.use_relax_interpreter(tapes["f"], n)  # 1,162 rows at n = 2: minutes to compile straight
    monkeypatch.setattr(RI, "RULE", "rows")
    n, tapes = _nl_tapes("hadamard_5")
    assert not RI.use_relax_interpreter(tapes["g"], n)  # the former rule leaves it straight-line

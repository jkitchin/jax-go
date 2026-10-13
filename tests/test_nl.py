"""Tests for the .nl reader."""

import pathlib
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import minimize
from jaxglobal.nl import NLError, parse_nl, read_nl

jax.config.update("jax_enable_x64", True)

# maximize  -(x0 - 1)^2 - exp(t) + 2 x1,  t = 0.5 x0 + sin(x1)  (defined variable v2)
# s.t.      0 <= x0 * x1 + x0 <= 4  (range),   x0 + x1 == 2  (linear equality)
#           0 <= x0 <= 3, -1 <= x1 <= 2
TEXT = """g3 1 1 0
 2 2 1 1 1
 1 1
 0 0
 2 2 2
 0 0 0 1
 0 0 0 0 0
 4 2
 0 0
 0 0 0 1 0
V2 1 0
0 0.5
o41
v1
C0
o2
v0
v1
C1
n0
O0 1
o0
o16
o5
o1
v0
n1
n2
o16
o44
v2
r
0 0 4
4 2
b
0 0 3
0 -1 2
x2
0 0.5
1 0.25
k1
2
J0 2
0 1
1 0
J1 2
0 1
1 1
G0 1
1 2
"""


def model_obj(x):
    t = 0.5 * x[0] + np.sin(x[1])
    return -((x[0] - 1) ** 2) - np.exp(t) + 2 * x[1]


def test_parse_text():
    M = parse_nl(TEXT)
    assert M.n == 2 and M.sense == -1 and M.bounded
    np.testing.assert_allclose(M.lo, [0, -1])
    np.testing.assert_allclose(M.hi, [3, 2])
    np.testing.assert_allclose(M.x0, [0.5, 0.25])
    assert M.info == dict(m=2, n_eq=1, n_ineq=2, n_defined=1)
    for x in np.random.default_rng(0).uniform([0, -1], [3, 2], (5, 2)):
        xj = jnp.asarray(x)
        np.testing.assert_allclose(M.objective(xj), model_obj(x), rtol=1e-12)
        np.testing.assert_allclose(M.f(xj), -model_obj(x), rtol=1e-12)
        np.testing.assert_allclose(M.bodies(xj), [x[0] * x[1] + x[0], x[0] + x[1]], rtol=1e-12)
        np.testing.assert_allclose(M.eq(xj), [x[0] + x[1] - 2], atol=1e-12)
        np.testing.assert_allclose(M.ineq(xj), [x[0] * x[1] + x[0] - 4, -(x[0] * x[1] + x[0])], atol=1e-12)


def _to_binary(text):
    """Re-encode the test model in the binary .nl format."""
    lines = text.splitlines()
    out = bytearray(("b" + lines[0][1:] + "\n" + "\n".join(lines[1:10]) + "\n").encode())
    i = lambda v: struct.pack("<i", v)  # noqa: E731
    d = lambda v: struct.pack("<d", v)  # noqa: E731
    out += b"V" + i(2) + i(1) + i(0) + i(0) + d(0.5) + b"o" + i(41) + b"v" + i(1)
    out += b"C" + i(0) + b"o" + i(2) + b"v" + i(0) + b"v" + i(1)
    out += b"C" + i(1) + b"n" + d(0.0)
    out += b"O" + i(0) + i(1) + b"o" + i(0) + b"o" + i(16) + b"o" + i(5) + b"o" + i(1) + b"v" + i(0) + b"n" + d(1.0)
    out += b"n" + d(2.0) + b"o" + i(16) + b"o" + i(44) + b"v" + i(2)
    out += b"r" + b"0" + d(0) + d(4) + b"4" + d(2)
    out += b"b" + b"0" + d(0) + d(3) + b"0" + d(-1) + d(2)
    out += b"x" + i(2) + i(0) + d(0.5) + i(1) + d(0.25)
    out += b"k" + i(1) + i(2)
    out += b"J" + i(0) + i(2) + i(0) + d(1) + i(1) + d(0)
    out += b"J" + i(1) + i(2) + i(0) + d(1) + i(1) + d(1)
    out += b"G" + i(0) + i(1) + i(1) + d(2)
    return bytes(out)


def test_binary_matches_text():
    A, B = parse_nl(TEXT), parse_nl(_to_binary(TEXT))
    np.testing.assert_array_equal(A.lo, B.lo)
    np.testing.assert_array_equal(A.x0, B.x0)
    x = jnp.asarray([1.3, 0.4])
    np.testing.assert_allclose(A.objective(x), B.objective(x), rtol=1e-15)
    np.testing.assert_allclose(A.bodies(x), B.bodies(x), rtol=1e-15)


def test_solve_from_nl():
    M = parse_nl(TEXT)
    r = minimize(M.f, M.lo, M.hi, ineq=M.ineq, eq=M.eq, max_iter=300)
    assert r.status == "optimal"
    # reference: dense grid on the equality line x1 = 2 - x0, feasible part of the range constraint
    x0 = np.linspace(0, 3, 300001)
    x = np.stack([x0, 2 - x0])
    ok = (x[1] >= -1) & (x[1] <= 2) & (x[0] * x[1] + x[0] >= 0) & (x[0] * x[1] + x[0] <= 4)
    best = model_obj(x[:, ok]).max()
    assert abs(M.sense * r.fun - best) < 1e-5


def test_unsupported():
    with pytest.raises(NLError):
        parse_nl(TEXT.replace("o41", "o99"))


@pytest.mark.skipif(not (pathlib.Path(__file__).parents[1] / "benchmarks/minlplib/nl/ex4_1_1.nl").exists(),
                    reason="MINLPLib files not downloaded")  # fmt: skip
def test_minlplib_file():
    M = read_nl(pathlib.Path(__file__).parents[1] / "benchmarks/minlplib/nl/ex4_1_1.nl")
    r = minimize(M.f, M.lo, M.hi)
    assert abs(r.fun - -7.487312365) < 1e-5


@pytest.mark.parametrize("field", range(5))
def test_integer_models_are_refused(field):
    """An integer variable in any of the five discrete-variable counts (linear binary, linear
    integer, nonlinear in both / constraints / objectives) is refused: solving the model as
    continuous would report its relaxation's optimum as the model's."""
    counts = [0] * 5
    counts[field] = 1
    text = TEXT.replace(" 0 0 0 0 0\n", " " + " ".join(map(str, counts)) + "\n", 1)
    for data in (text, _to_binary(text)):
        with pytest.raises(NLError, match="integer variables"):
            parse_nl(data)
        assert parse_nl(data, relax_integers=True).n_int == 1


def test_minlplib_integers_refused():
    """nvs01 (MINLPLib): both integers enter nonlinear constraints, so only the nonlinear
    integer counts see them (the linear binary/integer counts are 0)."""
    path = pathlib.Path(__file__).parent / "data" / "nvs01.nl"
    with pytest.raises(NLError, match="integer variables"):
        read_nl(path)
    M = read_nl(path, relax_integers=True)
    assert M.n == 3 and M.n_int == 2

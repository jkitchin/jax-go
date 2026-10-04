"""Nonconvex test problems written as ordinary JAX functions (no equations)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Problem(NamedTuple):
    name: str
    f: callable
    lo: np.ndarray
    hi: np.ndarray
    fstar: float  # known global minimum (None if unknown)
    note: str


def gramacy_lee(x):
    """1-D, many local minima, division by a variable."""
    x = x[0]
    return jnp.sin(10 * jnp.pi * x) / (2 * x) + (x - 1) ** 4


def six_hump_camel(x):
    a, b = x[0], x[1]
    return (4 - 2.1 * a**2 + a**4 / 3) * a**2 + a * b + (-4 + 4 * b**2) * b**2


def goldstein_price(x):
    a, b = x[0], x[1]
    t1 = 1 + (a + b + 1) ** 2 * (19 - 14 * a + 3 * a**2 - 14 * b + 6 * a * b + 3 * b**2)
    t2 = 30 + (2 * a - 3 * b) ** 2 * (18 - 32 * a + 12 * a**2 + 48 * b - 36 * a * b + 27 * b**2)
    return t1 * t2


def shubert(x):
    """Product of trigonometric sums; 760 local minima, 18 global."""
    i = jnp.arange(1.0, 6.0)
    terms = [jnp.sum(i * jnp.cos((i + 1) * x[d] + i)) for d in range(x.size)]
    return terms[0] * terms[1]


_H3_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
_H3_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
_H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_H6_P = 1e-4 * np.array([
    [1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381],
])  # fmt: skip
_H_C = np.array([1.0, 1.2, 3.0, 3.2])


def hartmann3(x):
    return -jnp.sum(_H_C * jnp.exp(-jnp.sum(_H3_A * (x - _H3_P) ** 2, axis=1)))


def hartmann6(x):
    return -jnp.sum(_H_C * jnp.exp(-jnp.sum(_H6_A * (x - _H6_P) ** 2, axis=1)))


_SH_B = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5.0])
_SH_C = np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7], [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]] * 2, dtype=float)


def shekel10(x):
    """Sum of reciprocals: 10 deep, narrow wells."""
    return -jnp.sum(1.0 / (jnp.sum((x[:, None] - _SH_C) ** 2, axis=0) + _SH_B))


def mccormick_fn(x):
    a, b = x[0], x[1]
    return jnp.sin(a + b) + (a - b) ** 2 - 1.5 * a + 2.5 * b + 1


def michalewicz(x):
    i = jnp.arange(1, x.size + 1)
    return -jnp.sum(jnp.sin(x) * jnp.sin(i * x**2 / jnp.pi) ** 20)


def eggholder(x):
    a, b = x[0], x[1]
    return -(b + 47) * jnp.sin(jnp.sqrt(jnp.abs(a / 2 + b + 47))) - a * jnp.sin(
        jnp.sqrt(jnp.abs(a - (b + 47)))
    )


def holder_table(x):
    a, b = x[0], x[1]
    return -jnp.abs(jnp.sin(a) * jnp.cos(b) * jnp.exp(jnp.abs(1 - jnp.sqrt(a**2 + b**2) / jnp.pi)))


def styblinski_tang(x):
    return 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x)


def piecewise(x):
    """Nonsmooth: jnp.where / max / abs on top of a nonconvex smooth part."""
    a, b = x[0], x[1]
    smooth = jnp.sin(3 * a) * jnp.cos(2 * b) + 0.1 * (a**2 + b**2)
    kink = jnp.where(a > 0.5, a - 0.5, 0.0) + jnp.abs(b + 0.3) * 0.5
    return smooth + kink + jnp.maximum(a * b, -0.2)


# A small "trained" neural network with fixed weights: minimizing a network's
# output over its input box (cf. verification / surrogate optimization).
_k1, _k2, _k3 = jax.random.split(jax.random.PRNGKey(0), 3)
_W1 = np.asarray(jax.random.normal(_k1, (2, 16)) * 1.5)
_b1 = np.asarray(jax.random.normal(_k2, (16,)))
_W2 = np.asarray(jax.random.normal(_k3, (16,)) / 4)


def tanh_net(x):
    return jnp.tanh(x @ _W1 + _b1) @ _W2


# Shekel: the often-quoted -10.5364 is f(4,4,4,4); the true minimizer is
# slightly off that point (~(4.0007, 4.0006, 3.9997, 3.9995)).
PROBLEMS = [
    Problem("gramacy_lee", gramacy_lee, np.array([0.5]), np.array([2.5]), -0.869011134989500, "1-D, sin(x)/x"),
    Problem("six_hump_camel", six_hump_camel, np.array([-3.0, -2.0]), np.array([3.0, 2.0]), -1.031628453489877, "polynomial, 6 local minima"),
    Problem("mccormick", mccormick_fn, np.array([-1.5, -3.0]), np.array([4.0, 4.0]), -1.913222954981037, "sin(x+y) + quadratic"),
    Problem("goldstein_price", goldstein_price, np.full(2, -2.0), np.full(2, 2.0), 3.0, "degree-8 polynomial"),
    Problem("shubert", shubert, np.full(2, -10.0), np.full(2, 10.0), -186.7309088310239, "product of trig sums, 18 global minima"),
    Problem("michalewicz_2d", michalewicz, np.zeros(2), np.full(2, np.pi), -1.801303410098554, "sin^20, steep valleys"),
    Problem("holder_table", holder_table, np.full(2, -10.0), np.full(2, 10.0), -19.20850256788675, "abs/exp/sqrt, 4 global minima"),
    Problem("eggholder", eggholder, np.full(2, -512.0), np.full(2, 512.0), -959.6406627106155, "sin(sqrt|.|), very rugged"),
    Problem("hartmann3", hartmann3, np.zeros(3), np.ones(3), -3.862782147819745, "sum of Gaussians"),
    Problem("styblinski_tang_3d", styblinski_tang, np.full(3, -5.0), np.full(3, 5.0), -39.16616570377142 * 3, "8 local minima"),
    Problem("shekel10_4d", shekel10, np.zeros(4), np.full(4, 10.0), -10.53644315348353, "reciprocals, 10 narrow wells"),
    Problem("hartmann6", hartmann6, np.zeros(6), np.ones(6), -3.322368011391339, "6-D sum of Gaussians"),
    Problem("piecewise_2d", piecewise, np.array([-2.0, -2.0]), np.array([2.0, 2.0]), None, "where/abs/max, nonsmooth"),
    Problem("tanh_net_2d", tanh_net, np.array([-3.0, -3.0]), np.array([3.0, 3.0]), None, "16-unit tanh network"),
]


# ---------------------------------------------------------------------------
# Constrained problems:  min f(x)  s.t.  ineq(x) <= 0,  eq(x) = 0,  lo <= x <= hi
# ---------------------------------------------------------------------------


class ConstrainedProblem(NamedTuple):
    name: str
    f: callable
    ineq: callable  # or None
    eq: callable  # or None
    lo: np.ndarray
    hi: np.ndarray
    fstar: float
    xstar: np.ndarray  # a known global minimizer (to within published digits)
    note: str


def g06(x):
    return (x[0] - 10) ** 3 + (x[1] - 20) ** 3


def g06_ineq(x):
    return jnp.stack([-((x[0] - 5) ** 2) - (x[1] - 5) ** 2 + 100, (x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81])


def g08(x):
    return -(jnp.sin(2 * jnp.pi * x[0]) ** 3) * jnp.sin(2 * jnp.pi * x[1]) / (x[0] ** 3 * (x[0] + x[1]))


def g08_ineq(x):
    return jnp.stack([x[0] ** 2 - x[1] + 1, 1 - x[0] + (x[1] - 4) ** 2])


def g11(x):
    return x[0] ** 2 + (x[1] - 1) ** 2


def g11_eq(x):
    return x[1] - x[0] ** 2


def g24(x):
    return -x[0] - x[1]


def g24_ineq(x):
    a, b = x[0], x[1]
    return jnp.stack([
        -2 * a**4 + 8 * a**3 - 8 * a**2 + b - 2,
        -4 * a**4 + 32 * a**3 - 88 * a**2 + 96 * a + b - 36,
    ])  # fmt: skip


def g04(x):
    return 5.3578547 * x[2] ** 2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141


def g04_ineq(x):
    x1, x2, x3, x4, x5 = x
    u = 85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5
    v = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3**2
    w = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
    return jnp.stack([u - 92, -u, v - 110, 90 - v, w - 25, 20 - w])


# Haverly (1978) pooling problem; z = [A, B, Cx, Cy, Px, Py, x, y, p]
def haverly(z):
    A, B, Cx, Cy, Px, Py, x, y, p = z
    return -(9 * x + 15 * y - 6 * A - 16 * B - 10 * (Cx + Cy))


def haverly_ineq(z):
    A, B, Cx, Cy, Px, Py, x, y, p = z
    return jnp.stack([p * Px + 2 * Cx - 2.5 * x, p * Py + 2 * Cy - 1.5 * y])


def haverly_eq(z):
    A, B, Cx, Cy, Px, Py, x, y, p = z
    return jnp.stack([Px + Py - A - B, x - Px - Cx, y - Py - Cy, p * (Px + Py) - 3 * A - B])


# Reactor network design (Ryoo & Sahinidis 1995 [ryoo1995global])
_K1 = 0.09755988
_K2 = 0.99 * _K1
_K3 = 0.0391908
_K4 = 0.9 * _K3


def reactor(x):
    return -x[3]


def reactor_ineq(x):
    return jnp.sqrt(x[4]) + jnp.sqrt(x[5]) - 4


def reactor_eq(x):
    return jnp.stack([
        x[0] + _K1 * x[0] * x[4] - 1,
        x[1] - x[0] + _K2 * x[1] * x[5],
        x[2] + x[0] + _K3 * x[2] * x[4] - 1,
        x[3] - x[2] + x[1] - x[0] + _K4 * x[3] * x[5],
    ])  # fmt: skip


CONSTRAINED_PROBLEMS = [
    ConstrainedProblem("G06", g06, g06_ineq, None, np.array([13.0, 0.0]), np.array([100.0, 100.0]),
                       -6961.81387558015, np.array([14.095, 0.8429607892154796]), "cubic, feasible set a thin crescent"),
    ConstrainedProblem("G08", g08, g08_ineq, None, np.array([0.0, 0.0]), np.array([10.0, 10.0]),
                       -0.0958250414180359, np.array([1.2279713526, 4.2453733661]), "sin^3 / polynomial"),
    ConstrainedProblem("G11", g11, None, g11_eq, np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
                       0.75, np.array([0.7071067811865476, 0.5]), "equality on a parabola, two minimizers"),
    ConstrainedProblem("G24", g24, g24_ineq, None, np.array([0.0, 0.0]), np.array([3.0, 4.0]),
                       -5.50801327159536, np.array([2.329520197477623, 3.17849307411774]), "linear f, quartic constraints, disconnected feasible set"),
    ConstrainedProblem("G04", g04, g04_ineq, None, np.array([78.0, 33, 27, 27, 27]), np.array([102.0, 45, 45, 45, 45]),
                       -30665.53867178332, np.array([78.0, 33.0, 29.9952560256815985, 45.0, 36.7758129057882073]), "5-D, bilinear double-sided constraints"),
    ConstrainedProblem("haverly", haverly, haverly_ineq, haverly_eq, np.array([0.0] * 8 + [1.0]),
                       np.array([300.0, 300, 100, 200, 300, 300, 100, 200, 3]),
                       -400.0, np.array([0.0, 100, 0, 100, 0, 100, 0, 200, 1]), "9-D pooling, bilinear equalities"),
    ConstrainedProblem("reactor", reactor, reactor_ineq, reactor_eq, np.array([0.0] * 4 + [1e-5] * 2),
                       np.array([1.0] * 4 + [16.0] * 2),
                       -0.388811, None, "6-D reactor network, sqrt + bilinear equalities"),
]  # fmt: skip

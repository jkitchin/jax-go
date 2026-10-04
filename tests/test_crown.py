"""CROWN linear bounds: per-activation relaxations, soundness on networks, exactness on affine functions, fallback."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal
from jaxglobal import minimize
from jaxglobal.crown import UNARY, _maxc, _minc, crown_auto, crown_bound, crown_supported
from jaxglobal.problems import PROBLEMS
from jaxglobal.synthetic import nn_surrogate

TANH_NET = next(p for p in PROBLEMS if p.name == "tanh_net_2d")
ACT = {
    "tanh": jnp.tanh, "logistic": jax.nn.sigmoid, "erf": jax.scipy.special.erf, "atan": jnp.arctan,
    "asinh": jnp.arcsinh, "exp": jnp.exp, "expm1": jnp.expm1, "cosh": jnp.cosh, "square": lambda z: z * z,
    "abs": jnp.abs, "log": jnp.log, "log1p": jnp.log1p, "sqrt": jnp.sqrt,
}  # fmt: skip


def _intervals(name, n, rng):
    ab = rng.normal(size=(n, 2)) * rng.choice([0.1, 1, 4], size=(n, 2))
    if name in ("log", "sqrt"):
        ab = np.abs(ab) + 1e-3
    elif name == "log1p":
        ab = np.abs(ab) - 0.9
    else:  # a third each of all-negative, all-positive and mixed intervals
        k = np.arange(n) % 3
        ab[k == 1], ab[k == 2] = -np.abs(ab[k == 1]), np.abs(ab[k == 2])
    return ab.min(1), ab.max(1)


def _check_lines(rule, g, l, u):
    al, bl, au, bu = (np.asarray(v)[:, None] for v in rule(jnp.asarray(l), jnp.asarray(u)))
    z = l[:, None] + np.linspace(0, 1, 201)[None] * (u - l)[:, None]
    gz = np.asarray(g(z))
    tol = 1e-9 * (1 + np.abs(gz))
    assert np.all(al * z + bl <= gz + tol)
    assert np.all(gz <= au * z + bu + tol)


@pytest.mark.parametrize("name", list(ACT))
def test_unary_lines_bound_the_function(name):
    l, u = _intervals(name, 1500, np.random.default_rng(1))
    _check_lines(UNARY[name], ACT[name], l, u)


@pytest.mark.parametrize("c", [0.0, 0.7, -1.3])
def test_relu_lines_bound_the_function(c):
    l, u = _intervals("tanh", 1500, np.random.default_rng(2))
    _check_lines(lambda a, b: _maxc(a, b, c), lambda z: np.maximum(z, c), l, u)
    _check_lines(lambda a, b: _minc(a, b, c), lambda z: np.minimum(z, c), l, u)


def test_sshape_is_exact_on_the_boundary_cases():
    # all-positive interval: chord below, tangent above; both touch g at the ends / middle
    al, bl, au, bu = (float(v) for v in UNARY["tanh"](jnp.array(0.5), jnp.array(2.0)))
    assert abs(al * 0.5 + bl - np.tanh(0.5)) < 1e-12 and abs(al * 2.0 + bl - np.tanh(2.0)) < 1e-12
    assert abs(au * 1.25 + bu - np.tanh(1.25)) < 1e-12


def _mlp(seed, act, layers=3, n_in=3, hidden=6):
    r = np.random.default_rng(seed)
    dims = [n_in] + [hidden] * layers + [1]
    Ws = [r.normal(size=(a, b)) * 1.5 / np.sqrt(a) for a, b in zip(dims[:-1], dims[1:])]
    bs = [r.normal(size=b) * 0.5 for b in dims[1:]]

    def f(x):
        for W, b in zip(Ws[:-1], bs[:-1]):
            x = act(x @ W + b)
        return (x @ Ws[-1] + bs[-1])[0]

    return f, n_in


def _bilinear(x):
    return x[0] * x[1] + 0.5 * x[1] * x[2] - jnp.tanh(x[0] - x[2]) * x[1] + x[2] ** 2


def _sound(f, lo, hi, n=20000, seed=0):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    X = np.random.default_rng(seed).uniform(lo, hi, (n, len(lo)))
    X = np.concatenate([X, lo[None], hi[None]])
    fmin = float(jax.vmap(f)(jnp.asarray(X)).min())
    lb = float(crown_bound(f, lo, hi))
    assert np.isfinite(lb) and lb <= fmin + 1e-9, (lb, fmin)
    return lb, fmin


@pytest.mark.parametrize("act", [jnp.tanh, jax.nn.relu, jax.nn.sigmoid])
@pytest.mark.parametrize("seed", [0, 1])
def test_random_mlp_bound_is_valid(act, seed):
    f, n = _mlp(seed, act)
    r = np.random.default_rng(seed)
    _sound(f, -np.ones(n), np.ones(n))
    for _ in range(3):  # sub-boxes (off-center, asymmetric)
        c, w = r.uniform(-1, 1, n), r.uniform(0.1, 1, n)
        _sound(f, c - w, c + w)


def test_tanh_net_and_surrogate_bounds_are_valid():
    _sound(TANH_NET.f, TANH_NET.lo, TANH_NET.hi)
    for p in (nn_surrogate(3, hidden=5, layers=2), nn_surrogate(3, hidden=5, layers=2, act="relu")):
        _sound(p.f, p.lo, p.hi)


def test_bilinear_bound_is_valid():
    r = np.random.default_rng(3)
    for _ in range(4):
        c, w = r.uniform(-1, 1, 3), r.uniform(0.1, 2, 3)
        _sound(_bilinear, c - w, c + w)


def test_deep_bounds_are_not_looser_on_a_deep_net():
    p = nn_surrogate(5)
    lo, hi = jnp.asarray(p.lo), jnp.asarray(p.hi)
    deep, shallow = (float(crown_bound(p.f, lo, hi, deep=d)) for d in (True, False))
    assert deep >= shallow - 0.1  # intermediate CROWN bounds help (not strictly monotone: tangent points move)


def test_affine_function_is_exact():
    r = np.random.default_rng(4)
    W, w, b = r.normal(size=(4, 3)), r.normal(size=3), 0.7

    def f(x):
        y = jnp.concatenate([x[:2] - 3.0 * x[2:], jnp.reshape(x, (2, 2))[:, 1] * 2.0 - x[:2] / 4.0])
        return jnp.sum((y @ W) * w) + b - 2 * x[0] + jnp.sum(jnp.broadcast_to(x[1], (3,)))

    lo, hi = np.array([-1.0, -2.0, 0.5, -0.3]), np.array([2.0, 1.0, 3.0, 0.9])
    # f is affine: its minimum over the box is at a vertex given by the gradient signs
    g = np.asarray(jax.grad(f)(jnp.zeros(4)))
    exact = float(f(jnp.zeros(4))) + np.minimum(g * lo, g * hi).sum()
    assert float(crown_bound(f, lo, hi)) == pytest.approx(exact, abs=1e-10)
    assert crown_supported(f, jnp.asarray(lo)) and not crown_auto(f, jnp.asarray(lo))


def test_unsupported_primitive_gives_minus_inf():
    def f(x):
        return jnp.sum(jnp.sin(x)) + jnp.tanh(x[0])

    lo, hi = -np.ones(2), np.ones(2)
    assert not crown_supported(f, jnp.asarray(lo))
    assert float(crown_bound(f, lo, hi)) == -np.inf
    # products of two variables through a matmul are not supported either
    assert float(crown_bound(lambda x: x @ x, lo, hi)) == -np.inf
    r = minimize(f, lo, hi, crown=True, time_limit=30)  # no crash, same answer as without
    assert r.converged and r.fun == pytest.approx(minimize(f, lo, hi, time_limit=30).fun, abs=1e-5)


def test_bound_jits_and_vmaps_over_boxes():
    from jaxglobal.crown import make_crown

    p = nn_surrogate(3, hidden=4, layers=2)
    bound = jax.jit(jax.vmap(make_crown(p.f)))
    lo = jnp.asarray(np.random.default_rng(5).uniform(-2, 0, (6, 3)))
    got = np.asarray(bound(lo, lo + 1.0))
    ref = [float(crown_bound(p.f, l, l + 1.0)) for l in lo]
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_option_validation_and_auto():
    with pytest.raises(ValueError, match="crown"):
        minimize(TANH_NET.f, TANH_NET.lo, TANH_NET.hi, crown="yes")
    six = next(p for p in PROBLEMS if p.name == "six_hump_camel")
    assert crown_auto(TANH_NET.f, jnp.asarray(TANH_NET.lo))
    assert not crown_auto(six.f, jnp.asarray(six.lo))


@pytest.mark.parametrize("crown", [True, "auto"])
def test_solve_with_crown_matches_without(crown):
    p = nn_surrogate(2, hidden=4, layers=2, seed=1)
    ref = minimize(p.f, p.lo, p.hi, time_limit=30)
    r = minimize(p.f, p.lo, p.hi, crown=crown, time_limit=30)
    assert r.converged and ref.converged
    assert r.fun == pytest.approx(ref.fun, abs=1e-5) and r.lower_bound <= ref.fun + 1e-5
    assert r.nodes <= ref.nodes

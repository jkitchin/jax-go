"""Structural derivative patterns and nonlinear variables (jaxglobal/structure.py)."""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal.bnb import nonlinear_variables
from jaxglobal.nl import read_nl
from jaxglobal.structure import structure

DATA = pathlib.Path(__file__).parent / "data"


def mixed():
    """Products, a constant multiple, a quotient by a variable and by a constant, exp, a square,
    a linear-only variable (x3) and an unused one (x5)."""

    def f(x):
        return x[0] * x[1] + 3.0 * x[3] + jnp.exp(0.5 * x[2]) - x[4] / 2.0

    def g(x):
        return jnp.stack([x[0] / (1.0 + x[4] ** 2) - 1.0, 2.0 * x[1] + x[3] - 4.0])

    def h(x):
        return jnp.atleast_1d(x[2] ** 2 + x[4] - 0.5)

    return f, g, h, 6


def models():
    yield "mixed", mixed()
    for name in ("nvs01", "kall_circles_c6a"):
        M = read_nl(DATA / f"{name}.nl", integers="relax")
        yield name, (M.f, M.ineq, M.eq, M.n)


@pytest.mark.parametrize("name,model", list(models()), ids=lambda v: v if isinstance(v, str) else "")
def test_patterns_cover_the_derivatives(name, model):
    """Every nonzero of the constraint Jacobian and of the Lagrangian Hessian (dense AD at random
    points and multipliers) lies in the structural patterns, which pounce requires to be supersets."""
    f, g, h, n = model
    S = structure(f, g, h, n)
    assert S is not None

    def cons(x):
        return jnp.concatenate([jnp.atleast_1d(fn(x)) for fn in (g, h) if fn is not None] or [jnp.zeros(0)])

    m = cons(jnp.ones(n)).size
    rng = np.random.default_rng(0)
    J, H = np.zeros((m, n), bool), np.zeros((n, n), bool)
    jac, lag = jax.jit(jax.jacfwd(cons)), jax.jit(jax.hessian(lambda x, y: f(x) + y @ cons(x)))
    for _ in range(4):
        x, y = jnp.asarray(rng.uniform(0.2, 2.0, n)), jnp.asarray(rng.normal(size=m))
        J |= np.asarray(jac(x)) != 0
        H |= np.asarray(lag(x, y)) != 0
    PJ, PH = np.zeros((m, n), bool), np.zeros((n, n), bool)
    PJ[S.jac], PH[S.hess] = True, True
    assert np.all(S.hess[0] >= S.hess[1])  # lower triangle
    PH |= PH.T
    assert not (J & ~PJ).any() and not (H & ~PH).any()


def test_nonlinear_variables_match_hessian_sampling():
    f, g, h, n = mixed()
    S = structure(f, g, h, n)
    assert S.nonlinear.tolist() == [True, True, True, False, True, False]
    sampled = nonlinear_variables([f, g, h], np.full(n, 0.2), np.full(n, 2.0))
    assert S.nonlinear.tolist() == sampled.tolist()


def test_vectorized_code_has_no_structure():
    W = jnp.asarray(np.random.default_rng(0).normal(size=(20, 3)))
    assert structure(lambda x: jnp.sum(jnp.tanh(W @ x) ** 2), None, None, 3) is None

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxglobal import minimize, relax
from jaxglobal.lp import make_constrained_bounder
from jaxglobal.problems import CONSTRAINED_PROBLEMS

IDS = [p.name for p in CONSTRAINED_PROBLEMS]


def boxes_around(p, x, k=6, seed=0):
    """Random sub-boxes of the domain that contain x."""
    rng = np.random.default_rng(seed)
    w = p.hi - p.lo
    for _ in range(k):
        a = np.maximum(p.lo, x - rng.uniform(0, 0.5, x.size) * w)
        b = np.minimum(p.hi, x + rng.uniform(0, 0.5, x.size) * w)
        yield jnp.asarray(a), jnp.asarray(b)


@pytest.mark.parametrize("p", [p for p in CONSTRAINED_PROBLEMS if p.eq is not None], ids=lambda p: p.name)
def test_concave_cuts_valid(p):
    """Affine overestimators of cc (used for -cc_h <= 0) lie above h on the box."""
    rh = relax(lambda x: jnp.atleast_1d(p.eq(x)))
    rng = np.random.default_rng(1)
    for _ in range(5):
        a, b = np.sort(rng.uniform(p.lo, p.hi, (2, p.lo.size)), axis=0)
        a, b = jnp.asarray(a), jnp.asarray(b)
        pt = jnp.asarray(rng.uniform(a, b))
        cc = rh(pt, a, b).cc
        J = jax.jacfwd(lambda x: rh(x, a, b).cc)(pt)
        xs = jnp.asarray(rng.uniform(a, b, (200, p.lo.size)))
        hv = jax.vmap(lambda x: jnp.atleast_1d(p.eq(x)))(xs)
        cut = cc + (xs - pt) @ J.T
        assert np.all(np.asarray(cut) >= np.asarray(hv) - 1e-8 * (1 + np.abs(np.asarray(hv))))


@pytest.mark.parametrize("p", [p for p in CONSTRAINED_PROBLEMS if p.xstar is not None], ids=lambda p: p.name)
def test_node_bound_below_optimum(p):
    """Any box containing a global minimizer must bound below f*."""
    bound = jax.jit(make_constrained_bounder(p.f, p.ineq, p.eq))
    for a, b in boxes_around(p, p.xstar):
        lb = float(bound(a, b)[0])
        assert lb <= p.fstar + 1e-5 * (1 + abs(p.fstar)), (lb, p.fstar)


@pytest.mark.parametrize("p", CONSTRAINED_PROBLEMS, ids=IDS)
def test_constrained_global_minimum(p):
    r = minimize(p.f, p.lo, p.hi, ineq=p.ineq, eq=p.eq, max_iter=3000)
    assert r.status == "optimal"
    tol = 1e-4 * (1 + abs(p.fstar))
    assert abs(r.fun - p.fstar) <= tol
    assert r.lower_bound <= p.fstar + tol


def test_infeasible_problem():
    r = minimize(lambda x: x[0] + x[1], [-1, -1], [1, 1], ineq=lambda x: 1.5 - x[0] ** 2 - x[1] ** 2 * 0.1)
    assert r.status == "infeasible"
    assert r.fun == np.inf

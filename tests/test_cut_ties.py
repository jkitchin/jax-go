"""Frank-Wolfe cuts must underestimate the relaxation when the gradient is compiled inside
``lax.scan`` with a traced box, as in ``make_bounder``.

``jnp.maximum``'s derivative weighs its operands by two separate comparisons; XLA may compute an
operand twice with different FMA contraction (after CSE / loop-invariant code motion in the
while loop), so at a near-tie both weights can be 0. ``mc_frac``'s ``max(base.cv, env)`` ties in
exact arithmetic whenever the numerator is at its upper bound, which FW iterates (vertex
combinations) hit on every box: the dropped term gave a cut 1400 above the relaxation and
duality-based range reduction then cut the global minimizer out of the box (scratch pump2.py,
K=26: status optimal at 130186 with SCIP's 130000.35 excluded)."""

import jax
import jax.numpy as jnp
import numpy as np

import jaxglobal  # noqa: F401  (enables x64)
from jaxglobal.univariate import make_relaxer

BLO = jnp.array([0.8869155651214526, 0.48628606117266093, 0.19387536584713297])
BHI = jnp.array([0.9, 0.49730266613865115, 0.19776668777453066])
X16 = jnp.array([0.9, 0.48685309231061635, 0.19393259116959471])  # the 17th FW iterate: x0 on its upper face


def _f(x):
    D = x[1]
    return (4 * x[0] * 0.3 / (jnp.pi * D**2) * D) ** 0.9  # Reynolds-number-like term (x0 / D) ** 0.9


def _in_scan_grad(fn, x0, n=2):
    rfu = make_relaxer(fn, False, False, convex=False, interpret=False)

    def run(blo, bhi, x0):
        data = rfu.prepare(blo, bhi)
        vg = jax.value_and_grad(lambda x: rfu(x, blo, bhi, data).cv)

        def step(x, k):
            v, g = jax.lax.optimization_barrier(vg(x))  # as make_bounder
            return x + 0 * g, (v, g)

        return jax.lax.scan(step, x0, jnp.arange(float(n)))[1]

    v, g = jax.jit(run)(BLO, BHI, x0)
    data = rfu.prepare(BLO, BHI)
    return float(v[0]), np.asarray(g[0]), jax.jit(jax.vmap(lambda x: rfu(x, BLO, BHI, data).cv))


def test_scan_gradient_is_a_subgradient_at_a_face_tie():
    v, g, cv = _in_scan_grad(_f, X16)
    Z = np.asarray(BLO + (BHI - BLO) * np.random.default_rng(0).uniform(size=(2000, 3)))
    cut = v + (Z - np.asarray(X16)) @ g
    assert np.max(cut - np.asarray(cv(Z))) <= 1e-9 * (1 + abs(v))
    assert g[1] < 0  # the relaxation decreases in x1 here (the bad gradient had the wrong sign)


def test_cvmax_ccmin_select_one_operand():
    from jaxglobal.relax import ccmin, cvmax

    a, b = jnp.array([1.0, 2.0, 2.0, jnp.nan]), jnp.array([2.0, 1.0, 2.0, 0.0])
    np.testing.assert_array_equal(cvmax(a, b), jnp.maximum(a, b))
    np.testing.assert_array_equal(ccmin(a, b), jnp.minimum(a, b))
    ga, gb = jax.grad(lambda a, b: cvmax(a, b).sum(), argnums=(0, 1))(a[:3], b[:3])
    np.testing.assert_array_equal(ga + gb, np.ones(3))  # weights of exactly one operand, also at the tie

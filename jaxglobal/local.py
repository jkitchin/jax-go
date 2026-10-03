"""Batched local NLP solves for upper bounds.

``make_local(f, ineq, eq, lo, hi, solver)`` returns ``solve(X0) -> X`` mapping a
batch of starting points to local solutions. Backends:

* ``"pounce"``: the pounce interior-point solver (Rust, Ipopt-derived) with
  exact JAX-built derivatives (``pounce.jax.from_jax``); the batch runs in
  parallel with ``pounce.solve_nlp_batch``.
* ``"optax"``: pure JAX, ``vmap``-ed over starts (described below).
* ``"scipy"``: L-BFGS-B / SLSQP one start at a time, kept for reference.

Results need not be feasible or even good: the caller checks feasibility and
keeps the best point, so a local solver can only help the upper bound.

The optax backend removes bounds by the reparameterization

    x = lo + (hi - lo) * (1 + sin z) / 2,

which reaches the bounds exactly (at z = +-pi/2, where dx/dz = 0) and stays
well conditioned there: f is quadratic in z around a bound-active optimum.
A sigmoid map would push bound optima to z = +-inf instead. are removed by the reparameterization

    x = lo + (hi - lo) * (1 + sin z) / 2,

which reaches the bounds exactly (at z = +-pi/2, where dx/dz = 0) and stays
well conditioned there: f is quadratic in z around a bound-active optimum.
A sigmoid map would push bound optima to z = +-inf instead.

Constraints use an augmented Lagrangian (Hestenes-Powell-Rockafellar):

    L(x) = f(x) + lam.h + rho/2 |h|^2 + 1/(2 rho) sum(max(0, mu + rho g)^2 - mu^2)

with optax.lbfgs as the inner solver. Every loop has a fixed trip count, so a
solve is a pure function and ``vmap``s over many starting points at once.
"""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _vec(fn):
    return (lambda x: jnp.atleast_1d(fn(x))) if fn is not None else (lambda x: jnp.zeros(0))


def _box_map(lo, hi):
    w = hi - lo

    def to_x(z):
        return lo + w * (1 + jnp.sin(z)) / 2

    def to_z(x):
        r = jnp.where(w > 0, 2 * (x - lo) / jnp.where(w > 0, w, 1.0) - 1, 0.0)
        return jnp.arcsin(jnp.clip(r, -1.0, 1.0))

    return to_x, to_z


def _lbfgs(fun, z0, iters):
    """optax L-BFGS (zoom line search) with a fixed number of iterations.

    Steps that produce non-finite values are rejected (the iterate is kept), so
    functions with singularities inside the box cannot poison the batch.
    """
    opt = optax.lbfgs()
    vg = optax.value_and_grad_from_state(fun)

    def step(carry, _):
        z, st = carry
        v, g = vg(z, state=st)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        u, st_new = opt.update(g, st, z, value=v, grad=g, value_fn=fun)
        z_new = optax.apply_updates(z, u)
        ok = jnp.all(jnp.isfinite(z_new)) & jnp.isfinite(fun(z_new))
        z = jnp.where(ok, z_new, z)
        st = jax.tree.map(lambda a, b: jnp.where(ok, a, b), st_new, st)
        return (z, st), None

    (z, _), _ = jax.lax.scan(step, (z0, opt.init(z0)), None, length=iters)
    return z


def make_local_solver(f, ineq=None, eq=None, inner=60, outer=12, rho0=10.0):
    """Return ``solve(x0, lo, hi) -> x``: a local minimizer near x0 within the box."""

    g, h = _vec(ineq), _vec(eq)
    constrained = ineq is not None or eq is not None

    def solve(x0, lo, hi):
        to_x, to_z = _box_map(lo, hi)
        z = to_z(x0)
        if not constrained:
            return to_x(_lbfgs(lambda z: f(to_x(z)), z, inner * 2))

        lam = jnp.zeros_like(h(x0))
        mu = jnp.zeros_like(g(x0))

        def viol(x):
            return jnp.maximum(jnp.max(jnp.maximum(g(x), 0.0), initial=0.0), jnp.max(jnp.abs(h(x)), initial=0.0))

        def outer_step(carry, _):
            z, lam, mu, rho, v_prev = carry

            def aug(z):
                x = to_x(z)
                hv, gv = h(x), g(x)
                return (f(x) + lam @ hv + 0.5 * rho * hv @ hv
                        + 0.5 / rho * jnp.sum(jnp.maximum(0.0, mu + rho * gv) ** 2 - mu**2))  # fmt: skip

            z = _lbfgs(aug, z, inner)
            x = to_x(z)
            lam = lam + rho * h(x)
            mu = jnp.maximum(0.0, mu + rho * g(x))
            v = viol(x)
            rho = jnp.where(v > 0.25 * v_prev, jnp.minimum(rho * 10.0, 1e8), rho)
            return (z, lam, mu, rho, v), None

        init = (z, lam, mu, jnp.asarray(rho0, dtype=x0.dtype), jnp.asarray(jnp.inf, dtype=x0.dtype))
        (z, *_), _ = jax.lax.scan(outer_step, init, None, length=outer)
        return to_x(z)

    return solve


def _pounce_local(f, ineq, eq, lo, hi, options=None):
    import pounce
    from pounce.jax import from_jax

    g, h = _vec(ineq), _vec(eq)
    xm = jnp.asarray(0.5 * (lo + hi))
    mi, me = g(xm).size, h(xm).size
    kw = dict(n=lo.size, lb=lo, ub=hi)
    if mi + me:
        kw.update(m=mi + me, cl=np.r_[np.full(mi, -1e20), np.zeros(me)], cu=np.zeros(mi + me))
        prob = from_jax(f, lambda x: jnp.concatenate([g(x), h(x)]), **kw)
    else:
        prob = from_jax(f, **kw)
    opts = dict(print_level=0, tol=1e-10, max_iter=200, sb="yes", **(options or {}))

    def solve(X0):
        res = pounce.solve_nlp_batch([prob] * len(X0), list(X0), options=opts)
        return np.stack([x for x, _ in res])

    return solve


def _scipy_local(f, ineq, eq, lo, hi):
    from scipy.optimize import minimize

    g, h = _vec(ineq), _vec(eq)
    fvg = jax.jit(jax.value_and_grad(f))
    gj = jax.jit(lambda x: (g(x), jax.jacfwd(g)(x)))
    hj = jax.jit(lambda x: (h(x), jax.jacfwd(h)(x)))
    bounds = list(zip(lo, hi))

    def fun(x):
        return tuple(np.asarray(v, dtype=np.float64) for v in fvg(x))

    def one(x0):
        if ineq is None and eq is None:
            return minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds).x
        cons = []
        if ineq is not None:
            cons.append({"type": "ineq", "fun": lambda x: -np.asarray(gj(x)[0]), "jac": lambda x: -np.asarray(gj(x)[1])})
        if eq is not None:
            cons.append({"type": "eq", "fun": lambda x: np.asarray(hj(x)[0]), "jac": lambda x: np.asarray(hj(x)[1])})
        return minimize(fun, x0, jac=True, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 200, "ftol": 1e-12}).x  # fmt: skip

    return lambda X0: np.stack([one(x0) for x0 in X0])


def make_local(f, ineq, eq, lo, hi, solver="auto"):
    """Batched local solver ``X0 (k, n) -> X (k, n)`` on the box [lo, hi]."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    if solver == "auto":
        solver = "pounce" if importlib.util.find_spec("pounce") else "optax"
    if solver == "pounce":
        return _pounce_local(f, ineq, eq, lo, hi)
    if solver == "scipy":
        return _scipy_local(f, ineq, eq, lo, hi)
    if solver == "optax":
        one = make_local_solver(f, ineq, eq)
        batched = jax.jit(jax.vmap(one, in_axes=(0, None, None)))
        def solve(X0, k=8):  # pad to a multiple of k starts so jit compiles once
            n = len(X0)
            X0 = np.concatenate([X0, np.repeat(X0[:1], -n % k, 0)])
            return np.asarray(batched(jnp.asarray(X0), jnp.asarray(lo), jnp.asarray(hi)))[:n]

        return solve
    raise ValueError(f"unknown local_solver {solver!r}")

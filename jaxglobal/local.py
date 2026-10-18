"""Batched local NLP solves for upper bounds.

``make_local(f, ineq, eq, lo, hi, solver)`` returns ``solve(X0, LO=None, HI=None) -> X``
mapping a batch of starting points to local solutions, within ``[lo, hi]`` or
per-start bounds ``LO``, ``HI`` (each (k, n); used to fix integer variables).
Backends:

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


def _pounce_local(f, ineq, eq, lo, hi, options=None, patterns=None):
    import pounce
    from pounce.jax import from_jax

    g, h = _vec(ineq), _vec(eq)
    xm = jnp.asarray(0.5 * (lo + hi))
    mi, me = g(xm).size, h(xm).size
    kw = dict(n=lo.size, lb=lo, ub=hi)
    if patterns is not None:
        # structural patterns (structure.py): no probing at build time (dense evaluation: colored
        # AD, sparse=True, made each solve 1.5 to 2.6 times slower on spectra2, syn05m02m, nous1)
        kw.update(hess_pattern=patterns.hess, **({"jac_pattern": patterns.jac} if mi + me else {}))
    if mi + me:
        kw.update(m=mi + me, cl=np.r_[np.full(mi, -1e20), np.zeros(me)], cu=np.zeros(mi + me))
        prob = from_jax(f, lambda x: jnp.concatenate([g(x), h(x)]), **kw)
    else:
        prob = from_jax(f, **kw)
    # bound_relax_factor 0: Ipopt's default (1e-8) lets a solution sit up to 1e-8 * max(1, |bound|)
    # past a bound; the caller clips it back into the box, which moves the constraint values by
    # slope x 1e-8 * |bound| and can push a feasible optimum past feas_tol (ex5_4_3: 1.6e-6, its
    # optimum found at the root was rejected for 50 s)
    opts = dict(print_level=0, tol=1e-10, max_iter=200, sb="yes", bound_relax_factor=0.0, **(options or {}))

    def with_bounds(lb, ub):  # the compiled functions are shared; only the bounds differ
        return pounce.Problem(n=lo.size, m=mi + me, problem_obj=prob.problem_obj, lb=lb, ub=ub,
                              cl=kw.get("cl"), cu=kw.get("cu"))  # fmt: skip

    def solve(X0, LO=None, HI=None, options=None):
        probs = [prob] * len(X0) if LO is None else [with_bounds(a, b) for a, b in zip(LO, HI)]
        res = pounce.solve_nlp_batch(probs, list(X0), options={**opts, **(options or {})})
        return np.stack([x for x, _ in res])

    return solve


def _scipy_local(f, ineq, eq, lo, hi):
    from scipy.optimize import minimize

    g, h = _vec(ineq), _vec(eq)
    fvg = jax.jit(jax.value_and_grad(f))
    gj = jax.jit(lambda x: (g(x), jax.jacfwd(g)(x)))
    hj = jax.jit(lambda x: (h(x), jax.jacfwd(h)(x)))
    def fun(x):
        return tuple(np.asarray(v, dtype=np.float64) for v in fvg(x))

    def one(x0, bounds):
        if ineq is None and eq is None:
            return minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds).x
        cons = []
        if ineq is not None:
            cons.append({"type": "ineq", "fun": lambda x: -np.asarray(gj(x)[0]), "jac": lambda x: -np.asarray(gj(x)[1])})
        if eq is not None:
            cons.append({"type": "eq", "fun": lambda x: np.asarray(hj(x)[0]), "jac": lambda x: np.asarray(hj(x)[1])})
        return minimize(fun, x0, jac=True, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 200, "ftol": 1e-12}).x  # fmt: skip

    def solve(X0, LO=None, HI=None, options=None):  # ``options`` (pounce's) are ignored
        LO, HI = (np.broadcast_to(lo, np.shape(X0)), np.broadcast_to(hi, np.shape(X0))) if LO is None else (LO, HI)
        return np.stack([one(x0, list(zip(a, b))) for x0, a, b in zip(X0, LO, HI)])

    return solve


NEAR_ZERO = 0.1  # |incumbent| below which local solves are repeated with the objective scaled up
MAX_OBJ_SCALE = 1e6


def near_zero_options(fun, abs_tol=1e-6):
    """Extra pounce options for an objective whose incumbent value ``fun`` is near zero, or None.

    An interior-point solve with the default barrier start (``mu_init = 0.1``)
    and an objective of size ~1e-6 is steered by the barrier, not by ``f``: on
    ex6_2_6 (optimum -2.6e-6 at a bound) every start ends at the interior KKT
    point +7.1e-7. Scaling the objective by ``1 / max(|fun|, abs_tol)`` (at most
    ``MAX_OBJ_SCALE``) and starting the barrier small lets ``f`` steer. Only
    used for extra solves: their results are checked like any other.
    """
    if not np.isfinite(fun) or abs(fun) >= NEAR_ZERO:
        return None
    scale = min(MAX_OBJ_SCALE, 1.0 / max(abs(fun), abs_tol))
    return {"obj_scaling_factor": float(scale), "mu_init": 1e-6}


def make_local(f, ineq, eq, lo, hi, solver="auto", patterns=None):
    """Batched local solver ``X0 (k, n) -> X (k, n)`` on the box [lo, hi].

    The solver is called as ``solve(X0, LO=None, HI=None, options=None)``: ``LO``/``HI``
    are per-start boxes; ``options`` are extra pounce (Ipopt-style) options for this
    call, e.g. ``near_zero_options(...)`` (the other backends ignore them).
    ``patterns`` (a ``structure.Structure``) gives pounce the derivative sparsity patterns."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    if solver == "auto":
        solver = "pounce" if importlib.util.find_spec("pounce") else "optax"
    if solver == "pounce":
        return _pounce_local(f, ineq, eq, lo, hi, patterns=patterns)
    if solver == "scipy":
        return _scipy_local(f, ineq, eq, lo, hi)
    if solver == "optax":
        one = make_local_solver(f, ineq, eq)
        batched = jax.jit(jax.vmap(one, in_axes=(0, 0, 0)))

        def solve(X0, LO=None, HI=None, options=None, k=8):  # pad to a multiple of k starts; ``options`` ignored
            n = len(X0)
            LO, HI = (np.broadcast_to(lo, np.shape(X0)), np.broadcast_to(hi, np.shape(X0))) if LO is None else (LO, HI)
            pad = lambda A: np.concatenate([A, np.repeat(A[:1], -n % k, 0)])  # noqa: E731
            return np.asarray(batched(jnp.asarray(pad(X0)), jnp.asarray(pad(LO)), jnp.asarray(pad(HI))))[:n]

        return solve
    raise ValueError(f"unknown local_solver {solver!r}")

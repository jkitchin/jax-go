"""Gaussian-process predictions with relaxations that keep the kernel structure.

``gp_mean`` is a plain function of the ``jaxglobal.kernels`` primitives.
``gp_std`` is a primitive: the posterior standard deviation

    sigma(x) = sqrt(max(v(x), eps)),   v(x) = sf2 - k(x)^T A k(x) = sf2 - ||W^T k(x)||^2,

with ``k_i(x) = sf2 * kernel(sum_j ((x_j - Z_ij) / l_j)**2)``, ``A`` the inverse
of the (noisy) training covariance and ``A = W W^T`` its Cholesky factor.
Relaxed as ``k^T A k``, the quadratic form is a sum of bilinear products, and
any bound through ``A`` inherits its conditioning (``~1/noise``). The rule
combines two relaxations, keeping the max of the convex and the min of the
concave sides:

* ``sqrt(max(sf2 - ||W^T k||^2, eps))``, squares of linear forms in ``k``:
  good from above;
* a Lipschitz bound from the GP itself. The posterior is a Gaussian process,
  so for any ``x`` and the box center ``c``,
  ``|sigma(x) - sigma(c)| <= sd_post(f(x) - f(c)) <= sd_prior(f(x) - f(c))
  = sqrt(2 sf2 (1 - kernel(d(x, c))))`` (the triangle inequality for standard
  deviations; conditioning never increases the variance of a linear
  functional). Over the box this gives ``sigma(c) -+ h`` with ``h`` from the
  farthest corner: first order in the box size, independent of ``A``.

``v >= 0`` in exact arithmetic (``A`` is positive semidefinite); ``eps`` keeps
the square root differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

from .kernels import KERNELS
from .relax import MC, RULES, as_mc, eval_jaxpr_mc, tighten


def _kvec(x, Z, lengthscale, sf2, kernel):
    d = jnp.sum(((x - Z) / lengthscale) ** 2, axis=-1)
    return sf2 * KERNELS[kernel](d)


def gp_mean(x, Z, lengthscale, alpha, sf2, kernel="matern52"):
    """Posterior mean ``k(x)^T alpha`` (``alpha = A y``) at one point ``x``."""
    return _kvec(x, Z, lengthscale, sf2, kernel) @ alpha


def _var(x, Z, lengthscale, W, sf2, kernel):
    return sf2 - jnp.sum((W.T @ _kvec(x, Z, lengthscale, sf2, kernel)) ** 2)


def _std(x, Z, lengthscale, W, sf2, *, kernel, eps):
    return jnp.sqrt(jnp.maximum(_var(x, Z, lengthscale, W, sf2, kernel), eps))


gp_std_p = Primitive("gp_std")
gp_std_p.def_impl(_std)
gp_std_p.def_abstract_eval(lambda x, Z, lengthscale, W, sf2, **_: x.update(shape=x.shape[:-1]))
mlir.register_lowering(gp_std_p, mlir.lower_fun(_std, multiple_results=False))
# derivatives in x only (the other operands are the trained GP's constants)
ad.defjvp(gp_std_p, lambda g, x, Z, lengthscale, W, sf2, **p: jax.jvp(
    lambda u: _std(u, Z, lengthscale, W, sf2, **p), (x,), (g,))[1], None, None, None, None)


def _batch(args, dims, **p):
    # batched evaluation goes through the formula: relaxations only ever see one point
    axes = tuple(d if isinstance(d, int) else None for d in dims)  # not_mapped -> None
    return jax.vmap(lambda *a: _std(*a, **p), in_axes=axes)(*args), 0


batching.primitive_batchers[gp_std_p] = _batch


def gp_std(x, Z, lengthscale, A, sf2, kernel="matern52", eps=1e-12):
    """Posterior standard deviation ``sqrt(max(sf2 - k(x)^T A k(x), eps))`` at one point ``x``.

    ``Z`` (N, n) training inputs, ``lengthscale`` (n,), ``A`` the inverse of the
    training covariance including noise (symmetric positive definite), ``sf2``
    the signal variance.
    """
    A = jnp.asarray(A)
    W = jnp.linalg.cholesky(0.5 * (A + A.T))  # A = W W^T
    f = jnp.asarray
    return gp_std_p.bind(f(x), f(Z), f(lengthscale), W, f(sf2), kernel=kernel, eps=float(eps))


def _rule(ctx, eqn, x, Z, lengthscale, W, sf2):
    kernel, eps = eqn.params["kernel"], eqn.params["eps"]
    x = as_mc(x)

    def run(fn, arg):
        closed = jax.make_jaxpr(fn)(jnp.zeros(arg.lo.shape, arg.lo.dtype))
        (out,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [arg], ctx)
        return as_mc(out)

    sig = run(lambda u: _std(u, Z, lengthscale, W, sf2, kernel=kernel, eps=eps), x)  # Cholesky form
    # Lipschitz bound: sigma(c) -+ sqrt(2 sf2 (1 - kernel(d_max))), d_max to the farthest corner
    fin = jnp.all(jnp.isfinite(x.lo) & jnp.isfinite(x.hi))
    c = jnp.where(jnp.isfinite(x.lo) & jnp.isfinite(x.hi), 0.5 * (x.lo + x.hi), 0.0)
    half = jnp.maximum(c - x.lo, x.hi - c)
    d_max = jnp.sum((half / lengthscale) ** 2)
    h = jnp.sqrt(jnp.maximum(2 * sf2 * (1 - KERNELS[kernel](d_max)), 0.0))
    sc = jnp.sqrt(jnp.maximum(_var(c, Z, lengthscale, W, sf2, kernel), 0.0))
    lo_l, hi_l = sc - h, sc + h + jnp.sqrt(eps)  # sqrt(max(v, eps)) <= sqrt(v) + sqrt(eps)
    ok = fin & jnp.isfinite(h) & jnp.isfinite(sc)
    return tighten(MC(jnp.where(ok, jnp.maximum(sig.lo, lo_l), sig.lo),
                      jnp.where(ok, jnp.minimum(sig.hi, hi_l), sig.hi),
                      jnp.where(ok, jnp.maximum(sig.cv, lo_l), sig.cv),
                      jnp.where(ok, jnp.minimum(sig.cc, hi_l), sig.cc)))  # fmt: skip


RULES["gp_std"] = _rule


# ---------------------------------------------------------------------------
# Expected improvement
# ---------------------------------------------------------------------------
#
# EI(mu, sigma) = (fmin - mu) Phi(z) + sigma phi(z), z = (fmin - mu) / sigma, is the perspective
# sigma * psi((fmin - mu) / sigma) of psi(z) = z Phi(z) + phi(z), which is convex (psi'' = phi > 0):
# so EI is jointly convex in (mu, sigma), nonincreasing in mu and nondecreasing in sigma
# [schweidtmann2021deterministic]. At sigma -> 0 it tends to max(fmin - mu, 0).

_SIG_MIN = 1e-300  # sigma floor: EI(mu, 0) is the continuous limit max(fmin - mu, 0)


def _ei(mu, sigma, fmin):
    s = jnp.maximum(sigma, _SIG_MIN)
    t = fmin - mu
    z = t / s
    return t * jax.scipy.stats.norm.cdf(z) + s * jax.scipy.stats.norm.pdf(z)


ei_p = Primitive("expected_improvement")
ei_p.def_impl(_ei)
ei_p.def_abstract_eval(lambda mu, sigma, fmin: mu.update(shape=jnp.broadcast_shapes(mu.shape, sigma.shape, fmin.shape)))
mlir.register_lowering(ei_p, mlir.lower_fun(_ei, multiple_results=False))


def _ei_z(mu, sigma, fmin):
    return (fmin - mu) / jnp.maximum(sigma, _SIG_MIN)


ad.defjvp(ei_p,
          lambda g, mu, sigma, fmin: -g * jax.scipy.stats.norm.cdf(_ei_z(mu, sigma, fmin)),
          lambda g, mu, sigma, fmin: g * jnp.where(sigma > _SIG_MIN, jax.scipy.stats.norm.pdf(_ei_z(mu, sigma, fmin)), 0.0),
          lambda g, mu, sigma, fmin: g * jax.scipy.stats.norm.cdf(_ei_z(mu, sigma, fmin)))  # fmt: skip
batching.defbroadcasting(ei_p)


def expected_improvement(mu, sigma, fmin):
    """Expected improvement below ``fmin`` of a Gaussian with mean ``mu`` and std ``sigma`` (elementwise).

    Same value as ``(fmin - mu) Phi(z) + sigma phi(z)``; relaxed as the jointly convex, monotone
    function it is, instead of through ``Phi``, ``phi`` and the division one by one.
    """
    f = jnp.asarray
    return ei_p.bind(f(mu, dtype=float), f(sigma, dtype=float), f(fmin, dtype=float))


def _ei_rule(ctx, eqn, mu, sigma, fmin):
    if isinstance(fmin, MC):  # fmin a variable: relax the formula
        closed = jax.make_jaxpr(_ei)(*(jnp.zeros(jnp.shape(v.lo if isinstance(v, MC) else v)) for v in (mu, sigma, fmin)))
        (out,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [mu, sigma, fmin], ctx)
        return as_mc(out)
    m, s = as_mc(mu), as_mc(sigma)
    mL, mU, sL, sU = m.lo, m.hi, jnp.maximum(s.lo, 0.0), jnp.maximum(s.hi, 0.0)
    f = lambda a, b: _ei(a, b, fmin)  # noqa: E731
    fLL, fLU, fUL, fUU = f(mL, sL), f(mL, sU), f(mU, sL), f(mU, sU)  # f(mu-corner, sigma-corner)
    # convex, nonincreasing in mu, nondecreasing in sigma
    cv = f(m.cc, jnp.maximum(s.cv, 0.0))
    # concave envelope over the (mu, sigma) box: the triangulation interpolant with the larger
    # center value, = the min of its two planes; every plane is nonincreasing in mu, nondecreasing
    # in sigma, so composing with (mu.cv, sigma.cc) stays concave and above EI
    dm, ds = mU - mL, sU - sL
    am, bs = jnp.where(dm > 0, 1.0 / jnp.where(dm > 0, dm, 1.0), 0.0), jnp.where(ds > 0, 1.0 / jnp.where(ds > 0, ds, 1.0), 0.0)
    pm, ps = m.cv - mL, jnp.maximum(s.cc, 0.0) - sL
    # diagonal LL-UU: triangles (LL, UL, UU) and (LL, LU, UU)
    a1 = jnp.minimum(fLL + (fUL - fLL) * am * pm + (fUU - fUL) * bs * ps,
                     fLL + (fUU - fLU) * am * pm + (fLU - fLL) * bs * ps)
    # diagonal LU-UL: triangles (LL, UL, LU) and (UU, LU, UL)
    a2 = jnp.minimum(fLL + (fUL - fLL) * am * pm + (fLU - fLL) * bs * ps,
                     fUU + (fUU - fLU) * am * (m.cv - mU) + (fUU - fUL) * bs * (jnp.maximum(s.cc, 0.0) - sU))
    cc = jnp.where(fLL + fUU >= fLU + fUL, a1, a2)
    return tighten(MC(fUL, fLU, cv, cc))


RULES["expected_improvement"] = _ei_rule


# ---------------------------------------------------------------------------
# Probability of improvement
# ---------------------------------------------------------------------------
#
# PI(mu, sigma) = Phi((t - mu) / sigma), t the target (fmin - xi). Not convex, but monotone:
# nonincreasing in mu, and in sigma nondecreasing for mu > t, nonincreasing for mu < t. So over a
# sigma interval [sL, sU] it is at most u(mu) = Phi((t - mu) / sL) for mu <= t, Phi((t - mu) / sU)
# for mu > t: a function of mu alone, nonincreasing, concave left of t and convex right of it, whose
# concave envelope composed with mu's convex relaxation bounds PI from above.


def _pi(mu, sigma, t):
    return jax.scipy.stats.norm.cdf((t - mu) / jnp.maximum(sigma, _SIG_MIN))


pi_p = Primitive("probability_of_improvement")
pi_p.def_impl(_pi)
pi_p.def_abstract_eval(lambda mu, sigma, t: mu.update(shape=jnp.broadcast_shapes(mu.shape, sigma.shape, t.shape)))
mlir.register_lowering(pi_p, mlir.lower_fun(_pi, multiple_results=False))


def _pi_dz(mu, sigma, t):  # phi(z) / s and z
    s = jnp.maximum(sigma, _SIG_MIN)
    z = (t - mu) / s
    return jax.scipy.stats.norm.pdf(z) / s, z


ad.defjvp(pi_p,
          lambda g, mu, sigma, t: -g * _pi_dz(mu, sigma, t)[0],
          lambda g, mu, sigma, t: jnp.where(sigma > _SIG_MIN, -g * _pi_dz(mu, sigma, t)[0] * _pi_dz(mu, sigma, t)[1], 0.0),
          lambda g, mu, sigma, t: g * _pi_dz(mu, sigma, t)[0])  # fmt: skip
batching.defbroadcasting(pi_p)


def probability_of_improvement(mu, sigma, target):
    """Probability that a Gaussian with mean ``mu`` and std ``sigma`` falls below ``target`` (elementwise).

    Same value as ``Phi((target - mu) / sigma)``; relaxed from its monotonicity in ``mu`` and
    ``sigma`` instead of through ``Phi`` and the division separately.
    """
    f = jnp.asarray
    return pi_p.bind(f(mu, dtype=float), f(sigma, dtype=float), f(target, dtype=float))


def _phi_cc_env(z, L, U):
    """Concave envelope of Phi on [L, U] at z (Phi convex below 0, concave above)."""
    Phi, phi = jax.scipy.stats.norm.cdf, jax.scipy.stats.norm.pdf
    L, U = jax.lax.stop_gradient(L), jax.lax.stop_gradient(U)

    def h(m):  # tangent at m (concave part) evaluated at L, minus Phi(L): increasing in m >= 0
        return Phi(m) + phi(m) * (L - m) - Phi(L)

    def bisect(_, ab):
        lo_, hi_ = ab
        mid = 0.5 * (lo_ + hi_)
        up = h(mid) >= 0
        return jnp.where(up, lo_, mid), jnp.where(up, mid, hi_)

    a0 = jnp.maximum(L, 0.0)
    _, mstar = jax.lax.fori_loop(0, 60, bisect, (a0, jnp.maximum(U, a0)))  # keep h(mstar) >= 0
    mstar = jax.lax.stop_gradient(mstar)
    tangent = jnp.where(z >= mstar, Phi(z), Phi(mstar) + phi(mstar) * (z - mstar))
    w = U - L
    chord = Phi(L) + jnp.where(w > 0, (Phi(U) - Phi(L)) / jnp.where(w > 0, w, 1.0), 0.0) * (z - L)
    # L >= 0: concave, the envelope is Phi; U <= 0, or no tangent from the concave part above
    # (L, Phi(L)): the chord; else Phi above the tangency point and its tangent below
    return jnp.where(L >= 0, Phi(z), jnp.where((U <= 0) | (h(jnp.maximum(U, 0.0)) < 0), chord, tangent))


def _phi_cv_env(z, L, U):
    """Convex envelope of Phi on [L, U] at z, from Phi(z) = 1 - Phi(-z)."""
    return 1.0 - _phi_cc_env(-z, -U, -L)


def _pi_rule(ctx, eqn, mu, sigma, t):
    if isinstance(t, MC):  # a variable target: the formula's relaxation only
        closed = jax.make_jaxpr(_pi)(*(jnp.zeros(jnp.shape(v.lo if isinstance(v, MC) else v)) for v in (mu, sigma, t)))
        (out,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [mu, sigma, t], ctx)
        return as_mc(out)
    m, s = as_mc(mu), as_mc(sigma)
    mL, mU = m.lo, m.hi
    sL, sU = jnp.maximum(s.lo, _SIG_MIN), jnp.maximum(s.hi, _SIG_MIN)
    Phi, phi = jax.scipy.stats.norm.cdf, jax.scipy.stats.norm.pdf

    def u(x):  # max of PI over sigma in [sL, sU] at mean x
        return jnp.where(x <= t, Phi((t - x) / sL), Phi((t - x) / sU))

    def lmin(x):  # min of PI over sigma
        return jnp.where(x <= t, Phi((t - x) / sU), Phi((t - x) / sL))

    hi, lo = u(mL), lmin(mU)  # exact: PI is nonincreasing in mu
    uU = u(mU)

    def du(x):  # derivative of u on its concave part (x <= t)
        return -phi((t - x) / sL) / sL

    def h(x):  # tangent of u at x, evaluated at mU, minus u(mU): decreasing on [mL, t], <= 0 at t
        return u(x) + du(x) * (mU - x) - uU

    # concave envelope of u on [mL, mU]:
    #   mU <= t: u is concave there, the envelope is u;
    #   otherwise, if a tangent from the concave part passes above (mU, u(mU)), u up to the
    #   tangency point and its tangent after it (taken at the last bisection point with h >= 0,
    #   so the tangent stays above u on both parts); else the chord from mL.
    a = jax.lax.stop_gradient(mL)
    b = jax.lax.stop_gradient(jnp.minimum(t, mU))
    tangent = (mL < t) & (h(a) >= 0)

    def bisect(_, ab):
        lo_, hi_ = ab
        mid = 0.5 * (lo_ + hi_)
        up = h(mid) >= 0
        return jnp.where(up, mid, lo_), jnp.where(up, hi_, mid)

    mstar, _ = jax.lax.fori_loop(0, 60, bisect, (a, b))
    mstar = jax.lax.stop_gradient(mstar)
    x = m.cv  # the envelope is nonincreasing: compose with mu's convex relaxation
    tan_env = jnp.where(x <= mstar, u(x), u(mstar) + du(mstar) * (x - mstar))
    w = mU - mL
    chord = u(mL) + jnp.where(w > 0, (uU - u(mL)) / jnp.where(w > 0, w, 1.0), 0.0) * (x - mL)
    env = jnp.where(mU <= t, u(x), jnp.where(tangent, tan_env, chord))
    # Phi composed with z = (t - mu) / sigma: the division's McCormick relaxation, then Phi's exact
    # envelopes (Phi is nondecreasing: concave side at z.cc, convex side at z.cv). Not the generic
    # erf rule: composed with z in a range of width ~1e3 (sigma tiny), its convex side can sit
    # ~1e-8 above Phi.
    sp = MC(jnp.maximum(s.lo, _SIG_MIN), jnp.maximum(s.hi, _SIG_MIN), jnp.maximum(s.cv, _SIG_MIN), jnp.maximum(s.cc, _SIG_MIN))
    closed = jax.make_jaxpr(lambda a, b: (t - a) / b)(jnp.zeros(jnp.shape(m.lo)), jnp.zeros(jnp.shape(sp.lo)))
    (z,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [m, sp], ctx)
    z = as_mc(z)
    zL, zU = z.lo, z.hi
    ccz = _phi_cc_env(jnp.clip(z.cc, zL, zU), zL, zU)
    cvz = _phi_cv_env(jnp.clip(z.cv, zL, zU), zL, zU)
    ok = jnp.isfinite(zL) & jnp.isfinite(zU) & jnp.isfinite(ccz) & jnp.isfinite(cvz)
    cc = jnp.minimum(jnp.minimum(env, hi), jnp.where(ok, ccz, hi))
    cv = jnp.maximum(lo, jnp.where(ok, cvz, lo))
    return tighten(MC(lo, hi, cv, cc))


RULES["probability_of_improvement"] = _pi_rule

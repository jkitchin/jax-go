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

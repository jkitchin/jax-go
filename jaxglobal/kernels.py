"""Gaussian-process covariance functions as primitives of the squared distance.

Write a GP kernel with these instead of its ``jnp`` formula to get tight
bounds. ``d = sum_j ((x_j - x'_j) / l_j)**2`` is the weighted squared
distance, and each function is ``k(d)`` with unit variance:

* ``matern12(d) = exp(-sqrt(d))``
* ``matern32(d) = (1 + sqrt(3 d)) exp(-sqrt(3 d))``
* ``matern52(d) = (1 + sqrt(5 d) + 5 d / 3) exp(-sqrt(5 d))``
* ``se(d) = exp(-d / 2)``

Written in ``jnp``, the Matérn kernels are relaxed factor by factor (a
polynomial in ``sqrt(d)`` times an exponential), which is very loose, and
their automatic second derivative goes through ``sqrt(d)``, so the alphaBB
Hessian is infinite at every training point. As primitives, each ``k`` is one
convex, nonincreasing function of ``d`` [schweidtmann2021deterministic], so
its McCormick rule is the convex function at the concave overestimator of
``d`` and the secant, its envelopes. Its derivatives ``k'`` (concave,
nondecreasing) and ``k''`` (convex, nonincreasing) are primitives too, so
autodiff never goes through ``sqrt(d)``. ``k''`` is finite at ``d = 0`` for
``matern52`` and ``se`` only; those two are marked smooth for alphaBB.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

from . import crown, relax, univariate


def _m12(d):
    s = jnp.sqrt(d)
    return jnp.exp(-s), -jnp.exp(-s) / (2 * s), (1 + s) * jnp.exp(-s) / (4 * s**3)


def _m32(d):
    s = jnp.sqrt(3 * d)
    return (1 + s) * jnp.exp(-s), -1.5 * jnp.exp(-s), 2.25 * jnp.exp(-s) / s


def _m52(d):
    s = jnp.sqrt(5 * d)
    return (1 + s + s * s / 3) * jnp.exp(-s), -(5 / 6) * (1 + s) * jnp.exp(-s), (25 / 12) * jnp.exp(-s)


def _se(d):
    e = jnp.exp(-0.5 * d)
    return e, -0.5 * e, 0.25 * e


def _family(name, formulas, smooth):
    """Primitives ``name``, ``name_d1``, ``name_d2`` for k, k', k'' and their rules."""
    fns = [lambda d, i=i: formulas(jnp.maximum(d, 0.0))[i] for i in range(3)]  # d < 0 only from rounding
    prims = [Primitive(name), Primitive(f"{name}_d1"), Primitive(f"{name}_d2")]
    for p, fn in zip(prims, fns):
        p.def_impl(fn)
        p.def_abstract_eval(lambda d: d)
        mlir.register_lowering(p, mlir.lower_fun(fn, multiple_results=False))
        batching.defvectorized(p)
    ad.defjvp(prims[0], lambda g, d: g * prims[1].bind(d))
    ad.defjvp(prims[1], lambda g, d: g * prims[2].bind(d))
    ad.defjvp(prims[2], lambda g, d: jax.jvp(fns[2], (d,), (g,))[1])  # k''' only for third derivatives

    k, k1, k2 = prims
    # every k is convex and nonincreasing in d, k' concave and nondecreasing, k'' convex and nonincreasing:
    # the box minimizer of k and k'' and the maximizer of k' are at d.hi
    relax.RULES[k.name] = lambda ctx, eqn, x: relax.convex_rule(fns[0], x, x.hi)
    relax.RULES[k1.name] = lambda ctx, eqn, x: relax.concave_rule(fns[1], x, x.hi)
    relax.RULES[k2.name] = lambda ctx, eqn, x: relax.convex_rule(fns[2], x, x.hi)
    relax.NONNEG.update({k.name, k2.name})
    univariate.EXACT.add(k.name)  # its McCormick rule is the envelope of k(variable)
    if smooth:  # k'' finite at d = 0: alphaBB applies, and CROWN's tangents are finite
        univariate.SMOOTH_ELEMENTWISE.update({k.name, k1.name, k2.name})
        crown.UNARY[k.name] = crown._convex(fns[0], fns[1])
    return k.bind


matern12 = _family("matern12", _m12, smooth=False)
matern32 = _family("matern32", _m32, smooth=False)
matern52 = _family("matern52", _m52, smooth=True)
se = _family("se", _se, smooth=True)

KERNELS = {"matern12": matern12, "matern32": matern32, "matern52": matern52, "se": se}

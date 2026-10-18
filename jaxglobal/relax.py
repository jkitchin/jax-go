"""McCormick relaxations of arbitrary JAX functions via jaxpr interpretation.

The user writes an ordinary Python function ``f(x)`` with ``jax.numpy``.  We
trace it to a jaxpr and re-interpret every primitive on ``MC`` objects:

    MC(lo, hi, cv, cc)

where, for every point ``p`` in the box ``[lo_box, hi_box]`` (the "box" of the
original variables):

    lo <= value(p) <= hi          (interval bounds, constant on the box)
    cv(p) <= value(p) <= cc(p)    (cv convex in p, cc concave in p)

The input variable is seeded as ``MC(lo_box, hi_box, p, p)``.  Because every
rule is written with ``jax.numpy``, the relaxation itself is a JAX function:
it can be ``jit``-ed, ``vmap``-ed over many boxes, and differentiated with
``jax.grad`` to get subgradients of the convex relaxation (used for cuts).

Rules come in four kinds:

* structural / positive-linear primitives (slice, reshape, add, reduce_sum,
  ...) are applied component-wise to (lo, hi, cv, cc);
* signed linear primitives (neg, sub, mul/dot by a constant) swap components
  by sign;
* bilinear ``mul`` uses the McCormick product rule;
* univariate primitives use the McCormick composition theorem with either a
  hand-written convex/concave envelope (exp, log, sqrt, x**2, |x|, 1/x, ...)
  or an *automatically derived* alphaBB relaxation (sin, cos, tanh, erf, ...)
  whose alpha comes from interval bounds on f'' obtained by tracing
  ``jax.grad(jax.grad(primitive))`` and interpreting *that* jaxpr in interval
  mode.

Arithmetic is ordinary floating point. With ``Ctx.safe`` every rule's result
is widened outward by a few ulps of its error scale (``widen``: lo and cv down,
hi and cc up), which makes the bounds valid up to the assumption that XLA's
transcendentals are within a few ulps; see ``docs/ROADMAP.md`` for what that
does and does not guarantee.
"""

from __future__ import annotations

import threading
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core as jcore

INF = jnp.inf


class MC(NamedTuple):
    """McCormick object: interval [lo, hi] plus convex/concave relaxations."""

    lo: jax.Array
    hi: jax.Array
    cv: jax.Array
    cc: jax.Array


class Unknown(NamedTuple):
    """A boolean that depends on the variables; we only know where it is decided."""

    sure_true: jax.Array
    sure_false: jax.Array


class Ctx(NamedTuple):
    interval_only: bool = False
    multivariate: bool = False  # Tsoukalas-Mitsos products (tighter, slower to compile)
    safe: bool = False  # widen every rule's result outward by a few ulps (see ``widen``)


def is_mc(v):
    return isinstance(v, MC)


def const_mc(c):
    c = jnp.asarray(c)
    return MC(c, c, c, c)


def as_mc(v):
    return v if is_mc(v) else const_mc(v)


def tighten(m: MC) -> MC:
    return MC(m.lo, m.hi, jnp.maximum(m.cv, m.lo), jnp.minimum(m.cc, m.hi))


def select_mc(cond, a: MC, b: MC) -> MC:
    return MC(*(jnp.where(cond, u, v) for u, v in zip(a, b)))


def unbounded_like(x: MC) -> MC:
    z = jnp.zeros_like(x.lo)
    return MC(z - INF, z + INF, z - INF, z + INF)


def safe_mc(x: MC, ok, lo_fill, hi_fill) -> MC:
    """Replace x by a harmless dummy box where ``ok`` is False (avoids NaN grads)."""
    mid = 0.5 * (lo_fill + hi_fill)
    return MC(
        jnp.where(ok, x.lo, lo_fill),
        jnp.where(ok, x.hi, hi_fill),
        jnp.where(ok, x.cv, mid),
        jnp.where(ok, x.cc, mid),
    )


def _mid(z, a, b):
    """Median of (z, a, b) assuming a <= b.

    Written with ``where`` rather than min/max: JAX splits gradients 50/50 at
    ties, and at a == b that would blend the gradient of the concave
    relaxation into a subgradient of the convex one (invalid cuts).
    """
    return jnp.where(z < a, a, jnp.where(z > b, b, z))


def _secant(f, lo, hi, z):
    flo, fhi = f(lo), f(hi)
    w = hi - lo
    pos = w > 0
    slope = jnp.where(pos, (fhi - flo) / jnp.where(pos, w, 1.0), 0.0)
    return flo + slope * (z - lo)


# ---------------------------------------------------------------------------
# Generic univariate composition rules
# ---------------------------------------------------------------------------


_MODE = threading.local()  # set by apply_rule: is the current rule evaluated for intervals only?


def _interval_only():
    """True while a rule runs under ``Ctx(interval_only=True)`` (FBBT, alphaBB's interval Hessian).

    Rules then skip cv/cc: MC(lo, hi, lo, hi) is always a valid (weak) relaxation, lo/hi come
    from the same code either way, and the skipped relaxations would only be traced, lowered
    and then dead-code-eliminated, which made the bounding kernels ~5-50x larger per operation.
    """
    return getattr(_MODE, "interval_only", False)


def convex_rule(f, x: MC, xmin) -> MC:
    """f convex on [x.lo, x.hi] with box minimizer ``xmin``."""
    flo, fhi = f(x.lo), f(x.hi)
    lo = f(xmin)
    hi = jnp.maximum(flo, fhi)
    if _interval_only():
        return MC(lo, hi, lo, hi)
    cv = f(_mid(xmin, x.cv, x.cc))
    xmax = jnp.where(fhi >= flo, x.hi, x.lo)
    cc = _secant(f, x.lo, x.hi, _mid(xmax, x.cv, x.cc))
    return tighten(MC(lo, hi, cv, cc))


def concave_rule(f, x: MC, xmax) -> MC:
    """f concave on [x.lo, x.hi] with box maximizer ``xmax``."""
    r = convex_rule(lambda z: -f(z), x, xmax)
    return MC(-r.hi, -r.lo, -r.cc, -r.cv)


def _argmin_convex(dg, lo, hi, iters=60):
    """Minimizer of a convex function on [lo, hi] by bisection on its derivative."""
    lo, hi = lax.stop_gradient(lo), lax.stop_gradient(hi)

    def body(_, s):
        a, b = s
        m = 0.5 * (a + b)
        neg = dg(m) < 0
        return jnp.where(neg, m, a), jnp.where(neg, b, m)

    a, b = lax.fori_loop(0, iters, body, (lo, hi))
    return lax.stop_gradient(0.5 * (a + b))


def alphabb_rule(ctx: Ctx, f, x: MC, rng, d2rng) -> MC:
    """Relaxation of a smooth univariate f from bounds on f''.

    ucv(z) = f(z) + a/2 (z-lo)(z-hi)  is convex and <= f on [lo, hi] when
    a >= max(0, -min f''); symmetrically for the concave overestimator.
    These are then composed with x's relaxations via McCormick's theorem.
    """
    flo, fhi = rng
    if ctx.interval_only:
        return MC(flo, fhi, flo, fhi)
    d2lo, d2hi = d2rng
    a = jnp.maximum(0.0, -d2lo)
    b = jnp.maximum(0.0, d2hi)
    ok_a, ok_b = jnp.isfinite(a), jnp.isfinite(b)
    a = jnp.where(ok_a, a, 0.0)
    b = jnp.where(ok_b, b, 0.0)
    # Widen x outward by a few ulps of its magnitude. The alphaBB terms give ucv/ucc a slope of
    # about a (hi - lo) / 2 that f itself does not have, so a rounding-level error in x's own
    # relaxation (e.g. a quotient's cc one ulp below the quotient) would otherwise come out
    # amplified: on z in [0.4, 863] it put Phi's convex side 4e-8 above Phi.
    pad = SAFE_K * float(jnp.finfo(jnp.asarray(x.lo).dtype).eps) * (jnp.abs(x.lo) + jnp.abs(x.hi))
    pad = jnp.where(jnp.isfinite(pad), pad, 0.0)
    x = MC(x.lo - pad, x.hi + pad, x.cv - pad, x.cc + pad)
    L, U = x.lo, x.hi
    m = 0.5 * (L + U)
    fp = jnp.vectorize(jax.grad(f))

    def ucv(z):
        return f(z) + 0.5 * a * (z - L) * (z - U)

    def ucc(z):
        return f(z) - 0.5 * b * (z - L) * (z - U)

    zmin = _argmin_convex(lambda z: fp(z) + a * (z - m), L, U)
    zmax = _argmin_convex(lambda z: -(fp(z) - b * (z - m)), L, U)
    cv = jnp.where(ok_a, ucv(_mid(zmin, x.cv, x.cc)), flo)
    cc = jnp.where(ok_b, ucc(_mid(zmax, x.cv, x.cc)), fhi)
    return tighten(MC(flo, fhi, cv, cc))


def d2_interval(prim, params, x: MC, safe=False):
    """Interval bounds of f'' over x's box, for f = the elementwise primitive.

    f'' is obtained by autodiff, traced to a jaxpr, and evaluated with this
    same interpreter in interval-only mode.
    """

    def fs(z):
        return prim.bind(z, **params)

    d2 = jax.vmap(jax.grad(jax.grad(fs)))
    shape = jnp.shape(x.lo)
    closed = jax.make_jaxpr(d2)(jnp.zeros(max(1, x.lo.size), x.lo.dtype))
    flat = MC(*(jnp.reshape(c, (-1,)) for c in (x.lo, x.hi, x.lo, x.hi)))
    (out,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [flat], Ctx(True, safe=safe))
    out = as_mc(out)
    return jnp.reshape(out.lo, shape), jnp.reshape(out.hi, shape)


# ---------------------------------------------------------------------------
# Elementary operations on MC objects
# ---------------------------------------------------------------------------


def mc_neg(x: MC) -> MC:
    return MC(-x.hi, -x.lo, -x.cc, -x.cv)


def mc_add(a, b) -> MC:
    a, b = as_mc(a), as_mc(b)
    return MC(a.lo + b.lo, a.hi + b.hi, a.cv + b.cv, a.cc + b.cc)


def mc_sub(a, b) -> MC:
    a, b = as_mc(a), as_mc(b)
    return MC(a.lo - b.hi, a.hi - b.lo, a.cv - b.cc, a.cc - b.cv)


def mc_scale(x: MC, c) -> MC:
    c = jnp.asarray(c)
    p = c >= 0
    return MC(
        jnp.where(p, c * x.lo, c * x.hi),
        jnp.where(p, c * x.hi, c * x.lo),
        jnp.where(p, c * x.cv, c * x.cc),
        jnp.where(p, c * x.cc, c * x.cv),
    )


def mc_square(x: MC) -> MC:
    return convex_rule(jnp.square, x, jnp.clip(0.0, x.lo, x.hi))


def mc_mul(a: MC, b: MC, multivariate: bool = False) -> MC:
    """McCormick product rule (Mitsos, Chachuat & Barton 2009 [mitsos2009mccormick])."""
    xL, xU, yL, yU = a.lo, a.hi, b.lo, b.hi
    prods = jnp.stack([xL * yL, xL * yU, xU * yL, xU * yU])
    lo, hi = prods.min(0), prods.max(0)
    if _interval_only():
        return MC(lo, hi, lo, hi)

    # min/max over {c*cv, c*cc} is decided by the sign of c alone; selecting
    # by sign keeps autodiff on the convex branch at ties (cv == cc).
    def mn(c, m):
        return jnp.where(c >= 0, c * m.cv, c * m.cc)

    def mx(c, m):
        return jnp.where(c >= 0, c * m.cc, c * m.cv)

    cv = jnp.maximum(
        mn(yL, a) + mn(xL, b) - xL * yL,
        mn(yU, a) + mn(xU, b) - xU * yU,
    )
    cc = jnp.minimum(
        mx(yL, a) + mx(xU, b) - xU * yL,
        mx(yU, a) + mx(xL, b) - xL * yU,
    )
    if not multivariate:
        return tighten(MC(lo, hi, cv, cc))
    # Multivariate McCormick (Tsoukalas & Mitsos 2014 [tsoukalas2014multivariate]):
    #   cv = min over (x, y) in [a.cv, a.cc] x [b.cv, b.cc] of max(F1, F2).
    # By minimax duality this equals max over lam in [0, 1] of the single facet
    # lam F1 + (1 - lam) F2 minimized over the rectangle, i.e. of
    #   C(lam) + mn(rho_x(lam), a) + mn(rho_y(lam), b),
    # a concave piecewise-linear function of lam whose breakpoints are where
    # rho_x or rho_y changes sign. Those depend only on the box, so the rule is
    # the two standard facets (lam = 0, 1) plus at most two more, and cv is a
    # max of convex functions of the point: autodiff gives valid subgradients
    # with no argmin, tolerance or custom JVP.
    fin = jnp.isfinite(xL) & jnp.isfinite(xU) & jnp.isfinite(yL) & jnp.isfinite(yU)
    z = jnp.zeros_like(xL)
    # finite stand-ins where a bound is infinite (those elements are masked
    # below), so the extra facets never put NaN into a gradient
    xL, xU, yL, yU = (jnp.where(fin, v, z) for v in (xL, xU, yL, yU))

    def frac(num, den):  # breakpoint in (0, 1), else a duplicate of lam = 0
        ok = (den != 0) & jnp.isfinite(num) & jnp.isfinite(den)
        t = num / jnp.where(ok, den, 1.0)
        return lax.stop_gradient(jnp.where(ok & (t > 0) & (t < 1), t, z))

    def facet_lo(lam):  # lam * (yL x + xL y - xL yL) + (1 - lam) * (yU x + xU y - xU yU)
        rx, ry = lam * yL + (1 - lam) * yU, lam * xL + (1 - lam) * xU
        c = -(lam * xL * yL + (1 - lam) * xU * yU)
        return mn(rx, a) + mn(ry, b) + c

    def facet_hi(mu):  # mu * (yL x + xU y - xU yL) + (1 - mu) * (yU x + xL y - xL yU)
        rx, ry = mu * yL + (1 - mu) * yU, mu * xU + (1 - mu) * xL
        c = -(mu * xU * yL + (1 - mu) * xL * yU)
        return mx(rx, a) + mx(ry, b) + c

    for lam in (frac(yU, yU - yL), frac(xU, xU - xL)):  # rho_x = 0, rho_y = 0
        cv = jnp.maximum(cv, jnp.where(fin, facet_lo(lam), -jnp.inf))
    for mu in (frac(yU, yU - yL), frac(xL, xL - xU)):
        cc = jnp.minimum(cc, jnp.where(fin, facet_hi(mu), jnp.inf))
    return tighten(MC(lo, hi, cv, cc))


def mc_recip(x: MC) -> MC:
    pos, neg = x.lo > 0, x.hi < 0
    xp = safe_mc(x, pos, 1.0, 2.0)
    rpos = convex_rule(lambda z: 1.0 / z, xp, xp.hi)
    xn = safe_mc(x, neg, -2.0, -1.0)
    rneg = concave_rule(lambda z: 1.0 / z, xn, xn.lo)
    return select_mc(pos, rpos, select_mc(neg, rneg, unbounded_like(x)))


def mc_exp(x: MC) -> MC:
    return convex_rule(jnp.exp, x, x.lo)


LOG_DELTA = 1e-6  # relative to hi: on a box touching 0, log's overestimator is its tangent below this


def mc_log(x: MC) -> MC:
    """``log`` (concave increasing). On a box that reaches 0 or below (``lo <= 0 < hi``) only the
    part ``x > 0`` is in log's domain, so no point of the problem has ``x <= 0`` there: the interval
    is ``[-inf, log(hi)]`` and cc is the concave function that is ``log`` on ``z >= d`` and its
    tangent at ``d = LOG_DELTA * hi`` below (finite with a finite slope, above ``log`` everywhere);
    cv is -inf. Before, the whole result was unbounded, so ``-log(x - y)`` on such a box had cv = -inf."""
    ok = x.lo > 0
    xs = safe_mc(x, ok, 1.0, 2.0)
    part = (x.lo <= 0) & (x.hi > 0) & jnp.isfinite(x.hi)
    xp = safe_mc(x, part, 1.0, 2.0)
    U = xp.hi
    if _interval_only():
        rp = MC(jnp.full_like(U, -INF), jnp.log(U), jnp.full_like(U, -INF), jnp.log(U))
    else:
        d = LOG_DELTA * U
        z = _mid(U, xp.cv, xp.cc)
        cc = jnp.where(z >= d, jnp.log(jnp.maximum(z, d)), jnp.log(d) + (z - d) / d)
        rp = MC(jnp.full_like(U, -INF), jnp.log(U), jnp.full_like(U, -INF), jnp.minimum(cc, jnp.log(U)))
    return select_mc(ok, concave_rule(jnp.log, xs, xs.hi), select_mc(part, rp, unbounded_like(x)))


XLOGX_DELTA = 1e-12  # relative to min(hi, 1): below it x*log(x)'s underestimator is its tangent there


def _xlogx0(z):
    """``z * log(z)`` for ``z >= 0``, continuously extended by 0 at ``z = 0`` (NaN-free gradients)."""
    pos = z > 0
    zs = jnp.where(pos, z, 1.0)
    return jnp.where(pos, zs * jnp.log(zs), 0.0)


def mc_xlogx(x: MC) -> MC:
    """Relaxation of ``x * log(x)`` (convex on ``x >= 0``, 0 at 0, minimum ``-1/e`` at ``1/e``).

    The product rule sees ``x`` times ``log(x)``, whose interval is unbounded below when the box
    touches 0, and gives -inf. Here ``x * log(x)`` is relaxed as one convex function of ``x``.
    It is defined only for ``x > 0`` (``log``'s domain), so the box is cut to ``[max(lo, 0), hi]``:
    a lower end slightly below 0 (FBBT's outward padding) does not make the bound -inf.

    * interval: ``[g(clip(1/e, L0, U)), max(g(L0), g(U))]`` with ``L0 = max(lo, 0)``;
    * cv: McCormick composition with ``g_d``, which is ``g`` on ``z >= d`` and its tangent at
      ``d`` below (``d = max(L0, XLOGX_DELTA * min(U, 1))``). ``g_d`` is convex and finite on the
      whole line, with finite slopes, so cuts at points with ``x`` at (or below) 0 stay finite; it
      underestimates ``g`` on ``z >= 0`` (a tangent of a convex function), by at most ``d`` and
      only on ``[0, d)``;
    * cc: the secant of ``g`` over ``[L0, U]`` (its concave envelope there).
    """
    ok = x.hi > 0
    xs = safe_mc(x, ok, 1.0, 2.0)
    L0, U = jnp.maximum(xs.lo, 0.0), xs.hi
    xmin = jnp.clip(jnp.exp(-1.0), L0, U)
    gL, gU = _xlogx0(L0), _xlogx0(U)
    lo, hi = _xlogx0(xmin), jnp.maximum(gL, gU)
    if _interval_only():
        r = MC(lo, hi, lo, hi)
    else:
        d = jnp.maximum(L0, XLOGX_DELTA * jnp.minimum(U, 1.0))
        ld = jnp.log(d)

        def g_d(z):
            return jnp.where(z >= d, _xlogx0(jnp.maximum(z, d)), d * ld + (1.0 + ld) * (z - d))

        cv = g_d(_mid(xmin, xs.cv, xs.cc))
        fin = jnp.isfinite(U)
        Uf = jnp.where(fin, U, L0 + 1.0)
        xmax = jnp.where(gU >= gL, Uf, L0)
        cc = jnp.where(fin, _secant(_xlogx0, L0, Uf, _mid(xmax, xs.cv, xs.cc)), INF)
        r = tighten(MC(lo, hi, cv, cc))
    return select_mc(ok, r, unbounded_like(x))


def xlogx_operand(eqn, logs):
    """Index of ``v`` if ``eqn`` is ``v * log(v)`` (either order), else None.

    ``logs`` maps the output variable (or tape slot) of every ``log`` seen so far to its operand.
    """
    return xlogx_pair(eqn.primitive.name, eqn.invars, logs)


def xlogx_pair(name, ins, logs):
    """``xlogx_operand`` on the operands ``ins`` (jaxpr variables or tape slots) of a ``name`` row."""
    if name != "mul" or len(ins) != 2:
        return None
    u, v = ins
    if isinstance(u, jcore.Literal) or isinstance(v, jcore.Literal) or not _hashable(u) or not _hashable(v):
        return None
    if logs.get(u) is v or (isinstance(v, int) and logs.get(u) == v):
        return 1
    if logs.get(v) is u or (isinstance(u, int) and logs.get(v) == u):
        return 0
    return None


def _hashable(v):
    try:
        hash(v)
    except TypeError:
        return False
    return True


def apply_xlogx(ctx, eqn, x):
    """``mc_xlogx(x)`` for the equation ``eqn`` (``x * log(x)``), outward-widened when ``ctx.safe``."""
    prev = _interval_only()
    _MODE.interval_only = ctx.interval_only
    try:
        out = mc_xlogx(as_mc(x))
    finally:
        _MODE.interval_only = prev
    return widen(eqn, [x], out) if ctx.safe else out


def mc_abs(x: MC) -> MC:
    return convex_rule(jnp.abs, x, jnp.clip(0.0, x.lo, x.hi))


def mc_max(a, b) -> MC:
    a, b = as_mc(a), as_mc(b)
    lo, hi = jnp.maximum(a.lo, b.lo), jnp.maximum(a.hi, b.hi)
    cv = jnp.maximum(a.cv, b.cv)
    # max(a, b) = (a + b + |a - b|) / 2 gives a concave overestimator
    cc = 0.5 * (a.cc + b.cc + mc_abs(mc_sub(a, b)).cc)
    cc = jnp.where(a.hi <= b.lo, b.cc, jnp.where(b.hi <= a.lo, a.cc, cc))
    return tighten(MC(lo, hi, cv, cc))


def mc_min(a, b) -> MC:
    return mc_neg(mc_max(mc_neg(as_mc(a)), mc_neg(as_mc(b))))


def mc_pow_const(x: MC, p) -> MC:
    """x**p for a constant (possibly non-integer) exponent, on x >= 0."""
    p = jnp.asarray(p, x.lo.dtype)
    ok = jnp.where(p > 0, x.lo >= 0, x.lo > 0)
    xs = safe_mc(x, ok, 1.0, 2.0)
    tiny = jnp.finfo(x.lo.dtype).tiny

    def f(z):
        return jnp.power(jnp.maximum(z, tiny), p)

    conv = convex_rule(f, xs, jnp.where(p > 0, xs.lo, xs.hi))
    conc = concave_rule(f, xs, xs.hi)
    r = select_mc((p >= 1) | (p < 0), conv, conc)
    r = select_mc(p == 0, const_mc(jnp.ones_like(x.lo)), r)
    return select_mc(ok, r, unbounded_like(x))


# ---------------------------------------------------------------------------
# Interval ranges for univariate primitives
# ---------------------------------------------------------------------------


def sin_range(lo, hi):
    tp = 2 * jnp.pi
    has_max = jnp.pi / 2 + tp * jnp.ceil((lo - jnp.pi / 2) / tp) <= hi
    has_min = -jnp.pi / 2 + tp * jnp.ceil((lo + jnp.pi / 2) / tp) <= hi
    wide = ~(hi - lo < tp)  # also catches inf / nan
    s1, s2 = jnp.sin(lo), jnp.sin(hi)
    rlo = jnp.where(has_min | wide, -1.0, jnp.minimum(s1, s2))
    rhi = jnp.where(has_max | wide, 1.0, jnp.maximum(s1, s2))
    return rlo, rhi


def cos_range(lo, hi):
    return sin_range(lo + jnp.pi / 2, hi + jnp.pi / 2)


def increasing_range(f):
    return lambda lo, hi: (f(lo), f(hi))


# ---------------------------------------------------------------------------
# Primitive rule table
# ---------------------------------------------------------------------------

RULES: dict[str, Callable] = {}


def rule(*names):
    def deco(fn):
        for n in names:
            RULES[n] = fn
        return fn

    return deco


def _bind_fields(eqn, invals):
    outs = []
    for k in range(4):
        args = [v[k] if is_mc(v) else v for v in invals]
        outs.append(eqn.primitive.bind(*args, **eqn.params))
    if eqn.primitive.multiple_results:  # e.g. unstack, split
        return [MC(*fields) for fields in zip(*outs)]
    return MC(*outs)


# Primitives that are monotone non-decreasing and affine (or pure data
# movement) in every argument can be applied component-wise.
STRUCTURAL = [
    "slice", "squeeze", "reshape", "broadcast_in_dim", "transpose",
    "concatenate", "pad", "dynamic_slice", "dynamic_update_slice", "gather",
    "rev", "convert_element_type", "copy", "copy_p", "reduce_sum", "cumsum",
    "add", "add_any", "expand_dims", "reduce_precision", "stack",
    "unstack", "split",
]  # fmt: skip
for _n in STRUCTURAL:
    RULES[_n] = lambda ctx, eqn, *inv: _bind_fields(eqn, inv)


@rule("neg")
def _neg(ctx, eqn, x):
    return mc_neg(x)


@rule("sub")
def _sub(ctx, eqn, a, b):
    return mc_sub(a, b)


def _same_var(eqn, i, j):
    u, v = eqn.invars[i], eqn.invars[j]
    return not isinstance(u, jcore.Literal) and u is v


@rule("mul")
def _mul(ctx, eqn, a, b):
    if not is_mc(a):
        return mc_scale(b, a)
    if not is_mc(b):
        return mc_scale(a, b)
    if _same_var(eqn, 0, 1):
        return mc_square(a)
    return mc_mul(a, b, ctx.multivariate)


@rule("div")
def _div(ctx, eqn, a, b):
    if not is_mc(b):
        return mc_scale(a, 1.0 / b)
    r = mc_recip(b)
    if not is_mc(a):
        return mc_scale(r, a)
    return mc_frac(a, b, mc_mul(a, r, ctx.multivariate))


def mc_frac(a: MC, b: MC, base: MC) -> MC:
    """Tighten ``base`` (McCormick of a * (1/b)) with the convex envelope of a/b.

    For a in [aL, aU] with aL >= 0 and b in [bL, bU] with bL > 0
    (Tawarmalani & Sahinidis 2001 [tawarmalani2001semidefinite]):

        a / b >= ((a + sqrt(aL aU)) / (sqrt(aL) + sqrt(aU)))^2 / b,

    since ``(sqrt(a) - sqrt(aL))(sqrt(a) - sqrt(aU)) <= 0``. The right side is
    convex (quadratic over linear), nondecreasing in a and nonincreasing in b,
    so composing it with a.cv and b.cc keeps it convex.
    """
    if _interval_only():
        return base
    ok = (a.lo >= 0) & (b.lo > 0) & jnp.isfinite(a.hi) & jnp.isfinite(b.hi)
    sl = jnp.sqrt(jnp.where(ok, a.lo, 0.0))
    su = jnp.sqrt(jnp.where(ok, a.hi, 1.0))
    den = jnp.where(ok & (sl + su > 0), sl + su, 1.0)
    num = (jnp.where(ok, a.cv, 0.0) + sl * su) / den
    env = num**2 / jnp.where(ok, b.cc, 1.0)
    cv = jnp.where(ok, jnp.maximum(base.cv, env), base.cv)
    return tighten(MC(base.lo, base.hi, cv, base.cc))


@rule("dot_general")
def _dot(ctx, eqn, a, b):
    def dg(u, v):
        return eqn.primitive.bind(u, v, **eqn.params)

    if is_mc(a) and is_mc(b):
        (ca, cb), (ba, bb) = eqn.params["dimension_numbers"]
        if a.lo.ndim == 1 and b.lo.ndim == 1 and not ba and not bb:
            p = mc_square(a) if _same_var(eqn, 0, 1) else mc_mul(a, b, ctx.multivariate)
            return MC(*(jnp.sum(c) for c in p))
        raise NotImplementedError(
            "dot_general between two variable-dependent arrays is only "
            "supported for 1-D inner products; write it as (a * b).sum()."
        )
    if is_mc(b):
        ap, an = jnp.maximum(a, 0), jnp.minimum(a, 0)
        return MC(
            dg(ap, b.lo) + dg(an, b.hi),
            dg(ap, b.hi) + dg(an, b.lo),
            dg(ap, b.cv) + dg(an, b.cc),
            dg(ap, b.cc) + dg(an, b.cv),
        )
    bp, bn = jnp.maximum(b, 0), jnp.minimum(b, 0)
    return MC(
        dg(a.lo, bp) + dg(a.hi, bn),
        dg(a.hi, bp) + dg(a.lo, bn),
        dg(a.cv, bp) + dg(a.cc, bn),
        dg(a.cc, bp) + dg(a.cv, bn),
    )


@rule("integer_pow")
def _ipow(ctx, eqn, x):
    n = eqn.params["y"]
    if n == 0:
        return const_mc(jnp.ones_like(x.lo))
    if n < 0:
        x, n = mc_recip(x), -n
    if n == 1:
        return x
    if n % 2 == 0:
        return convex_rule(lambda z: z**n, x, jnp.clip(0.0, x.lo, x.hi))

    def f(z):
        return z**n

    conv = convex_rule(f, x, x.lo)
    conc = concave_rule(f, x, x.hi)
    d2 = (n * (n - 1) * x.lo ** (n - 2), n * (n - 1) * x.hi ** (n - 2))
    if ctx.safe:  # closed-form f'' bounds: round outward like every other rule
        pad = SAFE_K * float(jnp.finfo(x.lo.dtype).eps) * (jnp.abs(d2[0]) + jnp.abs(d2[1]))
        d2 = (d2[0] - pad, d2[1] + pad)
    mixed = alphabb_rule(ctx, f, x, (x.lo**n, x.hi**n), d2)
    return select_mc(x.lo >= 0, conv, select_mc(x.hi <= 0, conc, mixed))


@rule("square")
def _sq(ctx, eqn, x):
    return mc_square(x)


@rule("one_minus_square")
def _one_minus_sq(ctx, eqn, x):
    # 1 - x**2 (jax >= 0.11 emits it in the derivatives of tanh, asin, acos, atanh):
    # the concave function and its secant, the exact envelopes
    return mc_sub(1.0, mc_square(x))


@rule("pow")
def _pow(ctx, eqn, a, b):
    if not is_mc(b):
        return mc_pow_const(a, b)
    # a**b = exp(b * log a)
    la = mc_log(as_mc(a)) if is_mc(a) else const_mc(jnp.log(a))
    return mc_exp(mc_mul(b, la, ctx.multivariate))


@rule("exp")
def _exp(ctx, eqn, x):
    return mc_exp(x)


@rule("exp2")
def _exp2(ctx, eqn, x):
    return convex_rule(jnp.exp2, x, x.lo)


@rule("expm1")
def _expm1(ctx, eqn, x):
    return convex_rule(jnp.expm1, x, x.lo)


@rule("log")
def _log(ctx, eqn, x):
    return mc_log(x)


@rule("log1p")
def _log1p(ctx, eqn, x):
    ok = x.lo > -1
    xs = safe_mc(x, ok, 0.0, 1.0)
    return select_mc(ok, concave_rule(jnp.log1p, xs, xs.hi), unbounded_like(x))


@rule("sqrt")
def _sqrt(ctx, eqn, x):
    return mc_pow_const(x, 0.5)


@rule("rsqrt")
def _rsqrt(ctx, eqn, x):
    return mc_pow_const(x, -0.5)


@rule("cosh")
def _cosh(ctx, eqn, x):
    return convex_rule(jnp.cosh, x, jnp.clip(0.0, x.lo, x.hi))


@rule("abs")
def _abs(ctx, eqn, x):
    return mc_abs(x)


@rule("max")
def _max(ctx, eqn, a, b):
    return mc_max(a, b)


@rule("min")
def _min(ctx, eqn, a, b):
    return mc_min(a, b)


def _reduce_fold(x: MC, axes, op):
    for ax in sorted(axes, reverse=True):
        parts = [
            MC(*(lax.index_in_dim(c, i, ax, keepdims=False) for c in x))
            for i in range(x.lo.shape[ax])
        ]
        acc = parts[0]
        for p in parts[1:]:
            acc = op(acc, p)
        x = acc
    return x


@rule("reduce_max")
def _reduce_max(ctx, eqn, x):
    return _reduce_fold(x, eqn.params["axes"], mc_max)


@rule("reduce_min")
def _reduce_min(ctx, eqn, x):
    return _reduce_fold(x, eqn.params["axes"], mc_min)


# Smooth univariate primitives relaxed automatically with alphaBB.
AUTO_UNIVARIATE = {
    "sin": sin_range,
    "cos": cos_range,
    "tanh": increasing_range(jnp.tanh),
    "logistic": increasing_range(jax.nn.sigmoid),
    "atan": increasing_range(jnp.arctan),
    "asinh": increasing_range(jnp.arcsinh),
    "sinh": increasing_range(jnp.sinh),
    "erf": increasing_range(jax.scipy.special.erf),
    "erfc": lambda lo, hi: (jax.scipy.special.erfc(hi), jax.scipy.special.erfc(lo)),  # decreasing (norm.cdf uses it)
}


def _auto_rule(ctx, eqn, x):
    rng = AUTO_UNIVARIATE[eqn.primitive.name](x.lo, x.hi)
    d2 = (None, None) if ctx.interval_only else d2_interval(eqn.primitive, eqn.params, x, ctx.safe)

    def f(z):
        return eqn.primitive.bind(z, **eqn.params)

    return alphabb_rule(ctx, f, x, rng, d2)


for _n in AUTO_UNIVARIATE:
    RULES[_n] = _auto_rule


# --- booleans, comparisons and select -------------------------------------


def _as_unknown(v):
    if isinstance(v, Unknown):
        return v
    v = jnp.asarray(v, bool)
    return Unknown(v, ~v)


@rule("gt", "ge", "lt", "le", "eq", "ne")
def _cmp(ctx, eqn, a, b):
    name = eqn.primitive.name
    if _same_var(eqn, 0, 1):  # e.g. ``x != x`` NaN checks
        val = name in ("ge", "le", "eq")
        return jnp.full(jnp.shape(as_mc(a).lo), val)
    a, b = as_mc(a), as_mc(b)
    if name in ("lt", "le"):
        a, b, name = b, a, {"lt": "gt", "le": "ge"}[name]
    if name == "gt":
        return Unknown(a.lo > b.hi, a.hi <= b.lo)
    if name == "ge":
        return Unknown(a.lo >= b.hi, a.hi < b.lo)
    point = (a.lo == a.hi) & (b.lo == b.hi) & (a.lo == b.lo)
    apart = (a.hi < b.lo) | (b.hi < a.lo)
    return Unknown(point, apart) if name == "eq" else Unknown(apart, point)


@rule("convert_element_type")
def _convert(ctx, eqn, x):
    if not isinstance(x, Unknown):
        return _bind_fields(eqn, (x,))
    if eqn.params["new_dtype"] == jnp.bool_:
        return x
    # bool -> number (e.g. the derivative of a select): 1 where decided true, 0 where decided false, else [0, 1]
    t = x.sure_true.astype(eqn.params["new_dtype"])
    hi = (~x.sure_false).astype(eqn.params["new_dtype"])
    return MC(t, hi, t, hi)


@rule("and")
def _and(ctx, eqn, a, b):
    a, b = _as_unknown(a), _as_unknown(b)
    return Unknown(a.sure_true & b.sure_true, a.sure_false | b.sure_false)


@rule("or")
def _or(ctx, eqn, a, b):
    a, b = _as_unknown(a), _as_unknown(b)
    return Unknown(a.sure_true | b.sure_true, a.sure_false & b.sure_false)


@rule("not")
def _not(ctx, eqn, a):
    a = _as_unknown(a)
    return Unknown(a.sure_false, a.sure_true)


@rule("select_n")
def _select(ctx, eqn, pred, *cases):
    if not isinstance(pred, Unknown):
        return _bind_fields(eqn, (pred, *cases))
    if len(cases) != 2:
        raise NotImplementedError("select_n with variable predicate and >2 cases")
    c0, c1 = as_mc(cases[0]), as_mc(cases[1])
    lo, hi = jnp.minimum(c0.lo, c1.lo), jnp.maximum(c0.hi, c1.hi)
    hull = MC(lo, hi, lo, hi)

    def pick(k):
        return jnp.where(pred.sure_true, c1[k], jnp.where(pred.sure_false, c0[k], hull[k]))

    return MC(*(pick(k) for k in range(4)))


# ---------------------------------------------------------------------------
# Outward rounding ("safe" mode)
# ---------------------------------------------------------------------------

SAFE_K = 4.0  # ulps: correctly rounded +,-,*,/ need 1; XLA's exp/log/tanh/sin are assumed within a few
# Data movement and exact selection: no rounding happens, so nothing to widen.
NOROUND = {
    "slice", "squeeze", "reshape", "broadcast_in_dim", "transpose", "concatenate", "pad", "dynamic_slice",
    "dynamic_update_slice", "gather", "rev", "convert_element_type", "copy", "copy_p", "expand_dims",
    "reduce_precision", "stack", "unstack", "split", "select_n", "neg",
}  # fmt: skip
# Sums of n terms: the error scales with n times the sum of the |terms|, not with |result|.
SUMS = {"reduce_sum", "cumsum", "dot_general"}
# Outputs that are mathematically >= 0 (always, or when every operand is): widening must not push lo below 0,
# or sqrt / log of a box that touches 0 (``x**2 + y**2`` near the origin) turns into NaN / -inf.
NONNEG = {"square", "abs", "exp", "exp2", "sqrt", "rsqrt", "cosh"}
NONNEG_IF = {"add", "add_any", "mul", "div", "reduce_sum", "cumsum", "dot_general", "max"}


def _nonneg(eqn, invals, out):
    """Elementwise mask where the true result is >= 0 and the rule's own lo already says so."""
    name = eqn.primitive.name
    if name in NONNEG or (name == "integer_pow" and eqn.params["y"] % 2 == 0):
        return out.lo >= 0
    if name in NONNEG_IF:
        ok = jnp.asarray(True)
        for v in invals:
            lo = v.lo if is_mc(v) else None if isinstance(v, Unknown) else jnp.asarray(v)
            if lo is None or not jnp.issubdtype(jnp.asarray(lo).dtype, jnp.floating):
                return None
            ok = ok & jnp.all(jnp.asarray(lo) >= 0)
        return (out.lo >= 0) & ok
    return None


def _scale(v):
    """Box-constant magnitude of an operand (None if not numeric); non-finite entries count as 0."""
    if is_mc(v):
        m = jnp.maximum(jnp.abs(v.lo), jnp.abs(v.hi))
    elif isinstance(v, Unknown):
        return None
    else:
        a = jnp.asarray(v)
        if not jnp.issubdtype(a.dtype, jnp.floating):
            return None
        m = jnp.abs(a)
    return jnp.where(jnp.isfinite(m), m, 0.0)


def _fit(s, shape):
    """``s`` broadcast to ``shape`` when it is elementwise-compatible, else its max (coarser, still valid)."""
    if jnp.shape(s) == tuple(shape):
        return s
    try:
        return jnp.broadcast_to(s, shape)
    except ValueError:
        return jnp.max(s)


def _sum_count(eqn, x):
    name = eqn.primitive.name
    if name == "reduce_sum":
        axes = eqn.params["axes"]
        return int(np.prod([x.shape[a] for a in axes])) if axes else 1
    if name == "cumsum":
        return int(x.shape[eqn.params["axis"]])
    (ca, _), _ = eqn.params["dimension_numbers"]
    return int(np.prod([x.shape[a] for a in ca])) if ca else 1


def _fabs(v):
    """Finite magnitude of a (possibly infinite) array."""
    a = jnp.abs(v)
    return jnp.where(jnp.isfinite(a), a, 0.0)


def widen(eqn, invals, out):
    """Outward-rounded ``out``: lo, cv down and hi, cc up by ``SAFE_K`` ulps of the rule's error scale.

    The margin is a function of the box only (``lo``/``hi`` of the result and
    of the operands), not of the evaluation point, so cv stays a convex
    function of the point with unchanged subgradients and only its value
    moves. The error scale is the magnitude of the result plus the sum and the
    product of the operand magnitudes (the terms a McCormick facet combines;
    their cancellation is what the result's own magnitude would miss), and the
    square of a unary operand's (alphaBB shifts by ``a (z - L)(z - U)``).
    Sums of n terms use n times the sum of the magnitudes of the terms.
    """
    name = eqn.primitive.name
    if not is_mc(out) or name in NOROUND:
        return out
    dt = out.lo.dtype
    if not jnp.issubdtype(dt, jnp.floating):
        return out
    eps = float(jnp.finfo(dt).eps)
    shape = jnp.shape(out.lo)
    s_out = jnp.maximum(_fabs(out.lo), _fabs(out.hi))
    ss = [s for s in map(_scale, invals) if s is not None]
    if name in SUMS and ss and len(ss) == len(invals):
        x0 = invals[0].lo if is_mc(invals[0]) else jnp.asarray(invals[0])
        S = (_sum_count(eqn, x0) + 1) * _fit(eqn.primitive.bind(*ss, **eqn.params), shape)
    elif ss:
        fit = [_fit(s, shape) for s in ss]
        S = sum(fit)
        prod = fit[0]
        for s in fit[1:]:
            prod = prod * s
        S = S + prod + (fit[0] ** 2 if len(fit) == 1 else 0.0)
    else:
        S = 0.0
    m = lax.stop_gradient(SAFE_K * eps * (s_out + S))
    dn = lambda v: jnp.where(jnp.isfinite(v), v - m, v)  # noqa: E731
    up = lambda v: jnp.where(jnp.isfinite(v), v + m, v)  # noqa: E731
    lo, cv = dn(out.lo), dn(out.cv)
    nn = _nonneg(eqn, invals, out)
    if nn is not None:
        lo, cv = jnp.where(nn, jnp.maximum(lo, 0.0), lo), jnp.where(nn, jnp.maximum(cv, 0.0), cv)
    return MC(lo, up(out.hi), cv, up(out.cc))


def apply_rule(ctx, eqn, invals):
    """``RULES[primitive](ctx, eqn, *invals)``, outward-widened when ``ctx.safe``."""
    prev = _interval_only()
    _MODE.interval_only = ctx.interval_only
    try:
        outs = RULES[eqn.primitive.name](ctx, eqn, *invals)
    finally:
        _MODE.interval_only = prev
    if not ctx.safe or eqn.primitive.multiple_results:
        return outs
    return widen(eqn, invals, outs)


# ---------------------------------------------------------------------------
# The interpreter
# ---------------------------------------------------------------------------

SUBJAXPR_PARAMS = ("jaxpr", "call_jaxpr", "fun_jaxpr")


_READS = {"slice", "squeeze", "reshape", "expand_dims", "convert_element_type"}  # element reads (see eval_jaxpr_mc)


def _relaxable(v):
    return isinstance(v, (MC, Unknown))


def eval_jaxpr_mc(jaxpr, consts, args, ctx: Ctx = Ctx()):
    env = {}

    def read(v):
        return v.val if isinstance(v, jcore.Literal) else env[v]

    for v, c in zip(jaxpr.constvars, consts):
        env[v] = c
    for v, a in zip(jaxpr.invars, args):
        env[v] = a

    # ``x * log(x)`` is relaxed as one function (``mc_xlogx``): ``logs`` maps the output of every log
    # to its operand, and ``canon`` merges repeated element reads (``x[0] * jnp.log(x[0])`` slices
    # ``x`` twice) so that the two operands are recognized as the same value
    logs, canon, seen = {}, {}, {}
    for eqn in jaxpr.eqns:
        invals = [read(v) for v in eqn.invars]
        name = eqn.primitive.name
        ins = [canon.get(v, v) if not isinstance(v, jcore.Literal) else v for v in eqn.invars]
        if name in _READS and len(ins) == 1 and not isinstance(ins[0], jcore.Literal):
            key = (name, ins[0], repr(sorted(eqn.params.items())))
            canon[eqn.outvars[0]] = seen.setdefault(key, eqn.outvars[0])
        if name == "log" and not isinstance(ins[0], jcore.Literal):
            logs[eqn.outvars[0]] = ins[0]
        xl = xlogx_pair(name, ins, logs) if name == "mul" else None
        if not any(_relaxable(v) for v in invals):
            outs = eqn.primitive.bind(*invals, **eqn.params)
        elif xl is not None and is_mc(invals[xl]):
            outs = apply_xlogx(ctx, eqn, invals[xl])
        elif name in RULES:
            outs = apply_rule(ctx, eqn, invals)
        else:
            sub = next((eqn.params[p] for p in SUBJAXPR_PARAMS if p in eqn.params), None)
            if sub is None:
                raise NotImplementedError(f"no relaxation rule for primitive '{name}'")
            outs = eval_jaxpr_mc(
                getattr(sub, "jaxpr", sub), getattr(sub, "consts", ()), invals, ctx
            )
        if eqn.primitive.multiple_results:
            for v, o in zip(eqn.outvars, outs):
                env[v] = o
        else:
            env[eqn.outvars[0]] = outs
    return [read(v) for v in jaxpr.outvars]


def relax(f, multivariate=False, safe=False):
    """Return ``rf(x, lo, hi) -> MC``: relaxations of ``f`` on the box [lo, hi] at x.

    ``multivariate=True`` relaxes products with the multivariate McCormick rule;
    ``safe=True`` rounds every result outward (``widen``).
    """
    ctx = Ctx(multivariate=multivariate, safe=safe)

    def rf(x, lo, hi):
        closed = jax.make_jaxpr(f)(x)
        (out,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [MC(lo, hi, x, x)], ctx)
        return as_mc(out)

    return rf


def interval(f, safe=False):
    """Return ``fi(lo, hi) -> (lo, hi)``: natural interval extension of ``f``."""
    rf = relax(f, safe=safe)

    def fi(lo, hi):
        out = rf(0.5 * (lo + hi), lo, hi)
        return out.lo, out.hi

    return fi

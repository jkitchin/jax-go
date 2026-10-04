"""CROWN-style backward linear bound propagation over the tape (Zhang et al. 2018).

Every nonlinear primitive of the tape gets, on the interval ``[l, u]`` of its
input, a linear lower and upper relaxation ``a_l z + b_l <= g(z) <= a_u z + b_u``.
Affine primitives (add, matmul with constants, reshape, ...) are exact. A
backward pass from the scalar output carries one coefficient array per slot:
at a nonlinear node the lower relaxation is used where the coefficient is
nonnegative and the upper one where it is negative, so the output is bounded
below by an affine function of the input, ``f(x) >= A.x + c``, and

    min_box f >= c + sum_i min(A_i lo_i, A_i hi_i).

Interval bounds of the intermediate slots (the ``[l, u]`` above) come from the
forward interval pass of ``fbbt``; with ``deep=True`` they are tightened by
running the same backward pass from every element of the slot, in layer order
(this is what makes the bound through composed activations tight). The
backward passes are ``vjp``-s of the affine nodes, so the whole bound is a pure
function of fixed-shape arrays: it ``jit``-s and ``vmap``-s over boxes.

Whether CROWN applies is decided when the tape is traced: any primitive on the
path from x to f without a rule makes the bound ``-inf`` (no error). Floating
point is ordinary, not outward rounded.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .fbbt import Const, _forward, make_tape, tape_cache
from .relax import is_mc

INF = jnp.inf


class Unsupported(Exception):
    """A primitive without a CROWN rule."""


# ---------------------------------------------------------------------------
# Relaxations of unary primitives: rule(l, u) -> (a_l, b_l, a_u, b_u)
# ---------------------------------------------------------------------------


def _secant(g, dg, l, u):
    w = u - l
    tiny = w <= 1e-9 * (1 + jnp.abs(l) + jnp.abs(u))
    m = 0.5 * (l + u)
    s = jnp.where(tiny, dg(m), (g(u) - g(l)) / jnp.where(tiny, 1.0, w))
    return s, jnp.where(tiny, g(m) - s * m, g(l) - s * l)


def _tangent(g, dg, d):
    a = dg(d)
    return a, g(d) - a * d


def _guard(ok, out):
    """Where ``ok`` is false the primitive has no relaxation: lines at -inf / +inf."""
    z = jnp.zeros_like(out[0])
    return (jnp.where(ok, out[0], z), jnp.where(ok, out[1], -INF), jnp.where(ok, out[2], z), jnp.where(ok, out[3], INF))


def _convex(g, dg, valid=None):
    def rule(l, u):
        out = _tangent(g, dg, 0.5 * (l + u)) + _secant(g, dg, l, u)
        return out if valid is None else _guard(valid(l, u), out)

    return rule


def _concave(g, dg, valid=None):
    cv = _convex(lambda z: -g(z), lambda z: -dg(z), valid)

    def rule(l, u):
        al, bl, au, bu = cv(l, u)
        return -au, -bu, -al, -bl  # valid=False rows: -(+inf) = -inf below, +inf above

    return rule


def _sshape(g, dg, iters=40):
    """Increasing functions convex on z <= 0 and concave on z >= 0 (tanh, sigmoid, erf, atan, asinh)."""

    def rule(l, u):
        m = 0.5 * (l + u)
        sec = _secant(g, dg, l, u)
        tan = _tangent(g, dg, m)
        # crossing: lower line through (u, g(u)) tangent at d in [l, 0] (d = lo has h <= 0, a valid line below g)
        hl = lambda d: g(d) + dg(d) * (u - d) - g(u)  # noqa: E731
        # upper line through (l, g(l)) tangent at d in [0, u] (d = hi has h >= 0, a valid line above g)
        hu = lambda d: g(d) + dg(d) * (l - d) - g(l)  # noqa: E731

        def bis(_, c):
            a, b, p, q = c
            mid = 0.5 * (a + b)
            neg = hl(mid) <= 0
            a, b = jnp.where(neg, mid, a), jnp.where(neg, b, mid)
            mid = 0.5 * (p + q)
            big = hu(mid) > 0
            return a, b, jnp.where(big, p, mid), jnp.where(big, mid, q)

        z = jnp.zeros_like(l)
        dl, _, _, du = lax.fori_loop(0, iters, bis, (l, z, z, u))
        lo_t = _tangent(g, dg, jnp.minimum(dl, 0.0))
        up_t = _tangent(g, dg, jnp.maximum(du, 0.0))
        lo_c = tuple(jnp.where(hl(l) >= 0, s, t) for s, t in zip(sec, lo_t))
        up_c = tuple(jnp.where(hu(u) <= 0, s, t) for s, t in zip(sec, up_t))
        pos, neg = l >= 0, u <= 0
        pick = lambda p, n, c: jnp.where(pos, p, jnp.where(neg, n, c))  # noqa: E731
        return (pick(sec[0], tan[0], lo_c[0]), pick(sec[1], tan[1], lo_c[1]),
                pick(tan[0], sec[0], up_c[0]), pick(tan[1], sec[1], up_c[1]))  # fmt: skip

    return rule


def _maxc(l, u, c):
    """max(z, c): identity above c, constant below, chord above and the better of {z, c} below when crossing."""
    above, below = l >= c, u <= c
    d = u - l
    s = (u - c) / jnp.where(d > 0, d, 1.0)
    one = u - c > c - l
    al = jnp.where(above, 1.0, jnp.where(below, 0.0, jnp.where(one, 1.0, 0.0)))
    bl = jnp.where(above | (one & ~below), 0.0, c)
    au = jnp.where(above, 1.0, jnp.where(below, 0.0, s))
    bu = jnp.where(above, 0.0, jnp.where(below, c, c - s * l))
    return al, bl, au, bu


def _minc(l, u, c):
    """min(z, c) = -max(-z, -c)."""
    al, bl, au, bu = _maxc(-u, -l, -c)
    return au, -bu, al, -bl


_TANH = _sshape(jnp.tanh, lambda z: 1 - jnp.tanh(z) ** 2)
_SIG = _sshape(jax.nn.sigmoid, lambda z: jax.nn.sigmoid(z) * (1 - jax.nn.sigmoid(z)))
_ERF = _sshape(jax.scipy.special.erf, lambda z: 2 / jnp.sqrt(jnp.pi) * jnp.exp(-z * z))
_ATAN = _sshape(jnp.arctan, lambda z: 1 / (1 + z * z))
_ASINH = _sshape(jnp.arcsinh, lambda z: 1 / jnp.sqrt(1 + z * z))

UNARY = {
    "tanh": _TANH,
    "logistic": _SIG,
    "erf": _ERF,
    "atan": _ATAN,
    "asinh": _ASINH,
    "exp": _convex(jnp.exp, jnp.exp),
    "expm1": _convex(jnp.expm1, jnp.exp),
    "cosh": _convex(jnp.cosh, jnp.sinh),
    "square": _convex(lambda z: z * z, lambda z: 2 * z),
    "abs": _convex(jnp.abs, jnp.sign),
    "log": _concave(jnp.log, lambda z: 1 / z, lambda l, u: l > 0),
    "log1p": _concave(jnp.log1p, lambda z: 1 / (1 + z), lambda l, u: l > -1),
    "sqrt": _concave(jnp.sqrt, lambda z: 0.5 / jnp.sqrt(z), lambda l, u: (l >= 0) & (u > 0)),
}


def _ipow(n):
    if n == 1:
        return lambda l, u: (jnp.ones_like(l), jnp.zeros_like(l), jnp.ones_like(l), jnp.zeros_like(l))
    if n >= 2 and n % 2 == 0:
        return _convex(lambda z: z**n, lambda z: n * z ** (n - 1))
    raise Unsupported(f"integer_pow y={n}")


# ---------------------------------------------------------------------------
# The plan: classify every tape node on a path from x
# ---------------------------------------------------------------------------

AFFINE = {
    "add", "add_any", "sub", "neg", "mul", "div", "dot_general", "broadcast_in_dim", "reshape", "reduce_sum",
    "squeeze", "expand_dims", "slice", "concatenate", "convert_element_type", "transpose", "rev", "copy", "copy_p",
    "reduce_precision", "select_n", "pad", "gather", "dynamic_slice", "cumsum",
}  # fmt: skip


def make_plan(tape):
    """List of (node, kind, extra) for the nodes that depend on x; raises Unsupported."""
    var = {0}
    isv = lambda a: not isinstance(a, Const) and a in var  # noqa: E731
    plan = []
    for node in tape.nodes:
        vi = [i for i, a in enumerate(node.ins) if isv(a)]
        if not vi:
            continue
        name = node.eqn.primitive.name
        if len(node.outs) != 1:
            raise Unsupported(name)
        var.update(node.outs)
        if name == "integer_pow":
            plan.append((node, "unary", _ipow(node.eqn.params["y"])))
        elif name in UNARY:
            plan.append((node, "unary", UNARY[name]))
        elif name in ("max", "min"):
            if len(vi) != 1:
                raise Unsupported(f"{name} of two variables")
            plan.append((node, "maxc", (vi[0], _maxc if name == "max" else _minc)))
        elif name == "mul" and len(vi) == 2:
            if node.ins[0] == node.ins[1]:
                plan.append((node, "unary", UNARY["square"]))
            else:
                plan.append((node, "bil", None))
        elif name in AFFINE:
            if name == "mul" or name == "dot_general":
                ok = len(vi) == 1
            elif name == "div":
                ok = vi == [0]
            elif name == "select_n":
                ok = 0 not in vi
            elif name == "concatenate":
                ok = True
            else:
                ok = name in ("add", "add_any", "sub") or vi == [0]
            if not ok:
                raise Unsupported(f"{name} of variables")
            plan.append((node, "aff", tuple(vi)))
        else:
            raise Unsupported(name)
    return plan


def _plan_of(f):
    def build(x):
        tape = make_tape(f, x)
        try:
            return tape, make_plan(tape)
        except Unsupported:
            return tape, None

    return tape_cache(build)


def crown_supported(f, x):
    """True if every primitive of ``f`` on a path from ``x`` has a CROWN rule."""
    return _plan_of(f)(jnp.asarray(x))[1] is not None


_ACT = {"tanh", "logistic", "erf", "atan", "asinh", "max", "min"}


def crown_auto(f, x):
    """True if CROWN applies to ``f`` and ``f`` composes activations (an S-shaped function or ReLU/max with a constant)."""
    plan = _plan_of(f)(jnp.asarray(x))[1]
    return plan is not None and sum(n.eqn.primitive.name in _ACT for n, _, _ in plan) > 0


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------


def _unb(v, shape):
    """Sum a broadcast result back to an operand of ``shape``."""
    shape = tuple(shape)
    if v.shape == shape:
        return v
    v = v.sum(axis=tuple(range(v.ndim - len(shape))))
    ax = tuple(i for i, s in enumerate(shape) if s == 1 and v.shape[i] != 1)
    return v.sum(axis=ax, keepdims=True) if ax else v


def _bilinear(xl, xu, yl, yu):
    """McCormick planes of x*y: (a_x, a_y, b) lower and upper, each the one that is tighter at the box center."""
    xm, ym = 0.5 * (xl + xu), 0.5 * (yl + yu)
    planes_lo = [(yl, xl, -xl * yl), (yu, xu, -xu * yu)]  # z >= yl x + xl y - xl yl, z >= yu x + xu y - xu yu
    planes_up = [(yu, xl, -xl * yu), (yl, xu, -xu * yl)]  # z <= yu x + xl y - xl yu, z <= yl x + xu y - xu yl
    val = lambda p: p[0] * xm + p[1] * ym + p[2]  # noqa: E731

    def pick(a, b, lower):
        sel = (val(a) >= val(b)) == lower
        return [jnp.where(sel, a[i], b[i]) for i in range(3)]

    lo, up = pick(planes_lo[0], planes_lo[1], True), pick(planes_up[0], planes_up[1], False)
    return tuple(lo) + tuple(up)


class _Pass:
    """One box: slot values, relaxations, and the backward pass over the plan."""

    def __init__(self, tape, plan, lo, hi, deep, cap):
        self.tape, self.plan, self.lo, self.hi, self.deep, self.cap = tape, plan, lo, hi, deep, cap
        self.vals, _ = _forward(tape, (lo, hi), {})
        self.rel, self.tight = {}, {}
        self.var = {0} | {s for node, _, _ in plan for s in node.outs}

    def value(self, a):
        return a.val if isinstance(a, Const) else self.vals[a]

    def isv(self, a):
        return not isinstance(a, Const) and a in self.var

    def box(self, slot):
        """Bounds of a variable slot: interval pass, tightened by CROWN when ``deep``."""
        if slot not in self.tight:
            v = self.vals[slot]
            l, u = v.lo, v.hi
            if self.deep and slot != 0 and l.size <= self.cap:
                m = l.size
                eye = jnp.eye(m, dtype=l.dtype).reshape((m,) + l.shape)
                lower = lambda lam: self.lower(lam, slot)[0]  # noqa: E731
                cl = jax.vmap(lower)(eye).reshape(l.shape)
                cu = -jax.vmap(lower)(-eye).reshape(l.shape)
                l2, u2 = jnp.maximum(l, cl), jnp.minimum(u, cu)
                ok = l2 <= u2
                l, u = jnp.where(ok, l2, l), jnp.where(ok, u2, u)
            self.tight[slot] = (l, u)
        return self.tight[slot]

    def relax(self):
        """Fill ``rel`` for every nonlinear node, in tape order."""
        for node, kind, ex in self.plan:
            o = node.outs[0]
            shp = self.tape.shapes[o][0]
            if kind == "unary":
                l, u = self.box(node.ins[0])
                self.rel[o] = ex(l, u)
            elif kind == "maxc":
                i, rule = ex
                l, u = self.box(node.ins[i])
                c = jnp.asarray(self.value(node.ins[1 - i]), l.dtype)
                b = lambda v: jnp.broadcast_to(v, shp)  # noqa: E731
                self.rel[o] = rule(b(l), b(u), b(c))
            elif kind == "bil":
                (xl, xu), (yl, yu) = self.box(node.ins[0]), self.box(node.ins[1])
                self.rel[o] = _bilinear(*(jnp.broadcast_to(v, shp) for v in (xl, xu, yl, yu)))

    def lower(self, lam, slot=None):
        """Affine lower bound of ``lam . value[slot]`` (slot None: the output): (bound, A, c)."""
        slot = self.out if slot is None else slot
        lams = {slot: lam}
        c = jnp.zeros((), self.lo.dtype)
        add = lambda s, v: lams.__setitem__(s, lams[s] + v if s in lams else v)  # noqa: E731
        for node, kind, ex in reversed(self.plan):
            o = node.outs[0]
            if o not in lams:
                continue
            lm = lams.pop(o)
            if kind in ("unary", "maxc"):
                al, bl, au, bu = self.rel[o]
                pos = lm >= 0
                b = jnp.where(pos, bl, bu)
                c = c + jnp.sum(jnp.where(lm == 0, 0.0, lm * b))
                g = lm * jnp.where(pos, al, au)
                src = node.ins[0 if kind == "unary" else ex[0]]
                add(src, _unb(g, self.tape.shapes[src][0]))
            elif kind == "bil":
                axl, ayl, bl, axu, ayu, bu = self.rel[o]
                pos = lm >= 0
                c = c + jnp.sum(jnp.where(lm == 0, 0.0, lm * jnp.where(pos, bl, bu)))
                for k, (ax, ay) in enumerate([(axl, axu), (ayl, ayu)]):
                    src = node.ins[k]
                    add(src, _unb(lm * jnp.where(pos, ax, ay), self.tape.shapes[src][0]))
            else:
                vi = ex
                ins = [self.value(a) for a in node.ins]
                zs = [jnp.zeros(*self.tape.shapes[node.ins[i]]) for i in vi]

                def F(*vs, ins=ins, vi=vi, node=node):
                    a = list(ins)
                    for i, v in zip(vi, vs):
                        a[i] = v
                    return node.eqn.primitive.bind(*a, **node.eqn.params)

                out0, vjp = jax.vjp(F, *zs)
                c = c + jnp.sum(lm * out0)
                for i, g in zip(vi, vjp(lm)):
                    add(node.ins[i], g)
        A = lams.get(0, jnp.zeros_like(self.lo))
        return c + jnp.sum(jnp.minimum(A * self.lo, A * self.hi)), A, c


def make_crown(f, deep=True, cap=512):
    """Return ``bound(lo, hi, return_x=False) -> lb`` for one box (``-inf`` if CROWN does not apply).

    ``f(x) >= lb`` on ``[lo, hi]`` from backward linear bound propagation. With
    ``return_x`` also the minimizer (a box vertex) of the affine underestimator.
    ``cap`` limits the size of the slots whose CROWN bounds are computed.
    """
    plan_of = _plan_of(f)

    def bound(lo, hi, return_x=False):
        tape, plan = plan_of(lo)
        out = tape.outs[0]
        res = (jnp.asarray(-INF, lo.dtype), lo)
        if plan is None:
            return res if return_x else res[0]
        if isinstance(out, Const):
            res = (jnp.sum(jnp.asarray(out.val, lo.dtype)), lo)
            return res if return_x else res[0]
        try:
            ps = _Pass(tape, plan, lo, hi, deep, cap)
            if out not in ps.var:  # f does not depend on x
                return (jnp.sum(ps.vals[out]), lo) if return_x else jnp.sum(ps.vals[out])
            ps.out = out
            ps.relax()
            lb, A, _ = ps.lower(jnp.ones(tape.shapes[out][0], lo.dtype))
        except NotImplementedError:
            return res if return_x else res[0]
        lb = jnp.where(jnp.isnan(lb), -INF, lb)
        return (lb, jnp.where(A >= 0, lo, hi)) if return_x else lb

    return bound


def crown_bound(f, lo, hi, deep=True):
    """Lower bound of ``f`` over the box ``[lo, hi]`` by CROWN (``-inf`` if not applicable)."""
    return make_crown(f, deep)(jnp.asarray(lo, float), jnp.asarray(hi, float))

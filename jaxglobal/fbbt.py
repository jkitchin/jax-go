"""Feasibility-based bound tightening (FBBT) on the jaxpr.

The functions ``f``, ``ineq`` and ``eq`` are traced into one jaxpr and
flattened (calls such as ``jit`` and ``custom_jvp_call`` are inlined) into a
tape of primitive equations. One round of FBBT is

* **forward**: interval bounds of every intermediate value, using the same
  primitive rules as the relaxations (``relax.RULES`` in interval-only mode),
  intersected with what earlier rounds learned about that value;
* **backward**: the outputs are constrained (``f <= cutoff``,
  ``ineq <= feas_tol``, ``|eq| <= feas_tol``) and each equation, visited in
  reverse, narrows the intervals of its inputs from the interval of its output
  (e.g. ``z = x + y  =>  x in Z - Y``; ``z = exp(x)  =>  x in log(Z)``).

Rounds run in a ``lax.fori_loop`` with a fixed trip count, so ``fbbt`` is a pure
function of fixed-shape arrays: it ``jit``-s and ``vmap``-s over boxes like the
bounders. A primitive without a backward rule simply passes no information to
its inputs, which is always valid. Whenever an interval becomes empty the box
contains no point with ``f <= cutoff`` that satisfies the constraints, and the
box is reported infeasible.

As with the relaxations, arithmetic is ordinary floating point. Backward
results are widened by a relative 1e-10 and ill-conditioned inverses (near the
saturation of tanh, erf, ...) are not used, but the result is not rigorous.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core as jcore

from .relax import MC, RULES, Ctx, inline_call, Unknown, _same_var, apply_rule, is_mc

INF = jnp.inf
CTX = Ctx(interval_only=True)
SAFE_CTX = Ctx(interval_only=True, safe=True)
PAD = 1e-10  # relative outward widening of backward results
EMPTY_TOL = 1e-9  # relative tolerance before a crossed interval counts as empty


# ---------------------------------------------------------------------------
# Flattening the jaxpr into a tape
# ---------------------------------------------------------------------------


class Const(NamedTuple):
    val: object


class Node(NamedTuple):
    eqn: object
    ins: tuple  # int slot or Const
    outs: tuple  # int slots


class Tape(NamedTuple):
    nodes: list
    shapes: dict  # slot -> (shape, dtype)
    outs: list  # output atoms


def _flatten(jaxpr, consts, in_atoms, nodes, shapes, seen):
    env = {}

    def read(v):
        return Const(v.val) if isinstance(v, jcore.Literal) else env[v]

    for v, c in zip(jaxpr.constvars, consts):
        env[v] = Const(c)
    for v, a in zip(jaxpr.invars, in_atoms):
        env[v] = a
    for eqn in jaxpr.eqns:
        ins = tuple(read(v) for v in eqn.invars)
        name = eqn.primitive.name
        # calls are inlined, scans unrolled, a cond on a constant index is its branch (as
        # ``relax.eval_jaxpr_mc`` does); other loops and branches raise NotImplementedError
        call = None if name in RULES else inline_call(eqn, ins, _const_value)
        if call is not None:
            outs = _flatten(*call, nodes, shapes, seen)
            for v, o in zip(eqn.outvars, outs):
                env[v] = o
            continue
        # common subexpressions (``x[0]`` written twice, a repeated term) are one node
        key = _cse_key(eqn, ins)
        if key is not None and key in seen:
            for v, k in zip(eqn.outvars, seen[key]):
                env[v] = k
            continue
        slots = []
        for v in eqn.outvars:
            k = len(shapes)
            shapes[k] = (v.aval.shape, v.aval.dtype)
            env[v] = k
            slots.append(k)
        nodes.append(Node(eqn, ins, tuple(slots)))
        if key is not None:
            seen[key] = tuple(slots)
    return [read(v) for v in jaxpr.outvars]


def _const_value(a):
    """The value of the tape atom ``a`` if it is a constant known at trace time, else None."""
    if isinstance(a, Const) and not isinstance(a.val, jax.core.Tracer):
        return a.val
    return None


def _cse_key(eqn, ins):
    """Hashable identity of an equation (primitive, params, inputs), or None."""
    if eqn.effects:
        return None
    try:
        args = tuple(("c", id(a.val)) if isinstance(a, Const) else ("s", a) for a in ins)
        key = (eqn.primitive, tuple(sorted((k, repr(v)) for k, v in eqn.params.items())), args)
        hash(key)
        return key
    except TypeError:
        return None


def _dce(nodes, outs):
    """Drop nodes none of whose results reach the outputs."""
    live = {a for a in outs if not isinstance(a, Const)}
    keep = []
    for node in reversed(nodes):
        if any(s in live for s in node.outs):
            keep.append(node)
            live.update(a for a in node.ins if not isinstance(a, Const))
    return keep[::-1]


def make_tape(fn, x):
    closed = jax.make_jaxpr(fn)(x)
    shapes = {0: (x.shape, x.dtype)}
    nodes = []
    outs = _flatten(closed.jaxpr, closed.consts, [0], nodes, shapes, {})
    return Tape(_dce(nodes, outs), shapes, outs)


def _has_tracer(tape):
    """True if a constant of the tape is a tracer of some enclosing transformation."""
    consts = [a.val for node in tape.nodes for a in node.ins if isinstance(a, Const)]
    consts += [a.val for a in tape.outs if isinstance(a, Const)]
    return any(isinstance(c, jax.core.Tracer) for c in consts)


def tape_cache(build):
    """Memoize ``build(x) -> (tape, ...)`` on the shape and dtype of ``x``.

    The relaxers and FBBT rebuild the same tape (and its structural analysis)
    every time they are traced; this keeps one per input shape. A tape whose
    constants are tracers (``fn`` closes over a value being transformed, e.g.
    a parameter under ``vmap``) belongs to that one trace and is not kept.
    """
    cache = {}

    def get(x):
        key = (tuple(jnp.shape(x)), jnp.result_type(x))
        if key not in cache:
            out = build(x)
            if _has_tracer(out[0]):
                return out
            cache[key] = out
        return cache[key]

    return get


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------


def _clean(lo, hi):
    """NaN endpoints carry no information."""
    return jnp.where(jnp.isnan(lo), -INF, lo), jnp.where(jnp.isnan(hi), INF, hi)


# Lower end of a primitive's domain (its operand): ``log(x)`` is only defined for ``x > 0``, so no
# point of the problem has ``x`` below it. The outward padding of a backward result must not cross it:
# ``x >= exp(-inf) = 0`` padded to ``x >= -1e-10`` put boxes partly outside log's domain, where the
# relaxations of ``log(x)`` and ``x * log(x)`` were unbounded.
DOMAIN_FLOOR = {"log": 0.0, "log1p": -1.0}


# Backward rules that only move elements: their results are exact, so they are not padded (padding
# them would also undo a domain floor on the way from a log's scalar operand back to ``x``).
EXACT_BACK = {"slice", "squeeze", "reshape", "broadcast_in_dim", "transpose", "rev", "concatenate", "stack",
              "expand_dims", "convert_element_type", "copy", "copy_p"}  # fmt: skip


def _floor(name, lo):
    """``lo`` raised to the domain floor of primitive ``name``'s operand (unchanged if it has none)."""
    f = DOMAIN_FLOOR.get(name)
    return lo if f is None else jnp.maximum(lo, f)


def _pad(lo, hi):
    lo = jnp.where(jnp.isfinite(lo), lo - PAD * (1 + jnp.abs(lo)), lo)
    return lo, jnp.where(jnp.isfinite(hi), hi + PAD * (1 + jnp.abs(hi)), hi)


def _empty(lo, hi):
    finite = jnp.isfinite(lo) & jnp.isfinite(hi)
    scale = jnp.where(finite, jnp.maximum(jnp.abs(lo), jnp.abs(hi)), 0.0)
    return lo > hi + EMPTY_TOL * (1 + scale)


def _meet2(a, b):
    """Intersection and its emptiness flags; slightly crossed endpoints (rounding) are reordered."""
    lo, hi = jnp.maximum(a[0], b[0]), jnp.minimum(a[1], b[1])
    bad = _empty(lo, hi)
    return jnp.where(bad, lo, jnp.minimum(lo, hi)), jnp.where(bad, hi, jnp.maximum(lo, hi)), bad


def _meet(a, b):
    lo, hi, _ = _meet2(a, b)
    return lo, hi


def _iv(v):
    if is_mc(v):
        return v.lo, v.hi
    v = jnp.asarray(v)
    return v, v


def _imul(a, b):
    p = jnp.stack(jnp.broadcast_arrays(a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]))
    p = jnp.where(jnp.isnan(p), 0.0, p)  # 0 * inf = 0 in interval arithmetic
    return p.min(0), p.max(0)


def _idiv(a, b):
    """a / b, or the whole line when b may contain 0."""
    ok = (b[0] > 0) | (b[1] < 0)
    rb = (1.0 / jnp.where(ok, b[1], 1.0), 1.0 / jnp.where(ok, b[0], 1.0))
    lo, hi = _imul(a, rb)
    return jnp.where(ok, lo, -INF), jnp.where(ok, hi, INF)


def _full(shape, dtype):
    return jnp.full(shape, -INF, dtype), jnp.full(shape, INF, dtype)


def _unbcast(c, shape):
    """Bounds for an operand of ``shape`` that was broadcast to c's shape."""
    lo, hi = c
    if lo.shape == tuple(shape):
        return lo, hi
    if len(shape) == 0 or all(s == 1 for s in shape):
        return jnp.reshape(lo.max(), shape), jnp.reshape(hi.min(), shape)
    return None


def _hole(Z, X, inv_hi, inv_lo):
    """x with g(x) in Z for even g (x^2n, |x|): |x| in [inv_lo, inv_hi]."""
    R, L = inv_hi, inv_lo
    plo, phi = jnp.maximum(X[0], L), jnp.minimum(X[1], R)
    nlo, nhi = jnp.maximum(X[0], -R), jnp.minimum(X[1], -L)
    pe, ne = ~_empty(plo, phi), ~_empty(nlo, nhi)
    lo = jnp.where(ne, nlo, jnp.where(pe, plo, INF))
    hi = jnp.where(pe, phi, jnp.where(ne, nhi, -INF))
    empty = Z[1] < 0
    return jnp.where(empty, INF, lo), jnp.where(empty, -INF, hi)


def _monotone(finv, Z, lo_ok=-INF, hi_ok=INF):
    """x in [finv(Zlo), finv(Zhi)] for increasing f.

    Endpoints of Z outside (lo_ok, hi_ok) are not inverted (ill-conditioned
    near saturation), so they give no information.
    """
    lo = jnp.where(Z[0] > lo_ok, finv(jnp.clip(Z[0], lo_ok, hi_ok)), -INF)
    hi = jnp.where(Z[1] < hi_ok, finv(jnp.clip(Z[1], lo_ok, hi_ok)), INF)
    return lo, hi


# ---------------------------------------------------------------------------
# Backward rules: (eqn, Zs, ins) -> list of candidate intervals per input
# ---------------------------------------------------------------------------

BACK = {}


def back(*names):
    def deco(fn):
        for n in names:
            BACK[n] = fn
        return fn

    return deco


def _shape(v):
    return jnp.shape(v.lo) if is_mc(v) else jnp.shape(v)


@back("add", "add_any")
def _b_add(eqn, Z, a, b):
    A, B = _iv(a), _iv(b)
    return [(Z[0] - B[1], Z[1] - B[0]), (Z[0] - A[1], Z[1] - A[0])]


@back("sub")
def _b_sub(eqn, Z, a, b):
    A, B = _iv(a), _iv(b)
    return [(Z[0] + B[0], Z[1] + B[1]), (A[0] - Z[1], A[1] - Z[0])]


@back("neg")
def _b_neg(eqn, Z, x):
    return [(-Z[1], -Z[0])]


def _sq_back(Z, X, n=2):
    R = jnp.power(jnp.maximum(Z[1], 0.0), 1.0 / n)
    L = jnp.power(jnp.maximum(Z[0], 0.0), 1.0 / n)
    return _hole(Z, X, R, L)


@back("mul")
def _b_mul(eqn, Z, a, b):
    if is_mc(a) and is_mc(b) and _same_var(eqn, 0, 1):
        c = _sq_back(Z, _iv(a))
        return [c, c]
    A, B = _iv(a), _iv(b)
    return [_idiv(Z, B), _idiv(Z, A)]


@back("div")
def _b_div(eqn, Z, a, b):
    A, B = _iv(a), _iv(b)
    return [_imul(Z, B), _idiv(A, Z)]


@back("square")
def _b_square(eqn, Z, x):
    return [_sq_back(Z, _iv(x))]


@back("one_minus_square")
def _b_one_minus_square(eqn, Z, x):
    return [_sq_back((1.0 - Z[1], 1.0 - Z[0]), _iv(x))]  # x**2 = 1 - z


@back("integer_pow")
def _b_ipow(eqn, Z, x):
    n = eqn.params["y"]
    X = _iv(x)
    if n == 0:
        return [None]
    if n < 0:
        Z, n = _idiv((jnp.ones_like(Z[0]), jnp.ones_like(Z[0])), Z), -n
        # x^-n in Z  <=>  x^n in 1/Z (when 0 is not in Z; else no information)
    if n % 2 == 0:
        return [_sq_back(Z, X, n)]

    def root(v):
        return jnp.sign(v) * jnp.power(jnp.abs(v), 1.0 / n)

    return [(root(Z[0]), root(Z[1]))]


@back("sqrt")
def _b_sqrt(eqn, Z, x):
    lo, hi = jnp.maximum(Z[0], 0.0) ** 2, jnp.where(Z[1] < 0, -INF, Z[1] ** 2)
    return [(lo, hi)]


@back("rsqrt")
def _b_rsqrt(eqn, Z, x):
    lo = jnp.where(Z[1] > 0, 1.0 / jnp.where(Z[1] > 0, Z[1], 1.0) ** 2, INF)
    hi = jnp.where(Z[0] > 0, 1.0 / jnp.where(Z[0] > 0, Z[0], 1.0) ** 2, INF)
    return [(lo, hi)]


@back("pow")
def _b_pow(eqn, Z, a, b):
    if is_mc(b) or not is_mc(a):
        return [None, None]
    p = jnp.asarray(b, a.lo.dtype)
    ok = jnp.where(p > 0, a.lo >= 0, a.lo > 0) & (p != 0)
    ps = jnp.where(p != 0, p, 1.0)
    z0, z1 = jnp.maximum(Z[0], 0.0), jnp.maximum(Z[1], 0.0)
    inc = (jnp.power(z0, 1.0 / ps), jnp.power(z1, 1.0 / ps))
    dec = (jnp.power(z1, 1.0 / ps), jnp.power(z0, 1.0 / ps))
    lo = jnp.where(p > 0, inc[0], dec[0])
    hi = jnp.where(p > 0, jnp.where(Z[1] < 0, -INF, inc[1]), dec[1])
    return [(jnp.where(ok, lo, -INF), jnp.where(ok, hi, INF)), None]


@back("exp")
def _b_exp(eqn, Z, x):
    return [(jnp.log(jnp.maximum(Z[0], 0.0)), jnp.log(jnp.maximum(Z[1], 0.0)))]


@back("exp2")
def _b_exp2(eqn, Z, x):
    return [(jnp.log2(jnp.maximum(Z[0], 0.0)), jnp.log2(jnp.maximum(Z[1], 0.0)))]


@back("expm1")
def _b_expm1(eqn, Z, x):
    return [(jnp.log1p(jnp.maximum(Z[0], -1.0)), jnp.log1p(jnp.maximum(Z[1], -1.0)))]


@back("log")
def _b_log(eqn, Z, x):
    return [(jnp.exp(Z[0]), jnp.exp(Z[1]))]


@back("log1p")
def _b_log1p(eqn, Z, x):
    return [(jnp.expm1(Z[0]), jnp.expm1(Z[1]))]


_SAT = 1e-6  # stay this far from the asymptotes of saturating functions


@back("tanh")
def _b_tanh(eqn, Z, x):
    return [_monotone(jnp.arctanh, Z, -1 + _SAT, 1 - _SAT)]


@back("logistic")
def _b_logistic(eqn, Z, x):
    return [_monotone(lambda z: jnp.log(z) - jnp.log1p(-z), Z, 0.0, 1 - _SAT)]


@back("erf")
def _b_erf(eqn, Z, x):
    return [_monotone(jax.scipy.special.erfinv, Z, -1 + _SAT, 1 - _SAT)]


@back("atan")
def _b_atan(eqn, Z, x):
    return [_monotone(jnp.tan, Z, -jnp.pi / 2 + _SAT, jnp.pi / 2 - _SAT)]


@back("sinh")
def _b_sinh(eqn, Z, x):
    return [_monotone(jnp.arcsinh, Z)]


@back("asinh")
def _b_asinh(eqn, Z, x):
    return [_monotone(jnp.sinh, Z)]


@back("abs")
def _b_abs(eqn, Z, x):
    return [_hole(Z, _iv(x), Z[1], jnp.maximum(Z[0], 0.0))]


def _bto(z, shape):
    """Broadcast z to ``shape``; a smaller (constant) operand's candidate is never used, so z stays as is."""
    return jnp.broadcast_to(z, shape) if len(shape) >= jnp.ndim(z) else z


@back("max")
def _b_max(eqn, Z, a, b):
    A, B = _iv(a), _iv(b)
    # both <= Zhi; the one that cannot reach Zlo forces the other above it
    return [
        (jnp.where(B[1] < Z[0], Z[0], -INF), _bto(Z[1], jnp.shape(A[0]))),
        (jnp.where(A[1] < Z[0], Z[0], -INF), _bto(Z[1], jnp.shape(B[0]))),
    ]


@back("min")
def _b_min(eqn, Z, a, b):
    A, B = _iv(a), _iv(b)
    return [
        (_bto(Z[0], jnp.shape(A[0])), jnp.where(B[0] > Z[1], Z[1], INF)),
        (_bto(Z[0], jnp.shape(B[0])), jnp.where(A[0] > Z[1], Z[1], INF)),
    ]


def _expand(v, axes, shape):
    return jnp.broadcast_to(jnp.expand_dims(v, tuple(axes)), shape)


@back("reduce_sum")
def _b_reduce_sum(eqn, Z, x):
    axes = eqn.params["axes"]
    X = _iv(x)
    shp = jnp.shape(X[0])
    slo, shi = X[0].sum(axis=axes), X[1].sum(axis=axes)
    # x_i in Z - (sum of the others)
    lo = _expand(Z[0], axes, shp) - (_expand(shi, axes, shp) - X[1])
    hi = _expand(Z[1], axes, shp) - (_expand(slo, axes, shp) - X[0])
    return [(lo, hi)]


@back("reduce_max")
def _b_reduce_max(eqn, Z, x):
    shp = _shape(x)
    return [(jnp.full(shp, -INF), _expand(Z[1], eqn.params["axes"], shp))]


@back("reduce_min")
def _b_reduce_min(eqn, Z, x):
    shp = _shape(x)
    return [(_expand(Z[0], eqn.params["axes"], shp), jnp.full(shp, INF))]


@back("reshape", "squeeze", "expand_dims", "convert_element_type", "copy", "copy_p", "reduce_precision")
def _b_reshape(eqn, Z, x, *rest):
    shp = _shape(x)
    return [(jnp.reshape(Z[0], shp), jnp.reshape(Z[1], shp))] + [None] * len(rest)


@back("broadcast_in_dim")
def _b_bcast(eqn, Z, x, *rest):
    shp = _shape(x)
    bd = eqn.params["broadcast_dimensions"]
    lo, hi = Z
    new = tuple(d for d in range(lo.ndim) if d not in bd)
    lo, hi = lo.max(axis=new), hi.min(axis=new)
    ones = tuple(i for i, s in enumerate(shp) if s == 1 and lo.shape[i] != 1)
    if ones:
        lo, hi = lo.max(axis=ones, keepdims=True), hi.min(axis=ones, keepdims=True)
    return [(jnp.reshape(lo, shp), jnp.reshape(hi, shp))] + [None] * len(rest)


@back("transpose")
def _b_transpose(eqn, Z, x):
    inv = tuple(int(i) for i in np.argsort(eqn.params["permutation"]))
    return [(jnp.transpose(Z[0], inv), jnp.transpose(Z[1], inv))]


@back("rev")
def _b_rev(eqn, Z, x):
    d = eqn.params["dimensions"]
    return [(lax.rev(Z[0], d), lax.rev(Z[1], d))]


@back("slice")
def _b_slice(eqn, Z, x):
    shp = _shape(x)
    st = eqn.params["strides"] or (1,) * len(shp)
    idx = tuple(slice(a, b, s) for a, b, s in zip(eqn.params["start_indices"], eqn.params["limit_indices"], st))
    lo = jnp.full(shp, -INF, Z[0].dtype).at[idx].set(Z[0])
    hi = jnp.full(shp, INF, Z[1].dtype).at[idx].set(Z[1])
    return [(lo, hi)]


@back("concatenate")
def _b_concat(eqn, Z, *xs):
    d = eqn.params["dimension"]
    out, k = [], 0
    for x in xs:
        m = _shape(x)[d]
        out.append(tuple(lax.slice_in_dim(z, k, k + m, axis=d) for z in Z))
        k += m
    return out


@back("stack")
def _b_stack(eqn, Z, *xs):
    ax = eqn.params["axis"]
    return [tuple(lax.index_in_dim(z, i, ax, keepdims=False) for z in Z) for i in range(len(xs))]


@back("dot_general")
def _b_dot(eqn, Z, a, b):
    """z = A x with A constant: x_j in (z_i - sum_{k != j} A_ik x_k) / A_ij."""
    if is_mc(a) == is_mc(b):
        return [None, None]
    var = a if is_mc(a) else b
    shp = _shape(var)
    size = int(jnp.size(var.lo))
    if size * max(1, int(jnp.size(Z[0]))) > 1_000_000:
        return [None, None]

    def lin(v):
        v = jnp.reshape(v, shp)
        args = (v, b) if is_mc(a) else (a, v)
        return jnp.reshape(eqn.primitive.bind(*args, **eqn.params), (-1,))

    M = jax.jacfwd(lin)(jnp.zeros(size, var.lo.dtype))  # (out, in)
    xl, xh = jnp.reshape(var.lo, (-1,)), jnp.reshape(var.hi, (-1,))
    clo, chi = jnp.minimum(M * xl, M * xh), jnp.maximum(M * xl, M * xh)
    clo, chi = jnp.where(M == 0, 0.0, clo), jnp.where(M == 0, 0.0, chi)  # 0 * inf
    rlo = clo.sum(1, keepdims=True) - clo  # sum over k != j
    rhi = chi.sum(1, keepdims=True) - chi
    rlo, rhi = _clean(rlo, rhi)
    zl, zh = jnp.reshape(Z[0], (-1, 1)), jnp.reshape(Z[1], (-1, 1))
    lo, hi = _idiv((zl - rhi, zh - rlo), (M, M))
    lo, hi = _clean(lo, hi)
    nz = M != 0
    lo = jnp.where(nz, lo, -INF).max(0)
    hi = jnp.where(nz, hi, INF).min(0)
    c = (jnp.reshape(lo, shp), jnp.reshape(hi, shp))
    return [c, None] if is_mc(a) else [None, c]


@back("select_n")
def _b_select(eqn, Z, pred, *cases):
    if isinstance(pred, Unknown):
        if len(cases) != 2:
            return [None] * (1 + len(cases))
        sel = [pred.sure_false, pred.sure_true]
    elif is_mc(pred):
        return [None] * (1 + len(cases))
    else:
        p = jnp.asarray(pred).astype(jnp.int32)
        sel = [p == k for k in range(len(cases))]
    out = [None]
    for s in sel:
        out.append((jnp.where(s, Z[0], -INF), jnp.where(s, Z[1], INF)))
    return out


# ---------------------------------------------------------------------------
# Forward / backward passes
# ---------------------------------------------------------------------------


def _forward(tape, x_bounds, prior, safe=False, guarded=frozenset(), fbad=None):
    """Interval value of every slot; ``prior`` holds earlier bounds per slot.

    A ``guarded`` slot (see ``guarded_slots``) may only be used in an arm of a ``jnp.where`` that
    is not taken, where its value is irrelevant (``log(x)`` at ``x < 0``): an empty interval there
    does not make the box empty. Its empty elements become unbounded and their flags go to
    ``fbad`` (a dict), which ``_backward`` counts only where the slot is used unconditionally.
    """
    vals = {0: MC(x_bounds[0], x_bounds[1], x_bounds[0], x_bounds[1])}
    empty = jnp.array(False)

    def read(a):
        return a.val if isinstance(a, Const) else vals[a]

    for node in tape.nodes:
        eqn = node.eqn
        invals = [read(a) for a in node.ins]
        if not any(isinstance(v, (MC, Unknown)) for v in invals):
            outs = eqn.primitive.bind(*invals, **eqn.params)
        else:
            outs = apply_rule(SAFE_CTX if safe else CTX, eqn, invals)
        if not eqn.primitive.multiple_results:
            outs = [outs]
        for s, o in zip(node.outs, outs):
            if is_mc(o):
                shp, dt = tape.shapes[s]
                lo, hi = _clean(*(jnp.broadcast_to(jnp.asarray(c, dt), shp) for c in (o.lo, o.hi)))
                if s in prior:
                    lo, hi, bad = _meet2((lo, hi), prior[s])  # a met interval is empty iff the meet was
                else:
                    bad = _empty(lo, hi)
                if s in guarded:
                    lo, hi = jnp.where(bad, -INF, lo), jnp.where(bad, INF, hi)
                    if fbad is not None:
                        fbad[s] = bad
                else:
                    empty |= jnp.any(bad)
                vals[s] = MC(lo, hi, lo, hi)
            else:
                vals[s] = o
    return vals, empty


# Elementwise primitives: output element i depends on input element i only, so the activity of
# output element i (``_backward``) is that of input element i.
ELEMENTWISE_BACK = {"add", "add_any", "sub", "mul", "div", "neg", "exp", "exp2", "expm1", "log", "log1p", "sqrt",
                    "rsqrt", "cbrt", "integer_pow", "square", "abs", "max", "min", "tanh", "logistic", "erf", "erfc",
                    "sin", "cos", "tan", "atan", "asinh", "sinh", "cosh", "pow", "one_minus_square", "sign",
                    "select_n", "clamp", "gt", "ge", "lt", "le", "eq", "ne", "and", "or", "not"}  # fmt: skip


# Primitives whose result is defined where an operand is NaN (``NaN > 0.5`` is False): the value
# of their operands does not reach the output, so a domain rule on an operand (``sqrt(x) > 0.5``
# narrowing ``x >= 0``) would cut points where the expression is defined.
NAN_STOP = {"gt", "ge", "lt", "le", "eq", "ne", "is_finite"}

# Equations whose operand elements reach the output elements by a known pattern (``_operand_mask``).
_RESHAPE_LIKE = {"reshape", "squeeze", "expand_dims", "convert_element_type", "copy", "copy_p", "reduce_precision"}


def stops_nan(node):
    """True if the equation's result is defined where its operands are NaN: a comparison, a cast
    to a non-float dtype, ``x ** 0``. Its operands' values do not reach the result."""
    name, p = node.eqn.primitive.name, node.eqn.params
    return name in NAN_STOP or (name == "convert_element_type" and not jnp.issubdtype(p["new_dtype"], jnp.inexact)) \
        or (name == "integer_pow" and p["y"] == 0)  # fmt: skip


def _unbcast_mask(m, shape, xp):
    """OR of the mask ``m`` over the dimensions an operand of ``shape`` was broadcast along."""
    shape = tuple(shape)
    if m.shape == shape:
        return m
    lead = m.ndim - len(shape)
    if lead < 0:
        return xp.broadcast_to(xp.any(m), shape)
    if lead:
        m = xp.any(m, axis=tuple(range(lead)))
    ones = tuple(i for i, s in enumerate(shape) if s == 1 and m.shape[i] != 1)
    if ones:
        m = xp.any(m, axis=ones, keepdims=True)
    return m if m.shape == shape else xp.broadcast_to(xp.any(m), shape)


def _set(z, idx, v, xp):
    if xp is np:
        z = z.copy()
        z[idx] = v
        return z
    return z.at[idx].set(v)


def _operand_mask(node, i, m, shape, xp=jnp, value=None):
    """Elements of operand ``i`` (of ``shape``) of ``node`` whose value reaches an element of its
    (single) output where the bool mask ``m`` is True; an operand that is not a slice, broadcast,
    reshape, ... of the output counts as reaching it wherever any element of ``m`` does (always
    safe: a False element only withholds narrowing). ``xp`` is ``np`` (static) or ``jnp``;
    ``value(slot)`` is the value of a slot that does not depend on x (an index), or None."""
    eqn = node.eqn
    name, p = eqn.primitive.name, eqn.params
    shape = tuple(shape)
    if stops_nan(node):
        return xp.zeros(shape, bool)
    if name in ELEMENTWISE_BACK or name in _ELEMENTWISE_NAN:
        return _unbcast_mask(m, shape, xp)
    if i == 0 and name in _RESHAPE_LIKE and int(np.prod(shape)) == int(np.prod(m.shape)):
        return xp.reshape(m, shape)
    if i == 0 and name == "broadcast_in_dim":
        bd = p["broadcast_dimensions"]
        new = tuple(d for d in range(m.ndim) if d not in bd)
        if new:
            m = xp.any(m, axis=new)
        ones = tuple(k for k, s in enumerate(shape) if s == 1 and m.shape[k] != 1)
        if ones:
            m = xp.any(m, axis=ones, keepdims=True)
        return xp.reshape(m, shape)
    if name == "transpose":
        return xp.transpose(m, tuple(int(k) for k in np.argsort(p["permutation"])))
    if name == "rev":
        return xp.flip(m, tuple(p["dimensions"])) if p["dimensions"] else m
    if name == "slice":
        st = p["strides"] or (1,) * len(shape)
        idx = tuple(slice(a, b, c) for a, b, c in zip(p["start_indices"], p["limit_indices"], st))
        return _set(xp.zeros(shape, bool), idx, m, xp)
    if name == "concatenate":
        d = p["dimension"]
        k = sum(int(v.aval.shape[d]) for v in eqn.invars[:i])
        idx = (slice(None),) * d + (slice(k, k + shape[d]),)
        return m[idx]
    if i == 0 and name in ("reduce_sum", "reduce_max", "reduce_min", "reduce_prod"):
        return xp.broadcast_to(xp.expand_dims(m, tuple(p["axes"])), shape)
    if name == "select_n":
        return _unbcast_mask(m, shape, xp)
    if i == 0 and name in _GATHER_LIKE:
        t = _transpose_mask(node, m, shape, xp, value)
        if t is not None:
            return t
    if name in _FULL_USE:  # every operand element reaches some output element: used if all are
        return xp.broadcast_to(xp.all(m), shape)
    # no known element pattern (scatter, dynamic_update_slice, a gather at variable indices, ...):
    # an operand element may be overwritten or dropped, so none counts as reaching the output
    return xp.zeros(shape, bool)


_GATHER_LIKE = {"gather", "dynamic_slice", "pad"}
# elementwise primitives without a backward rule that propagate NaN element by element
_ELEMENTWISE_NAN = {"floor", "ceil", "round", "atan2", "asin", "acos", "acosh", "atanh", "rem", "nextafter",
                    "lgamma", "digamma", "erf_inv", "reduce_precision", "real", "imag"}  # fmt: skip
# equations whose every output element reads every operand element it can (NaN propagates through
# ``0 * NaN``): an operand element reaches the output if all output elements are used
_FULL_USE = {"dot_general", "cumsum", "cumprod", "cummax", "cummin", "cumlogsumexp", "sort",
             "stack", "se", "se_d1", "se_d2", "matern12", "matern12_d1", "matern12_d2", "matern32",
             "matern32_d1", "matern32_d2", "matern52", "matern52_d1", "matern52_d2", "gp_mean", "gp_std", "gp_acquisition",
             "expected_improvement", "probability_of_improvement"}  # fmt: skip


def _transpose_mask(node, m, shape, xp, value=None):
    """Operand mask of a gather-like equation (linear in operand 0, picking elements) by transposing
    it at the mask; None if an integer operand (index) is not a constant."""
    eqn = node.eqn
    dt = eqn.invars[0].aval.dtype
    args = [None]
    for a, v in zip(node.ins[1:], eqn.invars[1:]):
        if jnp.issubdtype(v.aval.dtype, jnp.inexact):
            args.append(jnp.zeros(v.aval.shape, v.aval.dtype))
        else:
            v = a.val if isinstance(a, Const) else (None if value is None else value(a))
            if v is None or (xp is np and isinstance(v, jax.core.Tracer)):
                return None
            args.append(v)

    def run():
        def lin(v):
            return eqn.primitive.bind(v, *args[1:], **eqn.params)

        (t,) = jax.linear_transpose(lin, jax.ShapeDtypeStruct(shape, dt))(jnp.asarray(m).astype(dt))
        return t != 0

    try:
        if xp is np:
            with jax.ensure_compile_time_eval():
                return np.asarray(run())
        return run()
    except Exception:  # an equation that is not linear in its operand: no element pattern
        return None


def _constant_slots(tape):
    """Concrete values of the integer slots that do not depend on x (``gather`` indices)."""
    known = {}
    with jax.ensure_compile_time_eval():
        for node in tape.nodes:
            if len(node.outs) != 1 or jnp.issubdtype(tape.shapes[node.outs[0]][1], jnp.inexact):
                continue
            if not all(isinstance(a, Const) or a in known for a in node.ins):
                continue
            args = [a.val if isinstance(a, Const) else known[a] for a in node.ins]
            if any(isinstance(v, jax.core.Tracer) for v in args):
                continue
            try:
                known[node.outs[0]] = np.asarray(node.eqn.primitive.bind(*args, **node.eqn.params))
            except Exception:
                continue
    return known


def value_live(tape):
    """Per slot, the bool mask of its elements whose value reaches an output (statically).

    An element is dead if no output uses it (``jnp.log(x)[0]`` uses only element 0 of
    ``log(x)``) or if it reaches the outputs only through comparison operands (``NAN_STOP``):
    the expression is defined where it is NaN. Neither its domain nor an empty interval may then
    narrow or cut the box. ``jnp.where`` cases count as used (``guarded_slots`` handles them)."""
    known = _constant_slots(tape)
    live = {}
    for a in tape.outs:
        if not isinstance(a, Const):
            live[a] = np.ones(tape.shapes[a][0], bool)
    for node in reversed(tape.nodes):
        ms = [live.get(s) for s in node.outs]
        if all(m is None or not m.any() for m in ms):
            continue
        for i, a in enumerate(node.ins):
            if isinstance(a, Const):
                continue
            shp = tuple(tape.shapes[a][0])
            if len(node.outs) == 1:
                mi = np.asarray(_operand_mask(node, i, ms[0], shp, np, known.get), bool)
            else:  # ``split``, ...: used if every output is
                mi = np.full(shp, all(m is not None and m.all() for m in ms))
            live[a] = mi if a not in live else live[a] | mi
    return live


def guarded_slots(tape):
    """Slots whose value may be irrelevant at some points or elements.

    * used, directly or through other equations, only as a case of a ``select_n``
      (``jnp.where``): ``jnp.where(x > 0.5, jnp.log(x), 0.0)`` evaluates ``log(x)`` at every point
      but uses it only where ``x > 0.5``, so neither its domain (``x > 0``) nor its interval may
      narrow ``x`` at the other points;
    * with an element that does not reach the outputs, or reaches them only through a comparison
      (``value_live``).

    Every other slot is used unconditionally: its value enters an output at every point, so the
    backward rules may narrow its inputs. ``_backward`` tracks, per element, where a guarded slot
    is used.
    """
    users = {}
    for node in tape.nodes:
        for i, a in enumerate(node.ins):
            if not isinstance(a, Const):
                users.setdefault(a, []).append((node, i))
    sure = {a for a in tape.outs if not isinstance(a, Const)}
    for node in reversed(tape.nodes):
        if not all(s in sure or s not in users for s in node.outs):
            continue
        if stops_nan(node):  # ``sqrt(x) > 0.5`` is defined (False) where ``sqrt(x)`` is NaN
            continue
        cases = node.eqn.primitive.name == "select_n"
        p = node.ins[0] if cases else None
        for i, a in enumerate(node.ins):
            if isinstance(a, Const):
                continue
            if cases and i > 0:
                if not (isinstance(p, Const) and np.all(np.asarray(p.val).astype(np.int32) == i - 1)):
                    continue
            sure.add(a)
    live = value_live(tape)
    partial = {s for s in tape.shapes if s in live and not live[s].all()}
    partial |= {s for s in tape.shapes if s not in live}
    return frozenset(s for s in tape.shapes if s != 0 and (s not in sure or s in partial))


def _fit(m, shape, elementwise):
    """Activity mask ``m`` of an equation's output, as a mask for an operand-shaped array.

    Elementwise (same shape): element by element; otherwise an operand element counts as used
    only if every output element is (always safe: it only withholds narrowing)."""
    if m is True:
        return True
    if elementwise and (jnp.shape(m) == tuple(shape) or jnp.ndim(m) == 0):
        return jnp.broadcast_to(m, shape)
    return jnp.broadcast_to(jnp.all(m), shape)


def _backward(tape, vals, out_bounds, guarded=frozenset(), fbad=None):
    """One backward sweep. Narrowing from an equation's output reaches its operands only at the
    elements where the output is used unconditionally (``act``): a ``select_n`` passes activity to
    a case only where the predicate is decided for it, so nothing (no domain of ``log``, no
    interval of an arm) narrows the operands of the arm that is not taken."""
    fbad = fbad or {}
    b = {s: (v.lo, v.hi) for s, v in vals.items() if is_mc(v)}
    act = {}  # guarded slot -> bool mask of its elements used unconditionally (OR over its users)
    empty = jnp.array(False)  # a met interval stays crossed (never reordered) iff some meet was empty
    for a, z in zip(tape.outs, out_bounds):
        if not isinstance(a, Const) and a in b and z is not None:
            lo, hi, bad = _meet2(b[a], z)
            b[a], empty = (lo, hi), empty | jnp.any(bad)

    def cur(a):
        if isinstance(a, Const):
            return a.val
        if a in b:
            return MC(b[a][0], b[a][1], b[a][0], b[a][1])
        return vals[a]

    def shape(a):
        return jnp.shape(a.val) if isinstance(a, Const) else tape.shapes[a][0]

    known = _constant_slots(tape) if guarded else {}
    for node in reversed(tape.nodes):
        name = node.eqn.primitive.name
        elementwise = name in ELEMENTWISE_BACK
        # activity of this equation's output: True (statically) unless it is a guarded slot
        m = True
        for s in node.outs:
            if s in guarded:
                ms = act.get(s, False)
                if s in fbad:  # an empty forward interval matters only where the slot is used
                    empty |= jnp.any(fbad[s] & ms)
                m = ms if m is True else m & _fit(ms, jnp.shape(m), elementwise)
        if m is not True:
            m = jnp.asarray(m)
        if guarded:
            sel = None
            if name == "select_n":
                pv = cur(node.ins[0])
                if isinstance(pv, Unknown) and len(node.ins) == 3:
                    sel = [pv.sure_false, pv.sure_true]
                elif not is_mc(pv) and not isinstance(pv, Unknown):
                    sel = [jnp.asarray(pv).astype(jnp.int32) == k for k in range(len(node.ins) - 1)]
            for i, a in enumerate(node.ins):
                if isinstance(a, Const) or a not in guarded:
                    continue
                if len(node.outs) > 1:
                    mi = jnp.ones(shape(a), bool) if m is True else _fit(m, shape(a), elementwise)
                elif name == "select_n" and i > 0:
                    mi = False if sel is None else jnp.broadcast_to(sel[i - 1], shape(node.outs[0])) & (
                        True if m is True else m)
                    mi = _operand_mask(node, i, mi, shape(a), jnp, known.get) if mi is not False else False
                elif m is True:  # static: only the elements of ``a`` the output reads (``x[0]``)
                    mi = _operand_mask(node, i, np.ones(shape(node.outs[0]), bool), shape(a), np, known.get)
                else:
                    mi = _operand_mask(node, i, jnp.broadcast_to(m, shape(node.outs[0])), shape(a), jnp, known.get)
                if mi is not False:
                    act[a] = mi if a not in act else act[a] | mi
        rule = BACK.get(name)
        if rule is None or not all(s in b for s in node.outs):
            continue
        if name == "convert_element_type" and stops_nan(node):  # int(x) <= 0 holds for x < 1, not x <= 0
            continue
        ins = [cur(a) for a in node.ins]
        if not any(is_mc(v) for v in ins):
            continue
        Z = [b[s] for s in node.outs]
        if m is not True and name in EXACT_BACK and len(Z) == 1:
            # an element move: the elements not used carry no information (unbounded), and each
            # operand element is narrowed by the used elements it was moved to only
            Z = [(jnp.where(m, Z[0][0], -INF), jnp.where(m, Z[0][1], INF))]
            m = True
        cands = rule(node.eqn, Z[0] if len(Z) == 1 else Z, *ins)
        for a, c in zip(node.ins, cands):
            if c is None or isinstance(a, Const) or a not in b:
                continue
            c = _clean(*c)
            if m is True:
                c = _unbcast(c, jnp.shape(b[a][0]))
                if c is None:
                    continue
                pl, ph = c if name in EXACT_BACK else _pad(*c)
                pl = _floor(name, pl)
            else:  # narrow only the elements used unconditionally, the domain floor included
                pl, ph = c if name in EXACT_BACK else _pad(*c)
                pl = _floor(name, pl)
                mc = _fit(m, jnp.broadcast_shapes(jnp.shape(pl), jnp.shape(ph)), elementwise)
                c = _unbcast((jnp.where(mc, pl, -INF), jnp.where(mc, ph, INF)), jnp.shape(b[a][0]))
                if c is None:
                    continue
                pl, ph = c
            lo, hi, bad = _meet2(b[a], (pl, ph))
            b[a], empty = (lo, hi), empty | jnp.any(bad)
    return b, empty


def joint_tape(f, ineq, eq, x):
    """Tape of ``x -> [f, ineq, eq]`` (each as a 1-D array; ``None`` skipped)."""
    used = [fn for fn in (f, ineq, eq) if fn is not None]
    return make_tape(lambda x: [jnp.atleast_1d(fn(x)) for fn in used], x)


def output_bounds(tape, ineq, eq, cutoff, feas_tol):
    """Constraints on the tape outputs: f <= cutoff, ineq <= tol, |eq| <= tol."""
    zs = [(-INF, cutoff)]
    if ineq is not None:
        zs.append((-INF, feas_tol))
    if eq is not None:
        zs.append((-feas_tol, feas_tol))
    out = []
    for a, (zl, zh) in zip(tape.outs, zs):
        if isinstance(a, Const):
            out.append((zl, zh))
        else:
            shp, dt = tape.shapes[a]
            out.append((jnp.full(shp, zl, dt), jnp.full(shp, zh, dt)))
    return out


def round_integers(lo, hi, integer, tol=1e-6):
    """Bounds of the ``integer`` variables (indices) rounded inward; True if one has no integer left."""
    if not len(integer):
        return lo, hi, jnp.array(False)
    idx = jnp.asarray(integer)
    il, ih = jnp.ceil(lo[idx] - tol), jnp.floor(hi[idx] + tol)
    return lo.at[idx].set(il), hi.at[idx].set(jnp.maximum(il, ih)), jnp.any(il > ih)


def propagate(tape, lo, hi, out_bounds, rounds, safe=False, integer=()):
    """FBBT on a tape: ``rounds`` backward+forward sweeps from the box [lo, hi].

    Returns ``(vals, empty)``: ``vals`` maps every slot to its final value (an
    interval ``MC(lo, hi, lo, hi)``, an ``Unknown`` or a constant) from a last
    forward sweep, so every interval is consistent with the tightened box.
    """
    # Which slots hold intervals is a property of the tape: find it by an
    # abstract evaluation instead of tracing a first forward sweep whose
    # result the first round would recompute (a round starts with a sweep
    # against the earlier bounds, here none: the box and unbounded slots).
    guarded = guarded_slots(tape)
    shapes = jax.eval_shape(lambda l, h: _forward(tape, (l, h), {}, safe, guarded)[0], lo, hi)
    slots = sorted(s for s, v in shapes.items() if is_mc(v))
    empty = jnp.array(False)
    # a constant output that violates its constraint makes every box infeasible
    for a, z in zip(tape.outs, out_bounds):
        if isinstance(a, Const):
            v = jnp.asarray(a.val)
            empty |= jnp.any((v < z[0]) | (v > z[1]))
    live = [None if isinstance(a, Const) else z for a, z in zip(tape.outs, out_bounds)]

    def fix(s, c):
        shp, dt = tape.shapes[s]
        if not jnp.issubdtype(dt, jnp.inexact):  # bounds of an integer slot may be infinite
            dt = jnp.result_type(float)
        return tuple(jnp.broadcast_to(jnp.asarray(v, dt), shp) for v in c)

    def body(_, carry):
        bounds, empty = carry
        prior = dict(zip(slots, bounds))
        fbad = {}
        vals, e1 = _forward(tape, prior[0], prior, safe, guarded, fbad)
        b, e2 = _backward(tape, vals, live, guarded, fbad)
        new = [fix(s, b[s]) for s in slots]
        xl, xh, e3 = round_integers(*new[0], integer)  # slot 0 (x) is first: slots are sorted
        new[0] = (xl, xh)
        return tuple(new), empty | e1 | e2 | e3

    init = tuple(fix(s, (lo, hi) if s == 0 else (-INF, INF)) for s in slots)
    bounds, empty = lax.fori_loop(0, rounds, body, (init, empty))
    prior = dict(zip(slots, bounds))
    vals, e = _forward(tape, prior[0], prior, safe, guarded)
    return vals, empty | e


def make_fbbt(f, ineq=None, eq=None, rounds=5, feas_tol=1e-6, safe=False, interpret="auto", integer=()):
    """Return ``fbbt(lo, hi, cutoff) -> (lo, hi, infeasible)`` for one box.

    The returned box contains every point of ``[lo, hi]`` with
    ``f(x) <= cutoff``, ``ineq(x) <= feas_tol`` and ``|eq(x)| <= feas_tol``.
    ``infeasible`` means no such point exists.

    ``interpret``: ``"auto"`` runs large, mostly scalar tapes (models read from
    ``.nl`` files) through the tape interpreter of ``fbbt_interp``, whose
    compile time does not grow with the model; ``True`` / ``False`` force it
    on (when the tape can be scalarized) / off.

    ``integer`` (indices) are integer variables: their bounds are rounded inward
    after every round, so that rounding propagates within the same call.
    """
    integer = tuple(int(i) for i in integer)
    tape_of = tape_cache(lambda x: (joint_tape(f, ineq, eq, x),))
    interp = {}

    def interpreted(tape, n):
        if n not in interp:
            from .fbbt_interp import make_propagate, use_interpreter

            on = interpret is True or (interpret == "auto" and use_interpreter(tape, n))
            # Built lazily, usually inside the first jit trace.  Build-time
            # constants must stay concrete: the propagator is cached across
            # traces, and a tracer captured here leaks into the next trace.
            with jax.ensure_compile_time_eval():
                interp[n] = make_propagate(tape, ineq, eq, n, feas_tol, rounds, safe, integer=integer) if on else None
        return interp[n]

    def fbbt(lo, hi, cutoff=INF):
        (tape,) = tape_of(lo)
        if interpret is not False and jnp.ndim(lo) == 1 and not _has_tracer(tape):
            run = interpreted(tape, lo.shape[0])
            if run is not None:
                return run(lo, hi, cutoff)
        ob = output_bounds(tape, ineq, eq, jnp.asarray(cutoff, lo.dtype), feas_tol)
        vals, empty = propagate(tape, lo, hi, ob, rounds, safe, integer)
        x = vals[0]
        return jnp.minimum(x.lo, x.hi), jnp.maximum(x.lo, x.hi), empty

    return fbbt

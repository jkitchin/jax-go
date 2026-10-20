"""Tight relaxations of univariate sub-expressions (grid hull with curvature bound).

Many nonconvex functions are built from pieces that depend on a single
variable: separable sums (Styblinski-Tang, Michalewicz), polynomial factors
like ``(4 - 2.1 a^2 + a^4 / 3) a^2``, the factors of Shubert's product. The
McCormick rules relax such a piece operation by operation and lose most of the
tightness. Here every intermediate value whose elements each depend on one
variable element (found structurally on the flattened jaxpr) and that is built
from smooth primitives gets a nearly exact convex/concave envelope instead:

1. sample the piece g on a K-point grid of its variable's range;
2. bound g'' on each grid cell by interval evaluation of the second
   derivative (``jvp`` of ``jvp``) over that cell. With ``g'' <= M`` on a cell
   of width h, g is at least its chord minus ``max(M, 0) h^2 / 8``, so lowering
   the sample values by that amount gives a piecewise-linear function below g
   (and symmetrically, with the lower bound on g'', one above it);
3. the lower convex hull of the lowered samples is a convex underestimator of
   g on the whole range. It is computed as the maximum of all lines through
   two lowered samples, each shifted down until it lies below every sample;
   the concave overestimator is symmetric.

The hull lines depend only on the box, so they are computed once per box
(``prepare``) and reused for every point the bounder evaluates. The result is
intersected with the ordinary McCormick relaxation, so it can only tighten.
Kinks (abs, max, ``where`` on a variable condition) break the curvature
argument, so pieces containing them are left to the ordinary rules.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .fbbt import Const, _has_tracer, make_tape, tape_cache
from .relax import (AUTO_UNIVARIATE, MC, RULES, SAFE_K, STRUCTURAL, Ctx, Unknown, apply_rule, ccmin, cvmax, apply_xlogx, eval_jaxpr_mc, LogTerms, Monomials, lin_consumed, mul_consumed,
                    is_mc, tighten, xlogx_pair)

INTERVAL_CTX = Ctx(interval_only=True)
SAFE_INTERVAL_CTX = Ctx(interval_only=True, safe=True)

NONSMOOTH = {"abs", "max", "min", "sign", "floor", "ceil", "round", "rem", "clamp", "reduce_max", "reduce_min",
             "gt", "ge", "lt", "le", "eq", "ne", "and", "or", "not", "select_n", "argmax", "argmin"}  # fmt: skip
# data movement / linear maps: dependencies follow by applying the op to masks
MOVE = set(STRUCTURAL) | {"reduce_max", "reduce_min"}
AFFINE = set(STRUCTURAL) | {"neg", "sub"}
# univariate ops whose McCormick rule is already the exact envelope of op(variable)
EXACT = {"exp", "exp2", "expm1", "log", "log1p", "sqrt", "rsqrt", "cosh", "square", "one_minus_square"}
# smooth elementwise primitives this module knows how to differentiate twice
SMOOTH_ELEMENTWISE = {"mul", "div", "integer_pow", "square", "one_minus_square", "pow", "exp", "exp2", "expm1",
                      "log", "log1p", "sqrt", "rsqrt", "cosh", "neg", "sub"} | set(AUTO_UNIVARIATE)  # fmt: skip


def _dep_masks(tape, n):
    """Per slot: float array ``shape + (n,)``, > 0 where the element depends on x_j."""
    masks = {0: jnp.eye(n)}
    for node in tape.nodes:
        eqn, name = node.eqn, node.eqn.primitive.name
        var = [not isinstance(a, Const) and a in masks for a in node.ins]
        if not any(var):
            continue
        out_shapes = [tape.shapes[s][0] for s in node.outs]
        ins = [masks[a] if v else (a.val if isinstance(a, Const) else None) for a, v in zip(node.ins, var)]
        try:
            if name in MOVE or name == "dot_general":
                outs = _apply_masks(eqn, node, ins, var, tape)
            elif name in RULES:  # elementwise
                acc = jnp.zeros(out_shapes[0] + (n,))
                for m, v in zip(ins, var):
                    if v:
                        shp = jnp.shape(m)[:-1]
                        m = jnp.reshape(m, (1,) * (len(out_shapes[0]) - len(shp)) + shp + (n,))
                        acc = acc + jnp.broadcast_to(jnp.abs(m), out_shapes[0] + (n,))
                outs = [acc] * len(node.outs)
            else:
                raise NotImplementedError
        except Exception:  # unknown structure: depends on everything
            total = sum(jnp.sum(jnp.abs(m), axis=tuple(range(m.ndim - 1))) for m, v in zip(ins, var) if v)
            outs = [jnp.broadcast_to(total, s + (n,)) for s in out_shapes]
        for s, m in zip(node.outs, outs):
            masks[s] = jnp.abs(jnp.asarray(m, jnp.float64))
    return masks


def _apply_masks(eqn, node, ins, var, tape):
    """Apply a linear / data-movement primitive channel-wise to dependency masks."""
    consts = []
    for a, m, v in zip(node.ins, ins, var):
        if v:
            consts.append(None)
            continue
        val = jnp.asarray(a.val if isinstance(a, Const) else 0.0)
        if eqn.primitive.name == "dot_general":
            consts.append((val != 0).astype(jnp.float64))
        elif jnp.issubdtype(val.dtype, jnp.integer) or jnp.issubdtype(val.dtype, jnp.bool_):
            consts.append(val)  # indices
        else:
            consts.append(jnp.zeros_like(val))

    def op(*cols):
        it = iter(cols)
        args = [next(it) if v else c for v, c in zip(var, consts)]
        out = eqn.primitive.bind(*args, **eqn.params)
        return out if eqn.primitive.multiple_results else [out]

    if eqn.primitive.name == "dot_general" and all(var):
        a, b = ins
        oa = jax.vmap(lambda m: eqn.primitive.bind(m, jnp.ones(m.shape[:0] + jnp.shape(b)[:-1]), **eqn.params),
                      in_axes=-1, out_axes=-1)(a)  # fmt: skip
        ob = jax.vmap(lambda m: eqn.primitive.bind(jnp.ones(jnp.shape(a)[:-1]), m, **eqn.params),
                      in_axes=-1, out_axes=-1)(b)  # fmt: skip
        return [oa + ob]
    cols = [m for m, v in zip(ins, var) if v]
    return jax.vmap(op, in_axes=-1, out_axes=-1)(*cols)


class Block(NamedTuple):
    slot: int
    dep: np.ndarray  # variable index per element (flattened)


def find_blocks(tape, n):
    """Univariate, smooth, nonlinear slots whose value is used by a non-univariate one."""
    masks = _dep_masks(tape, n)
    nz = {s: np.asarray(m) > 0 for s, m in masks.items()}
    univ = {s: bool(np.all(m.sum(-1) <= 1)) for s, m in nz.items()}
    smooth = {0: True}
    nonlin = {0: False}
    loose = {0: False}  # contains an op whose McCormick rule is not already exact
    users = {}
    for node in tape.nodes:
        name = node.eqn.primitive.name
        vin = [a for a in node.ins if not isinstance(a, Const) and a in nz]
        for a in vin:
            users.setdefault(a, []).append(node)
        if not vin:
            continue
        sm = all(smooth.get(a, False) for a in vin) and (name in AFFINE or name in SMOOTH_ELEMENTWISE
                                                          or name == "dot_general")  # fmt: skip
        is_aff = (name in AFFINE or (name in ("mul", "dot_general") and len(vin) == 1)
                  or (name == "div" and node.ins[1] not in vin))  # fmt: skip
        nl = not is_aff or any(nonlin.get(a, False) for a in vin)
        if name == "integer_pow":
            exact = node.eqn.params["y"] > 0 and node.eqn.params["y"] % 2 == 0
        elif name == "mul" and len(vin) == 2 and node.ins[0] is node.ins[1]:
            exact = True  # x * x
        elif name == "div":
            exact = len(vin) == 1  # c / x
        else:
            exact = name in EXACT
        # loose: built from a non-exact op, a nonlinear op of a nonlinear value,
        # or an affine combination of two nonlinear values
        n_nl = sum(nonlin.get(a, False) for a in vin)
        ls = any(loose.get(a, False) for a in vin) or ((n_nl >= 2) if is_aff else (not exact or n_nl >= 1))
        for s in node.outs:
            smooth[s], nonlin[s], loose[s] = sm, nl, ls
    blocks = []
    for s in nz:
        if s == 0 or not (univ.get(s) and smooth.get(s) and nonlin.get(s) and loose.get(s)):
            continue
        if not nz[s].any():
            continue
        consumers = users.get(s, [])
        outputs = [a for a in tape.outs if not isinstance(a, Const)]
        frontier = s in outputs or any(not all(univ.get(o, False) for o in node.outs) for node in consumers)
        if frontier:
            dep = np.argmax(nz[s], axis=-1).reshape(-1)
            blocks.append(Block(s, dep))
    return blocks


def _eval_tape(tape, x, want):
    """Plain (numeric) evaluation of the tape; returns the values of ``want`` slots."""
    vals = {0: x}
    for node in tape.nodes:
        args = [a.val if isinstance(a, Const) else vals[a] for a in node.ins]
        out = node.eqn.primitive.bind(*args, **node.eqn.params)
        out = out if node.eqn.primitive.multiple_results else [out]
        for s, o in zip(node.outs, out):
            vals[s] = o
    return [jnp.reshape(vals[s], (-1,)) for s in want]


def _eval_tape_iv(tape, x, want):
    """Rigorous (outward-rounded) interval value of the ``want`` slots at the point ``x``: (lo, hi)."""
    vals = {0: MC(x, x, x, x)}
    for node in tape.nodes:
        args = [a.val if isinstance(a, Const) else vals[a] for a in node.ins]
        if not any(isinstance(a, (MC, Unknown)) for a in args):
            out = node.eqn.primitive.bind(*args, **node.eqn.params)
        else:
            out = apply_rule(SAFE_INTERVAL_CTX, node.eqn, args)
        out = out if node.eqn.primitive.multiple_results else [out]
        for s, o in zip(node.outs, out):
            vals[s] = o
    iv = [vals[s] if is_mc(vals[s]) else MC(*(jnp.asarray(vals[s]),) * 4) for s in want]
    return (jnp.concatenate([jnp.reshape(m.lo, (-1,)) for m in iv]),
            jnp.concatenate([jnp.reshape(m.hi, (-1,)) for m in iv]))  # fmt: skip


class Hull(NamedTuple):
    lo: jax.Array  # (m,) interval bounds per element
    hi: jax.Array
    ls: jax.Array  # (m, L) lower lines: slope, intercept
    li: jax.Array
    us: jax.Array  # (m, L) upper lines
    ui: jax.Array


def _lines(X, G, upper):
    """Lines through every pair of points, shifted to lie below (above) all points.

    X, G: (m, K). Returns slopes and intercepts (m, P).
    """
    K = X.shape[1]
    i, j = np.triu_indices(K, 1)
    dx = X[:, j] - X[:, i]
    ok = dx > 0
    s = jnp.where(ok, (G[:, j] - G[:, i]) / jnp.where(ok, dx, 1.0), 0.0)
    b = G[:, i] - s * X[:, i]
    sign = -1.0 if upper else 1.0

    def worst(k, acc):  # largest violation over the points, one point at a time (memory m * P)
        xk = jax.lax.dynamic_slice_in_dim(X, k, 1, axis=1)
        gk = jax.lax.dynamic_slice_in_dim(G, k, 1, axis=1)
        return jnp.maximum(acc, sign * (s * xk + b - gk))

    shift = jax.lax.fori_loop(0, K, worst, jnp.full(s.shape, -jnp.inf))
    scale = 1e-12 * (1 + jnp.max(jnp.abs(G), axis=1, keepdims=True))
    b = b - (shift + scale) if not upper else b + (shift + scale)
    bad = ~jnp.isfinite(s) | ~jnp.isfinite(b)
    return jnp.where(bad, 0.0, s), jnp.where(bad, -jnp.inf if not upper else jnp.inf, b)


def _interpreters(ctx, interpret):
    """``get(tape, blocks, x)``: the ``relax_interp`` relaxation of ``tape`` (cached per tape), or None."""
    cache = {}

    def get(tape, blocks, x):
        if interpret is False or jnp.ndim(x) != 1 or _has_tracer(tape):
            return None
        if id(tape) not in cache:
            from .relax_interp import interpreted

            cache[id(tape)] = (tape, interpreted(tape, x.shape[0], ctx, blocks, interpret))
        return cache[id(tape)][1]

    return get


def make_univariate(f, K=65, multivariate=False, safe=False, interpret="auto"):
    """Return ``prepare(lo, hi) -> data`` and ``relax_with(x, lo, hi, data) -> MC``.

    With ``safe`` the grid values come from outward-rounded interval evaluation,
    the curvature bounds from outward-rounded interval arithmetic, and the hull
    lines are lowered (raised) by the rounding error of evaluating them.
    ``interpret`` (``"auto"``, True, False): evaluate ``relax_with`` with the
    tape interpreter of ``relax_interp`` (large scalar tapes, see ``relax_interp.MIN_ROWS``).
    """
    ctx = Ctx(interval_only=False, multivariate=multivariate, safe=safe)
    ictx = SAFE_INTERVAL_CTX if safe else INTERVAL_CTX
    interpreter = _interpreters(ctx, interpret)
    taylors = {}

    def taylor(tape, blocks, x):
        """``relax_interp.make_taylor2`` of ``tape`` along one direction (cached), or None."""
        if interpret is False or jnp.ndim(x) != 1 or _has_tracer(tape):
            return None
        if id(tape) not in taylors:
            from .relax_interp import make_taylor2, use_relax_interpreter

            run = None
            if interpret is True or use_relax_interpreter(tape, x.shape[0]):
                with jax.ensure_compile_time_eval():
                    run = make_taylor2(tape, x.shape[0], ictx, 1)
            taylors[id(tape)] = (tape, run)
        return taylors[id(tape)][1]

    @tape_cache
    def setup(x):
        tape = make_tape(f, x)
        with jax.ensure_compile_time_eval():  # structure only: no tracing
            return tape, find_blocks(tape, x.size)

    def prepare(lo, hi):
        tape, blocks = setup(lo)
        if not blocks:
            return []
        want = [b.slot for b in blocks]
        w = hi - lo
        ts = jnp.linspace(0.0, 1.0, K)
        G = jax.vmap(lambda t: jnp.concatenate(_eval_tape(tape, lo + t * w, want)))(ts).T  # (m, K)
        dep = np.concatenate([b.dep for b in blocks])
        X = (lo[dep][:, None] + ts[None, :] * w[dep][:, None])  # (m, K)
        h = (w[dep] / (K - 1))[:, None]

        # second directional derivative along ones = g'' of each element's own variable
        def d2(z):
            fn = lambda z: jnp.concatenate(_eval_tape(tape, z, want))  # noqa: E731
            return jax.jvp(lambda z: jax.jvp(fn, (z,), (jnp.ones_like(z),))[1], (z,), (jnp.ones_like(z),))[1]

        try:
            run = taylor(tape, blocks, lo)
            if run is not None:  # interpreted: g'' of every slot along the all-ones direction
                ids = jnp.asarray(np.concatenate([np.asarray(run.program.ids[b.slot]).reshape(-1) for b in blocks]))
                ones = jnp.ones((lo.size, 1), lo.dtype)

                def cell(a, b):
                    r = run(a, b, ones)
                    return r[4][ids, 0, 0], r[5][ids, 0, 0]
            else:
                closed = jax.make_jaxpr(d2)(lo)

                def cell(a, b):
                    (out,) = eval_jaxpr_mc(closed.jaxpr, closed.consts, [MC(a, b, a, b)], ictx)
                    out = out if is_mc(out) else MC(out, out, out, out)
                    return jnp.broadcast_to(out.lo, (dep.size,)), jnp.broadcast_to(out.hi, (dep.size,))

            dlo, dhi = jax.vmap(lambda t0, t1: cell(lo + t0 * w, lo + t1 * w))(ts[:-1], ts[1:])  # (K-1, m)
            dlo, dhi = dlo.T, dhi.T
        except NotImplementedError:
            dlo = jnp.full((dep.size, K - 1), -jnp.inf)
            dhi = jnp.full((dep.size, K - 1), jnp.inf)
        dlo = jnp.where(jnp.isnan(dlo), -jnp.inf, dlo)
        dhi = jnp.where(jnp.isnan(dhi), jnp.inf, dhi)
        # on a cell of width h:  chord - max(g'', 0) h^2/8  <=  g  <=  chord + max(-g'', 0) h^2/8
        el = jnp.maximum(dhi, 0.0) * h**2 / 8  # (m, K-1)
        eu = jnp.maximum(-dlo, 0.0) * h**2 / 8
        pad = jnp.full((dep.size, 1), 0.0)
        El = jnp.maximum(jnp.concatenate([pad, el], 1), jnp.concatenate([el, pad], 1))
        Eu = jnp.maximum(jnp.concatenate([pad, eu], 1), jnp.concatenate([eu, pad], 1))
        nan = jnp.isnan(G)
        if safe:  # grid values are enclosures: plain float evaluation may be off by more than the hull margin
            G0, G1 = jax.vmap(lambda t: _eval_tape_iv(tape, lo + t * w, want))(ts)
            G0, G1 = G0.T, G1.T
            nan = nan | jnp.isnan(G0) | jnp.isnan(G1)
            Gl = jnp.where(nan, -jnp.inf, G0 - El)
            Gu = jnp.where(nan, jnp.inf, G1 + Eu)
        else:
            Gl = jnp.where(nan, -jnp.inf, G - El)
            Gu = jnp.where(nan, jnp.inf, G + Eu)
        ls, li = _lines(X, jnp.where(jnp.isfinite(Gl), Gl, -1e300), upper=False)
        us, ui = _lines(X, jnp.where(jnp.isfinite(Gu), Gu, 1e300), upper=True)
        okl = jnp.all(jnp.isfinite(Gl), axis=1, keepdims=True)
        oku = jnp.all(jnp.isfinite(Gu), axis=1, keepdims=True)
        li = jnp.where(okl, li, -jnp.inf)
        ui = jnp.where(oku, ui, jnp.inf)
        if safe:  # rounding of ``slope * x + intercept`` at any x of the box
            xm = jnp.maximum(jnp.abs(lo[dep]), jnp.abs(hi[dep]))[:, None]
            li = li - SAFE_K * float(jnp.finfo(lo.dtype).eps) * (jnp.abs(ls) * xm + jnp.abs(li))
            ui = ui + SAFE_K * float(jnp.finfo(lo.dtype).eps) * (jnp.abs(us) * xm + jnp.abs(ui))
        return Hull(jnp.where(okl[:, 0], Gl.min(1), -jnp.inf), jnp.where(oku[:, 0], Gu.max(1), jnp.inf), ls, li, us, ui)

    def relax_with(x, lo, hi, data):
        tape, blocks = setup(x)
        run = interpreter(tape, blocks, x)
        if run is not None:
            return run(x, lo, hi, data if blocks else None)
        override = {}
        k = 0
        for b in blocks:
            m = b.dep.size
            override[b.slot] = (b, k, m)
            k += m

        vals = {0: MC(lo, hi, x, x)}
        logs = {}  # slot of each log -> its operand slot (``x * log(x)``: ``relax.mc_xlogx``)
        monos = Monomials(mul_consumed((nd.eqn.primitive.name, nd.ins) for nd in tape.nodes), ctx)  # products of powers
        terms = LogTerms(lin_consumed((nd.eqn.primitive.name, nd.ins) for nd in tape.nodes), ctx)  # like x log x terms
        for node in tape.nodes:
            eqn = node.eqn
            invals = [a.val if isinstance(a, Const) else vals[a] for a in node.ins]
            name = eqn.primitive.name
            if name == "log" and not isinstance(node.ins[0], Const):
                logs[node.outs[0]] = node.ins[0]
            xl = xlogx_pair(name, node.ins, logs) if name == "mul" else None
            if not any(isinstance(v, (MC, Unknown)) for v in invals):
                outs = eqn.primitive.bind(*invals, **eqn.params)
            elif xl is not None and is_mc(invals[xl]):
                outs = apply_xlogx(ctx, eqn, invals[xl])
            else:
                outs = apply_rule(ctx, eqn, invals)
            if not eqn.primitive.multiple_results:
                outs = monos.observe(name, node.ins, invals, node.outs[0], outs)
                outs = [terms.observe(name, node.ins, invals, node.outs[0], outs, eqn)]
            for s, o in zip(node.outs, outs):
                if s in override and is_mc(o):
                    b, k, m = override[s]
                    shp = tape.shapes[s][0]
                    p = x[b.dep]
                    sl = slice(k, k + m)
                    cv = jnp.max(data.ls[sl] * p[:, None] + data.li[sl], axis=1)
                    cc = jnp.min(data.us[sl] * p[:, None] + data.ui[sl], axis=1)
                    std = MC(*(jnp.reshape(jnp.broadcast_to(c, shp), (-1,)) for c in o))
                    lo_ = jnp.maximum(std.lo, data.lo[sl])
                    hi_ = jnp.minimum(std.hi, data.hi[sl])
                    cv = jnp.where(jnp.isnan(std.cv), cv, cvmax(std.cv, cv))
                    cc = jnp.where(jnp.isnan(std.cc), cc, ccmin(std.cc, cc))
                    o = tighten(MC(*(jnp.reshape(c, shp) for c in (lo_, hi_, cv, cc))))
                vals[s] = o
        out = tape.outs[0]
        out = out.val if isinstance(out, Const) else vals[out]
        return out if is_mc(out) else MC(*(jnp.asarray(out),) * 4)

    return prepare, relax_with


def relax_univariate(f, K=65, safe=False):
    """Drop-in for ``relax(f)``: ``rf(x, lo, hi[, data])``, with ``rf.prepare(lo, hi)``.

    Without ``data`` the hull is rebuilt on every call (correct but slower).
    """
    prepare, relax_with = make_univariate(f, K, safe=safe)

    def rf(x, lo, hi, data=None):
        if data is None:
            data = prepare(lo, hi)
        return relax_with(x, lo, hi, data)

    rf.prepare = prepare
    return rf


class Relaxer(NamedTuple):
    """``prepare(lo, hi) -> data`` once per box, then ``__call__(x, lo, hi, data) -> MC``."""

    prepare: object
    call: object

    def __call__(self, x, lo, hi, data):
        return self.call(x, lo, hi, data)


def make_relaxer(fn, univariate=True, alphabb=False, K=65, multivariate=False, safe=False, interpret="auto",
                 convex=True, sides=("cv", "cc"), conditional=False):
    """Relaxation of ``fn``: McCormick, optionally tightened by univariate hull
    envelopes (``univariate``) and whole-function scaled-Gershgorin alphaBB
    (``alphabb``, smooth ``fn`` only; ``alphabb="eigen"`` adds the eigenvalue
    relaxation of exact quadratics), and (``convex``) by ``fn`` itself on the
    outputs that ``convexity.classify`` certifies convex (concave) on the box.
    ``safe`` rounds every bound outward. ``sides``: the sides (``"cv"``, ``"cc"``) the caller
    uses; a function certified on them is relaxed by itself alone (``convexity.wrap``), with
    ``conditional`` also when its verdicts are conditional (for boxes where they hold)."""
    from .alphabb import make_alphabb
    from .convexity import wrap
    from .relax import relax

    if convex:
        return wrap(fn, make_relaxer(fn, univariate, alphabb, K, multivariate, safe, interpret, convex=False), safe, sides,
                    conditional)

    if univariate:
        uprep, ucall = make_univariate(fn, K, multivariate, safe, interpret)
    else:
        rf = relax(fn, multivariate, safe)
        interpreter = _interpreters(Ctx(multivariate=multivariate, safe=safe), interpret)
        tape_of = tape_cache(lambda x: (make_tape(fn, x),))

        def ucall(x, lo, hi, data):
            run = interpreter(tape_of(x)[0], (), x) if interpret is not False else None
            return run(x, lo, hi) if run is not None else rf(x, lo, hi)

        uprep = lambda lo, hi: None  # noqa: E731
    if not alphabb:
        return Relaxer(uprep, ucall)
    aprep, aapply = make_alphabb(fn, eigen=alphabb == "eigen", safe=safe, interpret=interpret)

    def prepare(lo, hi):
        return uprep(lo, hi), aprep(lo, hi)

    def call(x, lo, hi, data):
        m = ucall(x, lo, hi, data[0])
        if data[1] is None:
            return m
        shp = jnp.shape(m.cv)
        acv, acc = aapply(x, lo, hi, data[1])
        acv, acc = jnp.reshape(acv, shp), jnp.reshape(acc, shp)
        cv = jnp.where(jnp.isnan(m.cv), acv, cvmax(m.cv, acv))
        cc = jnp.where(jnp.isnan(m.cc), acc, ccmin(m.cc, acc))
        return tighten(MC(m.lo, m.hi, cv, cc))

    return Relaxer(prepare, call)

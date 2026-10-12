"""McCormick relaxations as an interpreter over a scalar tape (compile time independent of the model).

``relax.eval_jaxpr_mc`` traces every rule of every equation into the kernel,
and ``jax.grad`` / ``jax.jacfwd`` of the relaxation trace it again, so for a
model read from an ``.nl`` file (fully scalar, ~1,000 operations) the bounding
kernels have hundreds of thousands of equations and take minutes to compile.
Here, as in ``fbbt_interp``, the tape is data: ``fbbt_interp.scalarize`` turns
it into rows ``(opcode, a, b, out)`` and a ``lax.fori_loop`` with a
``lax.switch`` on the opcode evaluates the rows in order.

Each slot carries ``MC(lo, hi, cv, cc)`` and, when derivatives are wanted, the
gradients of ``cv`` and ``cc`` with respect to ``x`` (forward mode: ``(S, n)``
arrays). An opcode's branch calls the same rule as ``eval_jaxpr_mc``
(``relax.apply_rule``) on scalar ``MC`` operands, and propagates gradients by
the chain rule with the rule's own Jacobian in ``(cv_a, cc_a, cv_b, cc_b)``,
which is what ``jax.jvp`` of the traced relaxation computes, so values and
subgradients are those of ``eval_jaxpr_mc`` (sums of more than two terms become
chains of ``add``: the same up to rounding). The result is a ``custom_jvp``
function of ``x``, so ``jax.grad`` and ``jax.jacfwd`` of it work unchanged.

Univariate hull envelopes (``univariate.make_univariate``) are applied at the
rows that compute a block's elements, as ``relax_with`` does.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.custom_derivatives import SymbolicZero

from .fbbt_interp import scalarize, use_interpreter
from .relax import MC, apply_rule, is_mc

F64 = jnp.float64
# Interpret only tapes this large. Straight-line relaxation kernels compile in a few seconds up to
# several hundred rows (480-row st_rv7 constraints: 2 s) and run 10-300x faster per call than the
# interpreter, which under vmap evaluates every opcode's branch at every row; at 1,162 rows
# (kriging_peaks) the traced gradient kernel takes minutes to compile and the interpreter 0.4 s.
MIN_ROWS = 600


def use_relax_interpreter(tape, n):
    return use_interpreter(tape, n, min_rows=MIN_ROWS)


def _stand(name, same, n_in, eqn):
    """What the rules read from an equation: primitive, params, and invars (only to tell x*x from x*y)."""
    v = object()
    return SimpleNamespace(primitive=eqn.primitive, params=eqn.params, invars=[v, v if same else object()][:n_in])


def _branches(prog, ctx, grads):
    out = []
    for name, _, same, ca, cb, n_in, eqn in prog.opcodes:
        stand = _stand(name, same, n_in, eqn)

        def rule(u, la, ha, lb, hb, stand=stand, ca=ca, cb=cb, n_in=n_in):
            args = [la if ca else MC(la, ha, u[0], u[1])]
            if n_in > 1:
                args.append(lb if cb else MC(lb, hb, u[2], u[3]))
            o = apply_rule(ctx, stand, args)
            o = o if is_mc(o) else MC(o, o, o, o)
            return jnp.stack([jnp.reshape(jnp.asarray(c, F64), ()) for c in o])

        # operands whose gradient is structurally zero (constants, the missing second operand)
        zero_cols = ([0, 1] if ca else []) + ([2, 3] if (n_in == 1 or cb) else [])

        if grads:

            def f(la, ha, cva, cca, lb, hb, cvb, ccb, T, rule=rule, zero_cols=zero_cols):
                u = jnp.stack([cva, cca, cvb, ccb])
                o, lin = jax.linearize(lambda u: rule(u, la, ha, lb, hb), u)
                J = jax.vmap(lin, out_axes=1)(jnp.eye(4, dtype=F64))[2:]  # d(cv, cc) / d(cv_a, cc_a, cv_b, cc_b)
                if zero_cols:
                    J = J.at[:, zero_cols].set(0.0)
                return o, J @ T
        else:

            def f(la, ha, cva, cca, lb, hb, cvb, ccb, rule=rule):
                return rule(jnp.stack([cva, cca, cvb, ccb]), la, ha, lb, hb)

        out.append(f)
    return out


def _combine(cv, cc, cvh, cch, lo, hi):
    """``relax_with``'s use of a hull: intersect, then tighten."""
    cv = jnp.where(jnp.isnan(cv), cvh, jnp.maximum(cv, cvh))
    cc = jnp.where(jnp.isnan(cc), cch, jnp.minimum(cc, cch))
    return jnp.stack([jnp.maximum(cv, lo), jnp.minimum(cc, hi)])


def _hull_rows(prog, tape, blocks, K):
    """Per row: index of the hull element it computes (``-1``: none), and the variable of each element."""
    hid = np.full(K, -1, np.int64)
    if not blocks:
        return hid, np.zeros(0, np.int64)
    row_of = {int(s): k for k, s in enumerate(prog.out)}
    deps, j = [], 0
    for b in blocks:
        slots = np.asarray(prog.ids[b.slot]).reshape(-1)
        for e, s in enumerate(slots):
            if int(s) in row_of:  # an element that is a variable or a constant has nothing to tighten
                hid[row_of[int(s)]] = j + e
        deps.append(b.dep)
        j += slots.size
    return hid, np.concatenate(deps)


def make_mc(tape, n, ctx, blocks=()):
    """``mc(x, lo, hi, hull=None) -> MC`` of the tape's (single) output by interpretation, or None.

    ``blocks`` and ``hull`` are ``univariate.find_blocks`` of this tape and the
    ``Hull`` its ``prepare`` returned for the box.
    """
    prog = scalarize(tape, n)
    if prog is None or len(prog.op) == 0 or len(tape.outs) != 1:
        return None
    try:
        fwd, fwd_g = _branches(prog, ctx, False), _branches(prog, ctx, True)
    except Exception:
        return None
    K = len(prog.op)
    op, A, B, OUT = (jnp.asarray(v) for v in (prog.op, prog.a, prog.b, prog.out))
    hid_np, dep_np = _hull_rows(prog, tape, list(blocks), K)
    HID = jnp.asarray(hid_np)
    has_hull = bool((hid_np >= 0).any())
    cval = jnp.asarray(np.nan_to_num(prog.const_val, nan=0.0))
    is_c = jnp.asarray(~np.isnan(prog.const_val))
    out_ids = jnp.asarray(prog.outs[0])
    oa = tape.outs[0]
    out_shape = tuple(np.shape(oa.val)) if not isinstance(oa, int) else tuple(tape.shapes[oa][0])

    def hull_values(x, hull, grads):
        """Hull lines at x for every hull element, plus a no-op entry at the end (index -1)."""
        dep = jnp.asarray(dep_np)
        p = x[dep]
        cvh = jnp.max(hull.ls * p[:, None] + hull.li, axis=1)
        cch = jnp.min(hull.us * p[:, None] + hull.ui, axis=1)
        pad = lambda v, c: jnp.concatenate([v, jnp.full((1,) + v.shape[1:], c, v.dtype)])  # noqa: E731
        vals = (pad(hull.lo, -jnp.inf), pad(hull.hi, jnp.inf), pad(cvh, -jnp.inf), pad(cch, jnp.inf))
        if not grads:
            return vals, None
        # d/dp of the max (min) of the lines, as jax.grad of the traced relax_with computes it
        dcv = jax.vmap(jax.grad(lambda p, s, i: jnp.max(s * p + i)))(p, hull.ls, hull.li)
        dcc = jax.vmap(jax.grad(lambda p, s, i: jnp.min(s * p + i)))(p, hull.us, hull.ui)
        onehot = jax.nn.one_hot(dep, n, dtype=F64)
        return vals, (pad(dcv[:, None] * onehot, 0.0), pad(dcc[:, None] * onehot, 0.0))

    def sweep(x, lo, hi, hull, grads):
        x, lo, hi = (jnp.asarray(v, F64) for v in (x, lo, hi))
        L = jnp.where(is_c, cval, 0.0).at[:n].set(lo)
        H = jnp.where(is_c, cval, 0.0).at[:n].set(hi)
        CV = jnp.where(is_c, cval, 0.0).at[:n].set(x)
        CC = CV
        use_hull = has_hull and hull is not None
        if use_hull:
            hv, hg = hull_values(x, hull, grads)
        if grads:
            S = cval.shape[0]
            GV = jnp.zeros((S, n), F64).at[:n].set(jnp.eye(n, dtype=F64))
            GC = GV

        def step(k, c):
            a, b, o = A[k], B[k], OUT[k]
            ins = (c[0][a], c[1][a], c[2][a], c[3][a], c[0][b], c[1][b], c[2][b], c[3][b])
            if grads:
                T = jnp.stack([c[4][a], c[5][a], c[4][b], c[5][b]])
                r, G = lax.switch(op[k], fwd_g, *ins, T)
            else:
                r = lax.switch(op[k], fwd, *ins)
            lo_, hi_, cv, cc = r[0], r[1], r[2], r[3]
            if use_hull:
                j = HID[k]
                lo_, hi_ = jnp.maximum(lo_, hv[0][j]), jnp.minimum(hi_, hv[1][j])
                if grads:
                    w, lin = jax.linearize(lambda u: _combine(u[0], u[1], u[2], u[3], lo_, hi_),
                                           jnp.stack([cv, cc, hv[2][j], hv[3][j]]))  # fmt: skip
                    Jh = jax.vmap(lin, out_axes=1)(jnp.eye(4, dtype=F64))
                    Gh = Jh @ jnp.concatenate([G, jnp.stack([hg[0][j], hg[1][j]])])
                    on = j >= 0
                    G = jnp.where(on, Gh, G)
                else:
                    w = _combine(cv, cc, hv[2][j], hv[3][j], lo_, hi_)
                on = j >= 0
                cv, cc = jnp.where(on, w[0], cv), jnp.where(on, w[1], cc)
            new = [c[0].at[o].set(lo_), c[1].at[o].set(hi_), c[2].at[o].set(cv), c[3].at[o].set(cc)]
            if grads:
                new += [c[4].at[o].set(G[0]), c[5].at[o].set(G[1])]
            return tuple(new)

        init = (L, H, CV, CC) + ((GV, GC) if grads else ())
        res = lax.fori_loop(0, K, step, init)
        m = tuple(jnp.reshape(v[out_ids], out_shape) for v in res[:4])
        if not grads:
            return m
        return m, tuple(jnp.reshape(v[out_ids], out_shape + (n,)) for v in res[4:])

    @jax.custom_jvp
    def mc(x, lo, hi, hull):
        return sweep(x, lo, hi, hull, False)

    def mc_jvp(primals, tangents):
        x, lo, hi, hull = primals
        xd, lod, hid, hulld = tangents
        if not all(isinstance(t, SymbolicZero) for t in jax.tree_util.tree_leaves((lod, hid, hulld))):
            raise NotImplementedError("the interpreted relaxation is differentiable in x only")
        m, (gv, gc) = sweep(x, lo, hi, hull, True)
        z = jnp.zeros(out_shape, F64)
        if isinstance(xd, SymbolicZero):
            return m, (z, z, z, z)
        return m, (z, z, gv @ xd, gc @ xd)

    mc.defjvp(mc_jvp, symbolic_zeros=True)

    def call(x, lo, hi, hull=None):
        return MC(*mc(x, lo, hi, hull))

    call.program = prog
    return call


def interpreted(tape, n, ctx, blocks=(), interpret="auto"):
    """``make_mc`` when ``interpret`` says so (``"auto"``: ``use_relax_interpreter``), else None."""
    if interpret is False or not (interpret is True or use_relax_interpreter(tape, n)):
        return None
    with jax.ensure_compile_time_eval():
        return make_mc(tape, n, ctx, blocks)


# ---------------------------------------------------------------------------
# Second-order interval forward mode (alphaBB's interval Hessian, the hull's g'')
# ---------------------------------------------------------------------------


def _taylor_step(name, same, ca, cb, n_in, eqn):
    """``F(va, ga, Ha, vb, gb, Hb) -> (v, g, H)``: one operation in second-order forward mode.

    ``g`` and ``H`` are the first and second derivatives along the seed
    directions; the primitive's own derivatives come from JAX autodiff.
    Constant operands are passed as values (their ``g``, ``H`` are ignored).
    """
    prim, params = eqn.primitive, eqn.params
    if same:  # x op x: one variable operand
        def p(z):
            return prim.bind(z, z, **params)
        var = [0]
    else:
        def p(*z):
            return prim.bind(*z, **params)
        var = [i for i, c in enumerate((ca, cb)[:n_in]) if not c]

    def F(va, ga, Ha, vb, gb, Hb):
        args = [va, vb][: 1 if same else n_in]
        gs, Hs = [ga, gb], [Ha, Hb]

        def pv(*z):
            full = list(args)
            for i, zi in zip(var, z):
                full[i] = zi
            return p(*full)

        z0 = [args[i] for i in var]
        val = pv(*z0)
        argn = tuple(range(len(var)))
        d1 = jax.grad(pv, argnums=argn)(*z0)
        d2 = jax.hessian(pv, argnums=argn)(*z0)
        g = sum(d1[k] * gs[i] for k, i in enumerate(var))
        H = sum(d1[k] * Hs[i] for k, i in enumerate(var))
        for k, i in enumerate(var):
            for m, j in enumerate(var):
                H = H + d2[k][m] * (gs[i][:, None] * gs[j][None, :])
        return val, g, H

    return F


def make_taylor2(tape, n, ictx, d):
    """``taylor(lo, hi, E) -> (V, G, H)``: interval value, first and second derivatives of every slot.

    Along the ``d`` seed directions ``E`` (``(n, d)``) over the box ``[lo, hi]``:
    ``V`` is ``(S, 2)``-like ``(lo, hi)`` pairs of shape ``(S,)``, ``G``
    ``(S, d)`` and ``H`` ``(S, d, d)`` likewise. Slots are those of
    ``scalarize(tape, n)`` (``taylor.program``). Returns None when the tape is
    not interpretable or an operation has no interval rule for its derivatives.
    """
    from .relax import eval_jaxpr_mc

    prog = scalarize(tape, n)
    if prog is None or len(prog.op) == 0:
        return None
    zs, zg, zH = jnp.zeros((), F64), jnp.zeros((d,), F64), jnp.zeros((d, d), F64)
    branches = []
    for name, _, same, ca, cb, n_in, eqn in prog.opcodes:
        F = _taylor_step(name, same, ca, cb, n_in, eqn)
        consts = (ca, cb if n_in > 1 else True)
        try:
            closed = jax.make_jaxpr(F)(zs, zg, zH, zs, zg, zH)
        except Exception:
            return None

        def br(la, ha, gla, gha, Hla, Hha, lb, hb, glb, ghb, Hlb, Hhb, closed=closed, consts=consts):
            ins = []
            for (vl, vh, gl, gh, Hl, Hh), c in zip(((la, ha, gla, gha, Hla, Hha), (lb, hb, glb, ghb, Hlb, Hhb)), consts):
                ins += [vl if c else MC(vl, vh, vl, vh), MC(gl, gh, gl, gh), MC(Hl, Hh, Hl, Hh)]
            outs = eval_jaxpr_mc(closed.jaxpr, closed.consts, ins, ictx)
            res = []
            for o, shape in zip(outs, ((), (d,), (d, d))):
                o = o if is_mc(o) else MC(o, o, o, o)
                lo = jnp.broadcast_to(jnp.asarray(o.lo, F64), shape)
                hi = jnp.broadcast_to(jnp.asarray(o.hi, F64), shape)
                lo, hi = jnp.where(jnp.isnan(lo), -jnp.inf, lo), jnp.where(jnp.isnan(hi), jnp.inf, hi)
                res += [lo, hi]
            return tuple(res)

        try:  # every derivative primitive needs an interval rule
            jax.eval_shape(br, *([zs, zs, zg, zg, zH, zH] * 2))
        except NotImplementedError:
            return None
        branches.append(br)

    K = len(prog.op)
    op, A, B, OUT = (jnp.asarray(v) for v in (prog.op, prog.a, prog.b, prog.out))
    cval = jnp.asarray(np.nan_to_num(prog.const_val, nan=0.0))
    is_c = jnp.asarray(~np.isnan(prog.const_val))
    S = cval.shape[0]

    def taylor(lo, hi, E):
        lo, hi, E = (jnp.asarray(v, F64) for v in (lo, hi, E))
        Vl = jnp.where(is_c, cval, 0.0).at[:n].set(lo)
        Vh = jnp.where(is_c, cval, 0.0).at[:n].set(hi)
        G = jnp.zeros((S, d), F64).at[:n].set(E)
        Hz = jnp.zeros((S, d, d), F64)

        def step(k, c):
            a, b, o = A[k], B[k], OUT[k]
            ins = []
            for s in (a, b):
                ins += [c[0][s], c[1][s], c[2][s], c[3][s], c[4][s], c[5][s]]
            r = lax.switch(op[k], branches, *ins)
            return tuple(ci.at[o].set(ri) for ci, ri in zip(c, r))

        return lax.fori_loop(0, K, step, (Vl, Vh, G, G, Hz, Hz))

    taylor.program = prog
    return taylor

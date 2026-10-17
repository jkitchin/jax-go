"""McCormick relaxations as an interpreter over a scalar tape (compile time independent of the model).

``relax.eval_jaxpr_mc`` traces every rule of every equation into the kernel,
and ``jax.grad`` / ``jax.jacfwd`` of the relaxation trace it again, so for a
model read from an ``.nl`` file (fully scalar, ~1,000 operations) the bounding
kernels have hundreds of thousands of equations and take minutes to compile.
Here, as in ``fbbt_interp``, the tape is data: ``fbbt_interp.scalarize`` turns
it into rows ``(opcode, a, b, out)``, ``fbbt_interp.schedule`` groups them
into steps of independent rows, and a ``lax.fori_loop`` over the steps
evaluates every opcode once per step, vectorized over that step's rows.

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

from .fbbt_interp import layout, scalarize, tape_size, use_interpreter
from .relax import MC, apply_rule, is_mc

F64 = jnp.float64

# Which tapes the McCormick relaxation (and the hull's g'') interprets.
# ``RULE = "rows"`` (default, unchanged): tapes of at least ``MIN_ROWS`` rows.
# ``RULE = "ops"`` (opt-in, ``--set relax_interp.RULE=ops``): also tapes whose straight-line node bound
# would evaluate at least ``MIN_OPS`` scalar rules per box (``kernel_ops``: wide models, where the n
# tangents at 2n + 1 + n_kelley cut points make a few hundred rows expensive to compile and run, e.g.
# hadamard_5's 503-row constraints at n = 26). ``MIN_OPS`` is a candidate, not yet calibrated: the A3
# ablation (docs/ROADMAP.md) has to show no node-rate loss on the instances it flips before it becomes
# the default.
# Straight-line relaxation kernels compile in a few seconds up to several hundred rows (480-row st_rv7
# constraints: 2 s) and usually run faster per call than the interpreter, which runs one loop step per
# schedule step; at 1,162 rows (kriging_peaks) the traced gradient kernel takes minutes to compile and
# the interpreter 0.4 s.
RULE = "rows"
MIN_ROWS = 600
MIN_OPS = 600_000
N_KELLEY = 4  # minimize's default n_kelley


def cut_points(n, n_kelley=N_KELLEY):
    """Points per box at which the node bound linearizes each relaxation: the center, the 2n axis
    points and one point per Kelley round (``lp.make_constrained_bounder``)."""
    return 2 * n + 1 + n_kelley


def kernel_ops(tape, n, n_kelley=N_KELLEY):
    """Scalar relaxation rules the straight-line node bound evaluates per box for this tape.

    Each of the ``rows`` scalar operations is a McCormick rule, evaluated with its forward-mode
    derivative in all ``n`` directions (``jax.jacfwd``: value plus ``n`` tangents) at every cut
    point (``cut_points``): ``rows * (n + 1) * cut_points(n, n_kelley)``.
    """
    rows, _ = tape_size(tape)
    return rows * ops_per_row(n, n_kelley)


def ops_per_row(n, n_kelley=N_KELLEY):
    """Rule evaluations per box for one scalar row: ``(n + 1) * cut_points(n, n_kelley)``."""
    return (n + 1) * cut_points(n, n_kelley)


def use_relax_interpreter(tape, n):
    """Interpret this tape in the relaxation kernels (see ``RULE``)."""
    if not use_interpreter(tape, n, min_rows=0):  # vectorized code: scalarizing multiplies the work
        return False
    rows, _ = tape_size(tape)
    if RULE == "rows":
        return rows >= MIN_ROWS
    return rows >= MIN_ROWS or rows * ops_per_row(n) >= MIN_OPS


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


def _put(c, v, start):
    """``c[start : start + len(v)] = v``."""
    return lax.dynamic_update_slice_in_dim(c, v, start, 0)


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
    lay = layout(prog)
    prog = lay.prog
    try:
        fwd, fwd_g = _branches(prog, ctx, False), _branches(prog, ctx, True)
    except Exception:
        return None
    K = len(prog.op)
    A, B, OUT = (jnp.asarray(v) for v in (prog.a, prog.b, prog.out))
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

    R = [jnp.asarray(r) for r in lay.rows]
    START = [jnp.asarray(st) for st in lay.start]
    S = cval.shape[0]

    def finish(r, G, j, hv, hg):
        """One row's result, intersected with its hull element ``j`` (``-1``: none)."""
        lo_, hi_, cv, cc = r[0], r[1], r[2], r[3]
        lo_, hi_ = jnp.maximum(lo_, hv[0][j]), jnp.minimum(hi_, hv[1][j])
        on = j >= 0
        if G is not None:
            w, lin = jax.linearize(lambda u: _combine(u[0], u[1], u[2], u[3], lo_, hi_),
                                   jnp.stack([cv, cc, hv[2][j], hv[3][j]]))  # fmt: skip
            Jh = jax.vmap(lin, out_axes=1)(jnp.eye(4, dtype=F64))
            G = jnp.where(on, Jh @ jnp.concatenate([G, jnp.stack([hg[0][j], hg[1][j]])]), G)
        else:
            w = _combine(cv, cc, hv[2][j], hv[3][j], lo_, hi_)
        return jnp.stack([lo_, hi_, jnp.where(on, w[0], cv), jnp.where(on, w[1], cc)]), G

    def sweep(x, lo, hi, hull, grads):
        x, lo, hi = (jnp.asarray(v, F64) for v in (x, lo, hi))
        base = jnp.where(is_c, cval, 0.0)
        # (S, 4) columns lo, hi, cv, cc: one 2-D array per box, which XLA updates in place in the
        # loop (four (S,) arrays were each copied every step under vmap)
        V = jnp.stack([base.at[:n].set(lo), base.at[:n].set(hi), base.at[:n].set(x), base.at[:n].set(x)], axis=1)
        use_hull = has_hull and hull is not None
        if use_hull:
            hv, hg = hull_values(x, hull, grads)
        if grads:  # columns: lo, hi, cv, cc, d cv / dx (n), d cc / dx (n)
            GV = jnp.zeros((S, n), F64).at[:n].set(jnp.eye(n, dtype=F64))
            V = jnp.concatenate([V, GV, GV], axis=1)

        def step(t, V):
            # the rows of a step are independent: read everything from the carry first, then write
            # it once (the step's blocks are adjacent), so XLA can update the carry in place
            rs = []
            for o in range(len(R)):  # every opcode once, vectorized over its rows of the step
                k = jnp.maximum(R[o][t], 0)
                va, vb = V[A[k]], V[B[k]]
                ins = (va[:, 0], va[:, 1], va[:, 2], va[:, 3], vb[:, 0], vb[:, 1], vb[:, 2], vb[:, 3])
                if grads:
                    T = jnp.stack([va[:, 4:4 + n], va[:, 4 + n:], vb[:, 4:4 + n], vb[:, 4 + n:]], axis=1)
                    r, G = jax.vmap(fwd_g[o])(*ins, T)
                else:
                    r, G = jax.vmap(fwd[o])(*ins), None
                if use_hull:
                    r, G = jax.vmap(lambda r, G, j: finish(r, G, j, hv, hg),
                                    in_axes=(0, 0 if grads else None, 0))(r, G, HID[k])  # fmt: skip
                rs.append(jnp.concatenate([r, G[:, 0], G[:, 1]], axis=1) if grads else r)
            return _put(V, jnp.concatenate(rs), START[0][t])

        V = lax.fori_loop(0, lay.steps, step, V)[out_ids]
        m = tuple(jnp.reshape(V[:, i], out_shape) for i in range(4))
        if not grads:
            return m
        return m, (jnp.reshape(V[:, 4:4 + n], out_shape + (n,)), jnp.reshape(V[:, 4 + n:], out_shape + (n,)))

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
    lay = layout(prog)
    prog = lay.prog
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

    A, B, OUT = (jnp.asarray(v) for v in (prog.a, prog.b, prog.out))
    cval = jnp.asarray(np.nan_to_num(prog.const_val, nan=0.0))
    is_c = jnp.asarray(~np.isnan(prog.const_val))
    S = cval.shape[0]
    R = [jnp.asarray(r) for r in lay.rows]
    START = [jnp.asarray(st) for st in lay.start]

    # packed columns: value lo, hi; gradient lo, hi; Hessian lo, hi
    cuts = np.cumsum([0, 1, 1, d, d, d * d, d * d])
    shapes = [(), (), (d,), (d,), (d, d), (d, d)]

    def unpack(v):  # (W, w) -> the six branch arguments
        return [jnp.reshape(v[:, cuts[i]:cuts[i + 1]], (-1,) + shapes[i]) for i in range(6)]

    def taylor(lo, hi, E):
        lo, hi, E = (jnp.asarray(v, F64) for v in (lo, hi, E))
        base = jnp.where(is_c, cval, 0.0)
        G = jnp.zeros((S, d), F64).at[:n].set(E)
        Hz = jnp.zeros((S, d * d), F64)
        V = jnp.concatenate([base.at[:n].set(lo)[:, None], base.at[:n].set(hi)[:, None], G, G, Hz, Hz], axis=1)

        def step(t, V):  # read, then one in-place write of the packed carry (see make_mc)
            rs = []
            for o in range(len(R)):
                k = jnp.maximum(R[o][t], 0)
                r = jax.vmap(branches[o])(*unpack(V[A[k]]), *unpack(V[B[k]]))
                rs.append(jnp.concatenate([jnp.reshape(ri, (ri.shape[0], -1)) for ri in r], axis=1))
            return _put(V, jnp.concatenate(rs), START[0][t])

        V = lax.fori_loop(0, lay.steps, step, V)
        return tuple(jnp.reshape(V[:, cuts[i]:cuts[i + 1]], (S,) + shapes[i]) for i in range(6))

    taylor.program = prog
    return taylor

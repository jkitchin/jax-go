"""FBBT as an interpreter over a scalar tape, so its compile time does not grow with the model.

``fbbt.propagate`` traces every equation of the tape into the kernel, so the
kernel, and XLA's compile time, grow with the model; for a model read from an
``.nl`` file, which is fully scalar, that is ~180 equations per scalar operation
(a 1,166-operation model gives a 200,000-equation FBBT kernel). Here the tape
is turned into data instead:

* **scalarize**: every element of every interval-valued slot becomes a scalar
  slot; structural equations (slice, squeeze, reshape, broadcast, stack, ...)
  only renumber slots, constants get fixed slots, and every elementwise
  equation becomes one row ``(opcode, a, b, out)`` per element;
* **interpret**: forward and backward sweeps are ``lax.fori_loop`` over the
  rows with a ``lax.switch`` on the opcode. Each opcode's branch calls the same
  rules as ``fbbt`` (``relax.RULES`` in interval mode forward, ``fbbt.BACK``
  backward) on scalar intervals, with constant operands passed as constants,
  so the result is that of ``propagate`` element by element (sums of more than
  two terms become chains of ``add``, the same bounds up to rounding).

The kernel's size depends on the number of distinct opcodes, not on the number
of rows. Its run time is one loop step per row, so ``make_fbbt`` uses it only
for large, mostly scalar tapes (see ``use_interpreter``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .fbbt import BACK, CTX, INF, SAFE_CTX, Const, _clean, _meet2, _pad
from .relax import MC, RULES, apply_rule, is_mc

# equations that only move elements: the interpreter renumbers slots, no row
STRUCTURAL = {"slice", "squeeze", "reshape", "broadcast_in_dim", "concatenate", "stack", "expand_dims",
              "transpose", "rev", "convert_element_type", "copy", "copy_p", "reduce_precision"}  # fmt: skip
# elementwise equations with one row per output element
ELEMENTWISE = {"add", "add_any", "sub", "mul", "div", "neg", "exp", "exp2", "expm1", "log", "log1p", "sqrt",
               "rsqrt", "integer_pow", "square", "abs", "max", "min", "tanh", "logistic", "erf", "erfc", "sin",
               "cos", "atan", "asinh", "sinh", "cosh", "pow", "one_minus_square"}  # fmt: skip


class Program(NamedTuple):
    op: np.ndarray  # (K,) opcode of each row
    a: np.ndarray  # (K,) first operand slot
    b: np.ndarray  # (K,) second operand slot (= a for unary rows)
    out: np.ndarray  # (K,) result slot
    const_val: np.ndarray  # (S,) value of constant slots, NaN for variable slots
    n: int  # the first n slots are x
    opcodes: list  # (name, params, same, const_a, const_b, n_in, eqn) per opcode
    outs: list  # slot ids (np arrays) of the tape outputs


def scalarize(tape, n):
    """The tape as a ``Program``, or None if it has an equation the interpreter does not handle."""
    # Structural equations and constant folding are evaluated on concrete arrays here. Inside an
    # enclosing jit/vmap trace JAX would stage them instead (omnistaging), so force eager evaluation.
    with jax.ensure_compile_time_eval():
        return _scalarize(tape, n)


def _scalarize(tape, n):
    ids = {0: np.arange(n)}
    const_val = [np.nan] * n
    const_ids = {}

    def const_slot(v):
        key = float(v)
        if key not in const_ids:
            const_ids[key] = len(const_val)
            const_val.append(key)
        return const_ids[key]

    def operand(a, shape):
        if isinstance(a, Const):
            v = np.asarray(a.val, np.float64)
            return np.vectorize(const_slot, otypes=[np.int64])(np.broadcast_to(v, shape))
        return np.broadcast_to(ids[a], shape)

    def is_const(s):
        return not np.isnan(const_val[s])

    rows, opcodes, opkey = [], [], {}
    for node in tape.nodes:
        eqn, name = node.eqn, node.eqn.primitive.name
        if len(node.outs) != 1:
            return None
        shape = tuple(tape.shapes[node.outs[0]][0])
        if name in STRUCTURAL:
            ins = [operand(a, np.shape(a.val) if isinstance(a, Const) else tape.shapes[a][0]) for a in node.ins]
            if name in ("convert_element_type", "copy", "copy_p", "reduce_precision"):
                out = ins[0]
            else:
                try:
                    out = np.asarray(eqn.primitive.bind(*[jnp.asarray(i, jnp.int64) for i in ins], **eqn.params))
                except Exception:
                    return None
            ids[node.outs[0]] = np.asarray(out, np.int64).reshape(shape)
            continue
        if name not in ELEMENTWISE or name not in RULES or len(node.ins) > 2:
            return None
        ins = [operand(a, shape) for a in node.ins]
        out = np.empty(shape, np.int64)
        for e in np.ndindex(shape):
            sa = int(ins[0][e])
            sb = int(ins[1][e]) if len(ins) > 1 else sa
            if is_const(sa) and is_const(sb):  # all constant: fold
                v = eqn.primitive.bind(*[jnp.asarray(const_val[s]) for s in ([sa, sb] if len(ins) > 1 else [sa])],
                                       **eqn.params)  # fmt: skip
                out[e] = const_slot(np.asarray(v))
                continue
            same = len(ins) > 1 and sa == sb
            key = (name, repr(sorted(eqn.params.items())), same, is_const(sa), len(ins) > 1 and is_const(sb), len(ins))
            if key not in opkey:
                opkey[key] = len(opcodes)
                opcodes.append(key + (eqn,))
            k = len(const_val)
            const_val.append(np.nan)
            rows.append((opkey[key], sa, sb, k))
            out[e] = k
        ids[node.outs[0]] = out
    outs = []
    for a in tape.outs:
        if isinstance(a, Const):
            outs.append(np.vectorize(const_slot, otypes=[np.int64])(np.atleast_1d(np.asarray(a.val, np.float64))))
        else:
            outs.append(np.asarray(ids[a]).reshape(-1))
    r = np.asarray(rows, np.int64).reshape(-1, 4)
    return Program(r[:, 0], r[:, 1], r[:, 2], r[:, 3], np.asarray(const_val, np.float64), n, opcodes, outs)


def _branches(prog, safe):
    ctx = SAFE_CTX if safe else CTX
    fwd, bwd = [], []
    for name, _, same, ca, cb, n_in, eqn in prog.opcodes:
        # what the rules read from an equation: primitive, params, and invars (only to tell x*x from x*y)
        v = object()
        stand = SimpleNamespace(primitive=eqn.primitive, params=eqn.params, invars=[v, v if same else object()][:n_in])

        def args(la, ha, lb, hb, ca=ca, cb=cb, n_in=n_in):
            x = [la if ca else MC(la, ha, la, ha)]
            if n_in > 1:
                x.append(lb if cb else MC(lb, hb, lb, hb))
            return x

        def f(la, ha, lb, hb, stand=stand, args=args):
            o = apply_rule(ctx, stand, args(la, ha, lb, hb))
            lo, hi = (o.lo, o.hi) if is_mc(o) else (o, o)
            return _clean(jnp.reshape(jnp.asarray(lo, jnp.float64), ()), jnp.reshape(jnp.asarray(hi, jnp.float64), ()))

        rule = BACK.get(name)

        def b(zl, zh, la, ha, lb, hb, stand=stand, args=args, rule=rule, ca=ca, cb=cb, n_in=n_in):
            none = (-INF * jnp.ones(()), INF * jnp.ones(()), jnp.array(False))
            if rule is None:
                return none + none
            cands = list(rule(stand, (zl, zh), *args(la, ha, lb, hb))) + [None]
            res = []
            for i, (c, const) in enumerate(zip(cands[:2], (ca, cb))):
                if i >= n_in or c is None or const:
                    res += list(none)
                else:
                    lo, hi = _clean(*(jnp.reshape(jnp.asarray(v, jnp.float64), ()) for v in c))
                    res += [lo, hi, jnp.array(True)]
            return tuple(res)

        fwd.append(f)
        bwd.append(b)
    return fwd, bwd


def make_propagate(tape, ineq, eq, n, feas_tol, rounds, safe=False):
    """``fbbt(lo, hi, cutoff) -> (lo, hi, empty)`` by interpreting the scalarized tape, or None."""
    prog = scalarize(tape, n)
    if prog is None or len(prog.op) == 0:
        return None
    fwd, bwd = _branches(prog, safe)
    op, A, B, O = (jnp.asarray(v) for v in (prog.op, prog.a, prog.b, prog.out))
    K = len(prog.op)
    cval = jnp.asarray(np.nan_to_num(prog.const_val, nan=0.0))
    is_c = jnp.asarray(~np.isnan(prog.const_val))
    out_ids = [jnp.asarray(o) for o in prog.outs]

    def forward(P):
        """One forward sweep from the prior bounds P = (lo, hi)."""

        def step(k, carry):
            lo, hi, empty = carry
            a, b, o = A[k], B[k], O[k]
            vl, vh = lax.switch(op[k], fwd, lo[a], hi[a], lo[b], hi[b])
            vl, vh, bad = _meet2((vl, vh), (P[0][o], P[1][o]))
            return lo.at[o].set(vl), hi.at[o].set(vh), empty | bad

        return lax.fori_loop(0, K, step, (P[0], P[1], jnp.array(False)))

    def backward(lo, hi, cutoff):
        empty = jnp.array(False)
        zs = [(-INF, cutoff)] + ([(-INF, feas_tol)] if ineq is not None else []) + ([(-feas_tol, feas_tol)] if eq is not None else [])
        for ids, (zl, zh) in zip(out_ids, zs):
            ml, mh, bad = _meet2((lo[ids], hi[ids]), (jnp.full(ids.shape, zl), jnp.full(ids.shape, zh)))
            keep = is_c[ids]  # a constant output that violates its bound makes the box empty (strictly, as fbbt)
            empty |= jnp.any(jnp.where(keep, (cval[ids] < zl) | (cval[ids] > zh), bad))
            lo, hi = lo.at[ids].set(jnp.where(keep, lo[ids], ml)), hi.at[ids].set(jnp.where(keep, hi[ids], mh))

        def update(lo, hi, s, cl, ch, use):
            pl, ph = _pad(cl, ch)
            ml, mh, bad = _meet2((lo[s], hi[s]), (pl, ph))
            use = use & ~is_c[s]
            return lo.at[s].set(jnp.where(use, ml, lo[s])), hi.at[s].set(jnp.where(use, mh, hi[s])), use & bad

        def step(j, carry):
            lo, hi, empty = carry
            k = K - 1 - j
            a, b, o = A[k], B[k], O[k]
            al, ah, ua, bl, bh, ub = lax.switch(op[k], bwd, lo[o], hi[o], lo[a], hi[a], lo[b], hi[b])
            lo, hi, e1 = update(lo, hi, a, al, ah, ua)
            lo, hi, e2 = update(lo, hi, b, bl, bh, ub)
            return lo, hi, empty | e1 | e2

        return lax.fori_loop(0, K, step, (lo, hi, empty))

    def fbbt(lo, hi, cutoff=INF):
        cutoff = jnp.asarray(cutoff, jnp.float64)
        S = cval.shape[0]
        L0 = jnp.where(is_c, cval, -INF).at[:n].set(lo)
        H0 = jnp.where(is_c, cval, INF).at[:n].set(hi)

        def round_(_, carry):
            Pl, Ph, empty = carry
            vl, vh, e1 = forward((Pl, Ph))
            bl, bh, e2 = backward(vl, vh, cutoff)
            return jnp.where(is_c, cval, bl), jnp.where(is_c, cval, bh), empty | e1 | e2

        Pl, Ph, empty = lax.fori_loop(0, rounds, round_, (L0, H0, jnp.array(False)))
        vl, vh, e = forward((Pl, Ph))
        xl, xh = vl[:n], vh[:n]
        return jnp.minimum(xl, xh), jnp.maximum(xl, xh), empty | e

    fbbt.program = prog
    return fbbt


def use_interpreter(tape, n, min_rows=50, max_expansion=1.5):
    """Interpret large tapes that are mostly scalar already (scalarizing vectorized code multiplies the work)."""
    rows = sum(int(np.prod(tape.shapes[s][0])) for node in tape.nodes for s in node.outs
               if node.eqn.primitive.name not in STRUCTURAL)  # fmt: skip
    eqns = sum(1 for node in tape.nodes if node.eqn.primitive.name not in STRUCTURAL)
    return rows >= min_rows and rows <= max_expansion * max(eqns, 1)

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
  steps of ``schedule`` (groups of independent rows), each step evaluating
  every opcode once, vectorized over its rows. Each opcode's branch calls the same
  rules as ``fbbt`` (``relax.RULES`` in interval mode forward, ``fbbt.BACK``
  backward) on scalar intervals, with constant operands passed as constants,
  so the result is that of ``propagate`` element by element (sums of more than
  two terms become chains of ``add``, the same bounds up to rounding).

The kernel's size depends on the number of distinct opcodes, not on the number
of rows. Its run time is one loop step per ``schedule`` step, so ``make_fbbt``
uses it only for large, mostly scalar tapes (see ``use_interpreter``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .fbbt import BACK, CTX, INF, SAFE_CTX, Const, _clean, _empty, _meet2, _pad
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
    ids: dict = None  # tape slot -> array of scalar slot ids (its shape)


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
    return Program(r[:, 0], r[:, 1], r[:, 2], r[:, 3], np.asarray(const_val, np.float64), n, opcodes, outs, ids)


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


# Tapes this large are swept step by step (``schedule``), smaller ones row by row: at 300+ rows
# the steps were 1.2-4x faster (kriging_peaks, 1,162 rows: 561 -> 126 ms for 16 boxes), under
# ~120 rows (a handful of rows per step, padded to twice as many) up to 2x slower for 256 boxes.
SCHEDULE_MIN_ROWS = 300


def make_propagate(tape, ineq, eq, n, feas_tol, rounds, safe=False, scheduled=None):
    """``fbbt(lo, hi, cutoff) -> (lo, hi, empty)`` by interpreting the scalarized tape, or None.

    ``scheduled`` (default: at least ``SCHEDULE_MIN_ROWS`` rows) sweeps step by
    step instead of row by row.
    """
    prog = scalarize(tape, n)
    if prog is None or len(prog.op) == 0:
        return None
    fwd, bwd = _branches(prog, safe)
    if scheduled is None:
        scheduled = len(prog.op) >= SCHEDULE_MIN_ROWS
    run = _propagate_steps(prog, fwd, bwd, ineq, eq, n, feas_tol, rounds) if scheduled else \
        _propagate_rows(prog, fwd, bwd, ineq, eq, n, feas_tol, rounds)  # fmt: skip
    run.program = prog
    return run


def _propagate_steps(prog, fwd, bwd, ineq, eq, n, feas_tol, rounds):
    """Step by step through ``layout``'s schedule: forward, each step evaluates its rows and
    writes them as one block; backward, the steps are taken in reverse, each opcode's rows
    narrowing their operands together (candidates for the same slot intersected by scatter
    max / min) before the next opcode's rows read them."""
    lay = layout(prog)
    prog = lay.prog
    A, B, OUT = (jnp.asarray(v) for v in (prog.a, prog.b, prog.out))
    R = [jnp.asarray(r) for r in lay.rows]
    VALID = [jnp.asarray(r >= 0) for r in lay.rows]
    START = jnp.asarray(lay.start[0])  # a step's blocks are adjacent, opcode 0's first
    T = lay.steps
    cval = jnp.asarray(np.nan_to_num(prog.const_val, nan=0.0))
    is_c = jnp.asarray(~np.isnan(prog.const_val))
    out_ids = [jnp.asarray(o) for o in prog.outs]

    def forward(P):
        """One forward sweep from the prior bounds ``P`` (``(S, 2)``: lo, hi)."""

        def step(t, c):
            V, empty = c
            rs, bads = [], []
            for o in range(len(R)):  # read everything, then write the step's block once (in place)
                k = jnp.maximum(R[o][t], 0)
                va, vb, p = V[A[k]], V[B[k]], P[OUT[k]]
                vl, vh = jax.vmap(fwd[o])(va[:, 0], va[:, 1], vb[:, 0], vb[:, 1])
                vl, vh, bad = _meet2((vl, vh), (p[:, 0], p[:, 1]))
                rs.append(jnp.stack([vl, vh], axis=1))
                bads.append(bad & VALID[o][t])
            V = lax.dynamic_update_slice_in_dim(V, jnp.concatenate(rs), START[t], 0)
            return V, empty | jnp.any(jnp.concatenate(bads))

        return lax.fori_loop(0, T, step, (P, jnp.array(False)))

    def backward(V, cutoff):
        lo, hi = V[:, 0], V[:, 1]
        empty = jnp.array(False)
        zs = [(-INF, cutoff)] + ([(-INF, feas_tol)] if ineq is not None else []) + ([(-feas_tol, feas_tol)] if eq is not None else [])
        for ids, (zl, zh) in zip(out_ids, zs):
            ml, mh, bad = _meet2((lo[ids], hi[ids]), (jnp.full(ids.shape, zl), jnp.full(ids.shape, zh)))
            keep = is_c[ids]  # a constant output that violates its bound makes the box empty (strictly, as fbbt)
            empty |= jnp.any(jnp.where(keep, (cval[ids] < zl) | (cval[ids] > zh), bad))
            lo, hi = lo.at[ids].set(jnp.where(keep, lo[ids], ml)), hi.at[ids].set(jnp.where(keep, hi[ids], mh))

        def step(j, c):
            V, empty = c
            t = T - 1 - j
            # opcode groups narrow the carry one after another (rows of the same opcode in a step
            # see each other's narrowing only in the next round): closer to the row-by-row sweep
            for o in range(len(R)):
                k = jnp.maximum(R[o][t], 0)
                a, b = A[k], B[k]
                z, va, vb = V[OUT[k]], V[a], V[b]
                al, ah, ua, bl, bh, ub = jax.vmap(bwd[o])(z[:, 0], z[:, 1], va[:, 0], va[:, 1], vb[:, 0], vb[:, 1])
                idx, cl, ch, use = [], [], [], []
                for s, l_, h_, u_ in ((a, al, ah, ua), (b, bl, bh, ub)):
                    pl, ph = _pad(l_, h_)
                    u_ = u_ & VALID[o][t] & ~is_c[s]
                    idx.append(s)
                    cl.append(jnp.where(u_, pl, -INF))
                    ch.append(jnp.where(u_, ph, INF))
                    use.append(u_)
                idx, cl, ch, use = (jnp.concatenate(v) for v in (idx, cl, ch, use))
                gl = V[:, 0].at[idx].max(cl)
                gh = V[:, 1].at[idx].min(ch)
                bad = _empty(gl[idx], gh[idx])  # _meet2 per slot: slightly crossed endpoints are reordered
                empty = empty | jnp.any(bad & use)
                sl = jnp.where(bad, gl[idx], jnp.minimum(gl[idx], gh[idx]))
                sh = jnp.where(bad, gh[idx], jnp.maximum(gl[idx], gh[idx]))
                V = jnp.stack([gl.at[idx].set(sl), gh.at[idx].set(sh)], axis=1)
            return V, empty

        return lax.fori_loop(0, T, step, (jnp.stack([lo, hi], axis=1), empty))

    def fbbt(lo, hi, cutoff=INF):
        cutoff = jnp.asarray(cutoff, jnp.float64)
        P0 = jnp.stack([jnp.where(is_c, cval, -INF).at[:n].set(lo), jnp.where(is_c, cval, INF).at[:n].set(hi)], axis=1)

        def round_(_, carry):
            P, empty = carry
            V, e1 = forward(P)
            V, e2 = backward(V, cutoff)
            return jnp.where(is_c[:, None], cval[:, None], V), empty | e1 | e2

        P, empty = lax.fori_loop(0, rounds, round_, (P0, jnp.array(False)))
        V, e = forward(P)
        xl, xh = V[:n, 0], V[:n, 1]
        return jnp.minimum(xl, xh), jnp.maximum(xl, xh), empty | e

    return fbbt


def _propagate_rows(prog, fwd, bwd, ineq, eq, n, feas_tol, rounds):
    """Row by row: one loop step per row, a ``lax.switch`` on its opcode."""
    op, A, B, OUT = (jnp.asarray(v) for v in (prog.op, prog.a, prog.b, prog.out))
    K = len(prog.op)
    cval = jnp.asarray(np.nan_to_num(prog.const_val, nan=0.0))
    is_c = jnp.asarray(~np.isnan(prog.const_val))
    out_ids = [jnp.asarray(o) for o in prog.outs]

    def forward(P):
        """One forward sweep from the prior bounds P = (lo, hi)."""

        def step(k, carry):
            lo, hi, empty = carry
            a, b, o = A[k], B[k], OUT[k]
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
            a, b, o = A[k], B[k], OUT[k]
            al, ah, ua, bl, bh, ub = lax.switch(op[k], bwd, lo[o], hi[o], lo[a], hi[a], lo[b], hi[b])
            lo, hi, e1 = update(lo, hi, a, al, ah, ua)
            lo, hi, e2 = update(lo, hi, b, bl, bh, ub)
            return lo, hi, empty | e1 | e2

        return lax.fori_loop(0, K, step, (lo, hi, empty))

    def fbbt(lo, hi, cutoff=INF):
        cutoff = jnp.asarray(cutoff, jnp.float64)
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

    return fbbt


def use_interpreter(tape, n, min_rows=50, max_expansion=1.5):
    """Interpret large tapes that are mostly scalar already (scalarizing vectorized code multiplies the work)."""
    rows = sum(int(np.prod(tape.shapes[s][0])) for node in tape.nodes for s in node.outs
               if node.eqn.primitive.name not in STRUCTURAL)  # fmt: skip
    eqns = sum(1 for node in tape.nodes if node.eqn.primitive.name not in STRUCTURAL)
    return rows >= min_rows and rows <= max_expansion * max(eqns, 1)


class Schedule(NamedTuple):
    rows: list  # per opcode: (T, W_o) row indices, -1 for padding
    steps: int  # T


def schedule(prog, step_cost=8.0, factors=(0.5, 1.0, 2.0, 4.0, 8.0)):
    """Group the rows into ``T`` steps of mutually independent rows, at most ``W_o`` per opcode.

    A step evaluates every opcode once, vectorized over its ``W_o`` slots, so an
    interpreter runs ``T`` loop iterations instead of one per row. Rows become
    ready when the rows producing their operands are done in an earlier step;
    ready rows are taken longest-remaining-path first (greedy list scheduling).
    The widths are ``factor * count_o / critical_path`` for the factor that
    minimizes ``T * (step_cost + sum_o W_o)``, the padded work.
    """
    import heapq

    K, nop = len(prog.op), len(prog.opcodes)
    row_of = np.full(len(prog.const_val), -1, np.int64)
    row_of[prog.out] = np.arange(K)
    deps = [sorted({int(row_of[s]) for s in (prog.a[k], prog.b[k]) if row_of[s] >= 0}) for k in range(K)]
    users = [[] for _ in range(K)]
    for k, ds in enumerate(deps):
        for d in ds:
            users[d].append(k)
    height = np.ones(K, np.int64)  # longest path from the row to an output, in rows
    for k in range(K - 1, -1, -1):
        for u in users[k]:
            height[k] = max(height[k], height[u] + 1)
    crit = int(height.max()) if K else 1
    count = np.bincount(prog.op, minlength=nop)

    def run(widths):
        left = np.array([len(d) for d in deps])
        heaps = [[] for _ in range(nop)]
        for k in range(K):
            if left[k] == 0:
                heapq.heappush(heaps[prog.op[k]], (-height[k], k))
        steps, done = [], 0
        while done < K:
            took = [[heapq.heappop(heaps[o])[1] for _ in range(min(widths[o], len(heaps[o])))] for o in range(nop)]
            steps.append(took)
            for rows in took:
                done += len(rows)
                for k in rows:
                    for u in users[k]:
                        left[u] -= 1
                        if left[u] == 0:
                            heapq.heappush(heaps[prog.op[u]], (-height[u], u))
        return steps

    best = None
    for f in factors:
        widths = [max(1, int(np.ceil(f * c / crit))) for c in count]
        steps = run(widths)
        cost = len(steps) * (step_cost + sum(widths))
        if best is None or cost < best[0]:
            best = (cost, widths, steps)
    _, widths, steps = best
    rows = []
    for o in range(nop):
        r = np.full((len(steps), widths[o]), -1, np.int64)
        for t, took in enumerate(steps):
            r[t, : len(took[o])] = took[o]
        rows.append(r)
    return Schedule(rows, len(steps))


class Layout(NamedTuple):
    prog: Program  # renumbered: slots of x and constants first, then one block per (step, opcode)
    rows: list  # per opcode: (T, W_o) row indices of the renumbered program, -1 for padding
    start: list  # per opcode: (T,) first slot of its block in each step
    steps: int


def layout(prog, sched=None):
    """Renumber the slots so each step writes each opcode's results to one contiguous block.

    An interpreter then stores a step's results with ``dynamic_update_slice``,
    which XLA does in place inside a loop, where a scatter to scattered slots
    copies the whole (batched) slot array every time. The block of opcode ``o``
    in step ``t`` has ``W_o`` slots; padding rows write to the unused ones.
    """
    sched = sched or schedule(prog)
    S = len(prog.const_val)
    is_row = np.zeros(S, bool)
    is_row[prog.out] = True
    keep = np.flatnonzero(~is_row)  # x first (slots 0..n-1 are never row outputs), then constants
    new = np.full(S, -1, np.int64)
    new[keep] = np.arange(keep.size)
    nxt = keep.size
    start = [np.zeros(sched.steps, np.int64) for _ in sched.rows]
    for t in range(sched.steps):
        for o, r in enumerate(sched.rows):
            start[o][t] = nxt
            for i, k in enumerate(r[t]):
                if k >= 0:
                    new[prog.out[k]] = nxt + i
            nxt += r.shape[1]
    const_val = np.full(nxt, np.nan)
    const_val[new[keep]] = prog.const_val[keep]
    ids = {s: new[np.asarray(v)] for s, v in (prog.ids or {}).items()}
    p2 = Program(prog.op, new[prog.a], new[prog.b], new[prog.out], const_val, prog.n, prog.opcodes,
                 [new[o] for o in prog.outs], ids)  # fmt: skip
    return Layout(p2, sched.rows, start, sched.steps)

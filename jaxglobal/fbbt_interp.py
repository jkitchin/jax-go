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

from .fbbt import BACK, CTX, DOMAIN_FLOOR, INF, SAFE_CTX, Const, _clean, _empty, _meet2, _pad, round_integers, stops_nan
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
        if len(node.outs) != 1 or stops_nan(node):  # rows would pass NaN through an int cast or x**0
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
# scheduled sweeps stop before ``rounds`` once a round moves no slot by more than this fraction of
# its width (``_rounds``): on smallinvDAX (1583 rows) the later rounds of the default 5 changed
# nothing and FBBT was 15x the cost of the node LP
STALL = 1e-3


def _live_rows(prog):
    """``prog`` without the rows whose result reaches no output.

    ``jnp.log(x)[0]`` computes ``log`` of every element of ``x``; the rows of the unused elements
    would apply log's domain (``x > 0``) to variables it does not constrain, and report a box
    without such points empty."""
    live = set(int(s) for o in prog.outs for s in np.ravel(o))
    keep = np.zeros(len(prog.op), bool)
    for k in range(len(prog.op) - 1, -1, -1):
        if int(prog.out[k]) in live:
            keep[k] = True
            live.add(int(prog.a[k]))
            live.add(int(prog.b[k]))
    if keep.all():
        return prog
    return prog._replace(op=prog.op[keep], a=prog.a[keep], b=prog.b[keep], out=prog.out[keep])


def balance_sums(prog):
    """``prog`` with every chain of two-operand ``add`` rows rebuilt as a balanced tree.

    A sum of ``L`` terms read from an ``.nl`` file is a chain ``((t1 + t2) + t3) + ...``:
    ``L - 1`` rows, each waiting for the previous one, so ``schedule`` needs ``L - 1`` steps for
    it (smallinvDAX: a 495-term quadratic, 469 of its 1583 rows' steps were that chain, and
    FBBT cost 740 ms per 16 boxes). Interval addition is associative up to rounding, so the
    forward value of the sum, and the backward narrowing of each term (the sum's interval minus
    that of all other terms), are the same for any bracketing. Here a maximal tree of add rows
    (both operands non-constant, every inner result used once and by the tree only) keeps its
    root's output slot and row count; its terms are paired lowest depth first (Huffman on the
    depths), so the tree's depth is about ``log2 L`` past its deepest term. The rows are emitted
    in topological order at the root's position, and inner slots belong to no other row or
    output, so nothing outside a tree changes. The inner slots now hold other partial sums, so
    the ``ids`` entries (tape slot -> scalar slots) that name one are dropped; the others, and the
    outputs, keep their meaning. Returns ``prog`` itself when nothing changes."""
    import heapq

    K = len(prog.op)
    if K == 0:
        return prog
    addable = np.array([oc[0] in ("add", "add_any") and not oc[2] and not oc[3] and not oc[4] and oc[5] == 2
                        for oc in prog.opcodes], bool)  # fmt: skip
    S = len(prog.const_val)
    uses = np.zeros(S, np.int64)
    np.add.at(uses, prog.a, 1)
    np.add.at(uses, prog.b, 1)
    np.add.at(uses, prog.b, -(prog.a == prog.b).astype(np.int64))  # a unary row (b = a) reads once
    for o in prog.outs:
        np.add.at(uses, np.ravel(np.asarray(o, np.int64)), 1)
    row_of = np.full(S, -1, np.int64)
    row_of[prog.out] = np.arange(K)
    is_add = addable[prog.op]
    # an inner row: an add whose result is read once, by another add row
    user = np.full(S, -1, np.int64)
    for k in range(K):
        user[prog.a[k]] = k
        user[prog.b[k]] = k
    inner = np.zeros(K, bool)
    for k in np.flatnonzero(is_add):
        s = prog.out[k]
        inner[k] = uses[s] == 1 and user[s] >= 0 and is_add[user[s]]
    if not inner.any():
        return prog
    depth = np.zeros(S, np.int64)  # steps until the slot is known, in the rebuilt program
    rows, moved = [], []
    for k in range(K):
        if inner[k]:
            continue
        if not is_add[k]:
            rows.append((prog.op[k], prog.a[k], prog.b[k], prog.out[k]))
            depth[prog.out[k]] = 1 + max(depth[prog.a[k]], depth[prog.b[k]])
            continue
        leaves, slots, stack = [], [], [prog.a[k], prog.b[k]]
        while stack:  # terms of the tree rooted at row k, and the inner slots it frees
            s = stack.pop()
            r = row_of[s]
            if r >= 0 and inner[r]:
                slots.append(s)
                stack += [prog.b[r], prog.a[r]]
            else:
                leaves.append(s)
        moved += slots
        heap = [(int(depth[s]), i, int(s)) for i, s in enumerate(leaves)]
        heapq.heapify(heap)
        i = len(heap)
        while len(heap) > 1:
            da, _, sa = heapq.heappop(heap)
            db, _, sb = heapq.heappop(heap)
            out = prog.out[k] if not heap else slots.pop()
            rows.append((prog.op[k], sa, sb, out))
            depth[out] = 1 + max(da, db)
            heapq.heappush(heap, (int(depth[out]), i, int(out)))
            i += 1
    if not moved:
        return prog
    r = np.asarray(rows, np.int64).reshape(-1, 4)
    moved = np.asarray(moved, np.int64)
    ids = None if prog.ids is None else \
        {t: v for t, v in prog.ids.items() if not np.isin(np.asarray(v), moved).any()}  # fmt: skip
    return prog._replace(op=r[:, 0], a=r[:, 1], b=r[:, 2], out=r[:, 3], ids=ids)


# What one schedule step costs, for ``_pick_program``, in rows of the cheapest opcode: measured
# on the warm scheduled sweep (16 boxes, CPU). A row costs about its opcode's forward and
# backward rules (``_row_weights``: a mul or integer_pow row ~3 add rows), a step ~8 cheap rows
# more (~4 gathers and scatters per opcode). Once a slot array outgrows ~512 KiB (more than
# LARGE_SLOTS slots for 16 boxes) XLA splits each of those into parallel tasks, a few
# microseconds apiece, and a step costs ~STEP_ROWS_LARGE rows instead: 300 steps of 9 scattered
# groups took 3 ms at 4,000 slots and 16 ms at 4,500. Fitted to 11 instances (see
# ``_pick_program``); STEP_ROWS_LARGE from 136 to 200 picks the faster program on all of them.
STEP_ROWS = 8.0
STEP_ROWS_LARGE = 168.0
LARGE_SLOTS = 4096


def _row_weights(fwd, bwd):
    """Per opcode, the relative cost of one row: the equations of its forward and backward rules,
    over the cheapest opcode's."""
    z = jnp.zeros(())
    e = np.array([len(jax.make_jaxpr(f)(*[z] * 4).jaxpr.eqns) + len(jax.make_jaxpr(g)(*[z] * 6).jaxpr.eqns)
                  for f, g in zip(fwd, bwd)], float)  # fmt: skip
    return e / max(e.min(), 1.0)


def _cost(prog, sched, w):
    """Estimated run time of ``prog`` swept by ``sched`` (row weights ``w``), in cheap rows."""
    W = [r.shape[1] for r in sched.rows]
    slots = len(prog.const_val) - len(prog.op) + sched.steps * sum(W)  # x, constants, the blocks
    return sched.steps * ((STEP_ROWS_LARGE if slots > LARGE_SLOTS else STEP_ROWS) + float(np.dot(w, W)))


def _pick_program(prog, fwd, bwd):
    """``(prog, schedule)``: ``prog`` or ``balance_sums(prog)``, whichever is estimated faster
    (``_cost``). ``fwd``, ``bwd``: the opcodes' rules (``_branches``), shared by both programs.

    Balancing shortens the critical path, but ``schedule`` then picks other widths, and the
    padding may push the slot array past LARGE_SLOTS: on kriging_peaks-red050 (critical path 64 ->
    21) the same 76 steps with 52 rows each instead of 33, and 4,114 slots instead of 2,670, 4.4x
    slower; on smallinvDAX (a 495-term chain) 469 -> 34 steps and 13x faster; on
    kriging_peaks-red500, 516 -> 104 steps but 39 -> 406 rows per step, 2.6x faster; on st_rv9,
    128 -> 22 steps but the mul rows per step 5 -> 51, 1.2x slower. Always balancing loses on
    red050, and unweighted rows pick st_rv9's balanced program. On 11 timed instances (the four kriging_peaks, clay0205m,
    st_rv9, heatexch_spec1, synheat, batch, smallinvDAX, du-opt) the pick was the faster program,
    or within noise of it."""
    sched = schedule(prog)
    bal = balance_sums(prog)
    if bal is prog:
        return prog, sched
    sb = schedule(bal)
    w = _row_weights(fwd, bwd)
    return (bal, sb) if _cost(bal, sb, w) < _cost(prog, sched, w) else (prog, sched)


def make_propagate(tape, ineq, eq, n, feas_tol, rounds, safe=False, scheduled=None, integer=()):
    """``fbbt(lo, hi, cutoff) -> (lo, hi, empty)`` by interpreting the scalarized tape, or None.

    ``scheduled`` (default: at least ``SCHEDULE_MIN_ROWS`` rows) sweeps step by
    step instead of row by row.
    """
    prog = scalarize(tape, n)
    if prog is None:
        return None
    prog = _live_rows(prog)
    if len(prog.op) == 0:
        return None
    fwd, bwd = _branches(prog, safe)
    if scheduled is None:
        scheduled = len(prog.op) >= SCHEDULE_MIN_ROWS
    if scheduled:  # the sweeps take one step per level of the tape: long add chains may be balanced
        prog, sched = _pick_program(prog, fwd, bwd)
        run = _propagate_steps(prog, fwd, bwd, ineq, eq, n, feas_tol, rounds, integer, sched)
    else:
        run = _propagate_rows(prog, fwd, bwd, ineq, eq, n, feas_tol, rounds, integer)
    run.program = prog
    return run


def _propagate_steps(prog, fwd, bwd, ineq, eq, n, feas_tol, rounds, integer=(), sched=None):
    """Step by step through ``layout``'s schedule: forward, each step evaluates its rows and
    writes them as one block; backward, the steps are taken in reverse, each opcode's rows
    narrowing their operands together (candidates for the same slot intersected by scatter
    max / min) before the next opcode's rows read them."""
    lay = layout(prog, sched)
    prog = lay.prog
    A, B, OUT = (jnp.asarray(v) for v in (prog.a, prog.b, prog.out))
    R = [jnp.asarray(r) for r in lay.rows]
    VALID = [jnp.asarray(r >= 0) for r in lay.rows]
    FLOOR = [DOMAIN_FLOOR.get(oc[0]) for oc in prog.opcodes]
    START = jnp.asarray(lay.start[0])  # a step's blocks are adjacent, opcode 0's first
    T = lay.steps
    BW = sum(r.shape[1] for r in lay.rows)  # slots of one step's block
    VALID_ALL = jnp.asarray(np.concatenate([r >= 0 for r in lay.rows], axis=1))
    cval = jnp.asarray(np.nan_to_num(prog.const_val, nan=0.0))
    is_c = jnp.asarray(~np.isnan(prog.const_val))
    out_ids = [jnp.asarray(o) for o in prog.outs]

    def forward(P):
        """One forward sweep from the prior bounds ``P`` (``(S, 2)``: lo, hi)."""

        def step(t, c):
            V, empty = c
            rs = []
            for o in range(len(R)):  # read everything, then write the step's block once (in place)
                k = jnp.maximum(R[o][t], 0)
                va, vb, p = V[A[k]], V[B[k]], V[OUT[k]]  # the step's block still holds P
                vl, vh = jax.vmap(fwd[o])(va[:, 0], va[:, 1], vb[:, 0], vb[:, 1])
                vl, vh, _ = _meet2((vl, vh), (p[:, 0], p[:, 1]))
                rs.append(jnp.stack([vl, vh], axis=1))
            V = lax.dynamic_update_slice_in_dim(V, jnp.concatenate(rs), START[t], 0)
            # emptiness from the block as written (_meet2 leaves an empty meet crossed): reading
            # the new V, not the old, lets XLA update V in place instead of copying it every step
            blk = lax.dynamic_slice_in_dim(V, START[t], BW, 0)
            return V, empty | jnp.any(_empty(blk[:, 0], blk[:, 1]) & VALID_ALL[t])

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
            L, H, empty = c  # lo and hi carried apart, so the scatters below update them in place
            t = T - 1 - j
            # opcode groups narrow the carry one after another (rows of the same opcode in a step
            # see each other's narrowing only in the next round): closer to the row-by-row sweep
            for o in range(len(R)):
                k = jnp.maximum(R[o][t], 0)
                a, b, out = A[k], B[k], OUT[k]
                al, ah, ua, bl, bh, ub = jax.vmap(bwd[o])(L[out], H[out], L[a], H[a], L[b], H[b])
                idx, cl, ch, use = [], [], [], []
                for s, l_, h_, u_ in ((a, al, ah, ua), (b, bl, bh, ub)):
                    pl, ph = _pad(l_, h_)
                    if s is a and FLOOR[o] is not None:  # the operand of log / log1p: see fbbt.DOMAIN_FLOOR
                        pl = jnp.maximum(pl, FLOOR[o])
                    u_ = u_ & VALID[o][t] & ~is_c[s]
                    idx.append(s)
                    cl.append(jnp.where(u_, pl, -INF))
                    ch.append(jnp.where(u_, ph, INF))
                    use.append(u_)
                # the barriers make every read of L and H come before either is written: else XLA
                # copies them (all S slots) to keep the old values for a read it may order later
                cl, ch = lax.optimization_barrier((jnp.concatenate(cl), jnp.concatenate(ch)))
                idx, use = jnp.concatenate(idx), jnp.concatenate(use)
                L = L.at[idx].max(cl)
                H = H.at[idx].min(ch)
                gl, gh = lax.optimization_barrier((L[idx], H[idx]))
                bad = _empty(gl, gh)  # _meet2 per slot: slightly crossed endpoints are reordered
                empty = empty | jnp.any(bad & use)
                L = L.at[idx].set(jnp.where(bad, gl, jnp.minimum(gl, gh)))
                H = H.at[idx].set(jnp.where(bad, gh, jnp.maximum(gl, gh)))
            return L, H, empty

        L, H, empty = lax.fori_loop(0, T, step, (lo, hi, empty))
        return jnp.stack([L, H], axis=1), empty

    def fbbt(lo, hi, cutoff=INF):
        cutoff = jnp.asarray(cutoff, jnp.float64)
        P0 = jnp.stack([jnp.where(is_c, cval, -INF).at[:n].set(lo), jnp.where(is_c, cval, INF).at[:n].set(hi)], axis=1)

        def round_(_, carry):
            P, empty = carry
            V, e1 = forward(P)
            V, e2 = backward(V, cutoff)
            xl, xh, e3 = round_integers(V[:n, 0], V[:n, 1], integer)  # the first n slots are x
            V = V.at[:n, 0].set(xl).at[:n, 1].set(xh)
            return jnp.where(is_c[:, None], cval[:, None], V), empty | e1 | e2 | e3

        P, empty = _rounds(round_, rounds, P0)
        V, e = forward(P)
        xl, xh = V[:n, 0], V[:n, 1]
        return jnp.minimum(xl, xh), jnp.maximum(xl, xh), empty | e

    return fbbt


def _progress(P, Q):
    """Whether a round moved any slot's bounds ``P -> Q`` (``(S, 2)``) by more than ``STALL`` of
    its width (or of 1 for a point or an unbounded slot): an infinite bound becoming finite counts."""
    w = jnp.where(jnp.isfinite(P[:, 1] - P[:, 0]), jnp.maximum(P[:, 1] - P[:, 0], 0.0), jnp.inf)
    tol = STALL * jnp.where(jnp.isfinite(w) & (w > 0), w, 1.0)
    d = jnp.abs(jnp.where(jnp.isfinite(Q), Q, 0.0) - jnp.where(jnp.isfinite(P), P, 0.0))
    newly = jnp.isfinite(Q) & ~jnp.isfinite(P)
    return jnp.any(newly) | jnp.any(d > tol[:, None])


def _rounds(round_, rounds, P0):
    """Up to ``rounds`` FBBT rounds ``round_(k, (P, empty))``; stops early once a round no longer
    makes progress (``_progress``) or finds the box empty. Fewer rounds only leave a box larger,
    so stopping is always sound."""

    def cond(c):
        k, _, empty, moved = c
        return (k < rounds) & moved & ~empty

    def body(c):
        k, P, empty, _ = c
        Q, e = round_(k, (P, empty))
        return k + 1, Q, e, _progress(P, Q)

    _, P, empty, _ = lax.while_loop(cond, body, (jnp.array(0), P0, jnp.array(False), jnp.array(True)))
    return P, empty


def _propagate_rows(prog, fwd, bwd, ineq, eq, n, feas_tol, rounds, integer=()):
    """Row by row: one loop step per row, a ``lax.switch`` on its opcode."""
    op, A, B, OUT = (jnp.asarray(v) for v in (prog.op, prog.a, prog.b, prog.out))
    K = len(prog.op)
    cval = jnp.asarray(np.nan_to_num(prog.const_val, nan=0.0))
    is_c = jnp.asarray(~np.isnan(prog.const_val))
    out_ids = [jnp.asarray(o) for o in prog.outs]
    FLOOR = jnp.asarray([DOMAIN_FLOOR.get(oc[0], -np.inf) for oc in prog.opcodes], jnp.float64)

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

        def update(lo, hi, s, cl, ch, use, floor=-INF):
            pl, ph = _pad(cl, ch)
            pl = jnp.maximum(pl, floor)  # the operand of log / log1p: see fbbt.DOMAIN_FLOOR
            ml, mh, bad = _meet2((lo[s], hi[s]), (pl, ph))
            use = use & ~is_c[s]
            return lo.at[s].set(jnp.where(use, ml, lo[s])), hi.at[s].set(jnp.where(use, mh, hi[s])), use & bad

        def step(j, carry):
            lo, hi, empty = carry
            k = K - 1 - j
            a, b, o = A[k], B[k], OUT[k]
            al, ah, ua, bl, bh, ub = lax.switch(op[k], bwd, lo[o], hi[o], lo[a], hi[a], lo[b], hi[b])
            lo, hi, e1 = update(lo, hi, a, al, ah, ua, FLOOR[op[k]])
            lo, hi, e2 = update(lo, hi, b, bl, bh, ub)
            return lo, hi, empty | e1 | e2

        return lax.fori_loop(0, K, step, (lo, hi, empty))

    def fbbt(lo, hi, cutoff=INF, all_slots=False):
        """With ``all_slots`` the bounds of every scalar slot of ``prog`` are returned, not just x."""
        cutoff = jnp.asarray(cutoff, jnp.float64)
        L0 = jnp.where(is_c, cval, -INF).at[:n].set(lo)
        H0 = jnp.where(is_c, cval, INF).at[:n].set(hi)

        def round_(_, carry):
            Pl, Ph, empty = carry
            vl, vh, e1 = forward((Pl, Ph))
            bl, bh, e2 = backward(vl, vh, cutoff)
            xl, xh, e3 = round_integers(bl[:n], bh[:n], integer)  # the first n slots are x
            bl, bh = bl.at[:n].set(xl), bh.at[:n].set(xh)
            return jnp.where(is_c, cval, bl), jnp.where(is_c, cval, bh), empty | e1 | e2 | e3

        Pl, Ph, empty = lax.fori_loop(0, rounds, round_, (L0, H0, jnp.array(False)))
        vl, vh, e = forward((Pl, Ph))
        if all_slots:
            return jnp.minimum(vl, vh), jnp.maximum(vl, vh), empty | e
        xl, xh = vl[:n], vh[:n]
        return jnp.minimum(xl, xh), jnp.maximum(xl, xh), empty | e

    return fbbt


def tape_size(tape):
    """``(rows, eqns)``: scalar rows ``scalarize`` would make (elements of non-structural outputs) and equations."""
    rows = sum(int(np.prod(tape.shapes[s][0])) for node in tape.nodes for s in node.outs
               if node.eqn.primitive.name not in STRUCTURAL)  # fmt: skip
    eqns = sum(1 for node in tape.nodes if node.eqn.primitive.name not in STRUCTURAL)
    return rows, eqns


# ``use_interpreter``'s rule for FBBT: "ops" interprets when the straight-line FBBT kernel would
# evaluate at least ``FBBT_MIN_OPS`` scalar rules per box (``fbbt_ops``); "rows" is the former fixed
# threshold of 50 rows (kept for ablations: ``--set fbbt_interp.RULE=rows``).
RULE = "ops"
MIN_ROWS = 50
FBBT_ROUNDS = 5  # minimize's default fbbt_rounds
FBBT_MIN_OPS = MIN_ROWS * (2 * FBBT_ROUNDS + 1)


def fbbt_ops(rows, rounds=FBBT_ROUNDS):
    """Scalar rule evaluations per box of the straight-line FBBT kernel: a forward and a backward
    sweep over the rows per round, and the final forward sweep."""
    return rows * (2 * rounds + 1)


def use_interpreter(tape, n, min_rows=None, max_expansion=1.5, rounds=FBBT_ROUNDS):
    """Interpret large tapes that are mostly scalar already (scalarizing vectorized code multiplies the work).

    ``min_rows`` (given): at least that many rows (``min_rows=0`` only asks whether the tape is
    mostly scalar). Otherwise the FBBT rule ``RULE``.
    """
    rows, eqns = tape_size(tape)
    if rows > max_expansion * max(eqns, 1):
        return False
    if min_rows is not None:
        return rows >= min_rows
    if RULE == "rows":
        return rows >= MIN_ROWS
    return fbbt_ops(rows, rounds) >= FBBT_MIN_OPS


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

"""FBBT with one compiled kernel for many problems: the scalar tape is an argument, not a constant.

``fbbt_interp`` interprets the scalarized tape (``fbbt_interp.scalarize``), so its kernel no longer
grows with the model, but the tape's arrays, its opcodes (one branch per distinct primitive,
parameters and operand kind) and the schedule's shapes are still baked into the kernel: every
problem compiles its own. Here they are data:

* a fixed **universal opcode table** (``KINDS``): every elementwise primitive the interpreter
  handles, in each operand variant (constant first or second operand, ``x op x``), and
  ``integer_pow`` for the exponents in ``POWERS``. A problem's opcodes are mapped onto it
  (``encode``); one it does not cover (another exponent, unusual parameters) makes ``encode``
  return None and the caller keeps its own kernel;
* the program (rows, operand slots, constants, which slots are outputs and of what kind, the
  integer mask) is padded to power-of-two **buckets** of rows, slots, variables and outputs, so
  problems of similar size share one executable;
* the sweeps are those of ``fbbt_interp._propagate_rows``: one loop step per row, a ``lax.switch``
  on its opcode. The opcode is the same for every box of a batch, so under ``vmap`` the switch
  stays a real branch and only one rule runs per row. The tolerance, the number of rounds and the
  number of real rows are arguments too.

The executable then depends only on the buckets and ``safe``: compiled once in a process (and,
with JAX's persistent compilation cache, once per machine), it serves every problem in the
bucket. The rules are those of the interpreter (``fbbt_interp._branches`` on canonical equations
of the universal opcodes), so the bounds are the interpreter's, element by element.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .fbbt import DOMAIN_FLOOR, INF, _meet2, _pad
from .fbbt_interp import Program, _branches, _live_rows, scalarize

POWERS = (2, 3, 4, 5, 6, 7, 8, -1, -2, -3)  # integer_pow exponents with an opcode of their own
INT_TOL = 1e-6  # as fbbt.round_integers

_UNARY = ["neg", "exp", "exp2", "expm1", "log", "log1p", "sqrt", "rsqrt", "square", "abs", "tanh", "logistic",
          "erf", "erfc", "sin", "cos", "atan", "asinh", "sinh", "cosh"]  # fmt: skip
_BINARY = ["add", "sub", "mul", "div", "max", "min", "pow"]
# (const_a, const_b, same): a binary row's operand variants (both constant rows are folded by scalarize)
_VARIANTS = [(False, False, False), (False, False, True), (True, False, False), (False, True, False)]


def _canonical_eqns():
    """A traced equation per primitive name (and per ``integer_pow`` exponent): what the rules read."""
    x = jnp.zeros((), jnp.float64)
    fns = {name: getattr(lax, name) for name in _UNARY if hasattr(lax, name)}
    fns.update({name: getattr(lax, name) for name in _BINARY})
    if hasattr(lax, "one_minus_square"):  # (jax's derivative of tanh)
        fns["one_minus_square"] = lax.one_minus_square
    eqns = {}
    for name, fn in fns.items():
        args = (x, x) if name in _BINARY else (x,)
        try:
            jaxpr = jax.make_jaxpr(fn)(*args).jaxpr
        except Exception:  # not in this jax version
            continue
        if len(jaxpr.eqns) == 1 and jaxpr.eqns[0].primitive.name == name:
            eqns[(name, None)] = jaxpr.eqns[0]
    for y in POWERS:
        eqns[("integer_pow", y)] = jax.make_jaxpr(lambda v, y=y: lax.integer_pow(v, y))(x).jaxpr.eqns[0]
    return eqns


@functools.lru_cache(maxsize=None)
def kinds():
    """The universal opcode table: ``(name, power, same, const_a, const_b, n_in, eqn)`` per opcode,
    in a fixed order (the order only has to be the same within a process and a jax version)."""
    from .relax import RULES

    out = []
    for (name, y), eqn in sorted(_canonical_eqns().items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        if name not in RULES:
            continue
        if name in _BINARY:
            out += [(name, y, same, ca, cb, 2, eqn) for ca, cb, same in _VARIANTS]
        else:
            out.append((name, y, False, False, False, 1, eqn))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _kind_index():
    return {k[:6]: i for i, k in enumerate(kinds())}


# The opcodes of the small ("core") kernel: those used by at least 15 of 343 MINLPLib models
# (global50, start50, int50, held out; counted 2026-11-03), plus ``x * c`` and ``x / c``, which
# .nl files never contain (they put the constant first) but JAX code usually does. A model using
# only these gets a kernel with 21 branches instead of all of ``kinds()``: it compiles in a
# fraction of the time.
CORE = (("mul", None, False, True, False, 2), ("add", None, False, False, False, 2),
        ("sub", None, False, False, True, 2), ("mul", None, False, False, False, 2),
        ("add", None, False, True, False, 2), ("neg", None, False, False, False, 1),
        ("integer_pow", 2, False, False, False, 1), ("sub", None, False, True, False, 2),
        ("sub", None, False, False, False, 2), ("add", None, False, False, True, 2),
        ("mul", None, True, False, False, 2), ("div", None, False, False, False, 2),
        ("log", None, False, False, False, 1), ("div", None, False, True, False, 2),
        ("pow", None, False, False, True, 2), ("integer_pow", 3, False, False, False, 1),
        ("exp", None, False, False, False, 1), ("integer_pow", 4, False, False, False, 1),
        ("sqrt", None, False, False, False, 1), ("mul", None, False, False, True, 2),
        ("div", None, False, False, True, 2))  # fmt: skip


@functools.lru_cache(maxsize=None)
def table(tier):
    """The opcodes of kernel ``tier``: "core" (``CORE``) or "full" (``kinds()``)."""
    full = kinds()
    if tier == "full":
        return full
    index = _kind_index()
    return tuple(full[index[k]] for k in CORE if k in index)


def _kind_of(opcode):
    """Index in ``kinds()`` of a ``scalarize`` opcode ``(name, params, same, ca, cb, n_in, eqn)``, or None."""
    name, _, same, ca, cb, n_in, eqn = opcode
    name = "add" if name == "add_any" else name
    y = None
    if name == "integer_pow":
        y = int(eqn.params["y"])
    elif name in _BINARY or name in _UNARY or name == "one_minus_square":
        canon = _canonical_eqns().get((name, None))
        if canon is None or repr(sorted(eqn.params.items())) != repr(sorted(canon.params.items())):
            return None  # parameters the canonical equation does not have
    if n_in == 1:
        same, ca, cb = False, False, False
    return _kind_index().get((name, y, bool(same), bool(ca), bool(cb), n_in))


def _bucket(v, least=8):
    """The power of two at least ``max(v, least)``."""
    return int(max(least, 1 << int(np.ceil(np.log2(max(int(v), 1))))))


class Data(NamedTuple):
    """A program as kernel arguments (all padded to their bucket)."""

    op: np.ndarray  # (K,) universal opcode per row
    a: np.ndarray  # (K,) operand slots
    b: np.ndarray
    out: np.ndarray  # (K,) result slot
    rows: np.ndarray  # () number of real rows
    cval: np.ndarray  # (S,) constant value (0 elsewhere)
    is_c: np.ndarray  # (S,) constant slot (padding slots count as constant 0)
    out_slot: np.ndarray  # (M,) output slots
    out_kind: np.ndarray  # (M,) 0 objective, 1 inequality, 2 equality, -1 padding
    integer: np.ndarray  # (N,) integer variable


jax.export.register_namedtuple_serialization(Data, serialized_name="jaxglobal.fbbt_generic.Data")


class Generic(NamedTuple):
    data: Data
    tier: str  # the opcode table: "core" or "full"
    n: int  # real variables (the first n of the N in the bucket)
    N: int


def encode(prog: Program, n, ineq, eq, integer=()):
    """``prog`` (``fbbt_interp.scalarize``, live rows) as a ``Generic`` program, or None when one of
    its opcodes is not in the universal table."""
    kind = [_kind_of(oc) for oc in prog.opcodes]
    if any(k is None for k in kind):
        return None
    full = kinds()
    core = {k[:6]: i for i, k in enumerate(table("core"))}
    tier = "core" if all(full[k][:6] in core for k in kind) else "full"
    if tier == "core":
        kind = [core[full[k][:6]] for k in kind]
    K, S0 = len(prog.op), len(prog.const_val)
    N = _bucket(n, least=8)
    # slots: x (n), padding variables (N - n), then the program's other slots shifted by N - n
    shift = N - n

    def renum(s):
        s = np.asarray(s, np.int64)
        return np.where(s < n, s, s + shift)

    Kb = _bucket(K, least=32)
    S = 2 * Kb if S0 + shift <= 2 * Kb else _bucket(S0 + shift)  # coarse buckets share more kernels
    cval = np.zeros(S)
    is_c = np.ones(S, bool)  # padding slots: constant 0
    is_c[:N] = False
    cv = np.asarray(prog.const_val, np.float64)
    other = np.arange(n, S0)
    is_c[other + shift] = ~np.isnan(cv[other])
    cval[other + shift] = np.nan_to_num(cv[other], nan=0.0)
    op = np.zeros(Kb, np.int32)
    op[:K] = np.asarray(kind, np.int32)[prog.op]
    a, b, out = (np.zeros(Kb, np.int32) for _ in range(3))
    a[:K], b[:K], out[:K] = renum(prog.a), renum(prog.b), renum(prog.out)
    groups = [0] + ([1] if ineq is not None else []) + ([2] if eq is not None else [])
    slots, kinds_ = [], []
    for g, o in zip(groups, prog.outs):
        o = renum(np.ravel(o))
        slots.append(o)
        kinds_.append(np.full(o.size, g))
    slots, kinds_ = np.concatenate(slots), np.concatenate(kinds_)
    M = _bucket(slots.size, least=8)
    out_slot = np.zeros(M, np.int32)
    out_kind = np.full(M, -1, np.int32)
    out_slot[: slots.size], out_kind[: slots.size] = slots, kinds_
    imask = np.zeros(N, bool)
    imask[np.asarray(integer, np.int64)] = True
    data = Data(op, a, b, out, np.asarray(K, np.int32), cval, is_c, out_slot, out_kind, imask)
    return Generic(data, tier, n, N)


@functools.lru_cache(maxsize=None)
def _rules(safe, tier="full"):
    """The branches of ``table(tier)`` (``fbbt_interp._branches``) and each opcode's domain floor."""
    t = table(tier)
    stand = Program(None, None, None, None, None, 0, [(k[0], None) + k[2:] for k in t], [])
    fwd, bwd = _branches(stand, safe)
    floor = np.array([DOMAIN_FLOOR.get(k[0], -np.inf) for k in t], np.float64)
    return fwd, bwd, floor


def _fbbt_one(lo, hi, cutoff, feas_tol, rounds, d: Data, safe, tier):
    """One box: ``fbbt_interp._propagate_rows``'s sweeps on the data ``d``. ``lo``, ``hi``: (N,)."""
    fwd, bwd, floor = _rules(safe, tier)
    FLOOR = jnp.asarray(floor)
    N = lo.shape[0]
    cval, is_c = jnp.asarray(d.cval), jnp.asarray(d.is_c)

    def forward(P):
        def step(k, carry):
            L, H, empty = carry
            a, b, o = d.a[k], d.b[k], d.out[k]
            vl, vh = lax.switch(d.op[k], fwd, L[a], H[a], L[b], H[b])
            vl, vh, bad = _meet2((vl, vh), (P[0][o], P[1][o]))
            return L.at[o].set(vl), H.at[o].set(vh), empty | bad

        return lax.fori_loop(0, d.rows, step, (P[0], P[1], jnp.array(False)))

    def backward(L, H):
        def target(j, carry):
            L, H, empty = carry
            s, kind = d.out_slot[j], d.out_kind[j]
            zl = jnp.where(kind == 2, -feas_tol, -INF)
            zh = jnp.where(kind == 0, cutoff, feas_tol)
            ml, mh, bad = _meet2((L[s], H[s]), (zl, zh))
            const = is_c[s]  # a constant output that violates its bound makes the box empty
            real = kind >= 0
            e = jnp.where(const, (cval[s] < zl) | (cval[s] > zh), bad)
            keep = const | ~real
            return (L.at[s].set(jnp.where(keep, L[s], ml)), H.at[s].set(jnp.where(keep, H[s], mh)),
                    empty | (real & e))  # fmt: skip

        L, H, empty = lax.fori_loop(0, d.out_slot.shape[0], target, (L, H, jnp.array(False)))

        def update(L, H, s, cl, ch, use, fl=-INF):
            pl, ph = _pad(cl, ch)
            pl = jnp.maximum(pl, fl)
            ml, mh, bad = _meet2((L[s], H[s]), (pl, ph))
            use = use & ~is_c[s]
            return L.at[s].set(jnp.where(use, ml, L[s])), H.at[s].set(jnp.where(use, mh, H[s])), use & bad

        def step(j, carry):
            L, H, empty = carry
            k = d.rows - 1 - j
            a, b, o = d.a[k], d.b[k], d.out[k]
            al, ah, ua, bl, bh, ub = lax.switch(d.op[k], bwd, L[o], H[o], L[a], H[a], L[b], H[b])
            L, H, e1 = update(L, H, a, al, ah, ua, FLOOR[d.op[k]])
            L, H, e2 = update(L, H, b, bl, bh, ub)
            return L, H, empty | e1 | e2

        return lax.fori_loop(0, d.rows, step, (L, H, empty))

    imask = jnp.asarray(d.integer)

    def round_integers(L, H):
        xl, xh = L[:N], H[:N]
        il, ih = jnp.ceil(xl - INT_TOL), jnp.floor(xh + INT_TOL)
        bad = jnp.any(imask & (il > ih))
        return L.at[:N].set(jnp.where(imask, il, xl)), H.at[:N].set(jnp.where(imask, jnp.maximum(il, ih), xh)), bad

    L0 = jnp.where(is_c, cval, -INF).at[:N].set(lo)
    H0 = jnp.where(is_c, cval, INF).at[:N].set(hi)

    def round_(_, carry):
        P, empty = carry
        vl, vh, e1 = forward((P[:, 0], P[:, 1]))
        bl, bh, e2 = backward(vl, vh)
        bl, bh, e3 = round_integers(bl, bh)
        return jnp.stack([jnp.where(is_c, cval, bl), jnp.where(is_c, cval, bh)], axis=1), empty | e1 | e2 | e3

    # up to ``rounds`` rounds, stopping at an exact fixpoint (a round that changes no bound): every
    # later round would return the same bounds, so the result is that of all ``rounds`` rounds.
    # (``fbbt_interp._rounds`` also stops once rounds move bounds only a little, which leaves boxes
    # looser; here a call with 30 rounds, ``domain.bound_domain``'s, must give its full result.)
    def cond(c):
        k, _, empty, moved = c
        return (k < rounds) & moved & ~empty

    def body(c):
        k, P, empty, _ = c
        Q, e = round_(k, (P, empty))
        same = (Q == P) | (jnp.isnan(Q) & jnp.isnan(P))
        return k + 1, Q, e, ~jnp.all(same)

    _, P, empty, _ = lax.while_loop(cond, body, (jnp.array(0), jnp.stack([L0, H0], axis=1), jnp.array(False),
                                                 jnp.array(True)))  # fmt: skip
    vl, vh, e = forward((P[:, 0], P[:, 1]))
    xl, xh = vl[:N], vh[:N]
    return jnp.minimum(xl, xh), jnp.maximum(xl, xh), empty | e


@functools.lru_cache(maxsize=None)
def kernel(safe=False, tier="full"):
    """The batched kernel ``(lo (B, N), hi (B, N), cutoff, feas_tol, rounds, data) -> (lo, hi, empty)``.

    One jitted function per ``safe``; JAX compiles it once per argument shape (bucket), and the
    executable depends on nothing else."""

    def run(lo, hi, cutoff, feas_tol, rounds, data):
        one = functools.partial(_fbbt_one, safe=safe, tier=tier)
        return jax.vmap(one, in_axes=(0, 0, None, None, None, None))(lo, hi, cutoff, feas_tol, rounds, data)

    return jax.jit(run)


_EXES = {}


def executable(safe, tier, args):
    """The compiled kernel for the shapes of ``args`` (``kernel``'s arguments): from this process,
    else the exported kernel in the disk cache (``kcache``: no tracing; XLA's persistent cache then
    has the compiled code), else traced, stored and compiled."""
    from . import kcache

    specs = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(jnp.shape(a), jnp.result_type(a)), args)
    key = (bool(safe), tier, tuple((s.shape, str(s.dtype)) for s in jax.tree_util.tree_leaves(specs)))
    if key in _EXES:
        return _EXES[key]
    fn = kernel(safe, tier)
    lowered = None
    if kcache.cache_root() is not None:
        import hashlib

        import jaxlib

        dev = jax.devices()[0]
        fp = "fbbt_generic-" + hashlib.sha256(repr((kcache._source_hash(), jax.__version__, jaxlib.__version__,
                                                    dev.platform, dev.device_kind, key)).encode()).hexdigest()[:32]  # fmt: skip
        exp = kcache.load(fp)
        if exp is None:
            try:
                exp = jax.export.export(fn)(*specs)
            except Exception:  # not exportable: trace below, uncached
                exp = None
            kcache.save(fp, exp)
        if exp:
            lowered = jax.jit(exp.call).trace(*specs).lower()
    if lowered is None:
        lowered = fn.trace(*specs).lower()
    _EXES[key] = lowered.compile()
    return _EXES[key]


class GenericFBBT:
    """``fbbt(lo, hi, cutoff) -> (lo, hi, empty)`` on batches ``(B, n)`` through the shared kernel."""

    def __init__(self, g: Generic, feas_tol, rounds, safe=False):
        self.g, self.safe = g, bool(safe)
        self.feas_tol, self.rounds = jnp.asarray(feas_tol, jnp.float64), jnp.asarray(rounds, jnp.int32)
        self.data = Data(*(jnp.asarray(v) for v in g.data))

    def pad(self, lo, hi):
        """``lo``, ``hi`` (B, n) padded to the bucket's N variables (padding variables fixed at 0)."""
        z = jnp.zeros(lo.shape[:-1] + (self.g.N - self.g.n,), jnp.float64)
        return jnp.concatenate([jnp.asarray(lo, jnp.float64), z], -1), jnp.concatenate([jnp.asarray(hi, jnp.float64), z], -1)

    def __call__(self, lo, hi, cutoff=INF):
        Lp, Hp = self.pad(lo, hi)
        args = (Lp, Hp, jnp.asarray(cutoff, jnp.float64), self.feas_tol, self.rounds, self.data)
        xl, xh, empty = executable(self.safe, self.g.tier, args)(*args)
        return xl[..., : self.g.n], xh[..., : self.g.n], empty


def make_generic_fbbt(f, ineq=None, eq=None, n=None, rounds=5, feas_tol=1e-6, safe=False, integer=()):
    """The shared-kernel FBBT for the problem, or None if its tape cannot be scalarized or uses an
    opcode outside the universal table."""
    from .fbbt import _has_tracer, joint_tape

    with jax.ensure_compile_time_eval():
        tape = joint_tape(f, ineq, eq, np.zeros(n))
    if _has_tracer(tape):
        return None
    prog = scalarize(tape, n)
    if prog is None:
        return None
    prog = _live_rows(prog)
    if len(prog.op) == 0:
        return None
    g = encode(prog, n, ineq, eq, integer)
    if g is None:
        return None
    return GenericFBBT(g, feas_tol, rounds, safe)

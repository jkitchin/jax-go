"""Export a JAX-defined problem to SCIP (PySCIPOpt) or to an AMPL ``.nl`` file.

``to_scip(f, lo, hi, ineq=..., eq=..., integer=...)`` builds a ``pyscipopt.Model``;
``to_nl(f, lo, hi, path, ...)`` writes a text ``.nl`` file that Couenne, BARON,
SCIP's AMPL reader and ``jaxglobal.nl.read_nl`` can read. Both take the problem
in the form ``minimize`` takes it: minimize ``f(x)`` over ``lo <= x <= hi``,
``ineq(x) <= 0``, ``eq(x) == 0``, with ``integer`` a boolean mask or indices.
``inspect(...)`` returns the export report without building anything.

How. The functions are traced to a jaxpr, and the jaxpr is evaluated
symbolically, element by element, into a hash-consed expression DAG (``Graph``):
each array becomes a numpy object array of node ids, constants are folded, and
structural equations (slice, reshape, broadcast, concatenate, gather and
dynamic_slice with constant indices, ...) only renumber elements, by binding the
primitive on integer index arrays as ``fbbt_interp.scalarize`` does. The DAG is
then lowered for the target (exact SCIP or ``.nl`` operators, exact rewrites,
reformulations with auxiliary variables) and written out.

Why not ``scalarize`` itself: its rows cover the scalar tapes of ``.nl`` models,
but it returns None on ``dot_general``, reductions and ``select_n``, and the tape
it reads (``fbbt.make_tape``) inlines a ``scan`` body once, as if it were a
call. The walker here handles those, unrolls ``scan`` / ``fori_loop`` and
while loops whose condition is a constant, and inlines ``jit`` and
``custom_jvp`` / ``custom_vjp`` calls.

Each operation is translated exactly, rewritten into an identity
(``tanh(x) = 1 - 2 / (exp(2x) + 1)``), reformulated with auxiliary variables
(``jnp.where``, comparisons and ``sign`` with binaries and big-M constants from
interval bounds over the box), or refused with ``ExportError`` naming the
primitive. The ``ExportReport`` (``model.data.report`` for SCIP, the return
value of ``to_nl``) lists which operations took which route, the assumptions
made (``isnan`` / ``isfinite`` of a model value are taken as False / True: a
solver only returns finite points), and what is not exact (``approximated``):

- A reformulated comparison cannot be strict. With ``strict_eps=0`` (the
  default), at a tie ``u == v`` either branch is allowed, so the model is a
  relaxation where the branches of a ``jnp.where`` (or the values of ``sign``)
  differ at the tie: a piecewise-constant function can then reach a value at a
  breakpoint that JAX never produces. ``strict_eps > 0`` makes the comparison
  strict by excluding ``0 < |u - v| < strict_eps`` on the wrong side, a
  restriction instead. Either way, evaluate ``f`` with JAX at the returned
  ``x``.
- ``reduce_precision`` and float64 -> float32 casts are exported as identities
  (the solvers compute in double precision).

Refused rather than exported wrongly: a ``jnp.where`` branch (or comparison
operand) that is undefined on part of the box, such as ``sqrt(x)`` with
``x < 0`` possible, because the reformulation evaluates both branches
everywhere and would cut that part of the box off (write
``jnp.sqrt(jnp.maximum(x, 0))``); big-M constants of 1e20 or more (SCIP's
infinity; above 1e8 they are reported and warned about, ``ExportBigMWarning``);
division by the constant 0; an exported intermediate whose interval bound
reaches 1e20 over the box while its operands are bounded (SCIP would treat
those points as invalid evaluations and cut them off). The ``tanh``,
``sigmoid``, ``asinh`` and ``acosh`` rewrites choose overflow-free forms from
the argument's interval so they do not hit this.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jcore

__all__ = ["ExportBigMWarning", "ExportError", "ExportReport", "ExportSizeWarning", "inspect", "solution", "to_nl",
           "to_scip"]  # fmt: skip


class ExportError(ValueError):
    """The problem uses an operation the target cannot represent exactly."""


class ExportSizeWarning(UserWarning):
    """The exported model is larger than ``max_size`` scalar operations."""


class ExportBigMWarning(UserWarning):
    """A big-M constant above ``BIGM_WARN``: the reformulation is numerically weak."""


BIGM_WARN = 1e8  # big-M constants above this are reported and warned about
BIGM_REFUSE = 1e20  # ... and refused from here on (SCIP treats 1e20 as infinity)
OVERFLOW_OPS = {"exp", "mul", "add", "sub", "sum", "neg", "abs", "sinh", "cosh"}  # finite operands, value can overflow
EXP_SAFE = 30.0  # rewrites keep exp arguments below this where they can (exp(30) ~ 1e13)
MAX_TERMS = 1000  # SCIP: lift a product operand to a variable before pyscipopt expands it past this


# ---------------------------------------------------------------------------
# Expression DAG
# ---------------------------------------------------------------------------

LOGICAL = {"lt", "le", "gt", "ge", "eq", "ne", "and", "or", "not"}  # 0/1-valued nodes


def _sel(p, f, t):
    return np.where(p != 0, t, f)


EVAL = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "neg": np.negative,
    "pow": np.power, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "sin": np.sin,
    "cos": np.cos, "tan": np.tan, "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh, "atan": np.arctan,
    "asin": np.arcsin, "acos": np.arccos, "atanh": np.arctanh, "asinh": np.arcsinh, "acosh": np.arccosh,
    "atan2": np.arctan2, "logistic": lambda a: 1.0 / (1.0 + np.exp(-a)), "log1p": np.log1p,
    "expm1": np.expm1, "max": np.maximum, "min": np.minimum, "sign": np.sign,
    "lt": lambda a, b: np.less(a, b) * 1.0, "le": lambda a, b: np.less_equal(a, b) * 1.0,
    "gt": lambda a, b: np.greater(a, b) * 1.0, "ge": lambda a, b: np.greater_equal(a, b) * 1.0,
    "eq": lambda a, b: np.equal(a, b) * 1.0, "ne": lambda a, b: np.not_equal(a, b) * 1.0,
    "and": lambda a, b: ((a != 0) & (b != 0)) * 1.0, "or": lambda a, b: ((a != 0) | (b != 0)) * 1.0,
    "not": lambda a: (a == 0) * 1.0, "select": _sel,
}  # fmt: skip


class Graph:
    """Hash-consed DAG; node ``i`` is ``(op, args, param)`` with ``args`` ids below ``i``.

    ``op`` is ``"var"`` (param: index), ``"const"`` (param: value), an operator of
    ``EVAL``, ``"sum"`` (n-ary) or ``"unsupported"`` (param: message).
    ``select(p, f, t)`` is ``t`` where ``p`` else ``f`` (``lax.select_n`` order).
    """

    def __init__(self):
        self.nodes = []
        self._key = {}
        self.assumptions = set()
        self.notes = Counter()
        self.approximated = {}  # op -> [count, how]
        self.partial = set()  # nodes whose expression is undefined somewhere (sqrt, log, div, ...)

    def _add(self, op, args=(), param=None):
        k = (op, tuple(args), param)
        i = self._key.get(k)
        if i is None:
            i = len(self.nodes)
            self.nodes.append(k)
            self._key[k] = i
            if _partial_op(self, op, args) or any(a in self.partial for a in args):
                self.partial.add(i)
        return i

    def const(self, v):
        return self._add("const", (), float(v))

    def var(self, j):
        return self._add("var", (), int(j))

    def unsupported(self, msg):
        return self._add("unsupported", (), msg)

    def is_const(self, i, v=None):
        op, _, p = self.nodes[i]
        return op == "const" and (v is None or p == v)

    def value(self, i):
        return self.nodes[i][2]

    def sum(self, terms):
        c, rest = 0.0, []
        for t in terms:
            if self.is_const(t):
                c += self.value(t)
            else:
                rest.append(t)
        if c != 0.0 or not rest:
            rest.append(self.const(c))
        return rest[0] if len(rest) == 1 else self._add("sum", tuple(rest))

    def op(self, name, *args):
        """Node ``name(*args)``, constant-folded and lightly simplified."""
        c = [self.value(a) if self.is_const(a) else None for a in args]
        if all(v is not None for v in c) and name in EVAL:
            with np.errstate(all="ignore"):
                return self.const(EVAL[name](*[np.float64(v) for v in c]))
        a = args[0]
        b = args[1] if len(args) > 1 else None
        if name == "add":
            if c[0] == 0.0:
                return b
            if c[1] == 0.0:
                return a
        elif name == "sub":
            if c[1] == 0.0:
                return a
            if c[0] == 0.0:
                return self.op("neg", b)
            if a == b:
                return self.const(0.0)
        elif name == "mul":
            if c[0] == 1.0:
                return b
            if c[1] == 1.0:
                return a
            # 0 * a = 0 only where a is defined (JAX: 0 * nan = nan)
            if (c[0] == 0.0 and b not in self.partial) or (c[1] == 0.0 and a not in self.partial):
                return self.const(0.0)
            if c[0] == -1.0:
                return self.op("neg", b)
            if c[1] == -1.0:
                return self.op("neg", a)
        elif name == "div":
            if c[1] == 1.0:
                return a
        elif name == "neg":
            if self.nodes[a][0] == "neg":
                return self.nodes[a][1][0]
        elif name == "pow":
            if c[1] == 1.0:
                return a
            if c[1] == 0.0:
                return self.const(1.0)
        elif name in ("max", "min"):
            if a == b:
                return a
            ident = -np.inf if name == "max" else np.inf
            if c[0] == ident:
                return b
            if c[1] == ident:
                return a
        elif name in ("eq", "le", "ge") and a == b:
            self.assumptions.add("x == x taken as True (isnan of a model value is False)")
            return self.const(1.0)
        elif name in ("ne", "lt", "gt") and a == b:
            self.assumptions.add("x != x taken as False (isnan of a model value is False)")
            return self.const(0.0)
        elif name == "and":
            if c[0] is not None:
                return b if c[0] else self.const(0.0)
            if c[1] is not None:
                return a if c[1] else self.const(0.0)
        elif name == "or":
            if c[0] is not None:
                return self.const(1.0) if c[0] else b
            if c[1] is not None:
                return self.const(1.0) if c[1] else a
        elif name == "not":
            if self.nodes[a][0] == "not":
                return self.nodes[a][1][0]
        elif name == "select":
            if c[0] is not None:
                return args[2] if c[0] else args[1]
            if args[1] == args[2]:
                return args[1]
        return self._add(name, args)


PARTIAL = {"sqrt", "log", "log1p", "div", "acosh", "asin", "acos", "atanh", "tan", "unsupported"}


def _partial_op(g, op, args):
    """Whether ``op(*args)`` itself is undefined for some real arguments."""
    if op == "pow":
        if not g.is_const(args[1]):
            return True
        k = g.value(args[1])
        return not (float(k).is_integer() and k >= 0)
    return op in PARTIAL


def _reachable(g, roots):
    """Ids of the nodes the ``roots`` depend on, in increasing (topological) order."""
    seen, stack = set(), list(roots)
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack.extend(g.nodes[i][1])
    return sorted(seen)


def _evaluate(g, roots, x):
    """Values of the ``roots`` at ``x`` (``(n,)`` or ``(B, n)``), numerically."""
    x = np.asarray(x, np.float64)
    val = {}
    with np.errstate(all="ignore"):
        for i in _reachable(g, roots):
            op, args, p = g.nodes[i]
            if op == "var":
                val[i] = x[..., p]
            elif op == "const":
                val[i] = np.float64(p)
            elif op == "sum":
                val[i] = sum(val[a] for a in args)
            elif op == "unsupported":
                raise ExportError(p)
            else:
                val[i] = EVAL[op](*[val[a] for a in args])
    return [val[r] for r in roots]


# ---------------------------------------------------------------------------
# Interval bounds over the box (for big-M constants and auxiliary-variable bounds)
# ---------------------------------------------------------------------------


def _imul(al, ah, bl, bh):
    p = np.nan_to_num(np.array([al * bl, al * bh, ah * bl, ah * bh]), nan=0.0, posinf=np.inf, neginf=-np.inf)
    return p.min(), p.max()


def _ipow(al, ah, p):
    if float(p).is_integer():
        k = int(p)
        if k == 0:
            return 1.0, 1.0
        if k < 0:
            ql, qh = _ipow(al, ah, -k)
            if ql <= 0 <= qh:
                return (-np.inf, np.inf) if k % 2 else (1.0 / qh if qh > 0 else np.inf, np.inf)
            return min(1.0 / ql, 1.0 / qh), max(1.0 / ql, 1.0 / qh)
        if k % 2:
            return al**k, ah**k
        if al >= 0:
            return al**k, ah**k
        if ah <= 0:
            return ah**k, al**k
        return 0.0, max(al**k, ah**k)
    al = max(al, 0.0)
    if ah < al:
        return -np.inf, np.inf
    return (al**p, ah**p) if p > 0 else (ah**p, al**p if al > 0 else np.inf)


def _itrig(al, ah, shift):
    """Bounds of ``sin(x + shift)`` on [al, ah]."""
    if not (np.isfinite(al) and np.isfinite(ah)) or ah - al >= 2 * np.pi:
        return -1.0, 1.0
    lo, hi = sorted((np.sin(al + shift), np.sin(ah + shift)))
    k0 = math.ceil((al + shift - np.pi / 2) / (2 * np.pi))
    if np.pi / 2 + 2 * np.pi * k0 <= ah + shift:
        hi = 1.0
    k1 = math.ceil((al + shift + np.pi / 2) / (2 * np.pi))
    if -np.pi / 2 + 2 * np.pi * k1 <= ah + shift:
        lo = -1.0
    return lo, hi


def _mono(fn, lo_dom=-np.inf, hi_dom=np.inf):
    def iv(al, ah):
        al, ah = max(al, lo_dom), min(ah, hi_dom)
        if ah < al:
            return -np.inf, np.inf
        return fn(al), fn(ah)

    return iv


_IV1 = {
    "neg": lambda al, ah: (-ah, -al), "exp": _mono(np.exp), "log": _mono(np.log, 0.0),
    "sqrt": _mono(np.sqrt, 0.0), "tanh": _mono(np.tanh), "sinh": _mono(np.sinh), "atan": _mono(np.arctan),
    "asinh": _mono(np.arcsinh), "log1p": _mono(np.log1p, -1.0), "expm1": _mono(np.expm1),
    "logistic": _mono(EVAL["logistic"]), "asin": _mono(np.arcsin, -1.0, 1.0),
    "acos": lambda al, ah: tuple(reversed(_mono(np.arccos, -1.0, 1.0)(al, ah))),
    "atanh": _mono(np.arctanh, -1.0, 1.0), "acosh": _mono(np.arccosh, 1.0),
    "sin": lambda al, ah: _itrig(al, ah, 0.0), "cos": lambda al, ah: _itrig(al, ah, np.pi / 2),
    "sign": lambda al, ah: (np.sign(al), np.sign(ah)),
    "abs": lambda al, ah: (al, ah) if al >= 0 else (-ah, -al) if ah <= 0 else (0.0, max(-al, ah)),
    "cosh": lambda al, ah: (np.cosh(min(max(0.0, al), ah)), max(np.cosh(al), np.cosh(ah))),
    "not": lambda al, ah: (0.0, 1.0),
}  # fmt: skip


def _itan(al, ah):
    if not (np.isfinite(al) and np.isfinite(ah)):
        return -np.inf, np.inf
    k = math.ceil((al - np.pi / 2) / np.pi)
    if np.pi / 2 + k * np.pi <= ah:
        return -np.inf, np.inf
    return np.tan(al), np.tan(ah)


def _idiv(al, ah, bl, bh):
    if bl <= 0 <= bh:
        return (0.0, 0.0) if al == ah == 0 else (-np.inf, np.inf)
    return _imul(al, ah, 1.0 / bh, 1.0 / bl)


def _intervals(g, order, lo, hi, fixed=None):
    """``{node: (lo, hi)}`` over the box ``[lo, hi]`` (plain float arithmetic: pad before use). ``fixed`` gives
    known enclosures for some nodes."""
    iv = {}
    fixed = fixed or {}
    with np.errstate(all="ignore"):
        for i in order:
            op, args, p = g.nodes[i]
            a = [iv[k] for k in args]
            if i in fixed:
                r = fixed[i]
            elif op == "var":
                r = (lo[p], hi[p])
            elif op == "const":
                r = (p, p)
            elif op in ("add", "sum"):
                r = (sum(t[0] for t in a), sum(t[1] for t in a))
            elif op == "sub":
                r = (a[0][0] - a[1][1], a[0][1] - a[1][0])
            elif op == "mul":
                r = _imul(*a[0], *a[1])
            elif op == "div":
                r = _idiv(*a[0], *a[1])
            elif op == "pow":
                if g.is_const(args[1]):
                    r = _ipow(*a[0], p=g.value(args[1]))
                else:  # x**y = exp(y log x)
                    lg = _mono(np.log, 0.0)(*a[0])
                    r = _mono(np.exp)(*_imul(*a[1], *lg))
            elif op == "tan":
                r = _itan(*a[0])
            elif op in _IV1:
                r = _IV1[op](*a[0])
            elif op == "max":
                r = (max(a[0][0], a[1][0]), max(a[0][1], a[1][1]))
            elif op == "min":
                r = (min(a[0][0], a[1][0]), min(a[0][1], a[1][1]))
            elif op == "select":
                pl, ph = a[0]
                f, t = a[1], a[2]
                r = t if pl > 0 else f if ph <= 0 else (min(f[0], t[0]), max(f[1], t[1]))
            elif op in LOGICAL:
                r = (0.0, 1.0)
            elif op == "atan2":
                r = (-np.pi, np.pi)
            else:
                r = (-np.inf, np.inf)
            rl, rh = float(r[0]), float(r[1])
            iv[i] = (-np.inf if np.isnan(rl) else rl, np.inf if np.isnan(rh) else rh)
    return iv


def _pad(lo, hi, rel=1e-6, abs_=1e-9):
    """[lo, hi] widened outward (the interval arithmetic above does not round outward)."""
    return lo - rel * abs(lo) - abs_, hi + rel * abs(hi) + abs_


# ---------------------------------------------------------------------------
# Tracing: jaxpr -> Graph
# ---------------------------------------------------------------------------

CALLS = {"pjit", "jit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call",
         "custom_vjp_call_jaxpr", "remat", "remat2", "checkpoint", "custom_lin"}  # fmt: skip
STRUCTURAL = {"slice", "squeeze", "reshape", "broadcast_in_dim", "concatenate", "expand_dims", "transpose",
              "rev", "pad", "gather", "stack", "unstack", "dynamic_slice", "dynamic_update_slice", "select_n", "scatter", "split",
              "copy", "copy_p"}  # fmt: skip
# positions of the operands that are data (moved around); the others are indices / predicates
DATA = {"gather": (0,), "dynamic_slice": (0,), "dynamic_update_slice": (0, 1), "scatter": (0, 2)}
IDENTITY = {"stop_gradient", "copy", "copy_p", "optimization_barrier"}
UNARY = {"neg", "exp", "log", "sqrt", "abs", "sign", "sin", "cos", "tan", "tanh", "logistic", "log1p", "expm1",
         "sinh", "cosh", "atan", "asin", "acos", "atanh", "asinh", "acosh"}  # fmt: skip
BINARY = {"add": "add", "add_any": "add", "sub": "sub", "mul": "mul", "div": "div", "max": "max", "min": "min",
          "pow": "pow", "atan2": "atan2", "lt": "lt", "le": "le", "gt": "gt", "ge": "ge", "eq": "eq", "ne": "ne"}  # fmt: skip
MAX_WHILE = 100_000


def _is_sym(v):
    return isinstance(v, np.ndarray) and v.dtype == object


class _Walker:
    def __init__(self, g):
        self.g = g

    def sym(self, v):
        """``v`` as an object array of node ids (constants become const nodes)."""
        if _is_sym(v):
            return v
        a = np.asarray(v)
        out = np.empty(a.shape, object)
        for idx in np.ndindex(a.shape):
            out[idx] = self.g.const(a[idx])
        return out

    def unsupported(self, eqn, msg):
        node = self.g.unsupported(msg)
        return [np.full(v.aval.shape, node, dtype=object) for v in eqn.outvars]

    def call(self, closed, args):
        jaxpr = getattr(closed, "jaxpr", closed)
        consts = getattr(closed, "consts", ())
        env = {}

        def read(v):
            return np.asarray(v.val, dtype=v.aval.dtype) if isinstance(v, jcore.Literal) else env[v]

        for v, c in zip(jaxpr.constvars, consts):
            env[v] = np.asarray(c)
        for v, a in zip(jaxpr.invars, args):
            env[v] = a
        for eqn in jaxpr.eqns:
            outs = self.eqn(eqn, [read(v) for v in eqn.invars])
            for v, o in zip(eqn.outvars, outs):
                env[v] = o
        return [read(v) for v in jaxpr.outvars]

    def eqn(self, eqn, ins):
        name, p = eqn.primitive.name, eqn.params
        if name in CALLS:
            sub = next((p[k] for k in ("jaxpr", "call_jaxpr", "fun_jaxpr") if k in p), None)
            if sub is None:
                return self.unsupported(eqn, f"{name}: no inner jaxpr")
            return self.call(sub, ins)
        if name == "scan":
            return self.scan(eqn, ins)
        if name == "while":
            return self.while_(eqn, ins)
        if name == "cond":
            if _is_sym(ins[0]):
                return self.unsupported(eqn, "cond (lax.cond) with a variable-dependent predicate: use jnp.where")
            branches = p["branches"]
            k = int(np.clip(int(np.asarray(ins[0])), 0, len(branches) - 1))
            return self.call(branches[k], ins[1:])
        if not any(_is_sym(v) for v in ins):  # constant: evaluate
            try:
                with jax.ensure_compile_time_eval():
                    out = eqn.primitive.bind(*[jnp.asarray(v, v_.aval.dtype) for v, v_ in zip(ins, eqn.invars)], **p)
            except Exception as e:  # noqa: BLE001
                return self.unsupported(eqn, f"{name}: constant evaluation failed ({e})")
            return [np.asarray(o) for o in out] if eqn.primitive.multiple_results else [np.asarray(out)]
        if name in IDENTITY:
            return list(ins)
        if name == "reduce_precision":
            how = "exported as the identity (the solvers compute in double precision)"
            self.g.approximated.setdefault("reduce_precision", [0, how])[0] += 1
            return list(ins)
        if name in STRUCTURAL and not (name == "select_n" and _is_sym(ins[0])):
            return self.structural(eqn, ins)
        if name == "convert_element_type":
            src, dst = eqn.invars[0].aval.dtype, p["new_dtype"]
            if jnp.issubdtype(dst, jnp.floating) or not jnp.issubdtype(src, jnp.floating):
                if jnp.issubdtype(src, jnp.floating) and np.dtype(dst).itemsize < np.dtype(src).itemsize:
                    key = f"convert_element_type {np.dtype(src).name} -> {np.dtype(dst).name}"
                    how = "exported as the identity (the solvers compute in double precision)"
                    self.g.approximated.setdefault(key, [0, how])[0] += 1
                return [ins[0]]
            return self.unsupported(eqn, f"convert_element_type from {src} to {dst} (rounds a model value)")
        g = self.g
        if name in UNARY:
            return self.elementwise(eqn, ins, lambda a: g.op(name, a))
        if name in BINARY:
            return self.elementwise(eqn, ins, lambda a, b: g.op(BINARY[name], a, b))
        if name == "integer_pow":
            k = g.const(p["y"])
            return self.elementwise(eqn, ins, lambda a: g.op("pow", a, k))
        if name == "square":
            return self.elementwise(eqn, ins, lambda a: g.op("pow", a, g.const(2)))
        if name == "rsqrt":
            return self.elementwise(eqn, ins, lambda a: g.op("pow", a, g.const(-0.5)))
        if name == "exp2":
            return self.elementwise(eqn, ins, lambda a: g.op("exp", g.op("mul", g.const(np.log(2.0)), a)))
        if name == "clamp":  # clamp(lo, x, hi)
            return self.elementwise(eqn, ins, lambda lo, x, hi: g.op("min", g.op("max", x, lo), hi))
        if name in ("and", "or", "not"):
            if eqn.outvars[0].aval.dtype != bool:
                return self.unsupported(eqn, f"bitwise {name} on integers")
            return self.elementwise(eqn, ins, lambda *a: g.op(name, *a))
        if name == "is_finite":
            g.assumptions.add("isfinite(x) taken as True (a solver returns finite points)")
            return [np.full(eqn.outvars[0].aval.shape, g.const(1.0), dtype=object)]
        if name == "select_n":  # symbolic predicate (constant predicates are structural)
            if len(ins) != 3:
                return self.unsupported(eqn, "select_n with more than two cases and a variable-dependent index")
            return self.elementwise(eqn, ins, lambda c, f, t: g.op("select", c, f, t))
        if name == "dot_general":
            return [self.dot_general(eqn, ins)]
        if name in ("reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "reduce_and", "reduce_or"):
            return [self.reduce(eqn, ins[0])]
        if name in ("cumsum", "cumprod", "cummax", "cummin"):
            return [self.cumulative(eqn, ins[0])]
        return self.unsupported(eqn, f"primitive '{name}' is not supported")

    def elementwise(self, eqn, ins, fn):
        shape = eqn.outvars[0].aval.shape
        args = [np.broadcast_to(self.sym(a), shape) for a in ins]
        out = np.empty(shape, object)
        for idx in np.ndindex(shape):
            out[idx] = fn(*(a[idx] for a in args))
        return [out]

    def structural(self, eqn, ins):
        name = eqn.primitive.name
        data = DATA.get(name, tuple(range(1, len(ins))) if name == "select_n" else tuple(range(len(ins))))
        for i, v in enumerate(ins):
            if i not in data and _is_sym(v):
                return self.unsupported(eqn, f"{name} with variable-dependent indices")
        table, args = [], list(ins)
        for i in data:
            s = self.sym(ins[i])
            args[i] = np.arange(len(table), len(table) + s.size, dtype=np.int64).reshape(s.shape)
            table.extend(s.ravel().tolist())
        try:
            with jax.ensure_compile_time_eval():
                out = eqn.primitive.bind(*[jnp.asarray(a) for a in args], **eqn.params)
        except Exception as e:  # noqa: BLE001
            return self.unsupported(eqn, f"{name}: could not renumber elements ({e})")
        outs = out if eqn.primitive.multiple_results else [out]
        table = np.array(table + [None], dtype=object)[:-1]
        res = []
        for o in outs:
            o = np.asarray(o)
            if o.size and (o.min() < 0 or o.max() >= len(table)):
                return self.unsupported(eqn, f"{name} reads out of bounds")
            res.append(table[o.reshape(-1)].reshape(o.shape))
        return res

    def dot_general(self, eqn, ins):
        g = self.g
        (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
        A, B = self.sym(ins[0]), self.sym(ins[1])
        lf = [d for d in range(A.ndim) if d not in lc and d not in lb]
        rf = [d for d in range(B.ndim) if d not in rc and d not in rb]
        nb = int(np.prod([A.shape[d] for d in lb], dtype=np.int64))
        K = int(np.prod([A.shape[d] for d in lc], dtype=np.int64))
        M = int(np.prod([A.shape[d] for d in lf], dtype=np.int64))
        N = int(np.prod([B.shape[d] for d in rf], dtype=np.int64))
        At = A.transpose(list(lb) + lf + list(lc)).reshape(nb, M, K)
        Bt = B.transpose(list(rb) + list(rc) + rf).reshape(nb, K, N)
        out = np.empty((nb, M, N), object)
        for b in range(nb):
            for m in range(M):
                for k in range(N):
                    out[b, m, k] = g.sum([g.op("mul", At[b, m, j], Bt[b, j, k]) for j in range(K)])
        return out.reshape(eqn.outvars[0].aval.shape)

    def reduce(self, eqn, a):
        g, name = self.g, eqn.primitive.name
        a = self.sym(a)
        axes = tuple(eqn.params["axes"])
        keep = [d for d in range(a.ndim) if d not in axes]
        r = a.transpose(keep + list(axes)).reshape([a.shape[d] for d in keep] + [-1])
        op = {"reduce_max": "max", "reduce_min": "min", "reduce_prod": "mul", "reduce_and": "and",
              "reduce_or": "or"}.get(name)  # fmt: skip
        out = np.empty(r.shape[:-1], object)
        for idx in np.ndindex(out.shape):
            terms = list(r[idx])
            if op is None:
                out[idx] = g.sum(terms)
            else:
                acc = terms[0]
                for t in terms[1:]:
                    acc = g.op(op, acc, t)
                out[idx] = acc
        return out

    def cumulative(self, eqn, a):
        g, name = self.g, eqn.primitive.name
        op = {"cumsum": "add", "cumprod": "mul", "cummax": "max", "cummin": "min"}[name]
        a = np.moveaxis(self.sym(a), eqn.params["axis"], 0)
        out = np.empty(a.shape, object)
        order = range(a.shape[0] - 1, -1, -1) if eqn.params.get("reverse") else range(a.shape[0])
        acc = None
        for t in order:
            acc = a[t] if acc is None else np.frompyfunc(lambda u, v: g.op(op, u, v), 2, 1)(acc, a[t])
            out[t] = acc
        return np.moveaxis(out, 0, eqn.params["axis"])

    def scan(self, eqn, ins):
        p = eqn.params
        L = p["length"]
        if "ft_in" in p:  # jax >= 0.10: flat-tree split of the operands and results
            consts, carry, xs = (list(a) for a in p["ft_in"].update(list(ins)).unpack())

            def split(out):
                c, y = p["ft_out"].update(list(out)).unpack()
                return list(c), list(y)
        else:
            nc, nk0 = p["num_consts"], p["num_carry"]
            consts, carry, xs = list(ins[:nc]), list(ins[nc : nc + nk0]), list(ins[nc + nk0 :])

            def split(out):
                return list(out[:nk0]), list(out[nk0:])

        nk = len(carry)
        ny = len(eqn.outvars) - nk
        ys = [[None] * L for _ in range(ny)]
        for t in range(L - 1, -1, -1) if p["reverse"] else range(L):
            out = self.call(p["jaxpr"], consts + carry + [x[t, ...] for x in xs])
            carry, yt = split(out)
            if len(yt) != ny:
                return self.unsupported(eqn, "scan: unexpected output structure")
            for j, y in enumerate(yt):
                ys[j][t] = y
        self.g.notes["scan iterations unrolled"] += L
        stacked = []
        for j, col in enumerate(ys):
            if L == 0:
                stacked.append(np.zeros(eqn.outvars[nk + j].aval.shape))
            elif any(_is_sym(y) for y in col):
                stacked.append(np.stack([self.sym(y) for y in col]))
            else:
                stacked.append(np.stack([np.asarray(y) for y in col]))
        return carry + stacked

    def while_(self, eqn, ins):
        p = eqn.params
        cn, bn = p["cond_nconsts"], p["body_nconsts"]
        cc, bc, carry = list(ins[:cn]), list(ins[cn : cn + bn]), list(ins[cn + bn :])
        for it in range(MAX_WHILE):
            c = self.call(p["cond_jaxpr"], cc + carry)[0]
            if _is_sym(c):
                return self.unsupported(
                    eqn, "while_loop with a variable-dependent condition (data-dependent trip count)"
                )
            if not bool(np.asarray(c)):
                self.g.notes["while iterations unrolled"] += it
                return carry
            carry = self.call(p["body_jaxpr"], bc + carry)
        return self.unsupported(eqn, f"while_loop ran more than {MAX_WHILE} iterations")


def _integer_mask(integer, n):
    """``integer`` (None, a boolean mask or indices) as a boolean mask, as ``minimize`` takes it."""
    mask = np.zeros(n, bool)
    if integer is None:
        return mask
    a = np.asarray(integer)
    if a.dtype == bool:
        if a.shape != (n,):
            raise ValueError(f"integer mask has shape {a.shape}, expected ({n},)")
        return a.copy()
    mask[a.astype(np.int64)] = True
    return mask


@dataclass
class _Problem:
    g: Graph
    n: int  # original variables (the first n)
    lo: np.ndarray  # bounds of all variables (original + auxiliary)
    hi: np.ndarray
    integer: np.ndarray
    obj: int
    cons: list  # (node, lo, hi, label): lo <= node <= hi
    labels: list  # variable labels


def _build(f, lo, hi, ineq=None, eq=None, integer=None):
    """Trace ``f`` / ``ineq`` / ``eq`` into a ``_Problem``."""
    lo = np.asarray(lo, np.float64).reshape(-1)
    hi = np.asarray(hi, np.float64).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError("lo and hi must have the same shape")
    n = lo.size
    fns = [fn for fn in (f, ineq, eq) if fn is not None]
    with jax.ensure_compile_time_eval():
        closed = jax.make_jaxpr(lambda x: [jnp.atleast_1d(fn(x)) for fn in fns])(jnp.zeros(n, jnp.float64))
    g = Graph()
    x = np.empty(n, object)
    for j in range(n):
        x[j] = g.var(j)
    w = _Walker(g)
    outs = [w.sym(o).reshape(-1) for o in w.call(closed, [x])]
    if outs[0].size != 1:
        raise ValueError(f"f must return a scalar, got {outs[0].size} values")
    cons, k = [], 1
    if ineq is not None:
        cons += [(int(s), -np.inf, 0.0, f"ineq[{i}]") for i, s in enumerate(outs[k])]
        k += 1
    if eq is not None:
        cons += [(int(s), 0.0, 0.0, f"eq[{i}]") for i, s in enumerate(outs[k])]
    mask = _integer_mask(integer, n)
    return _Problem(g, n, lo.copy(), hi.copy(), mask, int(outs[0][0]), cons, [f"x[{j}]" for j in range(n)])


# ---------------------------------------------------------------------------
# Lowering for a target
# ---------------------------------------------------------------------------

SCIP_EXACT = {"add", "sub", "mul", "div", "neg", "sum", "exp", "log", "sqrt", "abs", "sin", "cos"}
NL_EXACT = {"add", "sub", "mul", "div", "neg", "sum", "exp", "log", "sqrt", "abs", "sin", "cos", "tan", "tanh",
            "sinh", "cosh", "atan", "asin", "acos", "atanh", "asinh", "acosh", "atan2"}  # fmt: skip
NL_NATIVE_NONSMOOTH = {"max", "min", "select", "lt", "le", "gt", "ge", "eq", "ne", "and", "or", "not"}
REWRITES = {
    "tan": "sin(x) / cos(x)", "tanh": "1 - 2 / (exp(2x) + 1)", "logistic": "1 / (1 + exp(-x))",
    "log1p": "log(1 + x)", "expm1": "exp(x) - 1", "sinh": "(exp(x) - exp(-x)) / 2",
    "cosh": "(exp(x) + exp(-x)) / 2", "asinh": "log(x + sqrt(x^2 + 1))", "acosh": "log(x + sqrt(x^2 - 1))",
    "atanh": "log((1 + x) / (1 - x)) / 2", "pow(x, y)": "exp(y log x), x > 0 on the box",
    "max": "(a + b + |a - b|) / 2", "min": "(a + b - |a - b|) / 2",
    "sign": "if(x > 0, 1, if(x < 0, -1, 0))", "pow": "x^p",
}  # fmt: skip


@dataclass
class ExportReport:
    """What the export did with each operation of the model.

    ``exact``: operations written as the target's own operator.
    ``rewritten``: operations written as an exact identity in other operators (``how``).
    ``reformulated``: operations replaced by auxiliary variables and constraints (``how``).
    ``refused``: operations the target cannot represent (``to_*`` raise ``ExportError``).
    ``approximated``: operations whose export is not exact (``how``): comparisons
    and ``sign`` (relaxed at ties, or restricted near them with ``strict_eps``),
    ``reduce_precision`` and float64 -> float32 casts (identities). When it is not
    empty, evaluate ``f`` with JAX at the solver's ``x`` before trusting the
    objective value.
    """

    target: str
    n_vars: int = 0  # original variables
    n_aux: int = 0  # auxiliary variables added (continuous + binary)
    n_binary: int = 0  # of which binary
    n_cons: int = 0  # original constraints
    n_aux_cons: int = 0  # constraints added by reformulations
    n_ops: int = 0  # scalar operations in the exported model
    epigraph: bool = False  # objective moved into a constraint (SCIP: nonlinear objective)
    exact: Counter = field(default_factory=Counter)
    rewritten: dict = field(default_factory=dict)  # op -> [count, how]
    reformulated: dict = field(default_factory=dict)  # op -> [count, how]
    refused: list = field(default_factory=list)
    approximated: dict = field(default_factory=dict)  # op -> [count, how]
    assumptions: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    var_order: np.ndarray | None = None  # .nl: model variable of each .nl variable (first n_vars are x)

    def _count(self, kind, op, how):
        d = getattr(self, kind)
        e = d.setdefault(op, [0, how])
        e[0] += 1
        if how not in e[1].split("; "):  # the same op written in more than one form
            e[1] += "; " + how

    def __str__(self):
        lines = [f"export to {self.target}: {self.n_vars} variables (+{self.n_aux} auxiliary, {self.n_binary} binary), "
                 f"{self.n_cons} constraints (+{self.n_aux_cons}), {self.n_ops} operations"]  # fmt: skip
        if self.exact:
            lines.append("  exact: " + ", ".join(f"{k} x{v}" for k, v in sorted(self.exact.items())))
        for kind in ("rewritten", "reformulated", "approximated"):
            for k, (c, how) in sorted(getattr(self, kind).items()):
                lines.append(f"  {kind}: {k} x{c} -> {how}")
        if self.epigraph:
            lines.append("  objective: epigraph variable t >= f(x), minimize t")
        lines += [f"  refused: {r}" for r in self.refused]
        lines += [f"  assumption: {a}" for a in self.assumptions]
        lines += [f"  note: {s}" for s in self.notes]
        return "\n".join(lines)


class _Lowering:
    """Rebuild a problem's graph for a target, adding auxiliary variables where needed.

    ``mode``: ``"scip"`` (SCIP's operators), ``"nl"`` (.nl operators, nonsmooth
    ones native), ``"nl-reformulate"`` (.nl operators, nonsmooth ones
    reformulated as for SCIP) or ``"nl-portable"`` (only the .nl operators SCIP's
    and Couenne's readers accept: SCIP's set, nonsmooth ones reformulated).
    ``minmax``: ``"abs"`` or ``"bigm"`` for max / min where they are not native.
    ``strict_eps``: margin making reformulated strict comparisons strict (0: ties
    allowed either way).
    """

    TARGETS = {"scip": "SCIP", "nl": ".nl", "nl-reformulate": ".nl", "nl-portable": ".nl (portable)"}

    def __init__(self, prob, mode, minmax="abs", strict_eps=0.0):
        if minmax not in ("abs", "bigm"):
            raise ValueError(f"minmax must be 'abs' or 'bigm', not {minmax!r}")
        if not (np.isfinite(strict_eps) and strict_eps >= 0):
            raise ValueError(f"strict_eps must be finite and >= 0, not {strict_eps!r}")
        self.p, self.mode, self.minmax, self.eps = prob, mode, minmax, float(strict_eps)
        self.g = Graph()
        self.lo, self.hi = list(prob.lo), list(prob.hi)
        self.integer, self.labels = list(prob.integer), list(prob.labels)
        self.cons = []
        self.binary = 0
        self.bigm_max = 0.0
        self.unit = set()  # lowered nodes known to lie in [0, 1]
        self.report = ExportReport(target=self.TARGETS[mode])
        roots = [prob.obj] + [c[0] for c in prob.cons]
        self.order = _reachable(prob.g, roots)
        self.iv = _intervals(prob.g, self.order, prob.lo, prob.hi)
        self.undefined = {}  # node -> why its expression is undefined somewhere on the box
        for i in self.order:
            why = self._domain(i)
            if why is None:
                why = next((self.undefined[k] for k in prob.g.nodes[i][1] if k in self.undefined), None)
            if why is not None:
                self.undefined[i] = why
        self.map = {}
        for i in self.order:
            self.map[i] = self.lower(i)
        if not self.report.refused:
            self.overflow_check()
        r = self.report
        r.n_vars, r.n_cons = prob.n, len(prob.cons)
        r.assumptions = sorted(prob.g.assumptions)
        r.notes += [f"{k}: {v}" for k, v in sorted(prob.g.notes.items())]
        for k, (c, how) in prob.g.approximated.items():
            r.approximated.setdefault(k, [0, how])[0] += c
        if self.bigm_max > BIGM_WARN:
            msg = (f"largest big-M constant is {self.bigm_max:.3g} (> {BIGM_WARN:g}): the reformulation is "
                   "numerically weak; tighten the variable bounds")  # fmt: skip
            r.notes.append(msg)
            warnings.warn(msg, ExportBigMWarning, stacklevel=4)

    def _domain(self, i):
        """Why node ``i``'s own operation is undefined somewhere on the box (None if it is defined everywhere)."""
        g = self.p.g
        op, args, _ = g.nodes[i]
        if not args or op not in PARTIAL | {"pow"}:
            return None
        lo, hi = self.iv[args[0]]

        def why(cond):
            return f"{op} of an operand in [{lo:.4g}, {hi:.4g}]{cond}"

        if op == "sqrt" and lo < 0:
            return why(" (sqrt needs >= 0)")
        if op == "log" and lo <= 0:
            return why(" (log needs > 0)")
        if op == "log1p" and lo <= -1:
            return why(" (log1p needs > -1)")
        if op == "acosh" and lo < 1:
            return why(" (acosh needs >= 1)")
        if op in ("asin", "acos") and (lo < -1 or hi > 1):
            return why(f" ({op} needs [-1, 1])")
        if op == "atanh" and (lo <= -1 or hi >= 1):
            return why(" (atanh needs (-1, 1))")
        if op == "tan" and not np.all(np.isfinite(self.iv[i])):
            return why(" (tan has a pole there)")
        if op == "div":
            bl, bh = self.iv[args[1]]
            if bl <= 0 <= bh:
                return f"division by an operand in [{bl:.4g}, {bh:.4g}] (can be 0)"
        if op == "pow":
            if not g.is_const(args[1]):
                return why(" (x ** y needs x > 0)") if lo <= 0 else None
            k = g.value(args[1])
            if float(k).is_integer():
                return why(f" (x ** {k:g} needs x != 0)") if k < 0 and lo <= 0 <= hi else None
            if lo < 0 or (k < 0 and lo <= 0):
                return why(f" (x ** {k:g} needs x {'>' if k < 0 else '>='} 0)")
        return None

    def overflow_check(self):
        """Refuse when an exported intermediate reaches SCIP's infinity (1e20) over the box although its operands
        are bounded: the solver treats such points as invalid evaluations and cuts them off."""
        g = self.g
        roots = [self.map[self.p.obj]] + [self.map[c[0]] for c in self.p.cons] + [c[0] for c in self.cons]
        order = _reachable(g, roots)
        iv = _intervals(g, order, self.lo, self.hi, fixed={k: (0.0, 1.0) for k in self.unit})
        for i in order:
            op, args, _ = g.nodes[i]
            if op not in OVERFLOW_OPS and op not in ("div", "pow"):
                continue  # log, sqrt, tan, atanh, ...: an infinite bound is a pole or domain edge, not an overflow
            if not all(np.isfinite(iv[k][0]) and np.isfinite(iv[k][1]) for k in args):
                continue
            m = max(abs(iv[i][0]), abs(iv[i][1]))
            if np.isnan(m) or m < BIGM_REFUSE:
                continue
            if op == "pow" and g.is_const(args[1]) and g.value(args[1]) >= 0:
                pass  # x ** k, k >= 0: grows like exp
            elif op in ("div", "pow"):
                # 1/d, x ** -k or x ** y: an infinite bound is a pole or domain edge (d or x can be 0); a finite
                # bound >= 1e20 means the value really overflows
                if not np.isfinite(m):
                    continue
            msg = (
                f"an exported '{op}' reaches {m:.3g} (>= {BIGM_REFUSE:g}, SCIP's infinity) on the box although "
                "its operands are bounded: the solver would cut those points off; tighten the variable bounds"
            )
            self.report.refused.append(msg)
            return

    def defined(self, k, what):
        """Refuse when node ``k`` (a branch or operand the reformulation evaluates everywhere) is undefined somewhere."""
        if k in self.undefined:
            raise ExportError(
                f"{what} contains {self.undefined[k]}, which is undefined on part of the box: the reformulation "
                "evaluates it everywhere and would cut that part off (JAX masks it); make it defined everywhere, "
                "e.g. jnp.sqrt(jnp.maximum(x, 0)) or jnp.log(jnp.maximum(x, 1e-12))"
            )

    def bigm(self, what, *vals):
        """Check big-M constants / auxiliary bounds: refuse at ``BIGM_REFUSE``, remember the largest."""
        m = max(abs(float(v)) for v in vals)
        if not m < BIGM_REFUSE:
            raise ExportError(f"{what}: big-M constant {m:.3g} >= {BIGM_REFUSE:g} (SCIP's infinity); "
                              "tighten the variable bounds")  # fmt: skip
        self.bigm_max = max(self.bigm_max, m)

    def tie(self, op, n=1):
        if self.eps == 0:
            how = ("relaxed at ties: where the compared values are equal either branch is allowed (pass "
                   "strict_eps > 0, and evaluate f with JAX at the returned x)")  # fmt: skip
        else:
            how = (f"restricted near ties: values within strict_eps={self.eps:g} of the tie on the wrong side are "
                   "excluded (evaluate f with JAX at the returned x)")  # fmt: skip
        self.report._count("approximated", op, how)

    # helpers -------------------------------------------------------------
    def new_var(self, lo, hi, label, binary=False):
        j = len(self.lo)
        self.lo.append(0.0 if binary else lo)
        self.hi.append(1.0 if binary else hi)
        self.integer.append(binary)
        self.labels.append(label)
        self.binary += binary
        return self.g.var(j)

    def add_con(self, node, lo, hi, label):
        self.cons.append((node, lo, hi, label))

    def bounds(self, i, what):
        lo, hi = self.iv[i]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ExportError(f"{what}: no finite bound on an operand over the box (big-M needs one); "
                              "give the variables finite bounds")  # fmt: skip
        return _pad(lo, hi)

    def lin(self, *terms):
        """sum of c * node (node None for a constant c)."""
        g = self.g
        return g.sum([g.const(c) if v is None else g.op("mul", g.const(c), v) for c, v in terms])

    # one node ------------------------------------------------------------
    def lower(self, i):
        try:
            return self._lower(i)
        except ExportError as e:
            self.report.refused.append(str(e))
            return self.g.unsupported(str(e))

    def _lower(self, i):
        g, r = self.g, self.report
        op, args, p = self.p.g.nodes[i]
        a = [self.map[k] for k in args]
        nl = self.mode != "scip"
        native = self.mode == "nl"
        if op == "var":
            return g.var(p)
        if op == "const":
            if not np.isfinite(p) and nl:
                raise ExportError(f"non-finite constant {p} in the model")
            return g.const(p)
        if op == "unsupported":
            raise ExportError(p)
        if op == "div" and self.p.g.is_const(args[1], 0.0):
            raise ExportError("division by the constant 0")
        exact = SCIP_EXACT if self.mode in ("scip", "nl-portable") else NL_EXACT
        if op in exact:
            r.exact[op] += 1
            return g.sum(a) if op == "sum" else g.op(op, *a)
        if op == "pow":
            if self.p.g.is_const(args[1]):
                r.exact["pow"] += 1
                return g.op("pow", *a)
            if nl:
                r.exact["pow(x, y)"] += 1
                return g.op("pow", *a)
            if self.iv[args[0]][0] > 0:
                r._count("rewritten", "pow(x, y)", REWRITES["pow(x, y)"])
                return g.op("exp", g.op("mul", a[1], g.op("log", a[0])))
            raise ExportError("pow(x, y) with a variable exponent needs x > 0 over the box (SCIP has no x**y)")
        if op in REWRITES and op not in ("max", "min", "sign"):
            return self.rewrite(i, op, a[0])
        if op in ("atan", "asin", "acos", "atan2"):
            if self.mode == "nl-portable":
                raise ExportError(f"{op} is not read by SCIP's or Couenne's .nl readers (use portable=False)")
            raise ExportError(f"{op} has no SCIP expression (write to .nl instead)")
        if native and op in NL_NATIVE_NONSMOOTH:
            r.exact[op] += 1
            return g.op(op, *a)
        if op == "sign":
            if native:
                r._count("rewritten", "sign", REWRITES["sign"])
                one = g.const(1.0)
                zero = g.const(0.0)
                return g.op("select", g.op("gt", a[0], zero), g.op("select", g.op("lt", a[0], zero), zero,
                                                                     g.op("neg", one)), one)  # fmt: skip
            return self.sign(i, a[0])
        if op in ("max", "min"):
            if self.minmax == "abs":
                r._count("rewritten", op, REWRITES[op])
                d = g.op("abs", g.op("sub", a[0], a[1]))
                s = g.op("add", a[0], a[1])
                return g.op("mul", g.const(0.5), g.op("add" if op == "max" else "sub", s, d))
            return self.maxmin(i, op, args, a)
        if op in ("lt", "le", "gt", "ge"):
            return self.compare(i, op, args, a)
        if op in ("eq", "ne"):
            raise ExportError(f"comparison '{op}' between model values (exact equality) cannot be reformulated")
        if op in ("and", "or"):
            how = "w <= p, w <= q, w >= p + q - 1" if op == "and" else "w >= p, w >= q, w <= p + q"
            r._count("reformulated", op, how)
            w = self.new_var(0.0, 1.0, f"{op}[{i}]")
            P, Q = a
            if op == "and":
                self.add_con(self.lin((1, w), (-1, P)), -np.inf, 0.0, f"{op}[{i}]")
                self.add_con(self.lin((1, w), (-1, Q)), -np.inf, 0.0, f"{op}[{i}]")
                self.add_con(self.lin((1, w), (-1, P), (-1, Q)), -1.0, np.inf, f"{op}[{i}]")
            else:
                self.add_con(self.lin((1, w), (-1, P)), 0.0, np.inf, f"{op}[{i}]")
                self.add_con(self.lin((1, w), (-1, Q)), 0.0, np.inf, f"{op}[{i}]")
                self.add_con(self.lin((1, w), (-1, P), (-1, Q)), -np.inf, 0.0, f"{op}[{i}]")
            return w
        if op == "not":
            r._count("reformulated", "not", "1 - p")
            return self.lin((1.0, None), (-1, a[0]))
        if op == "select":
            return self.select(i, args, a)
        raise ExportError(f"operation '{op}' is not supported for {r.target}")

    def rewrite(self, i, op, x):
        """Exact rewrite of ``op`` applied to ``x``. Forms are chosen from ``x``'s interval so that no ``exp``
        overflows SCIP's infinity (1e20) where the original value is finite."""
        g = self.g
        xl, xh = self.iv[self.p.g.nodes[i][1][0]]

        def c(v):
            return g.const(v)

        def done(how):
            self.report._count("rewritten", op, how)

        def split(t):  # exp(min(t, 0)), exp(min(-t, 0)): both in (0, 1]
            m = g.op("abs", t)
            a, b = g.op("exp", g.op("sub", t, m)), g.op("exp", g.op("sub", g.op("neg", t), m))
            self.unit |= {a, b}  # in (0, 1]; interval arithmetic cannot see it (x - |x| loses the dependency)
            return a, b

        if op == "tan":
            done(REWRITES[op])
            return g.op("div", g.op("sin", x), g.op("cos", x))
        if op == "tanh":
            if xl >= 0:  # e = exp(-2x) <= 1
                done("(1 - exp(-2x)) / (1 + exp(-2x)) (x >= 0 on the box)")
                e = g.op("exp", g.op("mul", c(-2.0), x))
                return g.op("div", g.op("sub", c(1.0), e), g.op("add", c(1.0), e))
            if xh <= EXP_SAFE / 2:
                done(REWRITES[op])
                return g.op(
                    "sub", c(1.0), g.op("div", c(2.0), g.op("add", g.op("exp", g.op("mul", c(2.0), x)), c(1.0)))
                )
            done("(a - b) / (a + b), a = exp(x - |x|), b = exp(-x - |x|) (no exp overflow)")
            a, b = split(x)
            return g.op("div", g.op("sub", a, b), g.op("add", a, b))
        if op == "logistic":
            if xl >= -EXP_SAFE:
                done(REWRITES[op])
                return g.op("div", c(1.0), g.op("add", c(1.0), g.op("exp", g.op("neg", x))))
            if xh <= 0:  # e = exp(x) <= 1
                done("exp(x) / (1 + exp(x)) (x <= 0 on the box)")
                e = g.op("exp", x)
                return g.op("div", e, g.op("add", c(1.0), e))
            done("a / (a + b), a = exp((x - |x|)/2), b = exp((-x - |x|)/2) (no exp overflow)")
            a, b = split(g.op("mul", c(0.5), x))
            return g.op("div", a, g.op("add", a, b))
        if op not in ("asinh", "acosh"):
            done(REWRITES[op])

        def big(t):  # log(t) + log(1 + sqrt(1 + s t^-2)) for t >= 1: no t^2 (it overflows for t > 1e10)
            return g.op(
                "add",
                g.op("log", t),
                g.op(
                    "log",
                    g.op("add", c(1.0), g.op("sqrt", g.op("add", c(1.0), g.op("mul", c(s), g.op("pow", t, c(-2)))))),
                ),
            )

        if op == "log1p":
            return g.op("log", g.op("add", c(1.0), x))
        if op == "expm1":
            return g.op("sub", g.op("exp", x), c(1.0))
        if op in ("sinh", "cosh"):
            e, em = g.op("exp", x), g.op("exp", g.op("neg", x))
            return g.op("mul", c(0.5), g.op("sub" if op == "sinh" else "add", e, em))
        if op == "asinh":  # odd: x + sqrt(x^2 + 1) cancels for x << 0
            s = 1.0
            if xl >= 1 and xh > 1e9:
                done("log(x) + log(1 + sqrt(1 + x^-2)) (x >= 1 on the box)")
                return big(x)
            if xh <= -1 and xl < -1e4:
                done("-(log(-x) + log(1 + sqrt(1 + x^-2))) (x <= -1 on the box)")
                return g.op("neg", big(g.op("neg", x)))
            if xl >= -1e4:
                done(REWRITES[op])
                return g.op("log", g.op("add", x, g.op("sqrt", g.op("add", g.op("pow", x, c(2)), c(1.0)))))
            mag = g.op("log", g.op("add", g.op("abs", x), g.op("sqrt", g.op("add", g.op("pow", x, c(2)), c(1.0)))))
            if xh <= 0:
                done("-log(|x| + sqrt(x^2 + 1)) (x <= 0 on the box)")
                return g.op("neg", mag)
            if not np.isfinite(xl) or not np.isfinite(xh):
                done(REWRITES[op])
                how = "log(x + sqrt(x^2 + 1)) loses accuracy for x < -1e4 (no finite bound for sign(x))"
                self.report._count("approximated", "asinh", how)
                return g.op("log", g.op("add", x, g.op("sqrt", g.op("add", g.op("pow", x, c(2)), c(1.0)))))
            done("sign(x) log(|x| + sqrt(x^2 + 1)) (sign with two binaries)")
            return g.op("mul", self.sign(i, x, "asinh (sign(x) log(|x| + sqrt(x^2 + 1)))", ties=False), mag)
        if op == "acosh":
            if xh > 1e9:
                s = -1.0
                done("log(x) + log(1 + sqrt(1 - x^-2))")
                return big(x)
            done(REWRITES[op])
            return g.op("log", g.op("add", x, g.op("sqrt", g.op("sub", g.op("pow", x, c(2)), c(1.0)))))
        if op == "atanh":
            return g.op("mul", c(0.5), g.op("log", g.op("div", g.op("add", c(1.0), x), g.op("sub", c(1.0), x))))
        raise AssertionError(op)

    def compare(self, i, op, args, a):
        """Binary z = [u < v] (or <=) on d = u - v: z = 1 => d <= 0, z = 0 => d >= 0, with ``strict_eps``
        moving the strict side (lt: z = 1 => d <= -eps; le: z = 0 => d >= eps)."""
        self.report._count("reformulated", op, "binary z with big-M: z=1 => u <= v, z=0 => u >= v")
        self.tie(op)
        u, v = (0, 1) if op in ("lt", "le") else (1, 0)
        for k in (u, v):
            self.defined(args[k], f"an operand of comparison '{op}'")
        (ul, uh), (vl, vh) = (self.bounds(args[k], f"comparison '{op}'") for k in (u, v))
        dl, dh = ul - vh, uh - vl
        e = self.eps
        self.bigm(f"comparison '{op}'", dl - e, dh + e)
        d = self.g.op("sub", a[u], a[v])
        z = self.new_var(0, 1, f"{op}[{i}]", binary=True)
        lbl = f"{op}[{i}]"
        if op in ("lt", "gt"):  # strict: z = 1 => d <= -eps
            self.add_con(self.g.op("add", d, self.lin((dh + e, z))), -np.inf, dh, lbl)  # d <= dh (1-z) - eps z
            self.add_con(self.g.op("sub", d, self.lin((dl, z))), 0.0, np.inf, lbl)  # d >= dl z
        else:  # non-strict: z = 0 => d >= eps
            self.add_con(self.g.op("add", d, self.lin((dh, z))), -np.inf, dh, lbl)  # d <= dh (1-z)
            self.add_con(self.g.op("sub", d, self.lin((dl - e, z))), e, np.inf, lbl)  # d >= dl z + eps (1-z)
        return z

    def select(self, i, args, a):
        g = self.g
        P, F, T = a
        if g.is_const(F) and g.is_const(T):
            self.report._count("reformulated", "select", "f + (t - f) p with the predicate's binary")
            return self.lin((g.value(F), None), (g.value(T) - g.value(F), P))
        self.report._count("reformulated", "select", "aux y with big-M: p=1 => y = t, p=0 => y = f")
        self.defined(args[2], "the true branch of jnp.where / select_n")
        self.defined(args[1], "the false branch of jnp.where / select_n")
        (fl, fh), (tl, th) = self.bounds(args[1], "select_n (jnp.where)"), self.bounds(args[2], "select_n (jnp.where)")
        yl, yh = min(fl, tl), max(fh, th)
        self.bigm("select_n (jnp.where)", yl, yh, yh - tl, yl - th, yh - fl, yl - fh)
        y = self.new_var(yl, yh, f"select[{i}]")
        lbl = f"select[{i}]"
        # p = 1: y - t in [0, 0]; else |y - t| <= its range.  p = 0: y - f in [0, 0].
        mt_hi, mt_lo = yh - tl, yl - th  # y - t in [mt_lo, mt_hi]
        mf_hi, mf_lo = yh - fl, yl - fh
        yt, yf = g.op("sub", y, T), g.op("sub", y, F)
        self.add_con(g.op("add", yt, self.lin((mt_hi, P))), -np.inf, mt_hi, lbl)  # y - t <= mt_hi (1 - p)
        self.add_con(g.op("add", yt, self.lin((mt_lo, P))), mt_lo, np.inf, lbl)  # y - t >= mt_lo (1 - p)
        self.add_con(g.op("sub", yf, self.lin((mf_hi, P))), -np.inf, 0.0, lbl)  # y - f <= mf_hi p
        self.add_con(g.op("sub", yf, self.lin((mf_lo, P))), 0.0, np.inf, lbl)  # y - f >= mf_lo p
        return y

    def sign(self, i, x, what="sign", ties=True):
        """sign(x) = zp - zn with binaries; ``ties``: whether the value at x = 0 matters (else exact)."""
        self.report._count("reformulated", what, "binaries zp, zn: sign = zp - zn")
        e = self.eps if ties else 0.0
        if ties:
            self.tie("sign")
        xl, xh = self.bounds(self.p.g.nodes[i][1][0], what)
        self.bigm(what, xl - e, xh + e)
        zp = self.new_var(0, 1, f"sign+[{i}]", binary=True)
        zn = self.new_var(0, 1, f"sign-[{i}]", binary=True)
        lbl = f"sign[{i}]"
        self.add_con(self.lin((1, zp), (1, zn)), -np.inf, 1.0, lbl)
        self.add_con(self.g.op("sub", x, self.lin((xh, zp))), -np.inf, 0.0, lbl)  # x <= xh zp
        self.add_con(self.g.op("sub", x, self.lin((xl, zn))), 0.0, np.inf, lbl)  # x >= xl zn
        self.add_con(self.g.op("add", x, self.lin((xl - e, zp))), xl, np.inf, lbl)  # x >= xl (1 - zp) + eps zp
        self.add_con(self.g.op("add", x, self.lin((xh + e, zn))), -np.inf, xh, lbl)  # x <= xh (1 - zn) - eps zn
        return self.lin((1, zp), (-1, zn))

    def maxmin(self, i, op, args, a):
        self.report._count("reformulated", op, "aux y and binary z with big-M")
        (al, ah), (bl, bh) = (self.bounds(k, op) for k in args)
        self.bigm(op, al, ah, bl, bh, bh - al, ah - bl)
        g, A, B = self.g, a[0], a[1]
        lbl = f"{op}[{i}]"
        z = self.new_var(0, 1, f"{op}z[{i}]", binary=True)
        if op == "max":
            y = self.new_var(max(al, bl), max(ah, bh), lbl)
            ma, mb = max(0.0, bh - al), max(0.0, ah - bl)
            self.add_con(g.op("sub", y, A), 0.0, np.inf, lbl)
            self.add_con(g.op("sub", y, B), 0.0, np.inf, lbl)
            self.add_con(g.op("add", g.op("sub", y, A), self.lin((ma, z))), -np.inf, ma, lbl)  # y <= a + ma (1-z)
            self.add_con(g.op("sub", g.op("sub", y, B), self.lin((mb, z))), -np.inf, 0.0, lbl)  # y <= b + mb z
        else:
            y = self.new_var(min(al, bl), min(ah, bh), lbl)
            ma, mb = max(0.0, ah - bl), max(0.0, bh - al)
            self.add_con(g.op("sub", y, A), -np.inf, 0.0, lbl)
            self.add_con(g.op("sub", y, B), -np.inf, 0.0, lbl)
            self.add_con(g.op("sub", g.op("sub", y, A), self.lin((ma, z))), -ma, np.inf, lbl)  # y >= a - ma (1-z)
            self.add_con(g.op("add", g.op("sub", y, B), self.lin((mb, z))), 0.0, np.inf, lbl)  # y >= b - mb z
        return y

    # result --------------------------------------------------------------
    def problem(self):
        """The lowered ``_Problem`` (constant constraints checked and dropped)."""
        g, cons = self.g, []
        for node, lo, hi, label in [(self.map[c[0]],) + tuple(c[1:]) for c in self.p.cons] + self.cons:
            if g.is_const(node):
                v = g.value(node)
                if not (lo - 1e-9 <= v <= hi + 1e-9):
                    raise ExportError(f"constraint {label} is the constant {v}, outside [{lo}, {hi}]: infeasible")
                self.report.notes.append(f"constraint {label} is constant ({v}) and satisfied: dropped")
                continue
            cons.append((node, lo, hi, label))
        r = self.report
        r.n_aux = len(self.lo) - self.p.n
        r.n_binary = self.binary
        r.n_aux_cons = len(self.cons)
        roots = [self.map[self.p.obj]] + [c[0] for c in cons]
        r.n_ops = sum(1 for i in _reachable(g, roots) if g.nodes[i][0] not in ("var", "const"))
        return _Problem(g, self.p.n, np.asarray(self.lo, float), np.asarray(self.hi, float),
                        np.asarray(self.integer, bool), self.map[self.p.obj], cons, self.labels)  # fmt: skip


def _lower(f, lo, hi, ineq, eq, integer, mode, minmax, max_size, strict=True, strict_eps=0.0):
    prob = _build(f, lo, hi, ineq, eq, integer)
    low = _Lowering(prob, mode, minmax, strict_eps)
    if low.report.refused and strict:
        msgs = sorted(set(low.report.refused))
        raise ExportError(f"cannot export to {low.report.target}: " + "; ".join(msgs))
    out = low.problem() if not low.report.refused else None
    if out is not None and max_size is not None and low.report.n_ops > max_size:
        warnings.warn(f"exported model has {low.report.n_ops} scalar operations (> max_size={max_size:g})",
                      ExportSizeWarning, stacklevel=3)  # fmt: skip
    return out, low.report


def inspect(f, lo, hi, *, ineq=None, eq=None, integer=None, target="scip", minmax="abs", nonsmooth="native",
            portable=False, strict_eps=0.0):  # fmt: skip
    """The ``ExportReport`` of exporting to ``target`` (``"scip"`` or ``"nl"``), without raising on refusals."""
    if target not in ("scip", "nl"):
        raise ValueError(f"target must be 'scip' or 'nl', not {target!r}")
    mode = "scip" if target == "scip" else _nl_mode(nonsmooth, portable)
    _, report = _lower(f, lo, hi, ineq, eq, integer, mode, minmax, None, strict=False, strict_eps=strict_eps)
    return report


def _nl_mode(nonsmooth, portable):
    if nonsmooth not in ("native", "reformulate"):
        raise ValueError(f"nonsmooth must be 'native' or 'reformulate', not {nonsmooth!r}")
    return "nl-portable" if portable else ("nl" if nonsmooth == "native" else "nl-reformulate")


# ---------------------------------------------------------------------------
# SCIP
# ---------------------------------------------------------------------------


@dataclass
class SCIPExport:
    """Attached to the model as ``model.data``."""

    x: list  # pyscipopt variables of the original x
    report: ExportReport
    aux: list = field(default_factory=list)  # auxiliary variables


def to_scip(f, lo, hi, *, ineq=None, eq=None, integer=None, name="jaxglobal", minmax="abs", max_size=1e5,
            share_min=50, strict_eps=0.0):  # fmt: skip
    """A ``pyscipopt.Model`` for: minimize ``f(x)`` s.t. ``ineq(x) <= 0``, ``eq(x) == 0``, ``lo <= x <= hi``.

    The arguments are those of ``jaxglobal.minimize``. ``model.data`` is a
    ``SCIPExport`` with the variables of ``x`` and the ``ExportReport``;
    ``solution(model)`` returns the best solution as an array.

    ``minmax="abs"`` writes ``max`` / ``min`` (and ``relu``, ``clip``) as the exact
    identities ``(a + b +- |a - b|) / 2``; ``"bigm"`` uses a binary and big-M
    constraints instead. ``jnp.where`` and ``sign`` always use binaries with big-M
    constants from interval bounds over the box, and are refused when those are
    not finite. Reformulated comparisons (and ``sign``) are relaxed at ties
    unless ``strict_eps > 0`` (see the module docstring and
    ``report.approximated``). A shared subexpression with at least
    ``share_min`` operations becomes an auxiliary variable (``w == expr``), so
    unrolled loops do not grow into exponentially large expression trees, and a
    product operand becomes one before pyscipopt would expand the product past
    ``MAX_TERMS`` monomials. A nonlinear objective is moved into a constraint
    ``f(x) <= t`` with ``minimize t``. Warns (``ExportSizeWarning``) above
    ``max_size`` scalar operations and (``ExportBigMWarning``) for big-M
    constants above ``BIGM_WARN``. Raises ``ExportError`` naming the operation
    when something cannot be exported.
    """
    import pyscipopt as ps

    prob, report = _lower(f, lo, hi, ineq, eq, integer, "scip", minmax, max_size, strict_eps=strict_eps)
    g = prob.g
    model = ps.Model(name)
    xs = []
    for j in range(len(prob.lo)):
        lb = None if prob.lo[j] == -np.inf else float(prob.lo[j])
        ub = None if prob.hi[j] == np.inf else float(prob.hi[j])
        vt = "C"
        if prob.integer[j]:
            vt = "B" if (lb is not None and ub is not None and lb >= 0 and ub <= 1) else "I"
        xs.append(model.addVar(name=prob.labels[j], vtype=vt, lb=lb, ub=ub))
    roots = [prob.obj] + [c[0] for c in prob.cons]
    order = _reachable(g, roots)
    share = _shared(g, order, roots, share_min)
    iv = _intervals(g, order, prob.lo, prob.hi)
    E = {}
    n_lift = 0

    def aux(k, e, label):
        wl, wh = _pad(*iv[k])
        w = model.addVar(name=f"{label}[{k}]", lb=None if wl <= -1e20 else wl, ub=None if wh >= 1e20 else wh)
        model.addCons(w - e == 0, name=f"def_{label}[{k}]")
        xs.append(w)
        return w

    def nterms(e):
        return len(e.terms) if isinstance(e, ps.scip.Expr) else 1

    def lift(k):  # E[k] as a variable, so a product does not expand it
        nonlocal n_lift
        if nterms(E[k]) > 1:
            E[k] = aux(k, E[k], "p")
            n_lift += 1
        return E[k]

    for i in order:
        op, args, p = g.nodes[i]
        a = [E[k] for k in args]
        if op == "var":
            e = xs[p]
        elif op == "const":
            e = p
        elif op == "add":
            e = a[0] + a[1]
        elif op == "sub":
            e = a[0] - a[1]
        elif op == "mul":
            if nterms(a[0]) * nterms(a[1]) > MAX_TERMS:
                big, small = (0, 1) if nterms(a[0]) >= nterms(a[1]) else (1, 0)
                a[big] = lift(args[big])
                if nterms(a[0]) * nterms(a[1]) > MAX_TERMS:
                    a[small] = lift(args[small])
            e = a[0] * a[1]
        elif op == "div":
            if isinstance(a[1], float) and a[1] == 0.0:
                raise ExportError("division by the constant 0")
            e = a[0] * (1.0 / a[1]) if isinstance(a[1], float) else a[0] / a[1]
        elif op == "neg":
            e = -a[0]
        elif op == "sum":
            e = ps.quicksum(a)
        elif op == "pow":
            k = a[1]
            if float(k).is_integer() and k > 1 and float(nterms(a[0])) ** k > MAX_TERMS:
                a[0] = lift(args[0])
            e = a[0] ** (int(k) if float(k).is_integer() else k)
        elif op in ("exp", "log", "sqrt", "sin", "cos"):
            e = getattr(ps, op)(a[0])
        elif op == "abs":
            e = abs(a[0])
        else:  # pragma: no cover - the lowering only leaves the operators above
            raise ExportError(f"internal: operation '{op}' reached the SCIP writer")
        if i in share:
            e = aux(i, e, "w")
        E[i] = e
    if n_lift:
        report.n_aux += n_lift
        report.n_aux_cons += n_lift
        report._count("reformulated", "product operand", f"aux p == expr (product past {MAX_TERMS} monomials)")
    if share:
        report.n_aux += len(share)
        report.n_aux_cons += len(share)
        report._count("reformulated", "shared subexpression", f"aux w == expr (>= {share_min} operations, used twice+)")
    for k, (node, clo, chi, label) in enumerate(prob.cons):
        e = E[node]
        if clo == chi:
            model.addCons(e == clo, name=label)
            continue
        if clo > -np.inf:
            model.addCons(e >= clo, name=label)
        if chi < np.inf:
            model.addCons(e <= chi, name=label)
    obj = E[prob.obj]
    if isinstance(obj, float):
        model.setObjective(ps.Expr() + 0.0)
        model.addObjoffset(obj)
    elif isinstance(obj, ps.scip.Expr) and obj.degree() <= 1:
        model.setObjective(obj, "minimize")
    else:
        t = model.addVar(name="t_obj", lb=None, ub=None)
        model.addCons(obj - t <= 0, name="epigraph")
        model.setObjective(t, "minimize")
        report.epigraph = True
        report.n_aux += 1
        report.n_aux_cons += 1
    model.data = SCIPExport(x=xs[: prob.n], report=report, aux=xs[prob.n :])
    return model


def _shared(g, order, roots, share_min):
    """Nodes used more than once whose expanded expression tree has at least ``share_min`` operations."""
    if share_min is None:
        return set()
    uses = Counter(r for r in roots)
    size = {}
    for i in order:
        op, args, _ = g.nodes[i]
        for k in args:
            uses[k] += 1
        size[i] = 0 if op in ("var", "const") else 1 + sum(size[k] for k in args)
    out, size2 = set(), {}
    for i in order:  # sizes counting a shared node as a leaf once it is a variable
        op, args, _ = g.nodes[i]
        size2[i] = 0 if op in ("var", "const") else 1 + sum(0 if k in out else size2[k] for k in args)
        if uses[i] > 1 and size2[i] >= share_min:
            out.add(i)
    return out


def solution(model, sol=None):
    """``x`` of the best solution of a model built by ``to_scip`` (or of ``sol``), as an array."""
    sol = model.getBestSol() if sol is None else sol
    return np.array([model.getSolVal(sol, v) for v in model.data.x])


# ---------------------------------------------------------------------------
# AMPL .nl (text format)
# ---------------------------------------------------------------------------

NL_OPS = {
    "add": 0, "sub": 1, "mul": 2, "div": 3, "pow": 5, "neg": 16, "abs": 15, "sqrt": 39, "log": 43, "exp": 44,
    "sin": 41, "cos": 46, "tan": 38, "tanh": 37, "sinh": 40, "cosh": 45, "atan": 49, "asin": 51, "acos": 53,
    "atanh": 47, "asinh": 50, "acosh": 52, "atan2": 48, "lt": 22, "le": 23, "eq": 24, "ge": 28, "gt": 29,
    "ne": 30, "or": 20, "and": 21, "not": 34, "select": 35,
}  # fmt: skip


def to_nl(f, lo, hi, path, *, ineq=None, eq=None, integer=None, nonsmooth="native", minmax="abs", x0=None,
          max_size=1e5, portable=False, strict_eps=0.0, share_min=50):  # fmt: skip
    """Write the problem (arguments as for ``minimize``) to the text ``.nl`` file ``path``.

    ``nonsmooth="native"`` keeps ``max``, ``min``, ``jnp.where`` (``if``) and
    comparisons as .nl operators: exact, no extra variables, read by
    ``jaxglobal.nl.read_nl`` and ASL-based solvers that implement them.
    ``"reformulate"`` replaces them with binaries and big-M constraints as
    ``to_scip`` does (``strict_eps`` as there).

    Which operators a reader accepts varies. SCIP's .nl reader (``scip`` binary)
    takes ``+ - * / neg pow exp log sqrt abs sin cos`` and n-ary sums only;
    Couenne also takes ``tan tanh sinh cosh``. Neither takes ``min`` / ``max``,
    ``if`` and comparisons, ``atan asin acos atan2`` or the inverse hyperbolic
    functions. ``portable=True`` writes only operators both accept: the
    others are rewritten as for ``to_scip`` (``tanh`` with ``exp``, ...),
    nonsmooth ones are reformulated (``nonsmooth`` is ignored), and inverse
    trigonometric functions are refused. Variables are written
    continuous first, then integer (the order ASL requires); the report's
    ``var_order[k]`` is the model variable of .nl variable ``k`` (the first
    ``n_vars`` model variables are ``x``, the rest auxiliary). Shared
    subexpressions become defined variables (``V`` segments); with
    ``portable=True`` there are none (Couenne 0.5.8 misreads ``V`` segments,
    even in files AMPL writes; use ``portable=True`` with Couenne), and a shared subexpression with at least ``share_min``
    operations becomes an auxiliary variable with ``w == expr`` instead.
    Returns the ``ExportReport``.
    """
    mode = _nl_mode(nonsmooth, portable)
    prob, report = _lower(f, lo, hi, ineq, eq, integer, mode, minmax, max_size, strict_eps=strict_eps)
    if portable:
        prob = _lift_shared(prob, share_min, report)
    text, order = _nl_text(prob, x0, defined_vars=not portable)
    report.var_order = order
    with open(path, "w") as fh:
        fh.write(text)
    return report


def _fmt(v):
    v = float(v)
    return repr(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)


def _lift_shared(prob, share_min, report):
    """``prob`` with each large shared subexpression (``_shared``) replaced by a variable ``w`` and ``w == expr``."""
    g = prob.g
    roots = [prob.obj] + [c[0] for c in prob.cons]
    order = _reachable(g, roots)
    share = _shared(g, order, roots, share_min)
    if not share:
        return prob
    iv = _intervals(g, order, prob.lo, prob.hi)
    h, M, new = Graph(), {}, []
    lo, hi, integer, labels = list(prob.lo), list(prob.hi), list(prob.integer), list(prob.labels)
    for i in order:
        op, args, p = g.nodes[i]
        if op == "var":
            M[i] = h.var(p)
        elif op == "const":
            M[i] = h.const(p)
        elif op == "sum":
            M[i] = h.sum([M[k] for k in args])
        else:
            M[i] = h.op(op, *[M[k] for k in args])
        if i in share:
            wl, wh = _pad(*iv[i])
            lo.append(wl)
            hi.append(wh)
            integer.append(False)
            labels.append(f"w[{i}]")
            w = h.var(len(lo) - 1)
            new.append((h.op("sub", M[i], w), 0.0, 0.0, f"def_w[{i}]"))
            M[i] = w
    report.n_aux += len(share)
    report.n_aux_cons += len(share)
    report._count("reformulated", "shared subexpression", f"aux w == expr (>= {share_min} operations, used twice+)")
    cons = [(M[c[0]],) + tuple(c[1:]) for c in prob.cons] + new
    return _Problem(h, prob.n, np.asarray(lo, float), np.asarray(hi, float), np.asarray(integer, bool), M[prob.obj],
                    cons, labels)  # fmt: skip


def _nl_text(prob, x0=None, defined_vars=True):
    g = prob.g
    nv, m = len(prob.lo), len(prob.cons)
    order = np.concatenate([np.flatnonzero(~prob.integer), np.flatnonzero(prob.integer)])  # nl index -> var
    pos = np.empty(nv, np.int64)
    pos[order] = np.arange(nv)
    n_int = int(prob.integer.sum())
    roots = [prob.obj] + [c[0] for c in prob.cons]
    reach = _reachable(g, roots)
    uses = Counter(roots)
    deps = {}
    for i in reach:
        op, args, p = g.nodes[i]
        for k in args:
            uses[k] += 1
        deps[i] = (1 << int(pos[p])) if op == "var" else 0
        for k in args:
            deps[i] |= deps[k]
    defined = (
        [i for i in reach if uses[i] > 1 and g.nodes[i][0] not in ("var", "const") and g.nodes[i][0] not in LOGICAL]
        if defined_vars
        else []
    )
    # ASL classes of defined variables, numbered in this order: used by the objective and constraints (b),
    # by several constraints (c), then those used by a single constraint (c1) or the objective only (o1),
    # each written just before its C / O segment (Gay, "Writing .nl files"; Couenne relies on this)
    users = {i: 0 for i in reach}  # bit 0: objective, bit k + 1: constraint k
    users[prob.obj] |= 1
    for k, c in enumerate(prob.cons):
        users[c[0]] |= 1 << (k + 1)
    for i in reversed(reach):
        for k in g.nodes[i][1]:
            users[k] |= users[i]
    comb, comc, single = [], [], {}  # single: segment (k + 1 for constraint k, 0 objective) -> nodes
    for i in defined:
        u = users[i]
        if u & 1 and u >> 1:
            comb.append(i)
        elif u == 1:
            single.setdefault(0, []).append(i)
        elif u & (u - 1):
            comc.append(i)
        else:
            single.setdefault(u.bit_length() - 1, []).append(i)
    seq = comb + comc + [i for k in range(1, m + 1) for i in single.get(k, ())] + single.get(0, [])
    vidx = {i: nv + k for k, i in enumerate(seq)}
    n_c1 = sum(len(single.get(k, ())) for k in range(1, m + 1))

    def expr(root, out, top=True):
        """Prefix form of ``root``; logical nodes in numeric context become ``if(p, 1, 0)``."""
        stack = [(root, False, top)]
        while stack:
            item, logical_ctx, is_top = stack.pop()
            if isinstance(item, str):
                out.append(item)
                continue
            i = item
            op, args, p = g.nodes[i]
            if i in vidx and not is_top:
                out.append(f"v{vidx[i]}")
                continue
            if op == "var":
                out.append(f"v{pos[p]}")
                continue
            if op == "const":
                out.append(f"n{_fmt(p)}")
                continue
            if op in LOGICAL and not logical_ctx:  # numeric use of a 0/1 value
                out.append("o35")
                stack += [("n0", False, False), ("n1", False, False), (i, True, False)]
                continue
            if op == "sum" and len(args) <= 2:  # ASL's sumlist (o54) takes 3+ operands
                if len(args) == 2:
                    out.append("o0")
                stack += [(k, False, False) for k in reversed(args)]
                continue
            if op in ("sum", "max", "min"):
                out.append({"sum": "o54", "max": "o12", "min": "o11"}[op])
                out.append(str(len(args)))
                stack += [(k, False, False) for k in reversed(args)]
                continue
            if op == "select":  # if p then t else f
                out.append("o35")
                stack += [(args[1], False, False), (args[2], False, False), (args[0], True, False)]
                continue
            out.append(f"o{NL_OPS[op]}")
            logical_args = op in ("and", "or", "not")
            stack += [(k, logical_args, False) for k in reversed(args)]
        return out

    def bitlist(d):
        out, j = [], 0
        while d:
            if d & 1:
                out.append(j)
            d >>= 1
            j += 1
        return out

    nranges = sum(1 for _, lo, hi, _ in prob.cons if np.isfinite(lo) and np.isfinite(hi) and lo != hi)
    neq = sum(1 for _, lo, hi, _ in prob.cons if lo == hi)
    jac = [bitlist(deps[c[0]]) for c in prob.cons]
    grad = bitlist(deps[prob.obj])
    if m:
        nlvc = nlvo = nlvb = nv
        nlvbi, nlvci, nlvoi = n_int, 0, 0
    else:
        nlvc = nlvb = 0
        nlvo = nv
        nlvbi, nlvci, nlvoi = 0, 0, n_int
    L = [
        "g3 1 1 0\t# problem jaxglobal",
        f" {nv} {m} 1 {nranges} {neq} 0\t# vars, constraints, objectives, ranges, eqns, lcons",
        f" {m} 1\t# nonlinear constraints, objectives",
        " 0 0\t# network constraints: nonlinear, linear",
        f" {nlvc} {nlvo} {nlvb}\t# nonlinear vars in constraints, objectives, both",
        " 0 0 0 1\t# linear network variables; functions; arith, flags",
        f" 0 0 {nlvbi} {nlvci} {nlvoi}\t# discrete variables: binary, integer, nonlinear (b,c,o)",
        f" {sum(map(len, jac))} {len(grad)}\t# nonzeros in Jacobian, gradients",
        " 0 0\t# max name lengths: constraints, variables",
        f" {len(comb)} {len(comc)} 0 {n_c1} {len(single.get(0, ()))}\t# common exprs: b,c,o,c1,o1",
    ]
    for i in comb + comc:
        L.append(f"V{vidx[i]} 0 0")
        expr(i, L, top=True)
    for k, (node, _, _, _) in enumerate(prob.cons):
        for i in single.get(k + 1, ()):
            L.append(f"V{vidx[i]} 0 {k + 1}")
            expr(i, L, top=True)
        L.append(f"C{k}")
        expr(node, L)
    for i in single.get(0, ()):
        L.append(f"V{vidx[i]} 0 {m + 1}")
        expr(i, L, top=True)
    L.append("O0 0")
    expr(prob.obj, L)
    if x0 is None:
        x0 = np.where(np.isfinite(prob.lo) & np.isfinite(prob.hi), 0.5 * (prob.lo + prob.hi),
                      np.clip(0.0, prob.lo, prob.hi))  # fmt: skip
    else:
        x0 = np.concatenate([np.asarray(x0, float), np.clip(0.0, prob.lo[prob.n :], prob.hi[prob.n :])])
    L.append(f"x{nv}")
    L += [f"{k} {_fmt(x0[order[k]])}" for k in range(nv)]
    if m:
        L.append("r")
        for _, lo, hi, _ in prob.cons:
            L.append(_nl_range(lo, hi))
    L.append("b")
    for k in range(nv):
        L.append(_nl_range(prob.lo[order[k]], prob.hi[order[k]]))
    if m:
        counts = np.zeros(nv, np.int64)
        for js in jac:
            counts[js] += 1
        L.append(f"k{nv - 1}")
        L += [str(c) for c in np.cumsum(counts)[:-1]]
        for k, js in enumerate(jac):
            L.append(f"J{k} {len(js)}")
            L += [f"{j} 0" for j in js]
    if grad:
        L.append(f"G0 {len(grad)}")
        L += [f"{j} 0" for j in grad]
    return "\n".join(L) + "\n", order


def _nl_range(lo, hi):
    if lo == hi:
        return f"4 {_fmt(lo)}"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"0 {_fmt(lo)} {_fmt(hi)}"
    if np.isfinite(hi):
        return f"1 {_fmt(hi)}"
    if np.isfinite(lo):
        return f"2 {_fmt(lo)}"
    return "3"

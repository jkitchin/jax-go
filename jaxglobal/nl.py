"""Read AMPL ``.nl`` files (text format) into JAX functions.

``read_nl(path)`` returns an :class:`NLModel` with ``f``, ``ineq``, ``eq`` and
the variable bounds in the form ``minimize`` takes, so any model written by
AMPL, Pyomo, GAMS (convert) or downloaded from MINLPLib becomes a jaxglobal
problem. The expression trees are compiled to plain ``jax.numpy`` code: the
solver sees an ordinary traced function, exactly as for hand-written models.

Supported: text ("g") and binary ("b") formats, nonlinear expressions, linear
parts (J/G segments), defined variables (V segments), ranges, bounds and
initial points. Not supported: imported functions, suffixes, logical
constraints, complementarity, and integer variables (counted, not solved).
Reference: D. M. Gay, "Writing .nl Files" (Sandia report SAND2005-7907P).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

INF = 1e19  # values at or beyond this are infinite in .nl files

# opcode -> (name, arity); arity None means n-ary (count on the next line)
OPS = {
    0: ("add", 2), 1: ("sub", 2), 2: ("mul", 2), 3: ("div", 2), 4: ("rem", 2), 5: ("pow", 2),
    6: ("less", 2), 11: ("min", None), 12: ("max", None), 13: ("floor", 1), 14: ("ceil", 1),
    15: ("abs", 1), 16: ("neg", 1), 20: ("or", 2), 21: ("and", 2), 22: ("lt", 2), 23: ("le", 2),
    24: ("eq", 2), 28: ("ge", 2), 29: ("gt", 2), 30: ("ne", 2), 34: ("not", 1), 35: ("if", 3),
    37: ("tanh", 1), 38: ("tan", 1), 39: ("sqrt", 1), 40: ("sinh", 1), 41: ("sin", 1),
    42: ("log10", 1), 43: ("log", 1), 44: ("exp", 1), 45: ("cosh", 1), 46: ("cos", 1),
    47: ("atanh", 1), 48: ("atan2", 2), 49: ("atan", 1), 50: ("asinh", 1), 51: ("asin", 1),
    52: ("acosh", 1), 53: ("acos", 1), 54: ("sum", None), 57: ("round", 2), 58: ("trunc", 2),
    75: ("pow", 2), 76: ("sqr", 1), 77: ("pow", 2),
}  # fmt: skip

UNARY = {
    "floor": jnp.floor, "ceil": jnp.ceil, "abs": jnp.abs, "neg": jnp.negative, "tanh": jnp.tanh,
    "tan": jnp.tan, "sqrt": jnp.sqrt, "sinh": jnp.sinh, "sin": jnp.sin, "log10": jnp.log10,
    "log": jnp.log, "exp": jnp.exp, "cosh": jnp.cosh, "cos": jnp.cos, "atanh": jnp.arctanh,
    "atan": jnp.arctan, "asinh": jnp.arcsinh, "asin": jnp.arcsin, "acosh": jnp.arccosh,
    "acos": jnp.arccos, "sqr": lambda a: a**2, "not": lambda a: jnp.where(a != 0, 0.0, 1.0),
}  # fmt: skip
BINARY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b, "rem": jnp.fmod, "less": lambda a, b: jnp.maximum(a - b, 0.0),
    "atan2": jnp.arctan2, "lt": lambda a, b: jnp.where(a < b, 1.0, 0.0),
    "le": lambda a, b: jnp.where(a <= b, 1.0, 0.0), "eq": lambda a, b: jnp.where(a == b, 1.0, 0.0),
    "ge": lambda a, b: jnp.where(a >= b, 1.0, 0.0), "gt": lambda a, b: jnp.where(a > b, 1.0, 0.0),
    "ne": lambda a, b: jnp.where(a != b, 1.0, 0.0),
    "or": lambda a, b: jnp.where((a != 0) | (b != 0), 1.0, 0.0),
    "and": lambda a, b: jnp.where((a != 0) & (b != 0), 1.0, 0.0),
}  # fmt: skip


class NLError(ValueError):
    """The file uses a feature this reader does not support."""


@dataclass
class NLModel:
    name: str
    n: int
    lo: np.ndarray
    hi: np.ndarray
    x0: np.ndarray
    f: callable  # minimization objective (negated when the model maximizes)
    ineq: callable | None  # ineq(x) <= 0
    eq: callable | None  # eq(x) == 0
    sense: int  # +1 minimize, -1 maximize: model objective = sense * f(x)
    bodies: callable  # raw constraint bodies, for checking against other readers
    objective: callable  # raw model objective (not negated)
    n_int: int = 0
    info: dict = field(default_factory=dict)

    @property
    def bounded(self):
        return bool(np.all(np.isfinite(self.lo)) and np.all(np.isfinite(self.hi)))


NARGS = {"C": 1, "O": 2, "V": 3, "r": 0, "b": 0, "J": 2, "G": 2, "x": 1, "d": 1, "k": 1, "S": 2, "F": 4, "L": 1}


class _Text:
    """Token stream over the text ("g") format; comments are dropped."""

    def __init__(self, body):
        self.tok = [t for ln in body.splitlines() for t in ln.split("#", 1)[0].split()]
        self.i = 0

    def _next(self):
        self.i += 1
        return self.tok[self.i - 1]

    def done(self):
        return self.i >= len(self.tok)

    def int(self):
        return int(self._next())

    def float(self):
        return float(self._next())

    def seg(self):
        t = self._next()
        c = t[0]
        return c, [int(t[1:])] + [self.int() for _ in range(NARGS.get(c, 1) - 1)] if NARGS.get(c, 1) else []

    def leaf(self):
        t = self._next()
        return t[0], t[1:]

    def code(self):
        return self.int()


class _Binary:
    """Token stream over the binary ("b") format: 1-byte tags, int32, float64."""

    def __init__(self, body):
        self.b, self.i = body, 0

    def done(self):
        return self.i >= len(self.b)

    def _unpack(self, fmt, size):
        v = struct.unpack_from(fmt, self.b, self.i)[0]
        self.i += size
        return v

    def int(self):
        return self._unpack("<i", 4)

    def float(self):
        return self._unpack("<d", 8)

    def seg(self):
        c = chr(self.b[self.i])
        self.i += 1
        return c, [self.int() for _ in range(NARGS.get(c, 1))]

    def leaf(self):
        c = chr(self.b[self.i])
        self.i += 1
        if c == "n":
            return c, self.float()
        if c == "s":
            return c, self._unpack("<h", 2)
        if c == "l":
            return c, self._unpack("<q", 8)
        return c, self.int()

    def code(self):
        c = self.b[self.i]
        self.i += 1
        return c - ord("0")


def _parse_expr(R):
    c, v = R.leaf()
    if c in "nsl":
        return ("num", float(v))
    if c == "v":
        return ("var", int(v))
    if c == "o":
        code = int(v)
        if code not in OPS:
            raise NLError(f"unsupported .nl operator o{code}")
        name, arity = OPS[code]
        if arity is None:
            arity = R.int()
        return (name, *(_parse_expr(R) for _ in range(arity)))
    if c == "f":
        raise NLError("imported functions are not supported")
    if c == "h":
        raise NLError("string arguments are not supported")
    raise NLError(f"bad expression token {c}{v!r}")


def _range(R):
    """One range/bound entry -> (lo, hi) with +-inf."""
    k = R.code()
    if k == 0:
        lo, hi = R.float(), R.float()
    elif k == 1:
        lo, hi = -np.inf, R.float()
    elif k == 2:
        lo, hi = R.float(), np.inf
    elif k == 3:
        lo, hi = -np.inf, np.inf
    elif k == 4:
        lo = hi = R.float()
    else:
        raise NLError("complementarity constraints are not supported")
    lo = -np.inf if lo <= -INF else lo
    hi = np.inf if hi >= INF else hi
    return lo, hi


def parse_nl(data, name="model"):
    """Parse ``.nl`` contents (bytes or str, text or binary format)."""
    if isinstance(data, str):
        data = data.encode()
    head, body = [], data
    for _ in range(10):  # the 10-line header is text in both formats
        ln, body = body.split(b"\n", 1)
        head.append([int(t) for t in ln.decode().split("#", 1)[0].split()[(1 if not head else 0):]])
    kind = chr(data[0])
    if kind == "g":
        R = _Text(body.decode())
    elif kind == "b":
        R = _Binary(body)
    else:
        raise NLError(f"unknown .nl format {kind!r}")
    n, m, n_obj = head[1][:3]
    nbv, niv = head[6][:2]
    ncom = sum(head[9])

    exprs_c = [("num", 0.0)] * m
    lin_c = [dict() for _ in range(m)]
    exprs_o, lin_o, sense = [("num", 0.0)] * n_obj, [dict() for _ in range(n_obj)], [0] * n_obj
    defined = {}  # index (>= n) -> (linear dict, expr)
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    clo, chi = np.full(m, -np.inf), np.full(m, np.inf)
    x0 = np.zeros(n)

    def pairs(k):
        return {R.int(): R.float() for _ in range(k)}

    while not R.done():
        c, args = R.seg()
        if c == "C":
            exprs_c[args[0]] = _parse_expr(R)
        elif c == "O":
            sense[args[0]] = args[1]
            exprs_o[args[0]] = _parse_expr(R)
        elif c == "V":
            lin = pairs(args[1])
            defined[args[0]] = (lin, _parse_expr(R))
        elif c == "r":
            for i in range(m):
                clo[i], chi[i] = _range(R)
        elif c == "b":
            for i in range(n):
                lo[i], hi[i] = _range(R)
        elif c in "JG":
            (lin_c if c == "J" else lin_o)[args[0]].update(pairs(args[1]))
        elif c == "x":
            for j, v in pairs(args[0]).items():
                x0[j] = v
        elif c == "d":
            pairs(args[0])  # dual initial guess: not needed
        elif c == "k":
            for _ in range(args[0]):
                R.int()
        elif c == "S":
            raise NLError("suffixes are not supported")
        elif c == "F":
            raise NLError("imported functions are not supported")
        elif c == "L":
            raise NLError("logical constraints are not supported")
        else:
            raise NLError(f"unknown segment {c!r}")
    if len(defined) != ncom:
        raise NLError(f"expected {ncom} defined variables, found {len(defined)}")
    if n_obj > 1:
        raise NLError("multiple objectives are not supported")

    def compile_(x):
        """Evaluate all model expressions at x (one trace; defined vars cached)."""
        cache = {}

        def ev(e):
            op = e[0]
            if op == "num":
                return e[1]
            if op == "var":
                j = e[1]
                if j < n:
                    return x[j]
                if j not in cache:
                    lin, ex = defined[j]
                    cache[j] = ev(ex) + sum(a * ev(("var", k)) for k, a in lin.items())
                return cache[j]
            args = e[1:]
            if op == "sum":
                return sum(ev(a) for a in args)
            if op in ("min", "max"):
                fn = jnp.minimum if op == "min" else jnp.maximum
                out = ev(args[0])
                for a in args[1:]:
                    out = fn(out, ev(a))
                return out
            if op == "if":
                return jnp.where(ev(args[0]) != 0, ev(args[1]), ev(args[2]))
            if op == "pow":
                b = args[1]
                if b[0] == "num":  # constant exponent: integer powers trace as integer_pow
                    p = b[1]
                    return ev(args[0]) ** (int(p) if float(p).is_integer() else p)
                return ev(args[0]) ** ev(b)
            if op in ("round", "trunc"):
                raise NLError(f"{op} is not supported")
            if op in UNARY:
                return UNARY[op](ev(args[0]))
            return BINARY[op](ev(args[0]), ev(args[1]))

        def linear(lin):
            return sum(a * x[j] for j, a in lin.items())

        obj = (ev(exprs_o[0]) + linear(lin_o[0])) if n_obj else 0.0
        body = [ev(exprs_c[i]) + linear(lin_c[i]) for i in range(m)]
        return obj, body

    def objective(x):
        return jnp.asarray(compile_(x)[0], dtype=x.dtype)

    def bodies(x):
        return jnp.stack([jnp.asarray(b, dtype=x.dtype) for b in compile_(x)[1]]) if m else jnp.zeros(0, x.dtype)

    sgn = -1 if (n_obj and sense[0] == 1) else 1

    def f(x):
        return sgn * objective(x)

    eq_i = [i for i in range(m) if clo[i] == chi[i]]
    up_i = [i for i in range(m) if clo[i] != chi[i] and np.isfinite(chi[i])]
    lo_i = [i for i in range(m) if clo[i] != chi[i] and np.isfinite(clo[i])]

    def eq(x):
        b = bodies(x)
        return b[jnp.asarray(eq_i)] - jnp.asarray(clo[eq_i])

    def ineq(x):
        b = bodies(x)
        return jnp.concatenate([b[jnp.asarray(up_i, dtype=int)] - jnp.asarray(chi[up_i]),
                                jnp.asarray(clo[lo_i]) - b[jnp.asarray(lo_i, dtype=int)]])  # fmt: skip

    return NLModel(
        name=name, n=n, lo=lo, hi=hi, x0=x0, f=f,
        ineq=ineq if (up_i or lo_i) else None, eq=eq if eq_i else None,
        sense=sgn, bodies=bodies, objective=objective, n_int=nbv + niv,
        info=dict(m=m, n_eq=len(eq_i), n_ineq=len(up_i) + len(lo_i), n_defined=ncom),
    )  # fmt: skip


def read_nl(path):
    import pathlib

    p = pathlib.Path(path)
    return parse_nl(p.read_bytes(), name=p.stem)

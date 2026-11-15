"""Host dual simplex for the lifted node LP (``minimize(node_lp="simplex")``; off by default).

``quadratic.make_quadratic_bounder`` solves each round's LP ``min c.u, A u <= b, 0 <= u <= 1``
with its batched interior-point method (``quadratic._ipm``). With ``node_lp="simplex"`` the round
instead calls ``lp_callback``: a ``jax.pure_callback`` that solves the LP of every box of the
batch on the host with a dual simplex (HiGHS through ``highspy``, or SoPlex through
``pyscipopt.LP``; both optional), started from a basis:

* round 0 of a node builds the LP of its nonempty rows and starts from the basis its parent's
  round 0 ended with (children inherit it: ``Store``);
* rounds after the first add the rows that became nonempty (the round's tangent and pool rows)
  to the same LP object, kept between the callback's calls as a session whose id the bounder
  carries through its rounds, and re-solve from its current basis (the new rows' slacks basic);
* the basis a node's round 0 ends with is kept for its children.

The simplex's objective is never used as a bound. The callback returns the row duals ``y``;
the bounder evaluates ``quadratic._certificate(y)`` (weak duality with its rounding margins,
``safe`` included), which is a valid bound for any ``y``. An infeasible LP returns a dual ray,
whose scaled certificates are taken the same way (a box is pruned only when one reaches the
cutoff). A box whose solve failed (singular basis, iteration limit, solver error, no solver)
falls back to the IPM for that round.

Bases travel outside the kernel's arguments: the callback finds a box's parent basis in
``Store.inherit`` by the bytes of the box the kernel was called with, and leaves its own in
``Store.out``; ``bnb.minimize`` moves them between nodes (``Store.give`` / ``Store.take``). A
basis that is missing, stale or from another box only changes the simplex's starting point,
never a bound.
"""

from __future__ import annotations

import os
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

# status codes returned by the callback
OPTIMAL, INFEASIBLE, SKIPPED, FAILED = 0, 1, 2, 3
# basis status codes (the same in HiGHS and SCIP's LP interface)
LOWER, BASIC, UPPER = 0, 1, 2
SOLVER = os.environ.get("JAXGLOBAL_SIMPLEX", "auto")  # "highs", "soplex" or "auto" (the first installed)
ITER_FACTOR = 20  # simplex iteration limit: this many times (rows + columns)


def available(solver=None):
    """The host simplex ``solver`` (default ``SOLVER``) resolved to "highs" / "soplex", or None
    when it is not installed."""
    solver = SOLVER if solver is None else solver
    order = ("highs", "soplex") if solver == "auto" else (solver,)
    for s in order:
        try:
            if s == "highs":
                import highspy  # noqa: F401
            elif s == "soplex":
                import pyscipopt  # noqa: F401
            else:
                raise ValueError(f"unknown simplex solver {s!r} (highs, soplex or auto)")
            return s
        except ImportError:
            continue
    return None


def require(solver=None):
    """``available(solver)``, or an ImportError naming the packages."""
    s = available(solver)
    if s is None:
        raise ImportError("node_lp='simplex' needs highspy (pip install highspy) or pyscipopt "
                          "(pip install pyscipopt); neither is installed")
    return s


class Store:
    """Bases of a search, keyed by the bytes of the box a node was bounded on.

    ``give(lo, hi, bases)`` before a kernel call: box ``i`` starts from ``bases[i]`` (None: cold).
    ``take(lo, hi)`` after it: the basis each box's first LP ended with (None if it had none).
    The callback reads ``inherit`` and writes ``out`` (dict operations hold the GIL)."""

    MAX = 1 << 16  # entries not taken (calls outside ``evaluate``) are dropped beyond this

    def __init__(self):
        self.inherit, self.out = {}, {}
        self.stats = np.zeros(6, np.int64)  # solves by status (4), simplex iterations, warm starts
        self.seconds = np.zeros(2)  # in the callback, in the solver (summed over threads)
        self.lock = threading.Lock()

    @staticmethod
    def key(lo, hi):
        return np.ascontiguousarray(lo, np.float64).tobytes() + np.ascontiguousarray(hi, np.float64).tobytes()

    def clear(self):
        self.inherit.clear()
        self.out.clear()

    def give(self, lo, hi, bases):
        if bases is None:
            return
        if len(self.inherit) > self.MAX:
            self.inherit.clear()
        for i in range(len(lo)):
            if bases[i] is not None:
                self.inherit[self.key(lo[i], hi[i])] = bases[i]

    def take(self, lo, hi):
        res = np.empty(len(lo), dtype=object)
        for i in range(len(lo)):
            key = self.key(lo[i], hi[i])
            res[i] = self.out.pop(key, None)
            self.inherit.pop(key, None)
        if len(self.out) > self.MAX:
            self.out.clear()
        return res

    def count(self, status, iters, warm):
        with self.lock:
            self.stats[status] += 1
            self.stats[4] += iters
            self.stats[5] += warm


def _repair(cs, rs):
    """A basis of the right count (as many basic as rows) from column / row statuses ``cs``,
    ``rs``: the last basic candidates (rows, then columns) made nonbasic, or nonbasic row slacks
    (then columns) made basic. A singular result is left to the solver's own repair."""
    cs, rs = cs.copy(), rs.copy()
    m = rs.size
    nb = int(np.count_nonzero(cs == BASIC) + np.count_nonzero(rs == BASIC))
    if nb > m:
        bc, br = np.flatnonzero(cs == BASIC), np.flatnonzero(rs == BASIC)
        drop = nb - m
        kr = min(drop, br.size)
        rs[br[br.size - kr:]] = UPPER  # (a row is nonbasic at its bound b)
        if drop > kr:
            cs[bc[bc.size - (drop - kr):]] = LOWER
    elif nb < m:
        nr, nc = np.flatnonzero(rs != BASIC), np.flatnonzero(cs != BASIC)
        add = m - nb
        kr = min(add, nr.size)
        rs[nr[:kr]] = BASIC
        if add > kr:
            cs[nc[: add - kr]] = BASIC
    return cs, rs


class _HighsLP:
    """A HiGHS LP ``min c.u, A u <= b, 0 <= u <= 1``: dual simplex, no presolve, one thread. Rows
    added later (``add``) enter with their slacks basic, and ``solve`` restarts from the current
    basis (hot start). HiGHS releases the GIL while it solves."""

    _free = threading.local()  # instances are reused (creating one costs ~80 us)

    def __init__(self):
        import highspy

        self.hs = highspy
        pool = getattr(self._free, "pool", None)
        if pool:
            self.h = pool.pop()
        else:
            h = highspy._core._Highs()  # (the Python subclass enables callbacks: slower runs)
            for k, v in (("output_flag", False), ("presolve", "off"), ("threads", 1), ("solver", "simplex"),
                         ("simplex_strategy", 1), ("simplex_scale_strategy", 0)):  # fmt: skip
                h.setOptionValue(k, v)
            self.h = h
        self.S = sorted(highspy.HighsBasisStatus.__members__.values(), key=int)

    def release(self):
        pool = getattr(self._free, "pool", None)
        if pool is None:
            pool = self._free.pool = []
        if len(pool) < 64:
            pool.append(self.h)
        self.h = None

    def build(self, c, start, idx, val, b, basis, iter_limit):
        h, hs = self.h, self.hs
        N, m = c.size, b.size
        self.N, self.m = N, m
        h.setOptionValue("simplex_iteration_limit", int(iter_limit))
        st = h.passModel(N, m, val.size, 2, 1, 0.0, c, np.zeros(N), np.ones(N), np.full(m, -hs.kHighsInf), b,
                         start, idx, val, np.zeros(N, np.int32))  # fmt: skip
        if st == hs.HighsStatus.kError:
            return False
        if basis is not None:
            hb = hs.HighsBasis()
            hb.col_status = [self.S[v] for v in basis[0].tolist()]
            hb.row_status = [self.S[v] for v in basis[1].tolist()]
            hb.valid = True
            h.setBasis(hb)
        return True

    def add(self, start, idx, val, b):
        k = b.size
        st = self.h.addRows(k, np.full(k, -self.hs.kHighsInf), b, val.size, start, idx, val)
        self.m += k
        return st != self.hs.HighsStatus.kError

    def solve(self):
        """(status, z, y, iterations): ``y`` the duals of the rows (>= 0), or a dual ray."""
        h, hs = self.h, self.hs
        if h.run() == hs.HighsStatus.kError:
            return FAILED, None, None, 0
        ms = h.getModelStatus()
        it = int(h.getInfo().simplex_iteration_count)
        if ms == hs.HighsModelStatus.kOptimal:
            sol = h.getSolution()
            return OPTIMAL, np.asarray(sol.col_value), -np.asarray(sol.row_dual), it
        if ms == hs.HighsModelStatus.kInfeasible:
            _, has, ray = h.getDualRay()
            if has:
                return INFEASIBLE, None, np.asarray(ray, np.float64), it
        return FAILED, None, None, it

    def basis(self):
        hb = self.h.getBasis()
        return (np.fromiter(map(int, hb.col_status), np.int8, self.N),
                np.fromiter(map(int, hb.row_status), np.int8, self.m))  # fmt: skip


class _SoplexLP:
    """The same through SCIP's LP interface (SoPlex). It holds the GIL while it solves."""

    def __init__(self):
        from pyscipopt import LP

        self.lp = LP()

    def release(self):
        self.lp = None

    def _rows(self, start, idx, val):
        il, vl, s = idx.tolist(), val.tolist(), start.tolist() + [len(idx)]
        return [list(zip(il[s[r]:s[r + 1]], vl[s[r]:s[r + 1]])) for r in range(len(start))]

    def build(self, c, start, idx, val, b, basis, iter_limit):
        lp = self.lp
        N = c.size
        lp.addCols([[] for _ in range(N)], objs=c.tolist(), lbs=[0.0] * N, ubs=[1.0] * N)
        lp.addRows(self._rows(start, idx, val), lhss=[-lp.infinity()] * b.size, rhss=b.tolist())
        lp.setIntParam(10, int(iter_limit))  # SCIP_LPPAR_LPITLIM
        if basis is not None:
            lp.setBase(basis[0].tolist(), basis[1].tolist())
        return True

    def add(self, start, idx, val, b):
        self.lp.addRows(self._rows(start, idx, val), lhss=[-self.lp.infinity()] * b.size, rhss=b.tolist())
        return True

    def solve(self):
        lp = self.lp
        try:
            lp.solve(dual=True)
        except Exception:
            return FAILED, None, None, 0
        it = int(lp.getNIterations())
        if lp.isOptimal():
            return OPTIMAL, np.asarray(lp.getPrimal()), -np.asarray(lp.getDual()), it
        try:
            if not lp.isPrimalFeasible():
                ray = np.asarray(lp.getDualRay(), np.float64)
                if np.all(np.isfinite(ray)):
                    return INFEASIBLE, None, ray, it
        except Exception:
            pass
        return FAILED, None, None, it

    def basis(self):
        cs, rs = self.lp.getBase()
        return np.asarray(cs, np.int8), np.asarray(rs, np.int8)


class _Session:
    """A box's LP across the rounds of one kernel call: the solver, the pattern rows it holds (in
    its row order) and the values they were built from."""

    __slots__ = ("lp", "live", "vals", "cols", "b")


_SESSIONS = {}  # session id -> _Session (ids are carried through the bounder's rounds)
_SESSION_LOCK = threading.Lock()
_NEXT_SID = [0]
MAX_SESSIONS = 4096  # sessions left open by a call that never finished are dropped beyond this


def _open(sess):
    with _SESSION_LOCK:
        _NEXT_SID[0] += 1
        sid = _NEXT_SID[0]
        if len(_SESSIONS) >= MAX_SESSIONS:
            _SESSIONS.clear()
        _SESSIONS[sid] = sess
    return sid


def _close(sid):
    with _SESSION_LOCK:
        sess = _SESSIONS.pop(int(sid), None)
    if sess is not None:
        sess.lp.release()


def lp_callback(rows, m, N, store, n_rounds, solver=None):
    """``solve(key_lo, key_hi, k, c, vals, cols, b, active, basis, sid) -> (z, y, status, basis,
    sid)``, traceable and batched under ``vmap``, for round ``k`` (of ``0 .. n_rounds``) of the
    LP ``min c.u, A u <= b, 0 <= u <= 1`` whose nonzeros ``vals`` sit in rows ``rows`` (static)
    and columns ``cols``. ``y`` are the duals of the ``A`` rows (>= 0 at an optimum; 0 for empty
    rows), or a dual ray of an infeasible LP.

    Round 0 builds the LP of the nonempty rows, started from ``store``'s basis for the box
    ``(key_lo, key_hi)`` (``basis``: int8, columns then rows; any value < 0: none), and leaves
    its end basis there and in ``basis``. It returns a session id ``sid``: later rounds add the
    rows that became nonempty to that LP and re-solve from its current basis; a round whose
    earlier rows changed, or without its session, rebuilds from ``basis``. The session closes
    after round ``n_rounds`` or once the box is inactive."""
    name = require(solver)
    make = _HighsLP if name == "highs" else _SoplexLP
    rows = np.asarray(rows, np.int64)
    nb = N + m

    def csr(r, c, v):  # entries sorted by (row, column) -> (row starts, columns, values), duplicates summed
        key = r * N + c
        o = np.argsort(key, kind="stable")
        key, v = key[o], v[o]
        if key.size > 1 and np.any(key[1:] == key[:-1]):
            key, inv = np.unique(key, return_inverse=True)
            v = np.bincount(inv, v)
        return key // N, (key % N).astype(np.int32), v

    def build(sess, ci, vi, coli, bi, live, start_basis):
        nz = vi != 0.0
        rr, cc, vv = csr(rows[nz], coli[nz], vi[nz])
        pos = np.searchsorted(live, rr)  # row of the LP for each entry
        start = np.searchsorted(pos, np.arange(live.size)).astype(np.int32)
        bs = None
        if start_basis is not None:
            bs = _repair(start_basis[:N], start_basis[N:][live])
        sess.lp = make()
        sess.live, sess.vals, sess.cols, sess.b = live, vi, coli, bi
        return sess.lp.build(ci, start, cc, vv, bi[live], bs, ITER_FACTOR * (N + live.size))

    def host(key_lo, key_hi, k, c, vals, cols, b, active, basis, sid):
        t_in, t_solve = time.perf_counter(), 0.0
        c, vals, b = np.asarray(c, np.float64), np.asarray(vals, np.float64), np.asarray(b, np.float64)
        cols, active, basis, k, sid = np.asarray(cols), np.asarray(active), np.asarray(basis), np.asarray(k), np.asarray(sid)
        key_lo, key_hi = np.asarray(key_lo), np.asarray(key_hi)
        shp = c.shape[:-1]
        B = int(np.prod(shp, dtype=np.int64))
        c, vals, cols, b = c.reshape(B, N), vals.reshape(B, -1), cols.reshape(B, -1).astype(np.int64), b.reshape(B, m)
        active, basis = active.reshape(B), basis.reshape(B, nb)
        k, sid = np.broadcast_to(k, shp).reshape(B), np.broadcast_to(sid, shp).reshape(B).copy()
        key_lo, key_hi = key_lo.reshape(B, -1), key_hi.reshape(B, -1)
        Z, Y = np.full((B, N), 0.5), np.zeros((B, m))
        S, BS = np.full(B, SKIPPED, np.int32), basis.astype(np.int8).copy()
        seen = {}
        for i in range(B):
            if not active[i]:
                if sid[i] > 0:
                    _close(sid[i])
                    sid[i] = 0
                continue
            bkey = Store.key(key_lo[i], key_hi[i])
            j = seen.setdefault(bkey, i)
            if j != i:  # a copy of an earlier box (the padding of a call): the same LP
                Z[i], Y[i], S[i], BS[i] = Z[j], Y[j], S[j], BS[j]
                sid[i] = 0  # (the session belongs to box j)
                continue
            vi, bi, ci, coli = vals[i], b[i], c[i], cols[i]
            if not np.all(np.isfinite(ci)) or np.any(np.isnan(bi)) or np.any(bi == -np.inf) or not np.all(np.isfinite(vi)):
                S[i] = FAILED
                continue
            bi = np.where(np.isfinite(bi), bi, 1e30)
            livem = np.zeros(m, bool)
            livem[rows[vi != 0.0]] = True  # nonempty rows (dead and idle slots are all zero)
            live = np.flatnonzero(livem)
            sess = _SESSIONS.get(int(sid[i])) if k[i] > 0 and sid[i] > 0 else None
            warm = False
            t1 = time.perf_counter()
            try:
                if sess is not None:  # rows already in the LP must be as they were: add the new ones
                    oldm = np.zeros(m, bool)
                    oldm[sess.live] = True
                    old = oldm[rows]
                    new = np.flatnonzero(livem & ~oldm)
                    same = np.array_equal(vi[old], sess.vals[old]) and np.array_equal(coli[old], sess.cols[old]) \
                        and np.array_equal(bi[sess.live], sess.b[sess.live]) and bool(livem[sess.live].all())  # fmt: skip
                    if not same:
                        _close(sid[i])
                        sid[i], sess = 0, None
                    elif new.size:
                        nz = (livem & ~oldm)[rows] & (vi != 0.0)
                        rr, cc, vv = csr(rows[nz], coli[nz], vi[nz])
                        pos = np.searchsorted(new, rr)
                        ok = sess.lp.add(np.searchsorted(pos, np.arange(new.size)).astype(np.int32), cc, vv, bi[new])
                        sess.live = np.concatenate([sess.live, new])
                        sess.vals, sess.cols, sess.b = vi, coli, bi
                        if not ok:
                            raise RuntimeError("add rows")
                    warm = True
                if sess is None:
                    start = basis[i] if k[i] > 0 and basis[i][0] >= 0 else None
                    if k[i] == 0 and store is not None:
                        start = store.inherit.get(bkey)
                    start = start if start is not None and start.size == nb else None
                    warm = start is not None
                    sess = _Session()
                    if not build(sess, ci, vi, coli, bi, live, start):
                        raise RuntimeError("build")
                    if sid[i] > 0:
                        _close(sid[i])
                    sid[i] = _open(sess)
                st, z, y, it = sess.lp.solve()
                if st == OPTIMAL and k[i] == 0:  # the node's basis, for its children (and a rebuild)
                    cs, rs = sess.lp.basis()
                    full = np.full(nb, BASIC, np.int8)
                    full[:N], full[N + sess.live] = cs, rs
                    BS[i] = full
                    if store is not None:
                        store.out[bkey] = full
            except Exception:
                st, z, y, it = FAILED, None, None, 0
            t_solve += time.perf_counter() - t1
            S[i] = st
            if store is not None:
                store.count(st, it, warm)
            if st == OPTIMAL:
                Z[i] = np.clip(z, 0.0, 1.0)
            if st in (OPTIMAL, INFEASIBLE):
                Y[i, sess.live] = y[: sess.live.size]
            if st == FAILED or k[i] >= n_rounds:
                if sid[i] > 0:
                    _close(sid[i])
                sid[i] = 0
        if store is not None:
            with store.lock:
                store.seconds += (time.perf_counter() - t_in, t_solve)
        return (Z.reshape(shp + (N,)), Y.reshape(shp + (m,)), S.reshape(shp), BS.reshape(shp + (nb,)),
                sid.reshape(shp))  # fmt: skip

    out = (jax.ShapeDtypeStruct((N,), jnp.float64), jax.ShapeDtypeStruct((m,), jnp.float64),
           jax.ShapeDtypeStruct((), jnp.int32), jax.ShapeDtypeStruct((nb,), jnp.int8),
           jax.ShapeDtypeStruct((), jnp.int64))  # fmt: skip

    def solve(key_lo, key_hi, k, c, vals, cols, b, active, basis, sid):
        return jax.pure_callback(host, out, key_lo, key_hi, k, c, vals, cols, b, active, basis, sid,
                                 vmap_method="broadcast_all")  # fmt: skip

    return solve

"""Where does the time of one B&B node go?  (``profile_root.py`` covers setup and the root.)

Runs ``minimize`` on one .nl instance, recording every batch-kernel call, then replays recorded
boxes from the middle of the search through the real bound kernel and through instrumented copies
of the bounders. Nothing in the library is changed: the instrumented bounders are the library's
own source with a few lines patched in memory to return counters (``_patched``), so they follow
the library's logic exactly; if a patched line no longer exists the script stops with the line.

Reported:
  * the run: nodes/s, share of the wall time inside each kernel;
  * the real bound kernel, ms per box, by batch size and by number of Python threads dispatching
    the calls (a jitted call releases the GIL; ``bnb.minimize`` dispatches calls one at a time);
  * FBBT alone;
  * lifted LP path (``quadratic.make_quadratic_bounder``): LP size and row blocks, share of dead
    rows and fixed columns, LP solves per node, IPM iterations per round (each box's own, and what
    the vmapped ``while_loop`` executes: the maximum over the batch), cost of one IPM iteration
    and its split (normal-matrix scatter, Cholesky, the rest);
  * reduced-space path (``lp.make_constrained_bounder``): LP size, LP solves and IPM iterations
    per node, cost of cut generation (``jacfwd`` of the relaxations) alone;
  * ``--simplex`` (lifted path, needs pyscipopt): the same node LPs, row for row, solved by a
    dual simplex (SoPlex): cold, re-solved after each round's tangent rows, and re-solved after
    fixing one integer (a child node started from its parent's basis).

Usage:
  python benchmarks/profile_node.py benchmarks/minlporg/nl/m424_kClus_5_10_3_BM_1.nl --time-limit 45 --simplex
  NPROC=1 python benchmarks/profile_node.py ...     # XLA's intra-op pool capped at one thread

Run one at a time on a shared machine: the timings are wall times.
"""

from __future__ import annotations

import argparse
import collections
import inspect
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import jaxglobal  # noqa: E402
from jaxglobal import bnb, nl  # noqa: E402
from jaxglobal import lp as lpm  # noqa: E402
from jaxglobal import quadratic as qd  # noqa: E402
from jaxglobal.fbbt import make_fbbt  # noqa: E402


def timeit(fn, *args, reps=5):
    """Median wall seconds of a jitted call (compiled by a first call)."""
    jax.block_until_ready(fn(*args))
    w = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        w.append(time.perf_counter() - t)
    return float(np.median(w))


def _patched(fn, reps, ns):
    """Source of ``fn`` with each ``(old, new)`` of ``reps`` replaced, executed in ``ns``."""
    src = inspect.getsource(fn)
    for old, new in reps:
        if old not in src:
            sys.exit(f"profile_node: {fn.__name__} changed, this line is gone:\n  {old}")
        src = src.replace(old, new)
    exec(src, ns)


def lifted_bounder(Q, **kw):
    """``quadratic.make_quadratic_bounder`` returning also, per round, (IPM iterations, round's
    bound, whether the round ran)."""
    ns = dict(qd.__dict__)
    _patched(qd._ipm, [
        ("z, _, _, _, _, _, _, best, yb, _, _ = jax.lax.while_loop(lambda st: ~done(st), body_pc if pc else body, st)\n    return z, yb, best",
         "z, _, _, _, _, _, _, best, yb, kit, _ = jax.lax.while_loop(lambda st: ~done(st), body_pc if pc else body, st)\n    return z, yb, best, kit"),
    ], ns)  # fmt: skip
    _patched(qd.make_quadratic_bounder, [
        ("vals, b, cols, best, rlo, rhi, vprev, active = carry", "vals, b, cols, best, rlo, rhi, vprev, active = carry\n            act_in = active"),
        ("z, y, val = _ipm(", "z, y, val, kit = _ipm("),
        ("return (vals, b, cols, best, rlo, rhi, v, active), None", "return (vals, b, cols, best, rlo, rhi, v, active), (kit, lb, act_in)"),
        ("(_, _, _, best, rlo, rhi, x, _), _ = jax.lax.scan(", "(_, _, _, best, rlo, rhi, x, _), stats = jax.lax.scan("),
        ("return best, jnp.clip(x, lo, hi), infeasible, rlo, rhi", "return best, jnp.clip(x, lo, hi), infeasible, rlo, rhi, stats"),
    ], ns)  # fmt: skip
    return ns["make_quadratic_bounder"](Q, **kw)


def reduced_bounder(f, ineq, eq, **kw):
    """``lp.make_constrained_bounder`` returning also (Kelley rounds, IPM iterations of all its LPs,
    live rows of the last LP, IPM iterations of the first LP); prints the LP shape when traced."""
    ns = dict(lpm.__dict__)
    _patched(lpm._ipm_box, [("return st[2], st[-1], st[-2]", "return st[2], st[-1], st[-2], st[0]")], ns)
    _patched(lpm.ipm, [], ns)
    _patched(lpm.make_constrained_bounder, [
        ("z, y, lb = ipm(cobj, A, bvec, ipm_iters,",
         "print('   LP shape (rows, columns):', A.shape, 'plus', 2 * A.shape[1], 'implicit box rows')\n"
         "            z, y, lb, kit = ipm(cobj, A, bvec, ipm_iters,"),
        ("return best, rlo, rhi, jnp.clip(z[:n], 0.0, 1.0)", "return best, rlo, rhi, jnp.clip(z[:n], 0.0, 1.0), kit, jnp.sum(jnp.any(A != 0, axis=1))"),
        ("best, rlo, rhi, u = solve(const, coef, best0, lo, hi)", "best, rlo, rhi, u, kit0, live0 = solve(const, coef, best0, lo, hi)"),
        ("k, const, coef, best, rlo, rhi, u, _ = carry", "k, const, coef, best, rlo, rhi, u, _, kacc, _l = carry"),
        ("best, rlo, rhi, u = solve(const, coef, best, rlo, rhi)", "best, rlo, rhi, u, kit, live = solve(const, coef, best, rlo, rhi)"),
        ("return k + 1, const, coef, best, rlo, rhi, u, _kelley_go(prev, best, cutoff)",
         "return k + 1, const, coef, best, rlo, rhi, u, _kelley_go(prev, best, cutoff), kacc + kit, live"),
        ("init = (0, const, coef, best, rlo, rhi, u, _kelley_go(best0, best, cutoff, first=True))",
         "init = (0, const, coef, best, rlo, rhi, u, _kelley_go(best0, best, cutoff, first=True), kit0, live0)"),
        ("_, _, _, best, rlo, rhi, u, _ = jax.lax.while_loop(", "kr, _, _, best, rlo, rhi, u, _, kacc, live = jax.lax.while_loop("),
        ("return lb, lo + w * u, infeasible, rlo, rhi", "return lb, lo + w * u, infeasible, rlo, rhi, kr, kacc, live, kit0"),
    ], ns)  # fmt: skip
    return ns["make_constrained_bounder"](f, ineq, eq, **kw)


def record(M, tl, opts):
    """Run ``minimize``; returns (result, per-kernel call times, bound-kernel calls' arguments,
    the bound kernel, the ``_Kernels`` object)."""
    calls, saved, kernels, Ks = collections.defaultdict(list), [], {}, []
    call0, init0 = bnb._Precompiled.__call__, bnb._Kernels.__init__

    def call(self, *args, deadline=None):
        t = time.perf_counter()
        o = jax.block_until_ready(call0(self, *args, deadline=deadline))
        t1 = time.perf_counter()
        calls[self.name].append((t1 - max(self.dispatched, t), args[0].shape[0], max(self.dispatched, t) - t))
        if self.name.startswith("bound"):
            kernels[self.name] = self
            saved.append(tuple(np.asarray(a) for a in args))
        return o

    def init(self, *a, **k):
        init0(self, *a, **k)
        Ks.append(self)

    bnb._Precompiled.__call__, bnb._Kernels.__init__ = call, init
    try:
        r = jaxglobal.minimize(M.f, M.lo, M.hi, ineq=M.ineq, eq=M.eq, integer=M.integer, abs_tol=1e-6, rel_tol=1e-4,
                               feas_tol=1e-6, time_limit=tl, max_iter=10**7, **opts)
    finally:
        bnb._Precompiled.__call__, bnb._Kernels.__init__ = call0, init0
    return r, calls, saved, kernels, Ks


def lifted_report(M, Q, o, n, integer, LO, HI, cutoff, tight, simplex):
    tol, safe = o["feas_tol"], o["safe"]
    P = len(Q.pairs)
    N = n + P
    need = qd.square_needs_lower(Q)
    nk = o["n_kelley"] if need.any() else 0
    L = qd._layout(Q, n, nk, False, qd.N_SEP if Q.pool_rhs.size else 0)
    pat = L.pattern
    m = pat.m
    R, C = jnp.asarray(pat.rows), jnp.asarray(pat.cols)
    rounds = max(nk, L.n_sep) + 1
    print(f"== lifted LP: {N} columns ({n} x + {P} products, {int((Q.pairs[:, 0] == Q.pairs[:, 1]).sum())} squares), {m} rows: "
          f"McCormick {4 * P}, g {Q.n_g}, h {2 * Q.n_h}, RLT {2 * len(Q.rlt_var)}, static cuts {Q.cut_rhs.size}, "
          f"bound-factor {2 * len(Q.qr_g)}, Kelley tangent slots {nk}x{P}, separation slots {L.n_sep}x{L.K}; nnz {pat.rows.size}")
    key = np.asarray(pat.cols)[pat.pa] * N + np.asarray(pat.cols)[pat.pb]
    print(f"   IPM system: dense Cholesky of the {N}x{N} normal matrix (N^3/3 = {N**3 / 3:.3g} flops), assembled by "
          f"{pat.pa.size} scatter-adds; {len(np.unique(key))} of its {N * N} entries are structurally nonzero")

    def data(lo, hi):
        vlo, vhi, vals0, b0, _ = qd._lp(Q, L, lo, hi, tol, safe)
        scale = qd._scale(vlo, vhi, safe)
        A, bu = qd._unit(pat._replace(cols=jnp.asarray(pat.cols)), vals0, b0, vlo, scale, safe)
        cu = jnp.asarray(Q.E[0]) * scale
        return A, bu, cu / jnp.maximum(jnp.linalg.norm(cu), 1e-300), vhi <= vlo

    tlo, thi = tight(jnp.asarray(LO[:16]), jnp.asarray(HI[:16]))
    A, bu, chat, fixed = jax.jit(jax.vmap(data))(tlo, thi)
    live = np.asarray(jax.vmap(lambda a: jax.ops.segment_sum(jnp.abs(a), R, m) > 0)(A))
    print(f"   first LP of a node: live rows {live.sum(1).mean():.0f} of {m} ({100 * (1 - live.mean()):.0f}% dead or idle slots), "
          f"fixed columns {np.asarray(fixed).sum(1).mean():.0f} of {N} (still in the Cholesky)")

    ib = jax.jit(jax.vmap(lifted_bounder(Q, n_kelley=o["n_kelley"], tol=tol, safe=safe), in_axes=(0, 0, None)))
    ks, acts = [], []
    for i in range(0, len(LO) - 15, 16):
        out = ib(*tight(jnp.asarray(LO[i:i + 16]), jnp.asarray(HI[i:i + 16])), cutoff)
        kit, _, act = [np.asarray(a) for a in out[5]]  # (16, rounds)
        ks.append(kit)
        acts.append(act)
    kit, act = np.concatenate(ks), np.concatenate(acts)
    executed = float(np.mean([k.max(0).sum() for k in ks]))
    print(f"   rounds per node <= {rounds}; LP solves per node {act.sum(1).mean():.2f}; share of boxes active per round {np.round(act.mean(0), 2)}")
    print(f"   IPM iterations per round (mean over boxes) {np.round(kit.mean(0), 1)}; per node: own {kit.sum(1).mean():.1f}, "
          f"executed by the batched loop (max over the 16 boxes of a call) {executed:.1f}")

    res = {}
    for it in (5, 25):  # fixed trip counts (no early stop): the difference is 20 iterations
        def full(vals, b, c, it=it):
            cert = lambda y: qd._certificate(pat, vals, b, y, c, safe)[0]  # noqa: E731
            return qd._ipm(c, pat, vals, b, iters=it, cert=cert, mu_tol=-1.0, gap_tol=-1.0)[2]
        res[it] = timeit(jax.jit(jax.vmap(full)), A, bu, chat, reps=3)
    per = (res[25] - res[5]) / 20 / 16
    PA, PB, RPA, KEY = jnp.asarray(pat.pa), jnp.asarray(pat.pb), jnp.asarray(pat.rows[pat.pa]), jnp.asarray(key)
    REPS = 100

    def assemble(vals, d):
        return jnp.zeros(N * N).at[KEY].add(vals[PA] * vals[PB] * d[RPA]).reshape(N, N)

    def loop(piece):
        return jax.jit(jax.vmap(lambda *a: jax.lax.fori_loop(0, REPS, lambda i, acc: acc + piece(acc * 1e-30, *a), 0.0)))

    rng = np.random.default_rng(0)
    d0 = jnp.asarray(rng.uniform(0.5, 2.0, (16, m)))
    Ms = jax.jit(jax.vmap(lambda vals, d: assemble(vals, d) + jnp.eye(N)))(A, d0)
    r0 = jnp.asarray(rng.normal(size=(16, N)))

    def chol(e, Mm, r):
        Mi = Mm + e
        dm = jnp.sqrt(jnp.diag(Mi))
        fac = jax.scipy.linalg.cho_factor(Mi / dm[:, None] / dm[None, :], lower=True)
        return (jax.scipy.linalg.cho_solve(fac, r / dm) / dm)[0]

    t_as = timeit(loop(lambda e, vals, d: assemble(vals, d + e)[0, 0]), A, d0, reps=3) / REPS / 16
    t_ch = timeit(loop(chol), Ms, r0, reps=3) / REPS / 16
    print(f"   one IPM iteration: {per * 1e6:.0f} us per box (batch of 16): normal-matrix scatter {t_as * 1e6:.0f} us, "
          f"Cholesky factor + solve {t_ch * 1e6:.0f} us, rest {max(per - t_as - t_ch, 0) * 1e6:.0f} us")
    print(f"   => IPM per node: executed iterations x cost = {executed * per * 1e3:.2f} ms")
    if simplex:
        lifted_simplex(M, Q, L, n, integer, LO, HI, cutoff, tight, nk, tol, safe)


def lifted_simplex(M, Q, L, n, integer, LO, HI, cutoff, tight, nk, tol, safe):
    """The node LPs of the lifted path by a dual simplex (SoPlex through ``pyscipopt.LP``)."""
    try:
        from pyscipopt import LP
    except ImportError:
        print("   --simplex: pyscipopt is not installed")
        return
    pat = L.pattern
    P = len(Q.pairs)
    N = n + P
    t0row = 4 * P + Q.n_g + 2 * Q.n_h + 2 * len(Q.rlt_var) + Q.cut_rhs.size  # first tangent row (as quadratic._lp)
    patj = pat._replace(cols=jnp.asarray(pat.cols))

    @jax.jit
    def build(lo, hi):
        vlo, vhi, vals0, b0, _ = qd._lp(Q, L, lo, hi, tol, safe)
        scale = qd._scale(vlo, vhi, safe)
        cu = jnp.asarray(Q.E[0]) * scale
        return vlo, scale, vals0, b0, cu / jnp.maximum(jnp.linalg.norm(cu), 1e-300)

    unit = jax.jit(lambda vals, b, vlo, scale: qd._unit(patj, vals, b, vlo, scale, safe))
    tang = jax.jit(lambda lo, hi, vals, b, x, k: qd._lp(Q, L, lo, hi, tol, safe)[4](vals, b, x, k))

    def rows_of(A, bu, which):
        A, bu, ents, rhs = np.asarray(A), np.asarray(bu), [], []
        for r in which:
            idx = np.flatnonzero((pat.rows == r) & (A != 0))
            if idx.size:
                d = collections.defaultdict(float)
                for c, v in zip(pat.cols[idx], A[idx]):
                    d[int(c)] += float(v)
                ents.append(list(d.items()))
                rhs.append(float(bu[r]))
        return ents, rhs

    def solve(lp):
        t = time.perf_counter()
        lp.solve(dual=True)
        return time.perf_counter() - t, lp.getNIterations()

    stat = collections.defaultdict(list)
    ints = np.asarray(integer, np.int64)
    TLO, THI = tight(jnp.asarray(LO), jnp.asarray(HI))
    for bi in range(len(LO)):
        lo, hi = TLO[bi], THI[bi]
        vlo, scale, vals, b, chat = build(lo, hi)
        if not np.all(np.isfinite(np.asarray(vlo))):
            continue
        A, bu = unit(vals, b, vlo, scale)
        lp = LP()
        lp.addCols([[] for _ in range(N)], objs=[float(c) for c in np.asarray(chat)], lbs=[0.0] * N, ubs=[1.0] * N)
        ents, rhs = rows_of(A, bu, range(pat.m))
        lp.addRows(ents, lhss=[-lp.infinity()] * len(rhs), rhss=rhs)
        stat["cold"].append(solve(lp))
        if not lp.isOptimal():
            continue
        for k in range(nk):  # Kelley: the tangent rows of round k at the LP point, then the LP again
            v = np.asarray(vlo) + np.asarray(scale) * np.clip(np.asarray(lp.getPrimal()), 0, 1)
            vals, b = tang(lo, hi, vals, b, jnp.asarray(v), k)
            A, bu = unit(vals, b, vlo, scale)
            ents, rhs = rows_of(A, bu, range(t0row + k * P, t0row + (k + 1) * P))
            if ents:
                lp.addRows(ents, lhss=[-lp.infinity()] * len(rhs), rhss=rhs)
            stat["after a round's tangent rows"].append(solve(lp))
        z = np.asarray(lp.getPrimal())
        free = ints[np.asarray(hi)[ints] > np.asarray(lo)[ints]] if ints.size else ints
        if free.size:  # a child: the most fractional free integer column fixed at either end
            j = int(free[np.argmin(np.abs(z[free] - 0.5))])
            for val in (0.0, 1.0):
                lp.chgBound(j, val, val)
                stat["child from the parent's basis"].append(solve(lp))
    print("   dual simplex (SoPlex) on the same LPs, one core:")
    for k, v in stat.items():
        a = np.array(v)
        print(f"     {k:30s}: {len(a):4d} solves, {a[:, 1].mean():5.1f} simplex iterations, {a[:, 0].mean() * 1e6:6.0f} us per solve")


def reduced_report(M, o, n, LO, HI, cutoff, extra, tight):
    tol, safe = o["feas_tol"], o["safe"]
    kw = dict(univariate=o["univariate"], alphabb=o["alphabb"], multivariate=o["multivariate"], safe=safe, exact=o["exact"])
    print(f"== reduced-space cut LP{' with a cut pool of ' + str(extra[0].shape[0]) + ' rows' if extra else ''} ==")
    cb = reduced_bounder(M.f, M.ineq, M.eq, n_kelley=o["n_kelley"], tol=tol, **kw)
    ib = jax.jit(jax.vmap(lambda l, h: cb(l, h, cutoff, *extra)))
    outs = [ib(*tight(jnp.asarray(LO[i:i + 16]), jnp.asarray(HI[i:i + 16]))) for i in range(0, len(LO) - 15, 16)]
    kr, kacc, live, k0 = (np.concatenate([np.asarray(o_[j]) for o_ in outs]) for j in (5, 6, 7, 8))
    print(f"   per node (each box's own): LP solves {1 + kr.mean():.2f}, IPM iterations {kacc.mean():.1f} (first LP {k0.mean():.1f}); "
          f"live rows of the last LP {live.mean():.0f}")
    print(f"   Kelley rounds executed by the batched loop (max over the 16 boxes): {np.mean([np.asarray(o_[5]).max() for o_ in outs]):.2f} of {o['n_kelley']}")
    t16 = tight(jnp.asarray(LO[:16]), jnp.asarray(HI[:16]))
    print(f"   bounder alone: {timeit(ib, *t16) / 16 * 1e3:.2f} ms per box (batch of 16)")
    prepare, parts, _ = lpm._parts_fn(M.f, M.ineq, M.eq, **kw)

    def cuts_only(l, h):
        data = prepare(l, h)
        axis = jnp.concatenate([jnp.eye(n), -jnp.eye(n)]) * ((h - l) / 4)
        pts = jnp.concatenate([0.5 * (l + h)[None], 0.5 * (l + h) + axis])
        return jax.vmap(lambda p: lpm._cuts(parts, data, p, l, h, safe)[:2])(pts)

    w = timeit(jax.jit(jax.vmap(cuts_only)), *t16) / 16
    print(f"   cut generation alone (jacfwd of the relaxations at the {2 * n + 1} initial points): {w * 1e3:.2f} ms per box; "
          f"each Kelley round adds one point")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("nl", help="path of the .nl instance")
    ap.add_argument("--time-limit", type=float, default=45.0)
    ap.add_argument("--opts", default="{}", help="JSON of extra minimize options")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="Python threads of the dispatch test")
    ap.add_argument("--simplex", action="store_true", help="lifted path: dual simplex on the same LPs (pyscipopt)")
    a = ap.parse_args()
    opts = json.loads(a.opts)
    if "batch_sizes" in opts:
        opts["batch_sizes"] = tuple(opts["batch_sizes"])
    M = nl.read_nl(a.nl, integers="mask")
    n = M.lo.size
    integer = tuple(int(i) for i in np.flatnonzero(M.integer)) if M.integer is not None else ()
    r, calls, saved, kernels, Ks = record(M, a.time_limit, opts)
    print(f"{pathlib.Path(a.nl).stem}: n={n} integers={len(integer)} status={r.status} fun={r.fun:.6g} lb={r.lower_bound:.6g} "
          f"nodes={r.nodes} boxes bounded={r.evaluations} time={r.time:.1f}s nodes/s={r.nodes / r.time:.1f} NPROC={os.environ.get('NPROC', '-')}")
    for k, v in calls.items():
        t, b, wait = np.array(v).T
        print(f"   kernel {k}: {len(t)} calls, {t.sum():.2f} s ({100 * t.sum() / r.time:.0f}% of the wall time), "
              f"median {np.median(t) * 1e3:.1f} ms per call, {np.median(t / b) * 1e3:.2f} ms per box; "
              f"waited {wait.sum():.1f} s for its executable")
    if not saved or not Ks:
        sys.exit("no bound-kernel call recorded")
    K, (name, kern) = Ks[-1], next(iter(kernels.items()))
    full = [s for s in saved[len(saved) // 4:] if s[0].shape[0] >= 16][:4] or saved[-1:]
    LO, HI = np.concatenate([s[0][:16] for s in full]), np.concatenate([s[1][:16] for s in full])
    cutoff = jnp.asarray(saved[len(saved) // 2][2])
    extra = tuple(jnp.asarray(x) for x in full[-1][3:])
    Q = K.lifted_form(n) if (M.ineq is not None or M.eq is not None) else None
    print(f"   bound kernel '{name}' on {len(LO)} recorded boxes, cutoff {float(cutoff):.6g}")

    print("== the real bound kernel: batch size x dispatch threads ==")
    run = lambda ch: jax.block_until_ready(kern.jitted(ch[0], ch[1], cutoff, *extra))  # noqa: E731
    for B, th in [(16, 1), (16, a.threads), (4, 1), (4, a.threads), (1, 1), (1, a.threads)]:
        chunks = [(jnp.asarray(LO[i:i + B]), jnp.asarray(HI[i:i + B])) for i in range(0, len(LO) - B + 1, B)]
        if not chunks:
            continue
        run(chunks[0])
        best = np.inf
        for _ in range(3):
            t, p = time.perf_counter(), time.process_time()
            if th == 1:
                [run(ch) for ch in chunks]
            else:
                with ThreadPoolExecutor(th) as ex:
                    list(ex.map(run, chunks))
            w, cpu = time.perf_counter() - t, time.process_time() - p
            best, ratio = (w, cpu / w) if w < best else (best, ratio)
        print(f"   batch {B:3d} x {len(chunks):2d} calls, {th} thread(s): {best / len(chunks) / B * 1e3:7.2f} ms per box, "
              f"{len(chunks) * B / best:6.0f} boxes/s, cpu/wall {ratio:.2f}")

    o = K._opts
    if o["fbbt"] and not (Q is not None and qd.binary_only(Q, n)):
        tj = jax.jit(jax.vmap(make_fbbt(M.f, M.ineq, M.eq, rounds=o["fbbt_rounds"], feas_tol=o["feas_tol"], safe=o["safe"],
                                        integer=integer), in_axes=(0, 0, None)))  # fmt: skip
        print(f"== FBBT alone ({o['fbbt_rounds']} rounds): {timeit(tj, jnp.asarray(LO[:16]), jnp.asarray(HI[:16]), cutoff) / 16 * 1e3:.2f} ms per box ==")
        tight = lambda l, h: tj(l, h, cutoff)[:2]  # noqa: E731
    else:
        print("== FBBT is not part of this node kernel ==")
        tight = lambda l, h: (l, h)  # noqa: E731
    if Q is not None:
        lifted_report(M, Q, o, n, integer, LO, HI, cutoff, tight, a.simplex)
    elif M.ineq is not None or M.eq is not None:
        reduced_report(M, o, n, LO, HI, cutoff, extra, tight)
    else:
        print("== unconstrained: Frank-Wolfe bounder (bnb.make_bounder), not instrumented ==")


if __name__ == "__main__":
    main()

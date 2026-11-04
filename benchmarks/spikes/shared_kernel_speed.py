"""Spike: a scheduled shared McCormick sweep (one loop step per dependency level, all rows of a
level at once, every opcode of the table applied and selected), against the row-by-row shared
sweep and the per-problem straight-line kernel. Measures per-call time and compile time."""

import sys, time, os, functools, logging
from typing import NamedTuple
logging.disable(logging.INFO)
os.environ["JAXGLOBAL_CACHE"] = "0"
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/user/jax-go")
from jaxglobal import bound_generic as BG, lp, relax_interp
from jaxglobal.fbbt import joint_tape
from jaxglobal.fbbt_interp import scalarize, _live_rows, balance_sums
from jaxglobal.nl import read_nl

DataS = NamedTuple("DataS", [(f, object) for f in BG.Data._fields] + [("levels", object)])


def levels_of(d, K, Tb=None, Wb=None):
    """(T, W) row indices per dependency level, -1 padding."""
    S = d.cval.shape[0]
    lvl = np.zeros(S, int)
    rl = np.zeros(K, int)
    for k in range(K):
        rl[k] = 1 + max(lvl[d.a[k]], lvl[d.b[k]])
        lvl[d.out[k]] = rl[k]
    T = rl.max()
    rows = [np.flatnonzero(rl == t) for t in range(1, T + 1)]
    W = max(len(r) for r in rows)
    Tb, Wb = Tb or BG.fg._bucket(T, 4), Wb or BG.fg._bucket(W, 8)
    L = np.full((Tb, Wb), -1, np.int32)
    for t, r in enumerate(rows):
        L[t, : len(r)] = r
    return L, T


def sched_sweep(x, lo, hi, d, safe, tier):
    fwd = BG._rules(safe, tier)
    N = lo.shape[0]
    S = d.cval.shape[0]
    base = jnp.where(d.is_c, d.cval, 0.0)
    V = jnp.stack([base.at[:N].set(lo), base.at[:N].set(hi), base.at[:N].set(x), base.at[:N].set(x)], axis=1)
    eye = jnp.eye(N)
    G = jnp.zeros((S, 2, N)).at[:N, 0].set(eye).at[:N, 1].set(eye)
    V = jnp.concatenate([V, jnp.zeros((1, 4))])  # trash slot S for padding rows
    G = jnp.concatenate([G, jnp.zeros((1, 2, N))])

    def step(t, carry):
        V, G = carry
        k = d.levels[t]
        live = k >= 0
        kk = jnp.maximum(k, 0)
        a, b, o, op = d.a[kk], d.b[kk], jnp.where(live, d.out[kk], S), d.op[kk]
        va, vb = V[a], V[b]
        T = jnp.concatenate([G[a], G[b]], axis=1)  # (W, 4, N)
        outs = [jax.vmap(f)(va[:, 0], va[:, 1], va[:, 2], va[:, 3], vb[:, 0], vb[:, 1], vb[:, 2], vb[:, 3], T) for f in fwd]
        R = jnp.stack([r for r, _ in outs])  # (U, W, 4)
        J = jnp.stack([j for _, j in outs])  # (U, W, 2, N)
        sel = jax.nn.one_hot(op, len(fwd), dtype=bool).T  # (U, W)
        r = jnp.sum(jnp.where(sel[:, :, None], R, 0.0), axis=0)
        j = jnp.sum(jnp.where(sel[:, :, None, None], J, 0.0), axis=0)
        return V.at[o].set(r), G.at[o].set(j)

    V, G = jax.lax.fori_loop(0, d.levels.shape[0], step, (V, G))
    return V[:S], G[:S]


def bench(fn, *a, reps=20):
    jax.block_until_ready(fn(*a))
    t = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn(*a))
    return (time.perf_counter() - t) / reps * 1e3


relax_interp.RULE, relax_interp.MIN_ROWS = "rows", 10**9  # reference: straight-line McCormick (the per-problem kernel)
lp._parts_fn = functools.partial(lp._parts_fn, convex=False)
mode = sys.argv[2]
for name in sys.argv[1].split(","):
    M = read_nl(f"/home/user/jax-go/benchmarks/minlplib/nl/{name}.nl", integers="mask"); n = len(M.lo)
    lo, hi = np.maximum(M.lo, -1e3), np.minimum(M.hi, 1e3)
    L, H = np.tile(lo, (16, 1)), np.tile(hi, (16, 1))
    if mode == "straight":
        f = jax.jit(jax.vmap(lp.make_constrained_bounder(M.f, M.ineq, M.eq, tol=1e-6), in_axes=(0, 0, None)))
        t = time.perf_counter(); c = f.lower(jnp.asarray(L), jnp.asarray(H), jnp.inf).compile(); tc = time.perf_counter() - t
        ms = bench(c, jnp.asarray(L), jnp.asarray(H), jnp.inf)
        lb = float(np.asarray(c(jnp.asarray(L), jnp.asarray(H), jnp.inf)[0])[0])
    else:
        G = BG.make_generic_bounder(M.f, M.ineq, M.eq, n=n, tol=1e-6)
        if G is None:
            print(name, "not encodable"); continue
        d = G.data
        z = jnp.zeros((16, G.g.N - n))
        args = [jnp.concatenate([jnp.asarray(L), z], -1), jnp.concatenate([jnp.asarray(H), z], -1), jnp.asarray(jnp.inf), G.tol]
        if mode == "sched":
            lv, T = levels_of(G.g.data, int(G.g.data.rows))
            d = DataS(*d, jnp.asarray(lv))
            BG._sweep = sched_sweep
        fn = jax.jit(jax.vmap(functools.partial(BG._bound_one, safe=False, tier=G.g.tier, n_kelley=4, ipm_iters=40),
                              in_axes=(0, 0, None, None, None)))
        t = time.perf_counter(); c = fn.lower(*args, d).compile(); tc = time.perf_counter() - t
        ms = bench(c, *args, d)
        lb = float(np.asarray(c(*args, d)[0])[0])
    print(f"{name:22s} {mode:8s} compile {tc:6.2f}s  {ms:8.2f} ms/call (16 boxes)  root lb {lb:.7g}", flush=True)

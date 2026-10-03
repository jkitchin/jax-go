"""Where does the time go, and how does batched bounding scale?

1. Throughput of the jit(vmap) bounding kernel vs batch size (steady state,
   compile excluded).
2. Time split inside minimize(): bounding kernel vs scipy local solves vs the
   rest (host-side B&B bookkeeping, compilation), from cProfile.

Run:  PYTHONPATH=. python examples/throughput.py
"""

import cProfile
import json
import pstats
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxglobal import make_bounder, minimize
from jaxglobal.lp import make_constrained_bounder
from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS

P = {p.name: p for p in PROBLEMS + CONSTRAINED_PROBLEMS}


def random_boxes(p, k, rng):
    a, b = np.sort(rng.uniform(p.lo, p.hi, (2, k, p.lo.size)), axis=0)
    return jnp.asarray(a), jnp.asarray(b)


def throughput(name, sizes=(1, 16, 64, 256, 1024, 4096)):
    p = P[name]
    if hasattr(p, "ineq"):
        bound = jax.jit(jax.vmap(make_constrained_bounder(p.f, p.ineq, p.eq)))
    else:
        bound = jax.jit(jax.vmap(make_bounder(p.f)))
    rng = np.random.default_rng(0)
    out = {}
    for k in sizes:
        a, b = random_boxes(p, k, rng)
        t = time.perf_counter()
        jax.block_until_ready(bound(a, b))  # compile
        tc = time.perf_counter() - t
        reps = max(1, int(2000 / k))
        t = time.perf_counter()
        for _ in range(reps):
            jax.block_until_ready(bound(a, b))
        dt = (time.perf_counter() - t) / reps
        out[k] = dict(boxes_per_s=k / dt, call_ms=1e3 * dt, compile_s=tc)
        print(f"{name:12s} batch {k:5d}: {k / dt:9.0f} boxes/s  ({1e3 * dt:7.2f} ms/call, compile {tc:.1f}s)", flush=True)
    return out


def time_split(name):
    p = P[name]
    kw = dict(ineq=p.ineq, eq=p.eq) if hasattr(p, "ineq") else {}
    pr = cProfile.Profile()
    t = time.perf_counter()
    pr.enable()
    r = minimize(p.f, p.lo, p.hi, **kw)
    pr.disable()
    total = time.perf_counter() - t
    st = pstats.Stats(pr).stats
    cum = {}
    for (fn, _, func), (_, _, _, ct, _) in st.items():
        if fn.endswith("bnb.py") and func in ("local", "evaluate"):
            cum[func] = cum.get(func, 0) + ct
    local, bnd = cum.get("local", 0.0), cum.get("evaluate", 0.0)
    res = dict(total=total, bounding=bnd, local=local, other=total - bnd - local, nodes=r.nodes)
    print(f"{name:12s} total {total:6.2f}s  bounding {bnd:6.2f}s  local {local:6.2f}s  other {res['other']:6.2f}s", flush=True)
    return res


if __name__ == "__main__":
    out = dict(
        throughput={n: throughput(n) for n in ("hartmann6", "haverly")},
        split={n: time_split(n) for n in ("eggholder", "hartmann6", "G04", "haverly", "reactor")},
    )
    json.dump(out, open("examples/throughput_results.json", "w"), indent=1)

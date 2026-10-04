"""Time ``minimize`` on the built-in problems, each in a fresh process.

For every problem the worker solves it twice in one process: the first
("cold") call pays for tracing and XLA compilation, the second ("warm") call
shows what a repeat solve of the same functions costs. Results go to a JSON
file so two runs can be compared.

Usage:
  python benchmarks/builtin.py --out before.json
  python benchmarks/builtin.py --out after.json --compare before.json
  python benchmarks/builtin.py --only gramacy_lee,G04
  python benchmarks/builtin.py --cache-dir .jax_cache --only G04   # persistent XLA cache
  python benchmarks/builtin.py --opts '{"batch_sizes": [16, 64]}'
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = "BUILTIN-RESULT "


def worker(name, opts, cache_dir=""):
    sys.path.insert(0, str(ROOT))
    if cache_dir:
        import jax

        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    from jaxglobal import minimize
    from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS

    probs = {p.name: (p.f, None, None, p.lo, p.hi) for p in PROBLEMS}
    probs.update({p.name: (p.f, p.ineq, p.eq, p.lo, p.hi) for p in CONSTRAINED_PROBLEMS})
    f, g, h, lo, hi = probs[name]
    out = dict(name=name)
    for key in ("cold", "warm"):
        t = time.perf_counter()
        r = minimize(f, lo, hi, ineq=g, eq=h, **opts)
        out[key] = time.perf_counter() - t
    out.update(status=r.status, nodes=r.nodes, fun=r.fun, lower_bound=r.lower_bound)
    print(TAG + json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", default="")
    ap.add_argument("--opts", default="{}", help="JSON dict of extra minimize options")
    ap.add_argument("--cache-dir", default="", help="JAX persistent compilation cache directory (default: off)")
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker, json.loads(a.opts), a.cache_dir)

    sys.path.insert(0, str(ROOT))
    from jaxglobal.problems import CONSTRAINED_PROBLEMS, PROBLEMS

    names = [p.name for p in PROBLEMS] + [p.name for p in CONSTRAINED_PROBLEMS]
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    ref = {}
    if a.compare:
        ref = {r["name"]: r for r in json.loads(pathlib.Path(a.compare).read_text())}

    rows = []
    print(f"{'problem':20s} {'cold':>7s} {'warm':>7s} {'nodes':>6s}  status     gap" + ("   (ref cold / warm / nodes)" if ref else ""))
    for n in names:
        p = subprocess.run([sys.executable, __file__, "--worker", n, "--opts", a.opts, "--cache-dir", a.cache_dir], capture_output=True, text=True)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith(TAG)), None)
        if line is None:
            print(f"{n:20s} ERROR {p.stderr.strip()[-300:]}")
            continue
        r = json.loads(line[len(TAG):])
        rows.append(r)
        s = f"{n:20s} {r['cold']:7.2f} {r['warm']:7.2f} {r['nodes']:6d}  {r['status']:9s} {r['fun'] - r['lower_bound']:.1e}"
        if n in ref:
            q = ref[n]
            s += f"   ({q['cold']:6.2f} / {q['warm']:6.2f} / {q['nodes']:5d})"
        print(s, flush=True)
    tc, tw = sum(r["cold"] for r in rows), sum(r["warm"] for r in rows)
    s = f"{'total':20s} {tc:7.2f} {tw:7.2f}"
    if ref:
        common = [r["name"] for r in rows if r["name"] in ref]
        s += f"   ({sum(ref[c]['cold'] for c in common):6.2f} / {sum(ref[c]['warm'] for c in common):6.2f})"
    print(s)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

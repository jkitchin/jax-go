"""Scaling study: ``minimize`` on the synthetic families of ``jaxglobal/synthetic.py``.

Each instance runs in a fresh process: a cold solve (includes tracing and
compilation), optionally a warm repeat (kernels cached, so cold - warm
estimates compile time), then a root-only solve (``max_iter=1``) for the root
lower bound. Results go to JSON.

Usage:
  python benchmarks/synthetic.py --only univariate_sum_5,qcqp_5_2 --out r.json
  python benchmarks/synthetic.py --families pooling --timeout 600 --no-warm
  python benchmarks/synthetic.py --opts '{"obbt": false}' --compare r.json
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
TAG = "SYNTHETIC-RESULT "


def worker(name, opts, timeout, warm, cache_dir):
    sys.path.insert(0, str(ROOT))
    import jax

    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    from jaxglobal import minimize
    from jaxglobal.synthetic import suite

    fam, p = next((fam, p) for fam, p in suite() if p.name == name)
    g = getattr(p, "ineq", None)
    h = getattr(p, "eq", None)
    kw = dict(opts, ineq=g, eq=h)
    n_con = sum(int(c(0.5 * (p.lo + p.hi)).size) for c in (g, h) if c is not None)
    out = dict(name=name, family=fam, n=len(p.lo), n_con=n_con, fstar=p.fstar)
    t = time.perf_counter()
    r = minimize(p.f, p.lo, p.hi, time_limit=timeout, **kw)
    out["cold"] = time.perf_counter() - t
    out.update(status=r.status, fun=r.fun, lower_bound=r.lower_bound, gap=r.gap, nodes=r.nodes,
               iterations=r.iterations, evaluations=r.evaluations, search=r.time)
    print(TAG + json.dumps(out), flush=True)  # partial result in case a later phase dies
    if warm:
        t = time.perf_counter()
        minimize(p.f, p.lo, p.hi, time_limit=timeout, **kw)
        out["warm"] = time.perf_counter() - t
        out["compile_est"] = out["cold"] - out["warm"]
    t = time.perf_counter()
    r1 = minimize(p.f, p.lo, p.hi, **dict(kw, max_iter=1))
    out.update(root_time=time.perf_counter() - t, root_lb=r1.lower_bound, root_fun=r1.fun, root_gap=r1.gap)
    print(TAG + json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--only", default="")
    ap.add_argument("--families", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", default="")
    ap.add_argument("--opts", default="{}", help="JSON dict of extra minimize options")
    ap.add_argument("--timeout", type=float, default=300.0, help="per-solve time limit (s); the process is killed at 3x + 300")
    ap.add_argument("--stop-on-fail", action="store_true", help="skip larger instances of a family after one is not optimal")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--cache-dir", default="", help="JAX persistent compilation cache directory (default: off)")
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker, json.loads(a.opts), a.timeout, not a.no_warm, a.cache_dir)

    sys.path.insert(0, str(ROOT))
    from jaxglobal.synthetic import suite

    inst = suite(a.families.split(",") if a.families else None)
    if a.only:
        inst = [(fam, p) for fam, p in inst if p.name in a.only.split(",")]
    names, family = [p.name for fam, p in inst], {p.name: fam for fam, p in inst}
    failed = set()
    ref = {}
    if a.compare:
        ref = {r["name"]: r for r in json.loads(pathlib.Path(a.compare).read_text())}

    rows = []
    print(f"{'instance':22s} {'n':>4s} {'m':>3s} {'cold':>7s} {'warm':>7s} {'nodes':>6s} {'evals':>7s}  {'status':10s} {'gap':>8s} {'root gap':>9s}"
          + ("   (ref cold / nodes)" if ref else ""))
    for n in names:
        if family[n] in failed:
            continue
        cmd = [sys.executable, __file__, "--worker", n, "--opts", a.opts, "--timeout", str(a.timeout)]
        cmd += ["--no-warm"] * a.no_warm + (["--cache-dir", a.cache_dir] if a.cache_dir else [])
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3 * a.timeout + 300)
            stdout, err = p.stdout, p.stderr.strip()[-300:]
        except subprocess.TimeoutExpired as e:
            stdout, err = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "killed (hard timeout)"
        lines = [ln for ln in stdout.splitlines() if ln.startswith(TAG)]
        if not lines:
            print(f"{n:22s} ERROR {err}")
            failed.add(family[n] if a.stop_on_fail else None)
            rows.append(dict(name=n, status="error", message=err))
            continue
        r = json.loads(lines[-1][len(TAG):])
        if err and "root_gap" not in r:
            r["message"] = err
        rows.append(r)
        if r["status"] != "optimal" and a.stop_on_fail:
            failed.add(family[n])
        s = (f"{n:22s} {r['n']:4d} {r['n_con']:3d} {r['cold']:7.1f} {r.get('warm', float('nan')):7.1f} {r['nodes']:6d} "
             f"{r['evaluations']:7d}  {r['status']:10s} {r['gap']:8.1e} {r.get('root_gap', float('nan')):9.1e}")
        if n in ref and "cold" in ref[n]:
            s += f"   ({ref[n]['cold']:6.1f} / {ref[n]['nodes']:5d})"
        print(s, flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

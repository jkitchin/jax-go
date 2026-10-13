"""Run jaxglobal on a random sample of MINLPLib instances and fail on any wrong answer.

Each instance runs in a child process (``run.py``'s worker). The result is
classified against the reference bounds in ``minlplib/instances.csv`` with
``report.classify``; "wrong" (a claimed bound or optimum that contradicts the
reference) and a failed float64 primal verification both exit non-zero.

Usage:  python benchmarks/check_wrong.py --sample 12 --time-limit 20 --seed 0
"""

from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import report  # noqa: E402
import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=12, help="number of random instances")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-vars", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--safe", action="store_true", help="solve with minimize(safe=True)")
    a = ap.parse_args()
    if a.safe:
        os.environ["JAXGLOBAL_SAFE"] = "1"  # inherited by the worker processes

    from minlplib.fetch import references

    refs = references()
    pool = [r for r in run.instances(a.max_vars) if (run.LIB / "nl" / f"{r['name']}.nl").exists()]
    if not pool:
        sys.exit(f"no .nl files under {run.LIB / 'nl'}")
    picks = random.Random(a.seed).sample(pool, min(a.sample, len(pool)))

    def go(inst):
        r = run.solve("jaxglobal", inst, a.time_limit)
        return inst["name"], r, report.classify(r, refs[inst["name"]])

    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(go, picks))
    bad = []
    for name, r, cls in results:
        unverified = r.get("message") == "unverified"
        print(f"{name:28s} {cls:11s} {r['status']:11s} {float(r['time'] or 0):7.2f}s "
              f"primal {r['primal']} dual {r['dual']}" + ("  UNVERIFIED" if unverified else ""), flush=True)  # fmt: skip
        if cls == "wrong" or unverified:
            bad.append(name)
    counts = {c: sum(1 for _, _, k in results if k == c) for c in sorted({k for _, _, k in results})}
    print(f"{len(results)} instances (seed {a.seed}): {counts}")
    if bad:
        print("WRONG:", ", ".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()

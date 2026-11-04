"""Why instances fail even warm: classify the 2026-10-20 re-run's failures by cause.

For every instance of the four sets that jaxglobal did not solve warm (compiled kernels cached),
compare its incumbent and final bound with the MINLPLib reference optimum:

  unsupported         refused: no finite bounds were derived for some variables
  error               the run crashed (a traceback instead of a status)
  setup-bound         at most one node in the time limit: the root never finished
  bound-limited       incumbent within tolerance of the optimum, bound not (cannot prove it)
  incumbent-limited   bound within tolerance, incumbent not, or none (cannot find it)
  both                neither within tolerance

Tolerance: the solver's gap rule, max(1e-6, 1e-4 |optimum|) (with report.py's 1e-3 slack for
the incumbent). Writes results/failure_census.csv and prints the summary.

Usage:  python benchmarks/failure_census.py
"""

import csv
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report  # noqa: E402
from minlplib.fetch import references  # noqa: E402

R = references()


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


rows, seen = [], set()
for s in ["global50", "start50", "int50", "heldout"]:
    warm = {r["name"]: r for r in csv.DictReader(open(HERE / "results" / f"rerun_{s}_warm.csv"))}
    scip = {r["name"]: r for r in csv.DictReader(open(HERE / "results" / f"rerun_{s}_scip.csv"))}
    for name, r in warm.items():
        c = report.classify(r, R[name])
        if c == "solved" or name in seen:  # (an instance in two sets is counted once)
            continue
        seen.add(name)
        ref = R[name]
        sense = 1 if ref["objsense"] == "min" else -1
        best = sense * f(ref["primalbound"])
        p, d = sense * f(r["primal"]), sense * f(r["dual"])  # minimization form
        tol = max(1e-6, 1e-4 * abs(best)) if best == best else float("nan")
        nodes = int(f(r["nodes"])) if r["nodes"] not in ("", None) else 0
        if c == "unsupported":
            cause = "unsupported"
        elif r["status"] == "error" or "Traceback" in r["message"] or 'line ' in r["message"]:
            cause = "error"
        elif nodes <= 1:
            cause = "setup-bound"
        else:
            p_ok = p == p and abs(p - best) <= max(tol, report.tol(best))
            d_ok = d == d and best - d <= tol
            cause = "bound-limited" if p_ok and not d_ok else "incumbent-limited" if d_ok and not p_ok else \
                "both" if not (p_ok or d_ok) else "gap-rule"  # fmt: skip
        sc = scip.get(name)
        rows.append(dict(set=s, name=name, cause=cause, nvars=ref["nvars"], ncons=ref["ncons"],
                         ints=int(f(ref.get("nbinvars", 0)) + f(ref.get("nintvars", 0))) if ref.get("nbinvars") else 0,
                         nodes=nodes, time=round(f(r["time"]), 1), primal_gap=p - best if p == p else "",
                         dual_gap=best - d if d == d else "", scip=report.classify(sc, ref) if sc else "",
                         scip_time=round(f(sc["time"]), 2) if sc else "", message=r["message"][:80]))  # fmt: skip
out = HERE / "results" / "failure_census.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "failures (warm);", Counter(r["cause"] for r in rows))
print("SCIP solves:", {k: sum(r["scip"] == "solved" for r in rows if r["cause"] == k) for k in Counter(r["cause"] for r in rows)})
def _g(v):
    return f"{v:.3g}" if isinstance(v, float) else "-"


for k in ["error", "setup-bound", "bound-limited", "incumbent-limited", "both", "gap-rule", "unsupported"]:
    sel = [r for r in rows if r["cause"] == k]
    if sel:
        print(f"\n{k} ({len(sel)}):")
        for r in sorted(sel, key=lambda r: int(r["nvars"])):
            print(f"  {r['name']:30s} n={r['nvars']:>3s} m={r['ncons']:>3s} int={r['ints']:<3d} nodes={r['nodes']:<6d} "
                  f"primal gap {_g(r['primal_gap']):>10s} dual gap {_g(r['dual_gap']):>10s} SCIP {r['scip']:9s} {r['scip_time']!s:>6s}s  {r['message'][:45].splitlines()[0] if r['message'] else ''}")

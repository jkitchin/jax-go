"""Markdown table from flagship results.

Usage: python benchmarks/flagship_report.py results/flagship_<tag>.csv [more.csv ...]

One row per problem, one column per solver (verdict, time, and the gap when unsolved),
followed by the model-code line counts: the JAX function against the Pyomo builder.
"""

from __future__ import annotations

import csv
import pathlib
import sys

ORDER = ["jaxglobal", "scip", "couenne", "baron"]


def cell(r):
    if r is None:
        return "-"
    t = float(r["time"] or 0)
    v = r["verdict"]
    if v in ("correct", "wrong", "unknown") and r["status"] == "optimal":
        return f"{v} {t:.1f}s"
    p, d = r["primal"], r["dual"]
    bounds = f" [{float(d):.4g}, {float(p):.4g}]" if p not in ("", "nan") and d not in ("", "nan") else ""
    return f"{r['status']} {t:.0f}s{bounds}" if r["status"] != "error" else "error"


def main(paths):
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    solvers = [s for s in ORDER if any(r["solver"] == s for r in rows)]
    names = list(dict.fromkeys((r["cls"], r["name"]) for r in rows))
    by = {(r["solver"], r["name"]): r for r in rows}
    print("| class | problem | fstar | " + " | ".join(solvers) + " | LOC jax | LOC pyomo |")
    print("|---|---|---|" + "---|" * len(solvers) + "---|---|")
    for c, n in names:
        any_r = next(r for r in rows if r["name"] == n)
        fs = f"{float(any_r['fstar']):.6g}" if any_r["fstar"] else "?"
        cells = " | ".join(cell(by.get((s, n))) for s in solvers)
        print(f"| {c} | {n} | {fs} | {cells} | {any_r['loc_jax']} | {any_r['loc_pyomo']} |")
    n_ok = {s: sum(by.get((s, n), {}).get("verdict") == "correct" for _, n in names) for s in solvers}
    n_bad = {s: sum(by.get((s, n), {}).get("verdict") == "wrong" for _, n in names) for s in solvers}
    print()
    print("Correct / wrong / total: " + ", ".join(f"{s} {n_ok[s]}/{n_bad[s]}/{len(names)}" for s in solvers))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main([p if pathlib.Path(p).is_absolute() else pathlib.Path.cwd() / p for p in sys.argv[1:]])

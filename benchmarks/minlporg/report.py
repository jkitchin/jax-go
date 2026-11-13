"""Stage-1 report for the minlp.org set (issue #87): jaxglobal against SCIP, as a markdown table.

The reference for each instance is the better of minlp.org's best known value and SCIP's primal
value, with SCIP's dual bound (valid whatever SCIP's status). Each jaxglobal run is classified
with ``benchmarks/report.classify`` (solved / unsolved / wrong / unsupported, at its 1e-3
relative tolerance) and passes if it is solved within max(2 x SCIP time, 10 s), the rule of
docs/PLAN-scip-gap-v2.md.

Run:  python benchmarks/minlporg/report.py [--cold TAG] [--warm TAG] [--scip TAG]
"""

import argparse
import csv
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from report import classify  # noqa: E402


def load(tag):
    p = HERE.parent / "results" / f"{tag}.csv"
    return {(r["solver"], r["name"]): r for r in csv.DictReader(open(p))} if p.exists() else {}


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def fmt(v, digits=6):
    return "" if v != v else f"{v:.{digits}g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cold", default="minlporg_stage1_cold", help="results tag with jaxglobal and scip runs")
    ap.add_argument("--warm", default="minlporg_stage1_warm", help="results tag with a second jaxglobal run")
    ap.add_argument("--scip", default="", help="results tag with the scip runs (default: --cold)")
    ap.add_argument("--instances", default="instances.csv", help="instance list, e.g. instances_stage2.csv")
    a = ap.parse_args()
    cold, warm, scip = load(a.cold), load(a.warm), load(a.scip or a.cold)
    insts = list(csv.DictReader(open(HERE / a.instances)))
    print("| instance | type | vars (bin) | cons | reference | SCIP | jaxglobal cold | jaxglobal warm | class | pass (cold / warm) |")
    print("|---|---|---:|---:|---:|---|---|---|---|---|")
    counts = {}
    for inst in insts:
        name, s = inst["name"], (1 if inst["objsense"] == "min" else -1)
        sc = scip.get(("scip", name), {})
        best = [v for v in (num(inst["primalbound"]), num(sc.get("primal"))) if v == v]
        ref = dict(objsense=inst["objsense"], primalbound=str(min(best, key=lambda v: s * v)) if best else "",
                   dualbound=sc.get("dual", ""))  # fmt: skip
        st = num(sc.get("time"))
        cap = max(2 * st, 10.0) if sc.get("status") == "optimal" else math.nan
        cells, passes, cls = [], [], ""
        for res in (cold, warm):
            r = res.get(("jaxglobal", name))
            if not r:
                cells.append("")
                passes.append("")
                continue
            k = classify(r, ref)
            cls = cls or k
            t = num(r["time"])
            cells.append(f"{r['status']} {fmt(t, 3)} s, {fmt(num(r['primal']))}" if r["status"] != "unsupported" else "unsupported")
            passes.append("PASS" if k == "solved" and cap == cap and t <= cap else "-")
        counts[cls] = counts.get(cls, 0) + 1
        ref_txt = fmt(num(ref["primalbound"]))
        scip_txt = f"{sc.get('status', '')} {fmt(st, 3)} s, {fmt(num(sc.get('primal')))}" if sc else ""
        print(f"| {name} | {inst['probtype']} | {inst['nvars']} ({inst['nbinvars']}) | {inst['ncons']} | {ref_txt} | "
              f"{scip_txt} | {cells[0]} | {cells[1]} | {cls} | {passes[0]} / {passes[1]} |")  # fmt: skip
    print()
    print("classes (cold):", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()

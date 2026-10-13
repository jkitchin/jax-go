"""Select ``int50.txt``: 50 MINLPLib instances with integer variables (at most 50 variables).

From ``minlplib/instances_int.csv`` (``fetch.py --integer``), instances with a
reference solution closed to 1e-4, in four bands by number of integer variables
(1-5, 6-10, 11-20, 21-50). Within a band, instances are taken round-robin over
the cells (binary only or general integer) x (convex or not), in a random order
fixed by the seed, with at most 3 per model family.

Run:  python benchmarks/sets/make_int50.py > benchmarks/sets/int50.txt
"""

import csv
import pathlib
import random
import re

HERE = pathlib.Path(__file__).resolve().parent
BANDS = [(1, 5, 14), (6, 10, 12), (11, 20, 12), (21, 50, 12)]  # integers lo, hi, count
SEED = 2026


def family(name):
    f = re.sub(r"\d.*$", "", name)
    return name[:4] if f == "ex" else f  # ex1221, ex1252, ex14_1_1: ex12, ex14


def main():
    rows = list(csv.DictReader(open(HERE.parent / "minlplib" / "instances_int.csv")))
    ok = [r for r in rows if r["primalbound"] not in ("", "NA") and r["gap"] not in ("", "NA") and float(r["gap"]) <= 1e-4]
    rng = random.Random(SEED)
    used, out = {}, []
    for lo, hi, count in BANDS:
        band = [r for r in ok if lo <= int(r["nbinvars"]) + int(r["nintvars"]) <= hi]
        cells = {}
        for r in sorted(band, key=lambda r: r["name"]):
            key = ("general" if int(r["nintvars"]) else "binary", "convex" if r["convex"] == "True" else "nonconvex")
            cells.setdefault(key, []).append(r)
        for c in cells.values():
            rng.shuffle(c)
        taken = 0
        while taken < count and any(cells.values()):
            for key in sorted(cells):
                while cells[key] and used.get(family(cells[key][0]["name"]), 0) >= 3:
                    cells[key].pop(0)
                if cells[key] and taken < count:
                    r = cells[key].pop(0)
                    used[family(r["name"])] = used.get(family(r["name"]), 0) + 1
                    out.append((f"int{lo}-{hi}", r, key))
                    taken += 1
    print(f"# {len(out)} MINLPLib instances with integer variables; selection: benchmarks/sets/make_int50.py (seed {SEED})")
    print("# group        name                       nvars ncons  nbin  nint  kind     convexity")
    for g, r, key in out:
        print(f"{g:12s} {r['name']:26s} {int(r['nvars']):5d} {int(r['ncons']):5d} {int(r['nbinvars']):5d} {int(r['nintvars']):5d}"
              f"  {key[0]:8s} {key[1]}")


if __name__ == "__main__":
    main()

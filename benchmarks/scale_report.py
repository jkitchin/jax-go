"""Tables for the scale re-run (results/scale_int100_{cold,scip}.csv, see scale_2026_11.sh).

All 328 MINLPLib instances with integer variables and at most 100 variables (sets/int100.txt), 60 s, cold.
Rows are classified against the MINLPLib reference bounds with ``report.classify``. The stage-5 run of
2026-10-17 (results/int328_stage5.csv, jaxglobal only) is shown beside it for comparison.

Usage:  python benchmarks/scale_report.py
"""

import csv
import math
import pathlib
import statistics
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report  # noqa: E402
from minlplib.fetch import references  # noqa: E402

TL = 60.0
REFS = references()
BANDS = [(1, 10), (11, 20), (21, 50), (51, 100)]
RUNS = [("jaxglobal, 2026-10-17 (stage 5)", "int328_stage5", "jaxglobal"),
        ("jaxglobal, cold", "scale_int100_cold", "jaxglobal"),
        ("SCIP", "scale_int100_scip", "scip")]


def load(tag, solver):
    p = HERE / "results" / f"{tag}.csv"
    if not p.exists():
        return {}
    return {r["name"]: r for r in csv.DictReader(open(p)) if r["solver"] == solver}


def t(r):
    try:
        return min(float(r["time"]), TL)
    except (TypeError, ValueError):
        return TL


def sgm(times, shift=1.0):
    return math.exp(sum(math.log(x + shift) for x in times) / len(times)) - shift if times else float("nan")


def main():
    names = [ln.split()[0] for ln in open(HERE / "sets" / "int100.txt") if ln.strip() and not ln.startswith("#")]
    runs = [(label, load(tag, s)) for label, tag, s in RUNS]
    cls = [{n: report.classify(r, REFS[n]) for n, r in rows.items() if n in REFS} for _, rows in runs]

    head = "| run | rows | " + " | ".join(f"solved {a}-{b} vars" for a, b in BANDS) + \
        " | solved, all | wrong | unsupported | SGM s |"
    print(head)
    print("|---|---:|" + "---:|" * len(BANDS) + "---:|---:|---:|---:|")
    for (label, rows), c in zip(runs, cls):
        cells = []
        for a, b in BANDS:
            band = [n for n in names if a <= int(REFS[n]["nvars"]) <= b]
            cells.append(f"{sum(c.get(n) == 'solved' for n in band)} / {len(band)}")
        done = [n for n in names if n in rows]
        print(f"| {label} | {len(done)} | " + " | ".join(cells) +
              f" | {sum(c.get(n) == 'solved' for n in names)} | {sum(c.get(n) == 'wrong' for n in names)}"
              f" | {sum(c.get(n) == 'unsupported' for n in names)}"
              f" | {sgm([t(rows[n]) if c[n] == 'solved' else TL for n in done]):.1f} |")

    (_, old), (_, new), (_, scip) = runs
    cold, cs = cls[1], cls[2]
    print()
    for label, c in zip([r[0] for r in RUNS], cls):
        w = sorted(n for n, k in c.items() if k == "wrong")
        print(f"{label}: wrong {w}")
    gained = sorted(n for n in names if cold.get(n) == "solved" and cls[0].get(n) != "solved")
    lost = sorted(n for n in names if cls[0].get(n) == "solved" and n in new and cold.get(n) != "solved")
    print(f"solved now, not in stage 5 ({len(gained)}): {gained}")
    print(f"solved in stage 5, not now ({len(lost)}): {lost}")
    both = [n for n in names if cold.get(n) == "solved" and cs.get(n) == "solved"]
    print(f"solved by both jaxglobal (cold) and SCIP: {len(both)}; jaxglobal faster on "
          f"{sum(t(new[n]) < t(scip[n]) for n in both)}")
    only_jg = sorted(n for n in names if cold.get(n) == "solved" and cs.get(n) != "solved" and n in scip)
    print(f"solved by jaxglobal only ({len(only_jg)}): {only_jg}")
    big = [n for n in names if int(REFS[n]["nvars"]) > 50 and n in new]
    if big:
        nodes = [int(float(new[n]["nodes"])) for n in big if new[n].get("nodes") not in ("", None)]
        print(f"51-100 vars: median nodes reached {statistics.median(nodes) if nodes else 'n/a'} "
              f"({len(nodes)} rows with a node count); statuses "
              f"{dict(sorted(Counter(new[n]['status'] for n in big).items()))}")


if __name__ == "__main__":
    main()

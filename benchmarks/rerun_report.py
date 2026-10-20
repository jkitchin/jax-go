"""Tables for the October 2026 MINLPLib re-run (results/rerun_<set>_{cold,warm,scip}.csv, see rerun_2026_10.sh).

Every row is classified against the MINLPLib reference bounds with ``report.classify`` (solved / wrong / unsolved /
unsupported). An instance PASSES in a jaxglobal mode if jaxglobal solves it within max(2 x SCIP's time, 10 s), the rule
the README uses; SCIP's time is its own run in results/rerun_<set>_scip.csv, and an instance SCIP did not solve passes
when jaxglobal solves it within the 60 s limit.

Usage:  python benchmarks/rerun_report.py
"""

import csv
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report  # noqa: E402
from minlplib.fetch import references  # noqa: E402

SETS = ["global50", "start50", "int50", "heldout"]
TL = 60.0
REFS = references()


def load(tag):
    p = HERE / "results" / f"rerun_{tag}.csv"
    return {r["name"]: r for r in csv.DictReader(open(p))} if p.exists() else {}


def t(r):
    try:
        return min(float(r["time"]), TL)
    except (TypeError, ValueError):
        return TL


def sgm(times, shift=1.0):
    return math.exp(sum(math.log(x + shift) for x in times) / len(times)) - shift


def summarize(s):
    scip, out = load(f"{s}_scip"), {}
    cls = {"scip": {n: report.classify(r, REFS[n]) for n, r in scip.items()}}
    for mode in ("cold", "warm"):
        jg = load(f"{s}_{mode}")
        c = {n: report.classify(r, REFS[n]) for n, r in jg.items()}
        cls[mode] = c
        passed = []
        for n, k in c.items():
            if k != "solved":
                continue
            sc = scip.get(n)
            cap = max(2 * t(sc), 10.0) if sc and cls["scip"].get(n) == "solved" else TL
            if t(jg[n]) <= cap:
                passed.append(n)
        slow = [n for n, r in scip.items() if cls["scip"].get(n) == "solved" and t(r) >= 0.5]
        out[mode] = dict(
            n=len(jg),
            solved=sum(k == "solved" for k in c.values()),
            wrong=sorted(n for n, k in c.items() if k == "wrong"),
            unsupported=sum(k == "unsupported" for k in c.values()),
            passed=len(passed),
            hard_passed=sum(n in passed for n in slow),
            hard=len(slow),
            faster=sum(c[n] == "solved" and cls["scip"].get(n) == "solved" and t(jg[n]) < t(scip[n]) for n in c),
            sgm=sgm([t(jg[n]) if c[n] == "solved" else TL for n in c]),
            median_solved=statistics.median([t(jg[n]) for n in c if c[n] == "solved"] or [float("nan")]),
        )
    out["scip"] = dict(
        n=len(scip),
        solved=sum(k == "solved" for k in cls["scip"].values()),
        wrong=sorted(n for n, k in cls["scip"].items() if k == "wrong"),
        sgm=sgm([t(scip[n]) if cls["scip"][n] == "solved" else TL for n in scip]),
        unsolved=sorted(n for n, k in cls["scip"].items() if k != "solved"),
    )
    out["jg_unsolved"] = {m: sorted(n for n, k in cls[m].items() if k not in ("solved",)) for m in ("cold", "warm")}
    return out


def main():
    print("| set | instances | jaxglobal solved (cold / warm) | pass, cold | pass, warm | SCIP >= 0.5 s: pass (warm) | "
          "faster than SCIP (warm) | jaxglobal wrong | SCIP solved | SCIP wrong | SGM s jaxglobal cold / warm / SCIP |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    res = {s: summarize(s) for s in SETS}
    for s, r in res.items():
        c, w, q = r["cold"], r["warm"], r["scip"]
        print(f"| {s} | {c['n']} | {c['solved']} / {w['solved']} | {c['passed']} | {w['passed']} | {w['hard_passed']} / {w['hard']} | "
              f"{w['faster']} | {len(c['wrong']) + len(w['wrong'])} | {q['solved']} | {len(q['wrong'])} | "
              f"{c['sgm']:.1f} / {w['sgm']:.1f} / {q['sgm']:.2f} |")
    print()
    for s, r in res.items():
        print(f"{s}: jaxglobal wrong cold {r['cold']['wrong']} warm {r['warm']['wrong']}; SCIP wrong {r['scip']['wrong']}")
        print(f"   jaxglobal not solved (warm): {r['jg_unsolved']['warm']}")
        print(f"   SCIP not solved: {r['scip']['unsolved']}")
        print(f"   unsupported: {r['cold']['unsupported']}; median solve time (warm, solved): {r['warm']['median_solved']:.2f} s")


if __name__ == "__main__":
    main()

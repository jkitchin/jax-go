"""Tables for the October 2026 branching-rule comparison (results/branch_<rule>_*, see branching_2026_10.sh).

Part 1, the 25-problem suite (21 built-in + univariate_sum_10/20, nn_tanh_5x8x2, qcqp_10_5): nodes and warm time per
rule, geometric-mean node ratios against ``width`` over the problems every rule certified, and the total warm time.
Part 2, MINLPLib start50 / int50 (warm, 60 s): instances solved per rule (``report.classify`` against the reference
bounds), wrong answers, and the instances one rule solves that another does not.

Usage:  python benchmarks/branching_report.py
"""

import csv
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report  # noqa: E402
from minlplib.fetch import references  # noqa: E402

RULES = ["width", "pseudocost", "strong"]
RES = HERE / "results"


def suite(rule):
    rows = json.load(open(RES / f"branch_{rule}_builtin.json")) + json.load(open(RES / f"branch_{rule}_synthetic.json"))
    return {r["name"]: r for r in rows}


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def part1():
    s = {r: suite(r) for r in RULES}
    names = list(s["width"])
    print("| problem | " + " | ".join(f"nodes, {r}" for r in RULES) + " | " + " | ".join(f"warm s, {r}" for r in RULES) + " |")
    print("|---|" + "---:|" * (2 * len(RULES)))
    for n in names:
        nodes = [s[r][n].get("nodes") if s[r][n].get("status") == "optimal" else f"{s[r][n].get('status')}" for r in RULES]
        warm = [f"{s[r][n]['warm']:.2f}" if s[r][n].get("warm") is not None else "—" for r in RULES]
        print(f"| {n} | " + " | ".join(str(x) for x in nodes) + " | " + " | ".join(warm) + " |")
    ok = [n for n in names if all(s[r][n].get("status") == "optimal" for r in RULES)]
    print(f"\ncertified by every rule: {len(ok)} of {len(names)}")
    for r in RULES:
        ratio = gmean([max(s[r][n]["nodes"], 1) / max(s["width"][n]["nodes"], 1) for n in ok])
        tot = sum(s[r][n]["warm"] for n in ok if s[r][n].get("warm") is not None)
        fewer = sum(s[r][n]["nodes"] < s["width"][n]["nodes"] for n in ok)
        print(f"  {r:10s} nodes vs width (geomean) {ratio:.2f}; fewer nodes than width on {fewer}; total warm time {tot:.1f} s")


def part2():
    refs = references()
    for st in ("start50", "int50"):
        cls = {}
        for r in RULES:
            rows = {x["name"]: x for x in csv.DictReader(open(RES / f"branch_{r}_{st}.csv"))}
            cls[r] = {n: report.classify(x, refs[n]) for n, x in rows.items()}
        solved = {r: {n for n, k in cls[r].items() if k == "solved"} for r in RULES}
        wrong = {r: sorted(n for n, k in cls[r].items() if k == "wrong") for r in RULES}
        print(f"\n{st}: " + "; ".join(f"{r} solves {len(solved[r])} (wrong {wrong[r]})" for r in RULES))
        for a in RULES:
            for b in RULES:
                if a != b and solved[a] - solved[b]:
                    print(f"   {a} but not {b}: {sorted(solved[a] - solved[b])}")


if __name__ == "__main__":
    part1()
    part2()

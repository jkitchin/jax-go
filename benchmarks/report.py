"""Summarize a benchmark run: solved counts, shifted geometric mean, profiles.

Each result is checked against the MINLPLib reference bounds, not just the
solver's own claim:

  solved      claims optimal, primal value within TOL of the best known value
              and dual bound not past it
  wrong       claims optimal (or reports a bound) that contradicts the
              reference: a primal value better than the reference dual bound,
              or a dual bound worse than the reference primal value
  unsolved    time limit, error, or optimal claim on an instance whose
              reference gap is open and cannot be checked
  unsupported the solver cannot read or model the instance

TOL is 1e-3 relative (1e-6 absolute floor), looser than the 1e-4 gap the
solvers were given, so feasibility-tolerance noise is not counted as an error.

Usage:  python benchmarks/report.py [--tag baseline]
Writes  benchmarks/results/<tag>.md and <tag>_profile.png
"""

import argparse
import csv
import math
import pathlib
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
SHIFT = 1.0  # seconds, for the shifted geometric mean
BINS = [(1, 5), (6, 10), (11, 20), (21, 50)]


def tol(v):
    return max(1e-6, 1e-3 * abs(v))


def classify(r, ref):
    s = 1 if ref["objsense"] == "min" else -1  # compare in minimization form
    best, bound = s * float(ref["primalbound"] or "nan"), s * float(ref["dualbound"] or "nan")
    p, d = s * _f(r["primal"]), s * _f(r["dual"])
    if r["status"] == "unsupported":
        return "unsupported"
    # contradictions with the reference are errors whatever the solver claims
    if p == p and bound == bound and p < bound - tol(bound):
        return "wrong"  # "feasible" point better than the proven bound
    if d == d and best == best and d > best + tol(best):
        return "wrong"  # lower bound cuts off a known feasible value
    if r["status"] == "infeasible":
        return "wrong" if best == best else "solved"
    if r["status"] != "optimal":
        return "unsolved"
    if best == best and p == p and abs(p - best) <= tol(best):
        return "solved"
    if best == best and p == p and p > best + tol(best):
        return "wrong"  # claims optimal at a worse point
    return "unsolved"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--time-limit", type=float, default=None, help="defaults to the largest time recorded")
    a = ap.parse_args()
    from minlplib.fetch import references

    refs = references()
    rows = list(csv.DictReader(open(HERE / "results" / f"{a.tag}.csv")))
    solvers = list(dict.fromkeys(r["solver"] for r in rows))
    res = defaultdict(dict)
    for r in rows:
        res[r["solver"]][r["name"]] = (classify(r, refs[r["name"]]), _f(r["time"]), r)
    names = sorted(set.intersection(*(set(res[s]) for s in solvers)), key=lambda n: (int(refs[n]["nvars"]), n))
    tl = a.time_limit or max(t for s in solvers for _, t, _ in res[s].values() if t == t)
    tl = round(tl / 10) * 10 or tl

    def t_eff(s, n):
        c, t, _ = res[s][n]
        return min(t, tl) if c == "solved" else tl

    lines = [f"# Benchmark `{a.tag}`", "",
             f"{len(names)} continuous MINLPLib instances with ≤ 50 variables, time limit {tl:g} s, "
             f"relative gap 1e-4. Results are checked against the MINLPLib reference bounds "
             f"(see `benchmarks/report.py`).", "",
             "| solver | solved | wrong | unsolved | unsupported | SGM time (s) |", "|---|---|---|---|---|---|"]  # fmt: skip
    for s in solvers:
        c = Counter(res[s][n][0] for n in names)
        sgm = math.exp(sum(math.log(t_eff(s, n) + SHIFT) for n in names) / len(names)) - SHIFT
        lines.append(f"| {s} | {c['solved']} | {c['wrong']} | {c['unsolved']} | {c['unsupported']} | {sgm:.2f} |")
    lines += ["", "SGM: shifted geometric mean (shift 1 s) over all instances; unsolved, wrong and "
              "unsupported count as the time limit.", "", "## Solved by size", "",
              "| variables | instances | " + " | ".join(solvers) + " |",
              "|---|---|" + "---|" * len(solvers)]  # fmt: skip
    for lo, hi in BINS:
        sub = [n for n in names if lo <= int(refs[n]["nvars"]) <= hi]
        counts = [sum(res[s][n][0] == "solved" for n in sub) for s in solvers]
        lines.append(f"| {lo}–{hi} | {len(sub)} | " + " | ".join(map(str, counts)) + " |")
    wrong = [(s, n, res[s][n][2]) for s in solvers for n in names if res[s][n][0] == "wrong"]
    if wrong:
        lines += ["", "## Results that contradict the reference", "",
                  "| solver | instance | status | primal | dual | reference primal | reference dual |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
        for s, n, r in wrong:
            lines.append(f"| {s} | {n} | {r['status']} | {_f(r['primal']):.8g} | {_f(r['dual']):.8g} | "
                         f"{_f(refs[n]['primalbound']):.8g} | {_f(refs[n]['dualbound']):.8g} |")  # fmt: skip
    unsup = Counter((s, res[s][n][2]["message"][:60]) for s in solvers for n in names if res[s][n][0] == "unsupported")
    if unsup:
        lines += ["", "## Unsupported", ""] + [f"- {s}: {m or '(no message)'} ({k})" for (s, m), k in unsup.most_common()]
    lines += ["", f"![performance profile]({a.tag}_profile.png)", ""]
    (HERE / "results" / f"{a.tag}.md").write_text("\n".join(lines))
    print("\n".join(lines))
    profile(solvers, names, t_eff, tl, HERE / "results" / f"{a.tag}_profile.png")


def profile(solvers, names, t_eff, tl, path):
    """Dolan-More performance profile on time, plus fraction solved vs. time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    T = np.array([[t_eff(s, n) for n in names] for s in solvers])
    solved = T < tl
    best = np.where(solved.any(0), np.where(solved, T, np.inf).min(0), np.nan)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for i, s in enumerate(solvers):
        ratio = np.sort(np.where(solved[i], np.maximum(T[i], 1e-3) / np.maximum(best, 1e-3), np.inf))
        tau = np.logspace(0, 4, 400)
        a1.step(tau, [(ratio <= x).mean() for x in tau], where="post", label=s, lw=2)
        ts = np.logspace(-2, np.log10(tl), 400)
        a2.step(ts, [((T[i] <= x) & solved[i]).mean() for x in ts], where="post", label=s, lw=2)
    a1.set(xscale="log", xlabel="τ (time / best time)", ylabel="fraction of instances", title="Performance profile")
    a2.set(xscale="log", xlabel="time (s)", ylabel="fraction solved", title="Solved within time")
    for ax in (a1, a2):
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    a2.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()

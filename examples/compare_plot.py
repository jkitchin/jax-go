"""Figure + summary table from examples/compare_results.json (see compare.py)."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SOLVERS = ["jaxglobal", "scip", "couenne", "baron", "lindoglobal"]
LABEL = {"jaxglobal": "jaxglobal", "scip": "SCIP", "couenne": "Couenne", "baron": "BARON", "lindoglobal": "LindoGlobal"}
COLOR = {"jaxglobal": "#2a78d6", "scip": "#eb6834", "couenne": "#1baf7a", "baron": "#eda100", "lindoglobal": "#e87ba4"}
MARKER = {"jaxglobal": "o", "scip": "s", "couenne": "D", "baron": "^", "lindoglobal": "v"}


def classify(row, s):
    """('solved', time) or (failure kind, None)."""
    r = row[s]
    ref = row["fstar"] if row["fstar"] is not None else row["jaxglobal"]["fun"]
    st = r.get("status", "")
    if st.startswith("unsupported") or (s == "baron" and st == "noSolution"):
        return "unsupported", None
    if st in ("maxTimeLimit", "maxIterations", "max_iter"):
        return "time limit", None
    fun = r.get("fun")
    if fun is None:
        return "no solution", None
    if abs(fun - ref) > 1e-4 * (1 + abs(ref)):
        return "wrong optimum", None
    t = r.get("solver_time") or r["time"]  # GAMS: solver-reported time (excludes ~0.7 s launch)
    return "solved", t


def main():
    rows = json.load(open("examples/compare_results.json"))
    names = [r["name"] + (" *" if r["manual"] else "") for r in rows]

    fig, ax = plt.subplots(figsize=(11, 7.2))
    y = np.arange(len(rows))[::-1]
    fail_x = 150
    off = dict(zip(SOLVERS, np.linspace(-0.28, 0.28, len(SOLVERS))))
    for s in SOLVERS:
        xs, ys, fx, fy = [], [], [], []
        for yi, row in zip(y, rows):
            kind, t = classify(row, s)
            if kind == "solved":
                xs.append(t)
                ys.append(yi + off[s])
            else:
                fx.append(fail_x)
                fy.append(yi + off[s])
        ax.scatter(xs, ys, s=64, marker=MARKER[s], color=COLOR[s], edgecolor="white", linewidth=1.5,
                   label=LABEL[s], zorder=3)  # fmt: skip
        ax.scatter(fx, fy, s=64, marker=MARKER[s], facecolor="none", edgecolor=COLOR[s], linewidth=1.6, zorder=3)
    ax.axvline(60, color="0.55", lw=1, ls="--")
    ax.text(60, len(rows) - 0.3, " 60 s limit", color="0.35", fontsize=11, va="bottom")
    ax.axvspan(100, 250, color="0.94", zorder=0)
    ax.text(150, len(rows) - 0.3, "failed", ha="center", color="0.35", fontsize=11, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.008, 250)
    ax.set_yticks(y, names, fontsize=11)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    n_unc = sum(not r["constrained"] for r in rows)
    ax.axhline(y[n_unc - 1] - 0.5, color="0.75", lw=1)
    ax.set_xlabel("time to certified optimum (s, log scale)", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", color="0.9", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.45, -0.09), frameon=False, fontsize=12)
    fig.savefig("figures/benchmark.png", dpi=150, bbox_inches="tight")

    # summary table
    print("| solver | solved | unsupported | time limit | wrong / none | median time (solved) |")
    print("|---|---|---|---|---|---|")
    for s in SOLVERS:
        kinds = [classify(r, s) for r in rows]
        cnt = lambda k: sum(1 for kk, _ in kinds if kk == k)  # noqa: E731
        ts = [t for k, t in kinds if k == "solved"]
        print(f"| {LABEL[s]} | {cnt('solved')}/{len(rows)} | {cnt('unsupported')} | {cnt('time limit')} | "
              f"{cnt('wrong optimum') + cnt('no solution')} | {np.median(ts):.2f} s |")  # fmt: skip
    print()
    for r in rows:
        print(r["name"], {s: classify(r, s)[0] for s in SOLVERS if classify(r, s)[0] != "solved"},
              {s: r[s].get("status") for s in SOLVERS})  # fmt: skip


if __name__ == "__main__":
    main()

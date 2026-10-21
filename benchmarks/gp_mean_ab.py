"""A/B: the GP mean as the ``jnp`` formula ``k(x)^T alpha`` against the ``gp.gp_mean`` primitive.

For an A/B of another change (``gp_std``'s rule), run ``primitive`` in two checkouts.

Notebook 06's problems, its GP class (trained the same way) and its tolerances (abs/rel 1e-3).
Before the primitive, ``gp_mean`` was the formula, so ``formula`` is the old behaviour.
Each case runs in its own process; ``--repeat 2`` reports the first solve's time (compiling, or
loading kernels from the cache) and the second's (warm). ``JAXGLOBAL_CACHE=0`` for cold first times.

    python benchmarks/gp_mean_ab.py --list
    python benchmarks/gp_mean_ab.py CASE MODE [--repeat 2]      # one row of CSV to stdout
    python benchmarks/gp_mean_ab.py --all --jobs 2 > benchmarks/results/gp_mean_ab.csv
"""

import argparse
import csv
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = ["mean_N20", "mean_N50", "mean_N100", "mean_N200", "mean_N50_matern32", "mean_N50_se",
         "lcb", "ei", "pi", "pi0", "lcb_N50", "ei_N50", "chance", "mean_constraint"]  # fmt: skip
MODES = ["formula", "primitive"]
FIELDS = ["case", "mode", "status", "fun", "lower_bound", "nodes", "iterations", "time_s", "first_time_s"]


def notebook_namespace():
    """Notebook 06 up to its first case: imports, the GP class, peaks, lhs."""
    src = (ROOT / "notebooks" / "src" / "06-gaussian-processes.py").read_text()
    head = src[: src.index("# ## 1. Minimizing the GP mean")]
    defs = src[src.index("def peaks(x):") : src.index("X = lhs(50)")]
    ns = {"__name__": "nb06"}
    cwd = os.getcwd()
    os.chdir(ROOT / "notebooks")  # the notebook puts ROOT = cwd.parent on sys.path
    try:
        exec(compile(head + "\n" + defs, "06-gaussian-processes.py", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


def problem(case, mode):
    ns = notebook_namespace()
    np, jnp, GP, lhs, peaks = ns["np"], ns["jnp"], ns["GP"], ns["lhs"], ns["peaks"]
    from jaxglobal.gp import expected_improvement, gp_mean, probability_of_improvement

    def mean_of(gp):
        if mode == "formula":
            return gp.mean
        return lambda x: gp.ym + gp.ys * gp_mean(gp.scale(x), gp.Xs, gp.lam, gp.alpha, gp.sf2, kernel=gp.kernel)

    LO, HI = ns["LO"], ns["HI"]
    if case.startswith("mean_N"):
        parts = case.split("_")
        n, kernel = int(parts[1][1:]), (parts[2] if len(parts) > 2 else "matern52")
        X = lhs(n)
        return dict(f=mean_of(GP(X, peaks(X), LO, HI, kernel=kernel)), lo=LO, hi=HI)
    if case.split("_")[0] in ("lcb", "ei", "pi", "pi0"):
        n = int(case.split("_N")[1]) if "_N" in case else 20
        Xa = lhs(n)
        ya = np.asarray(peaks(Xa))
        gpa = GP(Xa, ya, LO, HI)
        mu, fmin = mean_of(gpa), float(ya.min())
        f = {"lcb": lambda x: mu(x) - 2.0 * gpa.std(x),
             "ei": lambda x: -expected_improvement(mu(x), gpa.std(x), fmin),
             "pi": lambda x: -probability_of_improvement(mu(x), gpa.std(x), fmin - 0.1 * gpa.ys),
             "pi0": lambda x: -probability_of_improvement(mu(x), gpa.std(x), fmin)}[case.split("_")[0]]  # fmt: skip
        return dict(f=f, lo=LO, hi=HI)
    CLO, CHI, C = np.zeros(2), np.ones(2), 5.0
    Xc = lhs(40, seed=1, lo=CLO, hi=CHI)
    rng = np.random.default_rng(1)
    sty = 150 * Xc[:, 0] + 80 * Xc[:, 1] + 40 * np.sin(6 * Xc[:, 0]) * np.cos(4 * Xc[:, 1])
    imp = 1 + 6 * Xc[:, 0] ** 2 + 3 * Xc[:, 1] + 0.5 * np.sin(5 * Xc[:, 1])
    gp_sty = GP(Xc, sty + 2.0 * rng.normal(size=40), CLO, CHI, noise=(1e-6, 0.1))
    gp_imp = GP(Xc, imp + 0.2 * rng.normal(size=40), CLO, CHI, noise=(1e-6, 0.1))
    m_sty, m_imp = mean_of(gp_sty), mean_of(gp_imp)
    if case == "chance":
        ineq = lambda x: m_imp(x) + 1.96 * gp_imp.std(x) - C  # noqa: E731
    else:
        ineq = lambda x: m_imp(x) - C  # noqa: E731
    return dict(f=lambda x: -m_sty(x), lo=CLO, hi=CHI, ineq=ineq)


def run_one(case, mode, repeat):
    from jaxglobal import minimize

    p = problem(case, mode)
    kw = {"ineq": p["ineq"]} if "ineq" in p else {}
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        r = minimize(p["f"], p["lo"], p["hi"], abs_tol=1e-3, rel_tol=1e-3, time_limit=300, **kw)
        times.append(time.perf_counter() - t)
    w = csv.DictWriter(sys.stdout, FIELDS)
    w.writerow(dict(case=case, mode=mode, status=r.status, fun=f"{r.fun:.6g}", lower_bound=f"{r.lower_bound:.6g}",
                    nodes=r.nodes, iterations=r.iterations, time_s=f"{times[-1]:.1f}",
                    first_time_s=f"{times[0]:.1f}"))


def run_all(jobs, repeat):
    from concurrent.futures import ThreadPoolExecutor

    def go(cm):
        cmd = [sys.executable, __file__, *cm, "--repeat", str(repeat)]
        out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "RUST_LOG": "warn"})
        rows = [ln for ln in out.stdout.splitlines() if ln.startswith(cm[0] + ",")]
        return rows[-1] if rows else f"{cm[0]},{cm[1]},error,,,,,,"

    print(",".join(FIELDS), flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        for row in ex.map(go, [(c, m) for c in CASES for m in MODES]):
            print(row, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?")
    ap.add_argument("mode", nargs="?", choices=MODES)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(CASES))
    elif a.all:
        run_all(a.jobs, a.repeat)
    else:
        run_one(a.case, a.mode, a.repeat)

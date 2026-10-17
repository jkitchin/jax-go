"""R0 of docs/PLAN-scip-gap-v2.md: the cold / warm re-baseline, in one command.

``run`` does every R0 measurement in order, each under the shared benchmark lock
(``mkdir /private/tmp/claude-501/jaxglobal-bench.lock``) and behind the load gate
of plan §3 (1-min load < 10):

  1. start50 cold    run.py --solvers jaxglobal --jobs 4 --time-limit 60, JAXGLOBAL_CACHE=0
  2. start50 prime   the same with an empty JAXGLOBAL_CACHE_DIR (fills the cache)
  3. start50 warm    the same again on the primed directory
  4-6. int50 cold / prime / warm
  7. the seven W0 instances, one at a time, profile_root.py --modes cold at 60 s and
     120 s, 3 repeats (mode recorded as cold_tl60_r1 ... cold_tl120_r3)

The lock is held for the whole of each set-level run and for each sweep of the
seven instances (one time limit, one repeat). Before each, the gate polls the
load every ``--poll`` s until it is < ``--max-load``; after ``--max-wait`` s it
proceeds anyway and the step is flagged ``gate=timeout`` in the load log.
Everything resumes: run.py skips instances already in its CSV, and the seven-
instance sweep skips (instance, mode) pairs already in its CSV.

Outputs (benchmarks/results/):
  scipgap_r0_{start50,int50}_{cold,prime,warm}.csv   run.py rows, with a per-instance load column
  scipgap_r0_seven_cold.csv                         profile_root.py rows
  scipgap_r0_loads.csv                              one row per step: load before / after, gate outcome

``summary`` prints the markdown tables for docs/ROADMAP.md from those CSVs:
solved / wrong / SGM per set and mode (``report.classify``) beside SCIP on the
same instances, and the median of the 3 repeats per seven-instance run.

Usage:
  python benchmarks/scipgap_r0.py run [--steps start50,int50,seven] [--max-wait 1800]
  python benchmarks/scipgap_r0.py summary
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import shutil
import signal
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
LOCK = pathlib.Path("/private/tmp/claude-501/jaxglobal-bench.lock")
CACHE = pathlib.Path("/private/tmp/claude-501/jaxglobal-r0-cache")
SEVEN = [
    "ex6_2_6",
    "ramsey",
    "portfol_buyin",
    "hadamard_5",
    "cvxnonsep_nsig30",
    "smallinvDAXr3b010-011",
    "batch",
]
SETS = ["start50", "int50"]
TLS = [60, 120]
REPEATS = 3
LOADLOG = RES / "scipgap_r0_loads.csv"


def set_names(s):
    return [
        ln.split()[1]
        for ln in (HERE / "sets" / f"{s}.txt").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def load1():
    return os.getloadavg()[0]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- lock and load gate


class Lock:
    """The shared benchmark lock (atomic mkdir); waits while another run holds it."""

    def __init__(self, step, poll):
        self.step, self.poll = step, poll

    def __enter__(self):
        for i in range(10**9):
            try:
                LOCK.mkdir()
                break
            except FileExistsError:
                if i % 10 == 0:
                    owner = (
                        (LOCK / "owner").read_text().strip()
                        if (LOCK / "owner").exists()
                        else "unknown owner"
                    )
                    log(f"{self.step}: waiting for {LOCK} ({owner})")
                time.sleep(self.poll)
        (LOCK / "owner").write_text(
            f"scipgap_r0.py pid {os.getpid()} step {self.step}\n"
        )
        return self

    def __exit__(self, *exc):
        shutil.rmtree(LOCK, ignore_errors=True)


def gate(step, max_load, max_wait, poll):
    """Wait until the 1-min load is below max_load (at most max_wait s). Returns (load, outcome)."""
    t = time.time()
    while (ld := load1()) >= max_load:
        if time.time() - t >= max_wait:
            log(
                f"{step}: load {ld:.1f} still >= {max_load} after {max_wait:.0f} s; proceeding (flagged)"
            )
            return ld, "timeout"
        log(f"{step}: load {ld:.1f} >= {max_load}, waiting")
        time.sleep(poll)
    return ld, "ok"


def record(step, start, load_before, outcome, load_after):
    new = not LOADLOG.exists()
    with open(LOADLOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(
                [
                    "step",
                    "start",
                    "end",
                    "load_before",
                    "gate",
                    "load_after",
                    "uptime_after",
                ]
            )
        up = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
        w.writerow(
            [
                step,
                start,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{load_before:.2f}",
                outcome,
                f"{load_after:.2f}",
                up,
            ]
        )


def step(name, fn, a):
    with Lock(name, a.poll):
        ld, outcome = gate(name, a.max_load, a.max_wait, a.poll)
        start = time.strftime("%Y-%m-%d %H:%M:%S")
        log(f"{name}: start, load {ld:.2f} ({outcome})")
        fn()
        record(name, start, ld, outcome, load1())
        log(f"{name}: done")


# ---------------------------------------------------------------- steps


def env_for(mode, s):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("JAXGLOBAL_", "JAX_COMPILATION_CACHE"))
    }
    env["PYTHONPATH"] = str(ROOT)
    if mode == "cold":
        env["JAXGLOBAL_CACHE"] = "0"
    else:
        env["JAXGLOBAL_CACHE_DIR"] = str(CACHE / s)
    return env


def run_set(s, mode, a):
    tag = f"scipgap_r0_{s}_{mode}"
    if mode == "prime" and not (RES / f"{tag}.csv").exists():
        shutil.rmtree(
            CACHE / s, ignore_errors=True
        )  # a fresh prime starts from an empty cache
    cmd = [sys.executable, str(HERE / "run.py"), "--solvers", "jaxglobal", "--time-limit", str(a.time_limit),
           "--jobs", str(a.jobs), "--tag", tag, "--names", ",".join(set_names(s))]  # fmt: skip
    subprocess.run(cmd, env=env_for(mode, s), check=True)


def run_seven(tl, rep, a):
    out = RES / "scipgap_r0_seven_cold.csv"
    label = f"_tl{tl}_r{rep}"
    done = set()
    if out.exists():
        done = {
            (r["instance"], r["mode"])
            for r in csv.DictReader(open(out))
            if r["category"] == "total"
        }
    env = env_for("cold", "")
    for name in SEVEN:
        if (name, "cold" + label) in done:
            continue
        cmd = [sys.executable, str(HERE / "profile_root.py"), "--names", name, "--modes", "cold",
               "--time-limit", str(tl), "--label", label, "--out", str(out),
               "--json-dir", str(CACHE / "json")]  # fmt: skip
        subprocess.run(cmd, env=env, check=True)


def cmd_run(a):
    chk = subprocess.run([sys.executable, "-c", "import jaxglobal; print(jaxglobal.__file__)"],
                         env=env_for("cold", ""), capture_output=True, text=True, cwd="/")  # fmt: skip
    assert pathlib.Path(chk.stdout.strip()).resolve().is_relative_to(ROOT), (
        f"jaxglobal not from {ROOT}: {chk.stdout}"
    )
    need = sorted({n for s in SETS for n in set_names(s)} | set(SEVEN))
    missing = [n for n in need if not (HERE / "minlplib" / "nl" / f"{n}.nl").exists()]
    assert not missing, (f"{len(missing)} .nl files missing from {HERE / 'minlplib' / 'nl'} (e.g. {missing[:3]}); in a "
                         "worktree, symlink the main checkout's benchmarks/minlplib/nl there")  # fmt: skip
    signal.signal(
        signal.SIGTERM, lambda *_: sys.exit(143)
    )  # so the lock is released on kill
    CACHE.mkdir(parents=True, exist_ok=True)
    steps = a.steps.split(",")
    for s in SETS:
        if s in steps:
            for mode in ("cold", "prime", "warm"):
                step(f"{s}_{mode}", lambda s=s, mode=mode: run_set(s, mode, a), a)
    if "seven" in steps:
        for rep in range(1, REPEATS + 1):
            for tl in TLS:
                step(
                    f"seven_tl{tl}_r{rep}",
                    lambda tl=tl, rep=rep: run_seven(tl, rep, a),
                    a,
                )


# ---------------------------------------------------------------- summary


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def set_table(refs, classify):
    tl = 60.0
    ref_csv = {
        "start50": (RES / "baseline.csv", "scip"),
        "int50": (RES / "int50_scip.csv", "scip-gams"),
    }
    lines = ["| set | run | solved | wrong | unsolved | unsupported | SGM (s) | load min / median / max |",
             "|---|---|---:|---:|---:|---:|---:|---|"]  # fmt: skip
    for s in SETS:
        names = set_names(s)
        runs = [
            (f"jaxglobal {m}", RES / f"scipgap_r0_{s}_{m}.csv", "jaxglobal")
            for m in ("cold", "prime", "warm")
        ]
        runs.append((f"SCIP (`{ref_csv[s][0].name}`)",) + ref_csv[s])
        for label, path, solver in runs:
            if not path.exists():
                lines.append(f"| {s} | {label} | pending | | | | | |")
                continue
            rows = {
                r["name"]: r
                for r in csv.DictReader(open(path))
                if r["solver"] == solver and r["name"] in names
            }
            cls = {
                n: classify(rows[n], refs[n]) if n in rows else "missing" for n in names
            }
            c = {
                k: sum(v == k for v in cls.values())
                for k in ("solved", "wrong", "unsolved", "unsupported", "missing")
            }
            te = [
                min(_f(rows[n]["time"]), tl) if cls[n] == "solved" else tl
                for n in names
            ]
            sgm = math.exp(sum(math.log(t + 1) for t in te) / len(te)) - 1
            loads = [
                _f(r.get("load"))
                for r in rows.values()
                if _f(r.get("load")) == _f(r.get("load"))
            ]
            ld = (
                f"{min(loads):.1f} / {statistics.median(loads):.1f} / {max(loads):.1f}"
                if loads
                else "n/a"
            )
            miss = f" ({c['missing']} missing)" if c["missing"] else ""
            lines.append(f"| {s} | {label}{miss} | {c['solved']} | {c['wrong']} | {c['unsolved']} | "
                         f"{c['unsupported']} | {sgm:.1f} | {ld} |")  # fmt: skip
    return lines


def seven_table():
    path = RES / "scipgap_r0_seven_cold.csv"
    if not path.exists():
        return ["(seven-instance runs pending)"]
    runs = {}
    for r in csv.DictReader(open(path)):
        d = runs.setdefault(
            (r["instance"], r["mode"]),
            dict(
                load=r["load_1_5_15"].split()[0] if r["load_1_5_15"] else "", stages=[]
            ),
        )
        if r["category"] == "total" and r["item"] == "minimize_wall":
            d["wall"] = _f(r["seconds"])
        elif r["category"] == "info" and r["item"] in (
            "status",
            "nodes",
            "fun",
            "lower_bound",
        ):
            d[r["item"]] = r["note"]
        elif r["category"] == "stage":
            d["stages"].append((r["item"], _f(r["seconds"])))
        elif r["category"] == "main_in_root_bound" and r["item"] == "run:bound":
            d["bound_ran"] = True
        elif r["category"] == "error":
            d["status"] = "error"
    lines = ["| instance | limit | wall s | overrun s | status (3 runs) | nodes | primal | dual | first bound s | load (1 min) |",
             "|---|---:|---:|---:|---|---:|---|---|---:|---|"]  # fmt: skip
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    for name in SEVEN:
        for tl in TLS:
            reps = [
                runs[k]
                for k in sorted(runs)
                if k[0] == name and k[1].startswith(f"cold_tl{tl}_r")
            ]
            if not reps:
                lines.append(f"| {name} | {tl} | pending | | | | | | | |")
                continue
            walls = [d["wall"] for d in reps if "wall" in d]
            fb = []
            for d in reps:  # first bound: stages in mark order up to the end of root_bound, if its kernel ran
                if d.get("bound_ran"):
                    names = [x for x, _ in d["stages"]]
                    if "root_bound" in names:
                        fb.append(
                            sum(
                                v
                                for _, v in d["stages"][: names.index("root_bound") + 1]
                            )
                        )
            fbs = (
                f"{med(fb):.1f}"
                if len(fb) == len(reps)
                else (f"{med(fb):.1f} ({len(fb)}/{len(reps)})" if fb else "never")
            )
            lines.append(
                f"| {name} | {tl} | {med(walls):.1f} | {med(walls) - tl:+.1f} | {', '.join(d.get('status', '?') for d in reps)} | "
                f"{med([_f(d.get('nodes')) for d in reps]):.0f} | {med([_f(d.get('fun')) for d in reps]):.6g} | "
                f"{med([_f(d.get('lower_bound')) for d in reps]):.6g} | {fbs} | {', '.join(d['load'] for d in reps)} |")  # fmt: skip
    return lines


def cmd_summary(a):
    sys.path.insert(0, str(HERE))
    from minlplib.fetch import references
    from report import classify

    refs = references()
    out = ["Set-level runs (60 s, 4 jobs, `report.classify`):", ""] + set_table(
        refs, classify
    )
    out += [
        "",
        "Seven W0 instances, cold, one at a time; medians of 3 repeats (primal / dual in minimize's sense):",
        "",
    ]
    out += seven_table()
    if LOADLOG.exists():
        out += [
            "",
            "Load per step (`scipgap_r0_loads.csv`):",
            "",
            "| step | start | load before | gate | load after |",
            "|---|---|---:|---|---:|",
        ]
        for r in csv.DictReader(open(LOADLOG)):
            out.append(
                f"| {r['step']} | {r['start']} | {r['load_before']} | {r['gate']} | {r['load_after']} |"
            )
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["run", "summary"])
    ap.add_argument("--steps", default="start50,int50,seven")
    ap.add_argument("--time-limit", type=float, default=60.0, help="set-level runs")
    ap.add_argument("--jobs", type=int, default=4, help="set-level runs")
    ap.add_argument("--max-load", type=float, default=10.0)
    ap.add_argument(
        "--max-wait",
        type=float,
        default=1800.0,
        help="s to wait for the load gate before proceeding",
    )
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args()
    cmd_run(a) if a.command == "run" else cmd_summary(a)


if __name__ == "__main__":
    main()

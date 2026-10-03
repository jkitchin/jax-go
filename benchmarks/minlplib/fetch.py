"""Download the continuous MINLPLib instances used for benchmarking.

Selection: no binary, integer or semicontinuous variables, an .nl file
available, and at most MAX_VARS variables. Writes nl/<name>.nl, gms/<name>.gms
(for solvers run through GAMS, e.g. BARON) and instances.csv (name, nvars, ncons, probtype, primal/dual bound, gap).

Run:  python benchmarks/minlplib/fetch.py
"""

import csv
import pathlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).parent
MAX_VARS = 50
UA = {"User-Agent": "jaxglobal-benchmark/0.1"}
FIELDS = ["name", "nvars", "ncons", "probtype", "objsense", "primalbound", "dualbound", "gap", "convex"]


def get(url, path):
    if path.exists():
        return
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        path.write_bytes(r.read())


def main():
    get("https://www.minlplib.org/instancedata.csv", HERE / "instancedata.csv")
    rows = list(csv.DictReader(open(HERE / "instancedata.csv"), delimiter=";"))
    sel = [r for r in rows
           if r["nbinvars"] == r["nintvars"] == r["nsemi"] == "0"
           and "nl" in r["formats"] and int(r["nvars"]) <= MAX_VARS]  # fmt: skip
    sel.sort(key=lambda r: (int(r["nvars"]), r["name"]))
    jobs = []
    for fmt in ("nl", "gms"):
        (HERE / fmt).mkdir(exist_ok=True)
        jobs += [(f"https://www.minlplib.org/{fmt}/{r['name']}.{fmt}", HERE / fmt / f"{r['name']}.{fmt}") for r in sel]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: get(*j), jobs))
    with open(HERE / "instances.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in sel)
    print(f"{len(sel)} instances")


if __name__ == "__main__":
    main()

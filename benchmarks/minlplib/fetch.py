"""Download the MINLPLib instances used for benchmarking.

Selection: an .nl file available, at most MAX_VARS variables, no semicontinuous
or SOS variables, and no binary or integer variables (default) or at least one
(``--integer``). Writes nl/<name>.nl, gms/<name>.gms (for solvers run through
GAMS, e.g. BARON) and instances.csv, or instances_int.csv for ``--integer``
(name, nvars, ncons, probtype, primal/dual bound, gap, convexity, and the
numbers of binary and integer variables).

Run:  python benchmarks/minlplib/fetch.py [--integer]
"""

import csv
import pathlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).parent
MAX_VARS = 50
UA = {"User-Agent": "jaxglobal-benchmark/0.1"}
FIELDS = ["name", "nvars", "ncons", "probtype", "objsense", "primalbound", "dualbound", "gap", "convex"]
INT_FIELDS = FIELDS + ["nbinvars", "nintvars"]


def references():
    """Reference rows by name from instances.csv and (if downloaded) instances_int.csv."""
    refs = {}
    for f in ("instances.csv", "instances_int.csv"):
        if (HERE / f).exists():
            refs.update({r["name"]: r for r in csv.DictReader(open(HERE / f))})
    return refs


def get(url, path):
    if path.exists():
        return
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        path.write_bytes(r.read())


def main(integer=False):
    get("https://www.minlplib.org/instancedata.csv", HERE / "instancedata.csv")
    rows = list(csv.DictReader(open(HERE / "instancedata.csv"), delimiter=";"))
    if integer:
        sel = [r for r in rows
               if int(r["nbinvars"]) + int(r["nintvars"]) > 0 and r["nsemi"] == r["nsos1"] == r["nsos2"] == "0"
               and "nl" in r["formats"] and int(r["nvars"]) <= MAX_VARS]  # fmt: skip
    else:
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
    fields = INT_FIELDS if integer else FIELDS
    with open(HERE / ("instances_int.csv" if integer else "instances.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in sel)
    print(f"{len(sel)} instances")


if __name__ == "__main__":
    main(integer="--integer" in sys.argv[1:])

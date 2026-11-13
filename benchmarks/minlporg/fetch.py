"""Download small nonlinear instances from the minlp.org library and convert them to .nl.

minlp.org (the CMU-IBM Cyber-Infrastructure for MINLP) publishes its instances as GAMS (.gms) or
MPS files. This script lists all instances (``library/instances.php``), keeps the nonlinear ones
(not LP/MILP) with at most MAX_VARS variables and MAX_CONS constraints whose input is a single
.gms or .mps file, downloads them to src/, and converts each to nl/<name>.nl with GAMS CONVERT
(the GAMS system that comes with the ``gamspy`` package and its demo license is enough at this
size). Before converting, ``option <modeltype> = <solver>;`` lines are removed (they name solvers
that may not be installed); a model with several solve statements is converted at its last
one. MPS files are read by SCIP (pyscipopt) and written as GAMS first; ``mps2gms`` changed the
quadratic MPS models in this set.

Writes instances.csv with the columns of benchmarks/minlplib/instances.csv (``primalbound`` is
minlp.org's "best known objective value"; there is no dual bound) plus the minlp.org instance
number, title and problem page. Instances that fail to convert are listed in skipped.csv with
the reason.

Run:  python benchmarks/minlporg/fetch.py [--max-vars N] [--max-cons N]                  (stage 1)
      python benchmarks/minlporg/fetch.py --max-vars 300 --max-cons 300 --above 60 \
          --per-problem 3 --out instances_stage2.csv                                    (stage 2)
"""

import argparse
import csv
import html
import pathlib
import re
import shutil
import subprocess
import tempfile
import sys
import urllib.request

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
SITE = "https://www.minlp.org/library/"
UA = {"User-Agent": "jaxglobal-benchmark/0.1"}
FIELDS = ["name", "nvars", "ncons", "probtype", "objsense", "primalbound", "dualbound", "gap", "convex",
          "nbinvars", "nintvars", "id", "title", "url"]  # fmt: skip
LINEAR = {"LP", "MILP", "MIP"}
TYPES = "lp|mip|rmip|nlp|dnlp|minlp|rminlp|qcp|miqcp|rmiqcp|mcp|cns|mpec|emp"  # "option <type> = <solver>;"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
        return r.read()


def listing():
    """All instances: dicts with id, title, nature, discrete, continuous, constraints, best, inputs, page."""
    s = get(SITE + "instances.php?rec_num=1000&limit=0").decode("utf-8", "ignore")
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", s, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) < 9:
            continue
        txt = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip() for c in cells]
        links = re.findall(r'download\.php\?file=([^&"]+)&(?:amp;)?location=([^"&]+)', row)
        page = re.findall(r"problem/index\.php\?i=(\d+)", row)
        out.append(dict(id=int(txt[0]), title=txt[2], nature=txt[4], discrete=_int(txt[5]), continuous=_int(txt[6]),
                        constraints=_int(txt[7]), best=txt[8], page=f"{SITE}problem/index.php?i={page[0]}" if page else "",
                        problem=page[0] if page else txt[0],
                        inputs=[(f, loc) for f, loc in links if "/input/" in loc]))  # fmt: skip
    return out


def _int(x):
    try:
        return int(x.replace(",", ""))
    except ValueError:
        return None


def select(rows, max_vars, max_cons, above=0, per_problem=0):
    """Nonlinear instances with at most ``max_vars`` variables and ``max_cons`` constraints and, with
    ``above``, more than ``above`` variables or constraints (stage 2: larger than stage 1); with
    ``per_problem``, only the smallest ``per_problem`` instances of each minlp.org problem."""
    keep = []
    for r in rows:
        d, c, m = r["discrete"], r["continuous"], r["constraints"]
        if r["nature"] in LINEAR or None in (d, c, m) or not 0 < d + c <= max_vars or m > max_cons:
            continue
        if above and d + c <= above and m <= above:
            continue
        keep.append(r)
    if per_problem:
        groups = {}
        for r in keep:
            groups.setdefault(r["problem"], []).append(r)
        keep = sorted((r for g in groups.values() for r in sorted(g, key=lambda r: r["discrete"] + r["continuous"])[:per_problem]),
                      key=lambda r: r["id"])  # fmt: skip
    return keep


def name_of(r, fname):
    stem = re.sub(r"_\d+_[0-9a-f]{7}$", "", pathlib.Path(fname).stem)  # drop minlp.org's version suffix
    return f"m{r['id']}_" + re.sub(r"[^A-Za-z0-9_]+", "_", stem)


def gams_dir():
    import gamspy_base

    return pathlib.Path(gamspy_base.directory)


def convert(src, nl):
    """GAMS (or MPS) file -> .nl with CONVERT; raises RuntimeError with the GAMS log tail on failure."""
    g = gams_dir()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        if src.suffix.lower() == ".mps":
            # through SCIP's MPS reader and GAMS writer: mps2gms changed the quadratic models here
            # (the four Haverly instances lost about 20% of their optimal value)
            import pyscipopt

            m = pyscipopt.Model()
            m.hideOutput()
            m.readProblem(str(src))
            m.writeProblem(str(tmp / "model.gms"), verbose=False)
        else:
            s = src.read_text(errors="ignore")
            (tmp / "model.gms").write_text(re.sub(rf"(?im)^\s*options?\s+({TYPES}|solver)\s*=\s*\w+\s*;", "", s))
        # the same options under every option-file number a model may select (``m.optfile = k``)
        for ext in ["opt"] + [f"op{k}" for k in range(2, 10)]:
            (tmp / f"convert.{ext}").write_text("AmplNL out.nl\n")
        solvers = [f"{t}=convert" for t in TYPES.split("|") if t not in ("mcp", "cns", "emp")]
        p = subprocess.run([str(g / "gams"), "model.gms", "lo=3", "optfile=1", *solvers], cwd=tmp, capture_output=True, text=True)
        if not (tmp / "out.nl").exists():
            raise RuntimeError("gams: " + " | ".join(ln for ln in p.stdout.splitlines() if "rror" in ln)[-300:])
        shutil.copy(tmp / "out.nl", nl)


def header(nl):
    """nvars, ncons, nbinvars, nintvars and the objective sense of an ASCII .nl file.

    The header counts discrete variables in five groups (linear binary, linear integer, and
    nonlinear in constraints / objective / both), so binaries are told from general integers by
    their bounds, from jaxglobal's reader."""
    import numpy as np

    from jaxglobal.nl import read_nl

    lines = nl.read_text(errors="ignore").splitlines()
    nv, nc = (int(v) for v in lines[1].split()[:2])
    m = re.search(r"(?m)^O\d+ (\d)", "\n".join(lines))
    M = read_nl(nl, integers="mask")
    integer = np.asarray(M.integer, bool)
    binary = integer & (np.asarray(M.lo) >= 0) & (np.asarray(M.hi) <= 1)
    return nv, nc, int(binary.sum()), int((integer & ~binary).sum()), ("max" if m and m.group(1) == "1" else "min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-vars", type=int, default=60)
    ap.add_argument("--max-cons", type=int, default=60)
    ap.add_argument("--above", type=int, default=0, help="only instances with more variables or constraints than this")
    ap.add_argument("--per-problem", type=int, default=0, help="at most this many (the smallest) per minlp.org problem")
    ap.add_argument("--out", default="instances.csv", help="instance list to write (skipped ones to skipped_<out>)")
    a = ap.parse_args()
    (HERE / "src").mkdir(exist_ok=True)
    (HERE / "nl").mkdir(exist_ok=True)
    rows, skipped = [], []
    for r in select(listing(), a.max_vars, a.max_cons, a.above, a.per_problem):
        models = [(f, loc) for f, loc in r["inputs"] if f.lower().endswith((".gms", ".mps"))]
        if len(models) != 1:
            skipped.append(dict(id=r["id"], title=r["title"], reason=f"inputs {[f for f, _ in r['inputs']]}: not one .gms/.mps model"))
            continue
        fname, loc = models[0]
        name = name_of(r, fname)
        src = HERE / "src" / f"{name}{pathlib.Path(fname).suffix.lower()}"
        if not src.exists():
            src.write_bytes(get(f"{SITE}problem/mod/download.php?file={fname}&location={loc}"))
        nl = HERE / "nl" / f"{name}.nl"
        try:
            if not nl.exists():
                convert(src, nl)
        except RuntimeError as e:
            skipped.append(dict(id=r["id"], title=r["title"], reason=str(e)))
            continue
        nv, nc, nb, ni, sense = header(nl)
        rows.append(dict(name=name, nvars=nv, ncons=nc, probtype=r["nature"], objsense=sense, primalbound=r["best"],
                         dualbound="", gap="", convex="", nbinvars=nb, nintvars=ni, id=r["id"], title=r["title"],
                         url=r["page"]))  # fmt: skip
        print(f"{name:45s} {r['nature']:6s} vars {nv:3d} (bin {nb}, int {ni}) cons {nc:3d} {sense} best {r['best']}")
    with open(HERE / a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    skip_name = "skipped.csv" if a.out == "instances.csv" else f"skipped_{a.out}"
    with open(HERE / skip_name, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["id", "title", "reason"])
        w.writeheader()
        w.writerows(skipped)
    print(f"{len(rows)} instances, {len(skipped)} skipped (see {skip_name})")


if __name__ == "__main__":
    main()

"""Verify references against Crossref and write docs/references.bib.

Input: docs/refs.json, a list of {"key", "doi", "title"} (DOI entries) or
{"key", "bibtex"} (entries without a Crossref DOI: software, arXiv, proceedings).
For each DOI, the Crossref record is fetched, its title is compared with the
expected title, and the BibTeX comes from Crossref's own transform, so no
bibliographic field is typed by hand. Mismatches and lookups that fail are
reported and left out of the .bib file.

Run:  python docs/verify_refs.py
"""

import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request

UA = "jaxglobal-refcheck/0.1 (reference verification script)"


def get(url, accept="application/json"):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2**attempt)
        except Exception:
            time.sleep(2**attempt)
    return None


# Crossref title glitches (MathML residue, lost glyphs, missing spaces), fixed after download.
# Each entry is checked against the publisher's page; Gendron & Crainic's
# "Branch-and-Branch" is the published title and is deliberately left alone.
FIXES = [
    (r"\$\$\\alpha \\hbox \{BB\}\$\$ α BB", r"$\\alpha$BB"),
    (r"\$\$\\alpha \$\$ α BB", r"$\\alpha$BB"),
    (r"\$\$\{\{\\mathcal C\}\^2\}\$\$ -continuous", r"$C^2$-continuous"),
    (r"title=\{\?BB:", r"title={$\\alpha$BB:"),
    (r"tighteningtechniques", r"tightening techniques"),
]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"<[^>]+>", "", s.lower()))


def main():
    refs = json.load(open("docs/refs.json"))
    out, report = [], []
    for r in refs:
        if "bibtex" in r:
            out.append(r["bibtex"].strip())
            report.append((r["key"], "no DOI (manual entry)", ""))
            continue
        q = urllib.parse.quote(r["doi"])
        meta = get(f"https://api.crossref.org/works/{q}")
        if meta is None:
            report.append((r["key"], "FAIL: DOI not found in Crossref", r["doi"]))
            continue
        msg = json.loads(meta)["message"]
        title = " ".join(msg.get("title", [""])) + " " + " ".join(msg.get("subtitle", []))
        score = difflib.SequenceMatcher(None, norm(r["title"]), norm(title)).ratio()
        if norm(r["title"]) in norm(title) or norm(title.strip()) in norm(r["title"]):
            score = 1.0
        if score < 0.8:
            report.append((r["key"], f"FAIL: title mismatch ({score:.2f})", f"crossref: {title.strip()!r}"))
            continue
        bib = get(f"https://api.crossref.org/works/{q}/transform/application/x-bibtex")
        if not bib:
            report.append((r["key"], "FAIL: no BibTeX from Crossref", r["doi"]))
            continue
        for pat, rep in FIXES:
            bib = re.sub(pat, rep, bib)
        bib = re.sub(r"^@(\w+)\{[^,]*,", lambda m: f"@{m.group(1)}{{{r['key']},", bib.strip(), count=1)
        out.append(bib)
        report.append((r["key"], f"ok ({score:.2f})", title.strip()[:70]))
        time.sleep(0.2)
    open("docs/references.bib", "w").write("\n\n".join(out) + "\n")
    for k, s, t in report:
        print(f"{k:28s} {s:34s} {t}")
    fails = [k for k, s, _ in report if s.startswith("FAIL")]
    print(f"\n{len(refs) - len(fails)}/{len(refs)} written; failures: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

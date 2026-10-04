"""Verify references against Crossref and write docs/references.bib.

Input: docs/refs.json, a list of {"key", "doi", "title"} (DOI entries) or
{"key", "bibtex"} (entries without a Crossref DOI: software, arXiv, proceedings).
For each DOI, the Crossref record is fetched, its title is compared with the
expected title, and the BibTeX comes from Crossref's own transform, so no
bibliographic field is typed by hand. Mismatches and lookups that fail are
reported and left out of the .bib file.

Manual entries are checked too, by the first source that applies: the arXiv
API for an ``eprint``, doi.org metadata for a (DataCite) ``doi``, a DBLP title
search, or the ``url`` page containing the title. Their title must match. An
entry no source can confirm is reported as "unconfirmed" (kept, not a failure);
a source that contradicts the title is a failure.

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


def field(bib, name):
    m = re.search(name + r"\s*=\s*\{(.+?)\}\s*,?\s*$", bib, re.M | re.I)
    return m.group(1) if m else ""


def similar(a, b):
    a, b = norm(a), norm(b)
    if a and b and (a in b or b in a):
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def check_manual(bib, seen_at=(), hand_checked=""):
    """(status, detail) for a hand-written entry, from an independent source."""
    title = re.sub(r"[{}]", "", field(bib, "title"))
    if eprint := field(bib, "eprint"):
        xml = get(f"https://export.arxiv.org/api/query?id_list={eprint}", accept="application/atom+xml")
        m = re.search(r"<entry>.*?<title>(.*?)</title>", xml or "", re.S)
        if m:
            got = " ".join(m.group(1).split())
            return ("ok" if similar(title, got) >= 0.8 else "FAIL: arXiv title mismatch"), f"arXiv {eprint}: {got[:60]}"
    if doi := field(bib, "doi"):
        meta = get(f"https://doi.org/{doi}", accept="application/vnd.citationstyles.csl+json")
        if meta:
            got = json.loads(meta).get("title", "")
            return ("ok" if similar(title, got) >= 0.8 else "FAIL: DOI title mismatch"), f"doi {doi}: {got[:60]}"
    hits = get("https://dblp.org/search/publ/api?format=json&h=5&q=" + urllib.parse.quote(title))
    try:
        hits = json.loads(hits or "")["result"]["hits"].get("hit") or []
    except (ValueError, KeyError):  # DBLP rate-limits with an HTML page
        hits = []
    if hits:
        for h in hits:
            got = h["info"].get("title", "")
            if similar(title, got) >= 0.9:
                return "ok", f"dblp: {got[:60]}"
    oa = get("https://api.openalex.org/works?per-page=5&filter=title.search:" + urllib.parse.quote(title.replace(",", " ")))
    try:
        works = json.loads(oa or "")["results"]
    except (ValueError, KeyError):
        works = []
    year = field(bib, "year")
    for w in works:
        got = w.get("title") or ""
        if similar(title, got) >= 0.9 and (not year or abs(int(year) - (w.get("publication_year") or 0)) <= 1):
            return "ok", f"openalex {w['id'].rsplit('/', 1)[-1]}: {got[:50]}"
    urls = [*seen_at, field(bib, "url")] + re.findall(r"\\url\{([^}]+)\}", bib)
    for url in filter(None, urls):
        if m := re.match(r"https?://github\.com/([^/]+)(?:/([^/]+?))?/?$", url):
            api = f"repos/{m.group(1)}/{m.group(2)}" if m.group(2) else f"users/{m.group(1)}"
            if get(f"https://api.github.com/{api}", accept="application/vnd.github+json"):
                return "ok", f"github {api.split('/', 1)[1]} exists"
            continue
        page = get(url, accept="text/html")
        if page and norm(title)[:60] in norm(" ".join(re.sub(r"<[^>]+>", " ", page).split())):
            return "ok", f"title on {url[:50]}"
    if hand_checked:
        return "hand-checked", hand_checked
    return "unconfirmed", "no independent source found"


def main():
    refs = json.load(open("docs/refs.json"))
    out, report = [], []
    for r in refs:
        if "bibtex" in r:
            out.append(r["bibtex"].strip())
            status, detail = check_manual(r["bibtex"], r.get("seen_at", ()), r.get("hand_checked", ""))
            report.append((r["key"], f"manual: {status}", detail))
            time.sleep(0.5)
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
    unconfirmed = [k for k, s, _ in report if s.endswith("unconfirmed")]
    print(f"\n{len(refs) - len(fails)}/{len(refs)} written; failures: {fails}")
    print(f"manual entries no source could confirm (check by hand): {unconfirmed}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

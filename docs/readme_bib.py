"""Link README citations to a generated bibliography at the end of README.md.

Plain citations ``[key]`` / ``[key1, key2]`` become author-year links
``([McCormick 1976](#ref-mccormick1976computability))`` and the ``## References``
section is rebuilt from every ``#ref-key`` link, using ``docs/references.bib``.
Safe to rerun after adding citations. Run: python docs/readme_bib.py
"""
import pathlib
import re
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
KEY = r"[a-z][a-z-]*\d{4}[a-z0-9]*"
CITE = re.compile(rf"\[({KEY}(?:\s*[,;]\s*{KEY})*)\](?!\()")
LINK = re.compile(r"\(#ref-([^)\s]+)\)")
HEADING = "## References"
ACCENT = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302", "~": "\u0303", "c": "\u0327"}


def parse_bib(text):
    """{key: {field: value}} for a BibTeX file (one-line or multi-line entries)."""
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,\s]+),", text):
        i, depth, fields = m.end(), 1, {"type": m.group(1).lower()}
        while depth > 0 and i < len(text):
            f = re.compile(r"\s*([A-Za-z]+)\s*=\s*").match(text, i)
            if not f:
                i += 1
                if text[i - 1] == "}":
                    depth -= 1
                continue
            j = f.end()
            if text[j] == "{":
                d, k = 0, j
                while True:
                    d += {"{": 1, "}": -1}.get(text[k], 0)
                    k += 1
                    if d == 0:
                        break
                val, i = text[j + 1:k - 1], k
            elif text[j] == '"':
                k = text.index('"', j + 1)
                val, i = text[j + 1:k], k + 1
            else:
                k = re.compile(r"[^,}\s]*").match(text, j).end()
                val, i = text[j:k], k
            fields[f.group(1).lower()] = val
        out[m.group(2)] = fields
    return out


def delatex(s):
    s = re.sub(r"\\([`'\"^~c])\{?\\?([A-Za-z])\}?",
               lambda m: unicodedata.normalize("NFC", m.group(2) + ACCENT[m.group(1)]), s)
    s = re.sub(r"\\url\{([^}]*)\}", r"<\1>", s)
    s = re.sub(r"\$\\alpha\$", "α", s)
    s = s.replace("--", "–").replace("\\&", "&").replace("&amp;", "&").replace("\\_", "_")
    return re.sub(r"\s+", " ", s.replace("{", "").replace("}", "")).strip()


def authors(e):
    names = [delatex(a) for a in re.split(r"\s+and\s+", e.get("author", "")) if a.strip()]
    return [(n.split(",")[0].strip(), n) for n in names]


def label(e):
    a = authors(e)
    year = e.get("year", "n.d.")
    if not a:
        return f"{delatex(e.get('title', '?'))} {year}"
    if len(a) == 1:
        return f"{a[0][0]} {year}"
    if len(a) == 2:
        return f"{a[0][0]} & {a[1][0]} {year}"
    return f"{a[0][0]} et al. {year}"


def short(name):
    """'Last, First Middle' -> 'Last, F. M.'"""
    if "," not in name:
        return name
    last, first = (p.strip() for p in name.split(",", 1))
    initials = " ".join(w[0] + "." for w in re.split(r"[\s]+", first) if w and w[0].isalpha())
    return f"{last}, {initials}".rstrip(", ")


def entry(key, e):
    a = [short(n) for _, n in authors(e)]
    etal = len(a) > 8 or (a and a[-1] == "others")
    a = [n for n in a if n != "others"]
    who = ", ".join(a[:8]) + (", et al." if etal else "")
    parts = [f"{who} ({e.get('year', 'n.d.')})." if who else f"({e.get('year', 'n.d.')})."]
    parts.append(delatex(e.get("title", "")).rstrip(".") + ".")
    venue = e.get("journal") or e.get("booktitle") or e.get("publisher") or e.get("howpublished")
    if venue:
        v = f"*{delatex(venue)}*"
        if e.get("volume"):
            v += f" {e['volume']}" + (f"({e['number']})" if e.get("number") else "")
        if e.get("pages"):
            v += f", {delatex(e['pages'])}"
        parts.append(v + ".")
    if e.get("doi"):
        parts.append(f"[doi:{e['doi']}](https://doi.org/{e['doi']})")
    elif e.get("url"):
        parts.append(f"<{e['url']}>")
    return f'<a id="ref-{key}"></a>' + " ".join(parts)


def main():
    bib = parse_bib((ROOT / "docs" / "references.bib").read_text())
    text = README.read_text()
    body = text.split("\n" + HEADING + "\n")[0].rstrip() + "\n"

    def link(m):
        keys = re.split(r"\s*[,;]\s*", m.group(1))
        missing = [k for k in keys if k not in bib]
        if missing:
            raise SystemExit(f"not in docs/references.bib: {missing}")
        return "(" + "; ".join(f"[{label(bib[k])}](#ref-{k})" for k in keys) + ")"

    body = CITE.sub(link, body)
    cited = dict.fromkeys(LINK.findall(body))
    missing = [k for k in cited if k not in bib]
    if missing:
        raise SystemExit(f"not in docs/references.bib: {missing}")
    order = sorted(cited, key=lambda k: (label(bib[k]).lower(), k))
    refs = "\n\n".join(entry(k, bib[k]) for k in order)
    README.write_text(f"{body}\n{HEADING}\n\nGenerated from `docs/references.bib` by "
                      f"`docs/readme_bib.py`.\n\n{refs}\n")
    print(f"{len(order)} references")


if __name__ == "__main__":
    main()

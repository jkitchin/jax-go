"""MkDocs hooks for the jaxglobal documentation site.

Before the build, copy the executed notebooks, the figures and the README's
Related work / References sections into the docs directory (these copies are
generated and git-ignored), scrubbing local absolute paths from them. After the
build, fail if any local path is left in the rendered site.
"""

import json
import pathlib
import re
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs-site"

NOTEBOOKS = {"notebooks": ROOT / "notebooks"}
FIGURES = ["logo.svg", "logo-square.svg", "relaxations.png", "convergence.png", "benchmark.png", "g24.png"]

# Local absolute paths: /Users/<name>/..., /home/<name>/..., /private/tmp/..., /private/var/...
LOCAL_PATH = re.compile(r"(/Users/[^/\s\"'\\]+|/home/[^/\s\"'\\]+|/private/(?:tmp|var)|/var/folders)(/[^\s\"'\\<>)]*)?")


def scrub(text):
    """Replace local absolute paths by a placeholder."""
    return LOCAL_PATH.sub("<local path>", text)


def _scrub_obj(obj):
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, list):
        return [_scrub_obj(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _scrub_obj(v) for k, v in obj.items()}
    return obj


def _copy_notebooks():
    for sub, src in NOTEBOOKS.items():
        dst = DOCS / sub
        dst.mkdir(exist_ok=True)
        for nb in sorted(src.glob("[0-9][0-9]-*.ipynb")):
            data = json.loads(nb.read_text())
            (dst / nb.name).write_text(json.dumps(_scrub_obj(data), indent=1))


def _copy_figures():
    dst = DOCS / "assets" / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        src = ROOT / "figures" / name
        if src.exists():
            shutil.copyfile(src, dst / name)


def _references_page():
    """Related work and References from the README (kept in sync by docs/readme_bib.py)."""
    text = (ROOT / "README.md").read_text()
    start = text.index("## Related work")
    body = text[start:]
    body = re.sub(r"\[`docs/references\.bib`\]\(docs/references\.bib\)", "`docs/references.bib`", body)
    # Any remaining relative link into the repository becomes plain code text.
    body = re.sub(r"\[([^\]]+)\]\((?!https?:|#)([^)]+)\)", r"\1", body)
    body = body.replace("(see *Lessons* above)", "(see [Lessons](guide/correctness.md#lessons-and-gotchas))")
    page = "# References\n\n" + body.replace("## References", "## Bibliography", 1)
    (DOCS / "references.md").write_text(scrub(page))


def on_pre_build(config, **kwargs):
    _copy_notebooks()
    _copy_figures()
    _references_page()


def on_post_build(config, **kwargs):
    site = pathlib.Path(config["site_dir"])
    bad = []
    for p in site.rglob("*"):
        if p.is_file() and p.suffix in (".html", ".js", ".json", ".xml", ".txt", ".ipynb", ".svg"):
            for m in LOCAL_PATH.finditer(p.read_text(errors="ignore")):
                bad.append(f"{p.relative_to(site)}: {m.group(0)}")
    if bad:
        raise SystemExit("local paths found in the built site:\n" + "\n".join(bad[:20]))

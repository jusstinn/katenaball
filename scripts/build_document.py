#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown-it-py==3.0.0", "playwright==1.65.2"]
# ///
r"""The technical document as ``<team_name>.pdf``, the name the submission form asks for.

    uv run scripts/build_document.py --team <team_name>            # writes submissions/<team_name>.pdf
    uv run scripts/build_document.py --team <team_name> --draft    # while TODO markers are still in the text

``uv run`` reads the dependency block above and runs this script in its own small environment, so the locked
project environment (``uv.lock``) is not touched. From an environment that already has ``markdown-it-py``,
``python scripts/build_document.py`` works too; ``playwright`` is needed only for the fallback in step 3.

What it does:

1. Reads ``docs/technical-document.md`` and writes the team name where the text says ``<team_name>``.
2. Stops if a ``TODO(`` marker is left, unless ``--draft`` is given: the markers are what only the real run
   can fill in (``docs/real-run.md``, step 4).
3. Converts the Markdown to one HTML page and prints it to A4 with a Chromium-family browser run headless
   (Edge, Chrome, Chromium or Brave; ``KATENA_BROWSER`` or ``--browser`` names another one). The browser gets
   a temporary profile and no network is needed. If its command line (``--print-to-pdf``) leaves no PDF, the
   same browser prints the same page again through Playwright, which drives it over a pipe (``browser_env``
   says why the command line can fail). Playwright downloads no browser here.
4. Prints the page count, the size and the SHA-256 of the PDF.

The PDF goes to ``submissions/``, which is git-ignored: the firewall check refuses a PDF in a commit, because
a reviewer cannot read one in a diff. The Markdown stays the source. Nothing here reads the provided match.

Exit status: 0 written, 1 a marker is left or no browser was found or the browser failed, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs" / "technical-document.md"
SUBMISSIONS = REPO / "submissions"
PLACEHOLDER = "<team_name>"
MARKER = "TODO("
# The team name names the file, so it must be a valid file name on Windows. The same rule as real_run.py's
# team_name, apart from the surrounding spaces stripped here (tests/test_build_document.py compares the two).
BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
WAIT_FOR_PDF = 30.0       # seconds a browser that returned at once gets to finish writing the PDF

CSS = """
@page { size: A4; margin: 15mm 14mm 16mm 14mm; @bottom-center { content: counter(page); font: 8pt "Segoe UI", Helvetica, Arial, sans-serif; color: #666; } }
* { box-sizing: border-box; }
body { font: 9.5pt/1.38 "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif; color: #111; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; padding-top: 5pt; border-top: 0.6pt solid #999; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 10pt 0 3pt; break-after: avoid; }
p { margin: 0 0 5pt; }
ul, ol { margin: 0 0 5pt; padding-left: 15pt; }
li { margin: 0 0 1.5pt; }
li > p { margin: 0 0 2pt; }
code { font: 8.4pt Consolas, "SF Mono", Menlo, "DejaVu Sans Mono", monospace; background: #f1f1f1; padding: 0 1.5pt; border-radius: 2pt; overflow-wrap: break-word; }
pre { background: #f4f4f4; padding: 5pt 6pt; margin: 0 0 6pt; border-radius: 3pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.8pt; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 7pt; font-size: 8.2pt; line-height: 1.28; }
th, td { border: 0.5pt solid #aaa; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { background: #ececec; }
tr { break-inside: avoid; }
td code, th code { font-size: 7.6pt; }
a { color: inherit; text-decoration: none; }
"""


def team_name(value: str) -> str:
    value = value.strip()
    if not value or BAD_NAME.search(value) or value.split(".")[0].strip().upper() in RESERVED:
        raise argparse.ArgumentTypeError('the team name names the PDF: none of < > : " / \\ | ? * and not a '
                                         "Windows device name such as CON or NUL")
    return value


def substitute(text: str, team: str) -> str:
    return text.replace(PLACEHOLDER, team)


def todo_lines(text: str) -> list[int]:
    """1-based numbers of the lines that still carry a marker, without the line that explains the marker."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if MARKER in line and "marks what only the final run" not in line:
            out.append(i)
    return out


def to_html(text: str, title: str) -> str:
    from markdown_it import MarkdownIt       # the script's own dependency (see the block at the top)

    body = MarkdownIt("commonmark", {"html": False}).enable("table").render(text)
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>\n<body>\n{body}</body></html>\n")


def browser_candidates() -> list[Path]:
    env = os.environ
    if sys.platform == "win32":
        roots = [env.get("ProgramFiles(x86)"), env.get("ProgramFiles"), env.get("LocalAppData")]
        tails = ["Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe",
                 "Chromium/Application/chrome.exe", "BraveSoftware/Brave-Browser/Application/brave.exe"]
        return [Path(r) / t for t in tails for r in roots if r]
    if sys.platform == "darwin":
        apps = ["Google Chrome", "Microsoft Edge", "Chromium", "Brave Browser"]
        return [Path(d) / f"{a}.app" / "Contents" / "MacOS" / a
                for a in apps for d in ("/Applications", str(Path.home() / "Applications"))]
    names = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"]
    return [Path(p) for n in names if (p := shutil.which(n))]


def find_browser(given: str | None = None) -> Path | None:
    named = given or os.environ.get("KATENA_BROWSER")
    if named:
        p = Path(named)
        if p.exists():
            return p
        w = shutil.which(named)
        return Path(w) if w else None
    return next((p for p in browser_candidates() if p.exists()), None)


def browser_env() -> dict:
    """This process's environment for the browser, without ``__COMPAT_LAYER``.

    Windows sets ``__COMPAT_LAYER`` (``DetectorsAppHealth`` was seen) on an app its compatibility detectors have
    flagged, and every process that app starts inherits it. With it set, msedge.exe hands the work to a new
    process and exits 0 at once: back after 0.06 s, the PDF a second later, or never once the page was deleted;
    without the variable, back after 1.1 s with the PDF written (Edge 157 on 2026-11-16, from a shell the Claude
    app had started). A PowerShell opened from the Start menu does not normally have it."""
    return {k: v for k, v in os.environ.items() if k.upper() != "__COMPAT_LAYER"}


def pdf_complete(path: Path) -> bool:
    """True when ``path`` is a PDF written to its end (``%%EOF`` in its last kilobyte)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            head = fh.read(5)
            fh.seek(max(0, size - 1024))
            tail = fh.read()
    except OSError:
        return False
    return head == b"%PDF-" and b"%%EOF" in tail


def wait_for_pdf(path: Path, seconds: float) -> bool:
    """Waits up to ``seconds`` for ``pdf_complete(path)``."""
    deadline = time.monotonic() + seconds
    while not pdf_complete(path):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)
    return True


def print_pdf(browser: Path, page: Path, out: Path, timeout: float = 180.0,
              wait: float = WAIT_FOR_PDF) -> tuple[int, str]:
    """Run the browser headless on ``page`` and print it to ``out``. Returns its exit status and last output.

    A browser that hands the work to another process returns before the PDF is there, so after exit status 0
    this waits up to ``wait`` seconds for the file while ``page`` still exists (``browser_env`` says when Edge
    does that). The browser's helper processes can hold a file of the temporary profile for a moment after it
    exits (seen on Windows with Edge on 2026-11-12, under load), so a profile that cannot be removed at once is
    left to the OS: the PDF is already written by then."""
    with tempfile.TemporaryDirectory(prefix="katena-doc-profile-", ignore_cleanup_errors=True) as profile:
        cmd = [str(browser), "--headless", "--disable-gpu", "--no-first-run", "--disable-extensions",
               f"--user-data-dir={profile}", "--no-pdf-header-footer", f"--print-to-pdf={out}", page.as_uri()]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout,
                               env=browser_env())
        except subprocess.TimeoutExpired:
            return 124, f"no PDF after {timeout:.0f} s"
        if r.returncode == 0:
            wait_for_pdf(out, wait)
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
        return r.returncode, tail[0]


def print_pdf_playwright(browser: Path, page: Path, out: Path, timeout: float = 180.0) -> tuple[int, str]:
    """The fallback: the same browser prints ``page`` to ``out`` through Playwright, A4 from the page's CSS and
    no header or footer, as on the command line. Returns 0 and "", or 1 and the reason."""
    try:
        from playwright.sync_api import sync_playwright      # the script's own dependency (the block at the top)
    except ImportError:
        return 1, "playwright is not installed (uv run scripts/build_document.py installs it)"
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(executable_path=str(browser), headless=True, env=browser_env(),
                                   timeout=timeout * 1000)
            try:
                tab = b.new_page()
                tab.goto(page.as_uri(), wait_until="load", timeout=timeout * 1000)
                tab.pdf(path=str(out), prefer_css_page_size=True, print_background=False)
            finally:
                b.close()
    except Exception as e:  # noqa: BLE001  a failed fallback is reported, never raised
        first = (str(e).strip().splitlines() or [""])[0]
        return 1, f"{type(e).__name__}: {first}"[:300]
    return 0, ""


def count_pages(pdf: bytes) -> int | None:
    n = len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", pdf))
    return n or None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", required=True, type=team_name, help="the registered team name; names the PDF")
    ap.add_argument("--draft", action="store_true", help="allow TODO markers in the text")
    ap.add_argument("--source", type=Path, default=SOURCE, help="the Markdown to print (default: the technical document)")
    ap.add_argument("--out", type=Path, default=None, help="the PDF (default: submissions/<team>.pdf)")
    ap.add_argument("--browser", default=None, help="a Chromium-family browser (default: the first one found)")
    ap.add_argument("--keep-html", action="store_true", help="also keep the HTML page beside the PDF")
    a = ap.parse_args(argv)

    text = substitute(a.source.read_text(encoding="utf-8"), a.team)
    left = todo_lines(text)
    if left and not a.draft:
        print(f"FAIL  {len(left)} TODO marker(s) left in {a.source.name}, at line(s) {', '.join(map(str, left))}. "
              "Fill them in (docs/real-run.md, step 4), or pass --draft.")
        return 1
    browser = find_browser(a.browser)
    if browser is None:
        print("FAIL  no Chromium-family browser found (Edge, Chrome, Chromium, Brave). "
              "Name one with --browser or KATENA_BROWSER.")
        return 1

    out = (a.out or SUBMISSIONS / f"{a.team}.pdf").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    page_html = to_html(text, f"{a.team}: technical document")
    with tempfile.TemporaryDirectory(prefix="katena-doc-", ignore_cleanup_errors=True) as tmp:
        page = Path(tmp) / "document.html"
        page.write_text(page_html, encoding="utf-8", newline="\n")
        if out.exists():
            out.unlink()
        rc, tail = print_pdf(browser, page, out)
        how, why = browser.name, ""
        if not pdf_complete(out):
            print(f"WARN  {browser.name} --print-to-pdf wrote no PDF (exit {rc}){': ' + tail if tail else ''}. "
                  "Trying the same browser through Playwright.")
            try:
                out.unlink(missing_ok=True)
            except OSError:
                pass
            _, why = print_pdf_playwright(browser, page, out)
            how = f"{browser.name} through Playwright"
    if not pdf_complete(out):
        print(f"FAIL  {browser.name} wrote no PDF (exit {rc}): {tail}; through Playwright: {why or 'no file'}")
        return 1
    if a.keep_html:
        out.with_suffix(".html").write_text(page_html, encoding="utf-8", newline="\n")

    pages = count_pages(out.read_bytes())
    print(f"wrote {out}")
    print(f"  pages   {pages if pages is not None else 'unknown'}")
    print(f"  bytes   {out.stat().st_size}")
    print(f"  sha256  {sha256_file(out)}")
    print(f"  browser {how}" + ("   DRAFT: TODO markers left at line(s) " + ", ".join(map(str, left)) if left else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

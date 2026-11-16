"""scripts/build_document.py without a browser and without its Markdown dependency: the team name, the TODO
markers, the browser lookup, the page count and the exit status. One test prints a real PDF when a
Chromium-family browser and markdown-it-py are both there, and is skipped otherwise (CI has neither)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts")]

import build_document as BD  # noqa: E402

TEXT = "# <team_name>: technical document\n\n`TODO(after the real run)` marks what only the final run can fill in.\n\n" \
       "| a | b |\n|---|---|\n| 1 | `make submit TEAM=<team_name>` |\n\nThe file: `TODO(after the real run)`: its hash.\n"


def test_the_team_name_replaces_the_placeholder_everywhere():
    out = BD.substitute(TEXT, "katena")
    assert BD.PLACEHOLDER not in out and out.startswith("# katena: technical document") and "TEAM=katena" in out


def test_todo_lines_skip_the_line_that_explains_the_marker():
    assert BD.todo_lines(TEXT) == [9]
    assert BD.todo_lines(TEXT.replace("`TODO(after the real run)`: its hash", "abc123")) == []


def test_the_real_document_names_its_markers():
    text = BD.SOURCE.read_text(encoding="utf-8")
    assert BD.PLACEHOLDER in text                                  # the title is filled in from --team
    assert "marks what only the final run" in text                 # the explaining line todo_lines skips
    lines = text.splitlines()
    assert all(BD.MARKER in lines[i - 1] for i in BD.todo_lines(text))


NAMES = [("katenaball", True), ("FC Katena_2-b.", True), ("Barça FC", True), ("Équipe d'Or & Co (2)", True),
         ("Katena F.C.", True), ("Console", True), ("", False), ("a/b", False), ("x:y", False), ("a\\b", False),
         ("a?", False), ('a"b', False), ("CON", False), ("nul.x", False), ("Lpt1", False)]


@pytest.mark.parametrize("name, ok", NAMES)
def test_team_name(name, ok):
    if ok:
        assert BD.team_name(f" {name} ") == name
    else:
        with pytest.raises(Exception):
            BD.team_name(name)


@pytest.mark.parametrize("name, ok", NAMES)
def test_the_pdf_and_the_csv_accept_the_same_team_names(name, ok):
    import argparse

    import real_run as RR
    try:
        RR.team_name(name)
        csv_ok = True
    except argparse.ArgumentTypeError:
        csv_ok = False
    assert csv_ok == ok


def test_the_browser_runs_without_the_compat_layer(monkeypatch):
    monkeypatch.setenv("__COMPAT_LAYER", "DetectorsAppHealth")
    monkeypatch.setenv("KATENA_TEST_KEEP", "1")
    env = BD.browser_env()
    assert "__COMPAT_LAYER" not in env and env["KATENA_TEST_KEEP"] == "1"


def test_pdf_complete_needs_the_header_and_the_end_marker(tmp_path):
    p = tmp_path / "x.pdf"
    assert not BD.pdf_complete(p)
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 5000)
    assert not BD.pdf_complete(p)                                  # still being written
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 5000 + b"\n%%EOF\n")
    assert BD.pdf_complete(p)
    p.write_bytes(b"<html>%%EOF")
    assert not BD.pdf_complete(p)


PDF = b"%PDF-1.4\n<< /Type /Pages /Count 1 >> << /Type /Page >>\n%%EOF\n"


def test_a_browser_that_returns_before_the_pdf_is_written_is_waited_for(tmp_path, monkeypatch):
    """msedge.exe with __COMPAT_LAYER set: exit 0 at once, the file a moment later from another process."""
    import subprocess
    import threading

    out = tmp_path / "late.pdf"

    def fake_run(cmd, **kw):
        assert "__COMPAT_LAYER" not in kw["env"]
        threading.Timer(0.5, lambda: out.write_bytes(PDF)).start()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setenv("__COMPAT_LAYER", "DetectorsAppHealth")
    monkeypatch.setattr(BD.subprocess, "run", fake_run)
    assert BD.print_pdf(tmp_path / "b.exe", tmp_path / "p.html", out, wait=10) == (0, "")
    assert BD.pdf_complete(out)


def test_no_wait_after_a_browser_that_failed(tmp_path, monkeypatch):
    import subprocess
    import time

    monkeypatch.setattr(BD.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 3, "", "oops\n"))
    t = time.monotonic()
    assert BD.print_pdf(tmp_path / "b.exe", tmp_path / "p.html", tmp_path / "x.pdf", wait=10) == (3, "oops")
    assert time.monotonic() - t < 5


def test_count_pages():
    pdf = b"<< /Type /Pages /Count 2 >> << /Type /Page /Parent 1 0 R >> <</Type/Page>> << /Type /PageLabels >>"
    assert BD.count_pages(pdf) == 2 and BD.count_pages(b"%PDF") is None


def test_find_browser(tmp_path, monkeypatch):
    exe = tmp_path / "browser.exe"
    exe.write_bytes(b"")
    monkeypatch.delenv("KATENA_BROWSER", raising=False)
    assert BD.find_browser(str(exe)) == exe
    assert BD.find_browser(str(tmp_path / "missing.exe")) is None
    monkeypatch.setenv("KATENA_BROWSER", str(exe))
    assert BD.find_browser() == exe
    monkeypatch.delenv("KATENA_BROWSER")
    monkeypatch.setattr(BD, "browser_candidates", lambda: [tmp_path / "none", exe])
    assert BD.find_browser() == exe
    monkeypatch.setattr(BD, "browser_candidates", lambda: [])
    assert BD.find_browser() is None


def test_markers_stop_the_build_before_anything_else(tmp_path, monkeypatch, capsys):
    src = tmp_path / "doc.md"
    src.write_text(TEXT, encoding="utf-8")
    out = tmp_path / "t.pdf"
    monkeypatch.setattr(BD, "find_browser", lambda given=None: pytest.fail("looked for a browser"))
    assert BD.main(["--team", "t", "--source", str(src), "--out", str(out)]) == 1
    assert "1 TODO marker(s) left" in capsys.readouterr().out and not out.exists()


def test_no_browser_is_a_failure_with_a_reason(tmp_path, monkeypatch, capsys):
    src = tmp_path / "doc.md"
    src.write_text(TEXT, encoding="utf-8")
    monkeypatch.setattr(BD, "find_browser", lambda given=None: None)
    assert BD.main(["--team", "t", "--draft", "--source", str(src), "--out", str(tmp_path / "t.pdf")]) == 1
    assert "no Chromium-family browser" in capsys.readouterr().out


def test_a_browser_that_writes_nothing_fails(tmp_path, monkeypatch, capsys):
    src = tmp_path / "doc.md"
    src.write_text(TEXT, encoding="utf-8")
    monkeypatch.setattr(BD, "find_browser", lambda given=None: tmp_path / "browser.exe")
    monkeypatch.setattr(BD, "to_html", lambda text, title: "<html></html>")
    monkeypatch.setattr(BD, "print_pdf", lambda browser, page, out: (7, "boom"))
    monkeypatch.setattr(BD, "print_pdf_playwright", lambda browser, page, out: (1, "TimeoutError: no page"))
    assert BD.main(["--team", "t", "--draft", "--source", str(src), "--out", str(tmp_path / "t.pdf")]) == 1
    out = capsys.readouterr().out
    assert "wrote no PDF (exit 7): boom; through Playwright: TimeoutError: no page" in out
    assert "FAIL" in out and "pages" not in out


def test_the_fallback_prints_through_playwright_and_reports_the_pages(tmp_path, monkeypatch, capsys):
    src = tmp_path / "doc.md"
    src.write_text(TEXT, encoding="utf-8")
    seen = []

    def fallback(browser, page, out):
        seen.append(page.read_text(encoding="utf-8"))            # the page still exists for the fallback
        out.write_bytes(PDF)
        return 0, ""

    monkeypatch.setattr(BD, "find_browser", lambda given=None: tmp_path / "msedge.exe")
    monkeypatch.setattr(BD, "to_html", lambda text, title: "<html>page</html>")
    monkeypatch.setattr(BD, "print_pdf", lambda browser, page, out: (0, ""))
    monkeypatch.setattr(BD, "print_pdf_playwright", fallback)
    pdf = tmp_path / "t.pdf"
    assert BD.main(["--team", "t", "--draft", "--source", str(src), "--out", str(pdf)]) == 0
    out = capsys.readouterr().out
    assert seen == ["<html>page</html>"] and BD.pdf_complete(pdf)
    assert "WARN  msedge.exe --print-to-pdf wrote no PDF (exit 0). Trying the same browser through Playwright." in out
    assert "pages   1" in out and "browser msedge.exe through Playwright   DRAFT" in out


def test_the_fallback_is_not_tried_when_the_command_line_worked(tmp_path, monkeypatch, capsys):
    src = tmp_path / "doc.md"
    src.write_text(TEXT, encoding="utf-8")
    monkeypatch.setattr(BD, "find_browser", lambda given=None: tmp_path / "msedge.exe")
    monkeypatch.setattr(BD, "to_html", lambda text, title: "<html></html>")
    monkeypatch.setattr(BD, "print_pdf", lambda browser, page, out: (out.write_bytes(PDF), (0, ""))[1])
    monkeypatch.setattr(BD, "print_pdf_playwright", lambda *a: pytest.fail("tried the fallback"))
    assert BD.main(["--team", "t", "--draft", "--source", str(src), "--out", str(tmp_path / "t.pdf")]) == 0
    assert "browser msedge.exe   DRAFT" in capsys.readouterr().out


def test_the_fallback_without_playwright_says_so(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_playwright(name, *a, **kw):
        if name.startswith("playwright"):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_playwright)
    rc, why = BD.print_pdf_playwright(tmp_path / "b.exe", tmp_path / "p.html", tmp_path / "x.pdf")
    assert rc == 1 and "playwright is not installed" in why


def test_the_dependency_block_names_both_pinned_dependencies():
    import tomllib

    text = (REPO / "scripts" / "build_document.py").read_text(encoding="utf-8")
    block = text.split("# /// script\n", 1)[1].split("# ///\n", 1)[0]
    meta = tomllib.loads("".join(line[2:] + "\n" for line in block.splitlines()))
    assert meta["requires-python"] == ">=3.11"
    assert sorted(d.split("==")[0] for d in meta["dependencies"]) == ["markdown-it-py", "playwright"]
    assert all("==" in d for d in meta["dependencies"])


def test_prints_a_pdf_when_a_browser_is_here(tmp_path, capsys):
    pytest.importorskip("markdown_it")
    if BD.find_browser() is None:
        pytest.skip("no Chromium-family browser on this machine")
    src = tmp_path / "doc.md"
    src.write_text(TEXT, encoding="utf-8")
    out = tmp_path / "team.pdf"
    assert BD.main(["--team", "team", "--draft", "--source", str(src), "--out", str(out), "--keep-html"]) == 0
    assert out.read_bytes().startswith(b"%PDF") and BD.count_pages(out.read_bytes()) == 1
    page = out.with_suffix(".html").read_text(encoding="utf-8")
    assert "<table>" in page and "team: technical document" in page and "&lt;team_name&gt;" not in page
    assert "pages   1" in capsys.readouterr().out

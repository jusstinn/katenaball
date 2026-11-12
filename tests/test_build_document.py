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


@pytest.mark.parametrize("name, ok", [("katenaball", True), ("FC Katena_2-b.", True), ("", False),
                                      ("a/b", False), ("x:y", False), ("-x", False)])
def test_team_name(name, ok):
    if ok:
        assert BD.team_name(f" {name} ") == name
    else:
        with pytest.raises(Exception):
            BD.team_name(name)


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
    assert BD.main(["--team", "t", "--draft", "--source", str(src), "--out", str(tmp_path / "t.pdf")]) == 1
    assert "wrote no PDF (exit 7): boom" in capsys.readouterr().out


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

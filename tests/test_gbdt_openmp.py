"""What ``gbdt.ensure_openmp`` tells a person to install when LightGBM cannot load, by platform. Found on
2026-11-12: a bare Ubuntu under WSL has no libgomp.so.1, and the message named Homebrew."""

from __future__ import annotations

import sys

import pytest

from katena.models import gbdt


@pytest.mark.parametrize("platform, words", [
    ("linux", ("libgomp.so.1", "apt-get install libgomp1", "dnf install libgomp")),
    ("darwin", ("brew install libomp",)),
    ("win32", ("Visual C++",)),
    ("freebsd14", ("OpenMP",))])
def test_the_hint_names_the_platforms_own_fix(platform, words):
    hint = gbdt.openmp_hint(platform)
    assert all(w in hint for w in words)
    assert ("brew" in hint) == (platform == "darwin")


def test_a_missing_runtime_stops_a_script_with_the_hint(monkeypatch):
    monkeypatch.setattr(gbdt, "lgb", None)
    monkeypatch.setattr(gbdt, "_LGB_ERROR", "libgomp.so.1: cannot open shared object file")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "pytest")           # ensure_openmp does nothing under pytest
    with pytest.raises(RuntimeError) as err:
        gbdt.ensure_openmp()
    assert "apt-get install libgomp1" in str(err.value) and "cannot open shared object file" in str(err.value)


def test_nothing_happens_when_lightgbm_loaded_or_under_pytest(monkeypatch):
    gbdt.ensure_openmp()                                  # under pytest: returns, whatever lgb is
    monkeypatch.setattr(gbdt, "lgb", object())
    monkeypatch.delitem(sys.modules, "pytest")
    gbdt.ensure_openmp()

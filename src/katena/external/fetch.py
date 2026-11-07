"""Downloading and integrity checks for the external corpora.

Everything downloaded is untrusted data: this module never executes or imports anything from it.
Each dataset lives in its own directory under ``external_data/``. SHA256 digests of every raw file
are pinned in ``scripts/external_manifest.json`` (trust on first use, then verified on every run).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

CHUNK = 1 << 20
FIGSHARE_ARTICLE = 28196177
METRICA_REPO = "https://github.com/metrica-sports/sample-data"
SKILLCORNER_REPO = "https://github.com/SkillCorner/opendata"
# Commits seen when the corpus was first built; pinned so a rebuild fetches the same bytes.
METRICA_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
SKILLCORNER_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"


class IntegrityError(RuntimeError):
    """A raw file does not match its pinned or published digest."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    """Pinned SHA256 digests, keyed by path relative to ``external_data/``."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}
        self.dirty = False

    def check(self, root: Path, file: Path) -> str:
        """Verify ``file`` against the manifest, pinning it if new. Returns the SHA256."""
        key = file.relative_to(root).as_posix()
        digest = sha256_file(file)
        size = file.stat().st_size
        known = self.entries.get(key)
        if known is None:
            self.entries[key] = {"sha256": digest, "size": size}
            self.dirty = True
        elif known["sha256"] != digest or known["size"] != size:
            raise IntegrityError(f"{key}: sha256 {digest} != pinned {known['sha256']}")
        return digest

    def save(self) -> None:
        if self.dirty:
            self.path.write_text(json.dumps(self.entries, indent=1, sort_keys=True) + "\n")
            self.dirty = False


def _download(url: str, dest: Path, headers: dict | None = None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "katena-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(CHUNK):
            f.write(chunk)
    os.replace(tmp, dest)


def fetch_idsse(root: Path, manifest: Manifest) -> list[Path]:
    """Download the 7 IDSSE/DFL matches (figshare API); verify against figshare's MD5."""
    raw = root / "idsse" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    api = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"
    with urllib.request.urlopen(api, timeout=60) as r:
        meta = json.load(r)
    (root / "idsse" / "figshare_article.json").write_text(json.dumps(meta, indent=1))
    out = []
    for fi in meta["files"]:
        dest = raw / fi["name"]
        if not dest.exists() or dest.stat().st_size != fi["size"]:
            _download(fi["download_url"], dest)
        if md5_file(dest) != fi["computed_md5"]:
            raise IntegrityError(f"{dest.name}: md5 mismatch vs figshare")
        manifest.check(root, dest)
        out.append(dest)
    manifest.save()
    return out


def _git_clone(url: str, commit: str, dest: Path, env_extra: dict | None = None) -> None:
    if (dest / ".git").exists():
        return
    env = {**os.environ, **(env_extra or {})}
    # Byte-exact checkout on every OS: Git for Windows defaults to core.autocrlf=true, which rewrites
    # LF as CRLF in text files (the Metrica CSVs) and breaks the pinned SHA256 in the manifest.
    subprocess.run(["git", "clone", "--quiet", "--config", "core.autocrlf=false", "--config", "core.eol=lf",
                    url, str(dest)], check=True, env=env)
    subprocess.run(["git", "-C", str(dest), "-c", "advice.detachedHead=false", "checkout", "--quiet", commit],
                   check=True, env=env)


def fetch_metrica(root: Path, manifest: Manifest) -> Path:
    repo = root / "metrica" / "repo"
    _git_clone(METRICA_REPO, METRICA_COMMIT, repo, {"GIT_LFS_SKIP_SMUDGE": "1"})
    for f in sorted((repo / "data").rglob("*")):
        if f.is_file():
            manifest.check(root, f)
    manifest.save()
    return repo


def _lfs_pointer(path: Path) -> tuple[str, int] | None:
    if path.stat().st_size > 512:
        return None
    text = path.read_text()
    if not text.startswith("version https://git-lfs"):
        return None
    oid = next(ln.split("sha256:")[1] for ln in text.splitlines() if ln.startswith("oid sha256:"))
    size = int(next(ln.split()[1] for ln in text.splitlines() if ln.startswith("size ")))
    return oid, size


def fetch_skillcorner(root: Path, manifest: Manifest) -> Path:
    """Clone the repo without LFS, then resolve tracking pointers via the public LFS batch API."""
    repo = root / "skillcorner" / "repo"
    _git_clone(SKILLCORNER_REPO, SKILLCORNER_COMMIT, repo, {"GIT_LFS_SKIP_SMUDGE": "1"})
    for f in sorted((repo / "data" / "matches").glob("*/*_tracking_extrapolated.jsonl")):
        ptr = _lfs_pointer(f)
        if ptr is not None:
            oid, size = ptr
            body = json.dumps({"operation": "download", "transfers": ["basic"],
                               "objects": [{"oid": oid, "size": size}]}).encode()
            req = urllib.request.Request(
                SKILLCORNER_REPO + ".git/info/lfs/objects/batch", data=body, method="POST",
                headers={"Accept": "application/vnd.git-lfs+json",
                         "Content-Type": "application/vnd.git-lfs+json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.load(r)
            obj = resp["objects"][0]
            if "error" in obj:
                raise IntegrityError(f"{f.name}: LFS error {obj['error']}")
            act = obj["actions"]["download"]
            tmp = f.with_suffix(".lfs")
            _download(act["href"], tmp, act.get("header") or {"User-Agent": "katena-fetch"})
            if sha256_file(tmp) != oid:
                tmp.unlink()
                raise IntegrityError(f"{f.name}: sha256 != LFS oid")
            os.replace(tmp, f)
        else:
            pass
        manifest.check(root, f)
    for f in sorted((repo / "data" / "matches").glob("*/*")):
        if f.is_file() and not f.name.endswith("_tracking_extrapolated.jsonl"):
            manifest.check(root, f)
    manifest.check(root, repo / "data" / "matches.json")
    manifest.save()
    return repo

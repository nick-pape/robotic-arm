"""Fetch the large reference artifacts that are deliberately not committed.

The small text sources (URDF, MJCF, licences) live in git. Meshes and STEP
files do not: they are ~16 MB of binary that already has a canonical home
upstream. This script pulls them at pinned commits and verifies every byte
against reference/manifest.json, so "not committed" never means "not pinned".

    uv run python scripts/fetch_reference.py           # fetch + verify
    uv run python scripts/fetch_reference.py --check   # verify only, no network
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / "reference"
MANIFEST = REFERENCE / "manifest.json"

MENAGERIE_REPO = "google-deepmind/mujoco_menagerie"
MENAGERIE_SHA = "da76818e269b82289eba39808e2fb91d679d6994"
MENAGERIE_DIR = "seeed_rebot_devarm/assets"

ROBSTRIDE_REPO = "RobStride/Product_Information"
ROBSTRIDE_SHA = "0f4ad74fdb67023e75bbcbeebecd6f1a003ce000"
ROBSTRIDE_FILES = {
    "step/RS06-new.step": "Product Literature/RS06/RS06-new.step",
    "step/RS00.step": "Product Literature/RS00/RS00.step",
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "robotic-arm-fetch"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _raw(repo: str, sha: str, path: str) -> str:
    quoted = urllib.parse.quote(path)
    return f"https://raw.githubusercontent.com/{repo}/{sha}/{quoted}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def planned_files() -> dict[str, str]:
    """Map destination path (relative to reference/) -> source URL."""
    tree_url = (
        f"https://api.github.com/repos/{MENAGERIE_REPO}/contents/"
        f"{MENAGERIE_DIR}?ref={MENAGERIE_SHA}"
    )
    entries = json.loads(_get(tree_url))
    files = {
        f"mjcf/assets/{e['name']}": _raw(
            MENAGERIE_REPO, MENAGERIE_SHA, f"{MENAGERIE_DIR}/{e['name']}"
        )
        for e in entries
        if e["type"] == "file"
    }
    files.update(
        {
            dest: _raw(ROBSTRIDE_REPO, ROBSTRIDE_SHA, src)
            for dest, src in ROBSTRIDE_FILES.items()
        }
    )
    return files


def load_manifest() -> dict[str, str]:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def check_only() -> int:
    manifest = load_manifest()
    if not manifest:
        print("no manifest; run without --check first", file=sys.stderr)
        return 1
    missing, bad = [], []
    for rel, want in sorted(manifest.items()):
        path = REFERENCE / rel
        if not path.exists():
            missing.append(rel)
        elif _sha256(path.read_bytes()) != want:
            bad.append(rel)
    for rel in missing:
        print(f"MISSING  {rel}")
    for rel in bad:
        print(f"MISMATCH {rel}")
    if missing or bad:
        print(f"\n{len(missing)} missing, {len(bad)} mismatched", file=sys.stderr)
        return 1
    print(f"all {len(manifest)} reference files verified")
    return 0


def fetch() -> int:
    manifest = load_manifest()
    files = planned_files()
    print(f"fetching {len(files)} files")
    fresh, cached, failed = {}, 0, []
    for rel, url in sorted(files.items()):
        dest = REFERENCE / rel
        want = manifest.get(rel)
        if want and dest.exists() and _sha256(dest.read_bytes()) == want:
            fresh[rel] = want
            cached += 1
            continue
        try:
            data = _get(url)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed.append(f"{rel}: {exc}")
            continue
        got = _sha256(data)
        if want and got != want:
            failed.append(f"{rel}: checksum drift, upstream changed under a pinned SHA")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        fresh[rel] = got
        print(f"  {rel} ({len(data):,} B)")

    for line in failed:
        print(f"FAILED {line}", file=sys.stderr)
    if failed:
        return 1

    if fresh != manifest:
        MANIFEST.write_text(json.dumps(dict(sorted(fresh.items())), indent=2) + "\n")
        print(f"wrote manifest with {len(fresh)} entries")
    print(f"done: {len(fresh) - cached} downloaded, {cached} already current")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without network")
    args = parser.parse_args()
    sys.exit(check_only() if args.check else fetch())

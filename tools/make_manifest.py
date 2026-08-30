#!/usr/bin/env python3
"""
Regenerate ``data/manifest.json`` from a directory of data files.

The manifest pins every static data file satkit downloads by size and
SHA-256 and lists the URLs it may be fetched from (see ``data/README.md``).
This tool recomputes ``size``/``sha256`` from the files on disk and keeps
``urls``, ``source``, ``license``, ``tier`` and ``default`` from the existing
manifest, so a data refresh is a reproducible two-step process::

    python tools/make_manifest.py --data-dir ~/Library/Application\\ Support/satkit-data
    git diff data/manifest.json          # review what changed

Add a new file by first adding a stub entry (name + urls + source + license)
to the manifest, then running this tool to fill in size and hash. Bump
``data_version`` (and ``release_base``) with ``--data-version data-v2`` when
the GitHub release tag changes.

Only files already listed in the manifest are touched; unknown files in the
data directory are ignored (the refresh files EOP-All.csv / SW-All.csv are
never pinned).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="directory holding the data files")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="manifest to update (default: data/manifest.json)")
    ap.add_argument("--data-version", help="new data_version / release tag (also rewrites release_base and the release URLs)")
    ap.add_argument("--check", action="store_true", help="exit 1 if the manifest would change (CI drift check)")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    data_dir = args.data_dir.expanduser()
    changed = []

    if args.data_version and args.data_version != manifest["data_version"]:
        old, new = manifest["data_version"], args.data_version
        manifest["data_version"] = new
        manifest["release_base"] = manifest["release_base"].replace(old, new)
        for e in manifest["files"]:
            e["urls"] = [u.replace(f"/{old}/", f"/{new}/") for u in e["urls"]]
        changed.append(f"data_version {old} -> {new}")

    for e in manifest["files"]:
        path = data_dir / e["name"]
        if not path.is_file():
            print(f"warning: {e['name']} not in {data_dir}; keeping existing size/sha256", file=sys.stderr)
            continue
        size, sha = path.stat().st_size, sha256_of(path)
        if size != e.get("size") or sha != e.get("sha256"):
            changed.append(f"{e['name']}: size {e.get('size')} -> {size}, sha256 {str(e.get('sha256'))[:12]}… -> {sha[:12]}…")
            e["size"], e["sha256"] = size, sha

    if args.check:
        for c in changed:
            print("would change:", c)
        return 1 if changed else 0

    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for c in changed:
        print("changed:", c)
    if not changed:
        print("manifest unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())

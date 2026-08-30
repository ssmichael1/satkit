#!/usr/bin/env python3
"""
Regenerate the compiled-in ("embedded") core data files under data/embedded/.

satkit compiles a small subset of its data directly into the library so that
frame transforms and gravity work with no data directory and no network:

* IERS Conventions (2010) Tables 5.2a / 5.2b / 5.2d  (nutation / CIO series)
* the four gravity models, truncated to degree <= EMBED_MAX_DEGREE
  (the evaluator uses degree <= 40; the extra headroom keeps the files
  useful if that cap is raised)

Each file is gzip'd (level 9, mtime 0 so the output is reproducible) and
stored as data/embedded/<name>.gz; the gravity files keep every header line
(licence / attribution / tide_system) and drop only the `gfc` rows with
n > EMBED_MAX_DEGREE. SOURCES.json records, for each embedded file, the
SHA-256 of the *full* source file it was derived from (which must match
data/manifest.json), the truncation degree, and the SHA-256 of the embedded
(inflated) bytes, so the blobs are reproducible and auditable.

Usage:
    python tools/embed_data.py [--data-dir DIR] [--check]

--check verifies that regenerating from DIR would produce byte-identical
blobs (exit 1 otherwise).
"""

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "embedded"
MANIFEST = REPO / "data" / "manifest.json"

EMBED_MAX_DEGREE = 70
IERS_TABLES = ["tab5.2a.txt", "tab5.2b.txt", "tab5.2d.txt"]
GRAVITY = ["EGM96.gfc", "ITU_GRACE16.gfc", "JGM2.gfc", "JGM3.gfc"]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def default_data_dir() -> Path:
    if os.environ.get("SATKIT_DATA"):
        return Path(os.environ["SATKIT_DATA"])
    home = Path.home()
    for p in (home / "Library" / "Application Support" / "satkit-data", home / ".satkit-data"):
        if p.is_dir():
            return p
    return home / ".satkit-data"


def truncate_gfc(text: str, max_degree: int) -> str:
    """Keep all header lines; keep `gfc` rows with n <= max_degree; drop the rest."""
    out, in_header = [], True
    for line in text.splitlines(keepends=True):
        tok = line.split()
        if in_header:
            out.append(line)
            if tok and tok[0] == "end_of_head":
                in_header = False
            continue
        if tok and tok[0] == "gfc":
            if int(tok[1]) <= max_degree:
                out.append(line)
        else:
            out.append(line)
    return "".join(out)


def gzip_deterministic(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)


def build(data_dir: Path):
    manifest = json.loads(MANIFEST.read_text())
    pinned = {e["name"]: e for e in manifest["files"]}
    blobs, sources = {}, {}
    for name in IERS_TABLES + GRAVITY:
        src = data_dir / name
        if not src.is_file():
            sys.exit(f"missing source file {src} (run satkit.utils.update_datafiles() first)")
        raw = src.read_bytes()
        src_sha = sha256(raw)
        if name in pinned and pinned[name]["sha256"] != src_sha:
            sys.exit(
                f"{name}: source sha256 {src_sha} does not match data/manifest.json "
                f"({pinned[name]['sha256']})"
            )
        entry = {"source": name, "source_sha256": src_sha, "source_size": len(raw)}
        if name in GRAVITY:
            payload = truncate_gfc(raw.decode("latin-1"), EMBED_MAX_DEGREE).encode("latin-1")
            entry["truncated_to_degree"] = EMBED_MAX_DEGREE
        else:
            payload = raw
        entry["embedded_sha256"] = sha256(payload)
        entry["embedded_size"] = len(payload)
        blobs[name + ".gz"] = gzip_deterministic(payload)
        sources[name] = entry
    return blobs, sources


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument(
        "--check", action="store_true", help="verify the committed blobs match a regeneration"
    )
    args = ap.parse_args()
    blobs, sources = build(args.data_dir)
    sources_json = (
        json.dumps({"embed_max_degree": EMBED_MAX_DEGREE, "files": sources}, indent=1) + "\n"
    )
    if args.check:
        bad = [f for f, data in blobs.items() if not (OUT / f).is_file() or (OUT / f).read_bytes() != data]
        if not (OUT / "SOURCES.json").is_file() or (OUT / "SOURCES.json").read_text() != sources_json:
            bad.append("SOURCES.json")
        if bad:
            sys.exit("embedded data out of date: " + ", ".join(bad))
        print("embedded data up to date")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for fname, data in blobs.items():
        (OUT / fname).write_bytes(data)
        total += len(data)
        print(f"  {fname:24s} {len(data):8d} bytes (inflated {sources[fname[:-3]]['embedded_size']})")
    (OUT / "SOURCES.json").write_text(sources_json)
    print(f"wrote {len(blobs)} blobs, {total} bytes total, to {OUT}")


if __name__ == "__main__":
    main()

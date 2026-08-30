#!/usr/bin/env python3
"""
Download satkit's static data files into a directory, verified against
``data/manifest.json`` (the same manifest the Rust library embeds).

Standalone — needs only ``requests`` — so CI can populate the data cache
before the Rust build. Mirrors ``satkit.utils.update_datafiles()``:

* every static file is tried from ``SATKIT_DATA_URL`` (if set) first, then
  from the manifest's URLs in order (GitHub release asset, origin server,
  legacy bucket), and is only kept when its size and SHA-256 match;
* a file already present with the right hash is skipped;
* the regularly refreshed files (EOP, space weather) are fetched from the
  manifest's ``refresh`` URLs on every run, unverified; a failed refresh keeps
  the existing copy and prints a warning instead of failing the run.

Usage: ``python python/test/download_data.py [dest_dir] [--refresh-only]``
(default ``astro-data``). ``--refresh-only`` skips the manifest files and only
re-fetches EOP / space weather — run it on every CI job, including cache hits,
so a cached data directory never carries a stale EOP table.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "manifest.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_urls(entry: dict) -> list:
    mirror = os.environ.get("SATKIT_DATA_URL", "").strip().rstrip("/")
    urls = [f"{mirror}/{entry['name']}"] if mirror else []
    return urls + list(entry["urls"])


def fetch_verified(entry: dict, dest_dir: Path) -> str:
    """Return the URL the file came from, or "present" if already verified."""
    dest = dest_dir / entry["name"]
    if dest.is_file() and dest.stat().st_size == entry["size"] and sha256_of(dest) == entry["sha256"]:
        return "present"
    part = dest.with_name(dest.name + ".part")
    attempts = []
    for url in candidate_urls(entry):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                h = hashlib.sha256()
                size = 0
                with part.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
            if size != entry["size"]:
                raise ValueError(f"size mismatch (expected {entry['size']}, got {size})")
            if h.hexdigest() != entry["sha256"]:
                raise ValueError("sha256 mismatch")
            part.replace(dest)
            return url
        except Exception as e:  # noqa: BLE001 - report every source, then move on
            part.unlink(missing_ok=True)
            attempts.append(f"{url}: {e}")
    raise SystemExit(f"could not download {entry['name']} from any source:\n  " + "\n  ".join(attempts))


def fetch_refresh(url: str, dest_dir: Path) -> str:
    """Re-fetch a daily-refreshed file; on failure keep the existing copy."""
    name = url.rsplit("/", 1)[-1]
    dest = dest_dir / name
    part = dest.with_name(name + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with part.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        part.replace(dest)
        return f"refreshed from {url}"
    except Exception as exc:  # noqa: BLE001 - any failure keeps the old file
        part.unlink(missing_ok=True)
        if dest.exists():
            return f"WARNING: refresh failed ({exc}); keeping existing copy"
        return f"WARNING: refresh failed ({exc}); file absent"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refresh_only = "--refresh-only" in sys.argv[1:]
    dest_dir = Path(args[0] if args else "astro-data")
    dest_dir.mkdir(exist_ok=True, parents=True)
    manifest = json.loads(MANIFEST.read_text())
    print(f"satkit data {manifest['data_version']} -> {dest_dir}")
    if not refresh_only:
        for entry in manifest["files"]:
            if not entry.get("default", True):
                continue
            src = fetch_verified(entry, dest_dir)
            print(f"  {entry['name']}: {src}")
    for url in manifest.get("refresh", []):
        print(f"  {url.rsplit('/', 1)[-1]}: {fetch_refresh(url, dest_dir)}")


if __name__ == "__main__":
    main()

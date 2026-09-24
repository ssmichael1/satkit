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
  manifest's ``refresh`` URLs, unverified; a failed refresh keeps the existing
  copy and prints a warning instead of failing the run.

The refresh follows `CelesTrak's usage policy
<https://celestrak.org/usage-policy.php>`_, which asks clients to download
each file once per update: a local copy younger than its cadence (3 h for
space weather, 24 h for EOP; ``--max-age-hours`` overrides) is left alone with
**no request at all**, and otherwise the request carries ``If-Modified-Since``
so an unchanged file costs a ``304`` rather than several MB. The freshness
state lives in a ``<name>.http-cache`` sidecar, the same format the Rust
client writes, so the two agree about a shared data directory.

Usage: ``python python/test/download_data.py [dest_dir] [--refresh-only]
[--max-age-hours N] [--force-refresh]`` (default ``astro-data``).
``--refresh-only`` skips the manifest files and only re-fetches EOP / space
weather; ``--force-refresh`` ignores the cadence and the sidecar.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "manifest.json"

# Identify the client. CelesTrak asks callers to be identifiable so it can
# reach a project whose automation misbehaves, and an anonymous default
# (``python-requests/x.y``) is shared with every other script on the internet,
# which makes any throttling decision land on ours.
USER_AGENT = "satkit-ci (+https://github.com/ssmichael1/satkit)"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT

# CelesTrak's publication cadence per file, in seconds; the default covers
# any feed added to the manifest later.
REFRESH_MIN_AGE = {"EOP-All.csv": 24 * 3600, "SW-All.csv": 3 * 3600}
DEFAULT_MIN_AGE = 3 * 3600


def marker_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".http-cache")


def read_marker(dest: Path):
    """``(checked_at_unix, last_modified_or_None)``, or None if unusable."""
    try:
        lines = marker_path(dest).read_text().splitlines()
        checked_at = int(lines[0].strip())
    except (OSError, ValueError, IndexError):
        return None
    last_modified = lines[1].strip() if len(lines) > 1 and lines[1].strip() else None
    return checked_at, last_modified


def write_marker(dest: Path, last_modified) -> None:
    try:
        marker_path(dest).write_text(f"{int(time.time())}\n{last_modified or ''}\n")
    except OSError:
        pass  # best effort: the next run simply fetches unconditionally


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
            with SESSION.get(url, stream=True, timeout=120) as r:
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


def fetch_refresh(url: str, dest_dir: Path, max_age: int = None, force: bool = False) -> str:
    """Re-fetch a regularly updated file; on failure keep the existing copy.

    Makes the smallest request that keeps the copy current: none at all while
    it is inside its cadence, a conditional GET after that.
    """
    name = url.rsplit("/", 1)[-1]
    dest = dest_dir / name
    part = dest.with_name(name + ".part")
    # A sidecar without the file it describes says nothing about what is on
    # disk, so a missing file is always fetched in full.
    marker = read_marker(dest) if dest.is_file() else None
    min_age = REFRESH_MIN_AGE.get(name, DEFAULT_MIN_AGE) if max_age is None else max_age

    if not force and marker is not None:
        age = max(0, int(time.time()) - marker[0])
        if age < min_age:
            return f"current ({age / 3600:.1f} h old); no request made"

    headers = {}
    if not force and marker is not None and marker[1]:
        # Echo the server's own Last-Modified. The local mtime is when *we*
        # wrote the file, which is later than the data's timestamp and would
        # suppress updates the server does have.
        headers["If-Modified-Since"] = marker[1]
    try:
        with SESSION.get(url, stream=True, timeout=120, headers=headers) as r:
            if r.status_code == 304:
                write_marker(dest, marker[1])
                return "unchanged on the server (304)"
            r.raise_for_status()
            with part.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            last_modified = r.headers.get("Last-Modified")
        part.replace(dest)
        write_marker(dest, last_modified)
        return f"refreshed from {url}"
    except Exception as exc:  # noqa: BLE001 - any failure keeps the old file
        part.unlink(missing_ok=True)
        if dest.exists():
            return f"WARNING: refresh failed ({exc}); keeping existing copy"
        return f"WARNING: refresh failed ({exc}); file absent"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("dest_dir", nargs="?", default="astro-data")
    ap.add_argument(
        "--refresh-only",
        action="store_true",
        help="skip the manifest files; only consider EOP / space weather",
    )
    ap.add_argument(
        "--max-age-hours",
        type=float,
        default=None,
        metavar="N",
        help="treat a refreshed file younger than N hours as current "
        "(default: CelesTrak's cadence, 3 h for space weather, 24 h for EOP)",
    )
    ap.add_argument(
        "--force-refresh",
        action="store_true",
        help="ignore the cadence and the sidecar; always transfer the file",
    )
    ns = ap.parse_args()
    max_age = None if ns.max_age_hours is None else int(ns.max_age_hours * 3600)
    dest_dir = Path(ns.dest_dir)
    dest_dir.mkdir(exist_ok=True, parents=True)
    manifest = json.loads(MANIFEST.read_text())
    print(f"satkit data {manifest['data_version']} -> {dest_dir}")
    if not ns.refresh_only:
        for entry in manifest["files"]:
            if not entry.get("default", True):
                continue
            src = fetch_verified(entry, dest_dir)
            print(f"  {entry['name']}: {src}")
    for url in manifest.get("refresh", []):
        outcome = fetch_refresh(url, dest_dir, max_age=max_age, force=ns.force_refresh)
        print(f"  {url.rsplit('/', 1)[-1]}: {outcome}")


if __name__ == "__main__":
    main()

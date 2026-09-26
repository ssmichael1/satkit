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
* the regularly refreshed files are fetched unverified: space weather from
  the manifest's ``refresh`` URLs, Earth orientation from the first of the
  manifest's ``eop`` sources that answers (the IERS ``finals2000A.all``
  mirrors, then CelesTrak's ``EOP-All.csv``); a failed refresh keeps the
  existing copy and prints a warning instead of failing the run.
  ``--all-eop-sources`` (CI uses it) keeps the primary source refreshed as
  above and also fetches the *other* sources, but only when they are missing:
  CelesTrak's CSV is wanted for its 1962-1972 rows, which satkit puts in front
  of the IERS table and which never change, so an existing copy is never
  re-requested. CI therefore contacts CelesTrak once per new data cache, not
  per run.

The refresh follows `CelesTrak's usage policy
<https://celestrak.org/usage-policy.php>`_, which asks clients to download
each file once per update: a local copy younger than its cadence (3 h for
space weather, 24 h for EOP; ``--max-age-hours`` overrides) is left alone with
**no request at all**, and otherwise the request carries ``If-Modified-Since``
so an unchanged file costs a ``304`` rather than several MB. The same gate
applies to the IERS mirrors. The freshness state lives in a
``<name>.http-cache`` sidecar, the same format the Rust client writes, so the
two agree about a shared data directory.

Usage: ``python python/test/download_data.py [dest_dir] [--refresh-only]
[--max-age-hours N] [--force-refresh] [--all-eop-sources]`` (default
``astro-data``).
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

# Publication cadence per file, in seconds (CelesTrak's for its feeds; the
# IERS finals file is likewise updated daily); the default covers any feed
# added to the manifest later.
REFRESH_MIN_AGE = {
    "finals2000A.all": 24 * 3600,
    "EOP-All.csv": 24 * 3600,
    "Kp_ap_Ap_SN_F107_since_1932.txt": 3 * 3600,
    "45-day-forecast.txt": 24 * 3600,
    "msafe-f10-prd.txt": 7 * 24 * 3600,
}
# NASA publishes the MSAFE forecast under a month-specific name with no
# stable "latest" URL; mirror the library and walk back from the current month.
MSAFE_LOCAL = "msafe-f10-prd.txt"
MSAFE_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def msafe_candidate_urls(n: int = 6):
    y, m = time.gmtime().tm_year, time.gmtime().tm_mon
    for _ in range(n):
        yield f"https://www.nasa.gov/wp-content/uploads/{y}/{m:02d}/{MSAFE_MONTHS[m - 1]}{y}f10-prd.txt"
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)


def refresh_msafe(dest_dir: Path, max_age: int, force: bool) -> None:
    dest = dest_dir / MSAFE_LOCAL
    if not force and dest.is_file():
        marker = read_marker(dest)
        if marker and time.time() - marker[0] < max_age:
            print(f"  {MSAFE_LOCAL}: fresh, no request")
            return
    for url in msafe_candidate_urls():
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code != 200 or "F10.7" not in r.text:
                continue
            dest.write_text(r.text)
            write_marker(dest, None)
            print(f"  {MSAFE_LOCAL}: downloaded from {url}")
            return
        except requests.RequestException:
            continue
    print(f"  warning: {MSAFE_LOCAL} not refreshed (no monthly file answered); keeping any existing copy")
DEFAULT_MIN_AGE = 3 * 3600


def marker_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".http-cache")


def read_marker(dest: Path):
    """``(checked_at_unix, last_modified_or_None)``, or None if unusable.

    None means "fetch unconditionally". The marker records the size and
    whole-second mtime of the file it describes, so a copy swapped in by hand
    cannot inherit the previous file's ``Last-Modified`` and be reported as
    current by a ``304``. Sub-second mtime is ignored: it does not survive the
    tar round trip of a CI cache restore.
    """
    try:
        lines = marker_path(dest).read_text().splitlines()
        checked_at, size, mtime = (int(x) for x in lines[0].split()[:3])
        st = dest.stat()
    except (OSError, ValueError, IndexError):
        return None
    if st.st_size != size or int(st.st_mtime) != mtime:
        return None
    last_modified = lines[1].strip() if len(lines) > 1 and lines[1].strip() else None
    return checked_at, last_modified


def write_marker(dest: Path, last_modified) -> None:
    try:
        st = dest.stat()
        marker_path(dest).write_text(
            f"{int(time.time())} {st.st_size} {int(st.st_mtime)}\n{last_modified or ''}\n"
        )
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


def _refresh(url: str, dest: Path, max_age: int = None, force: bool = False) -> str:
    """Bring ``dest`` up to date from ``url``; raise on any failure.

    Makes the smallest request that keeps the copy current: none at all while
    it is inside its cadence, a conditional GET after that.
    """
    name = dest.name
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
        # No feed is legitimately empty; an empty body must not replace a
        # good file (the Rust client rejects this in `check_content`), and
        # neither must a notice page served in place of the file.
        if part.stat().st_size == 0:
            raise ValueError("the response body was empty")
        head = part.open("rb").read(256).lstrip().lower()
        if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
            raise ValueError("the response is an HTML page, not the data file")
        part.replace(dest)
        write_marker(dest, last_modified)
        return f"refreshed from {url}"
    except Exception:
        part.unlink(missing_ok=True)
        raise


def fetch_refresh(url: str, dest_dir: Path, max_age: int = None, force: bool = False) -> str:
    """Re-fetch a regularly updated file; on failure keep the existing copy."""
    dest = dest_dir / url.rsplit("/", 1)[-1]
    try:
        return _refresh(url, dest, max_age=max_age, force=force)
    except Exception as exc:  # noqa: BLE001 - any failure keeps the old file
        if dest.exists():
            return f"WARNING: refresh failed ({exc}); keeping existing copy"
        return f"WARNING: refresh failed ({exc}); file absent"


def fetch_eop(sources: list, dest_dir: Path, max_age: int = None, force: bool = False) -> str:
    """Bring the Earth orientation file up to date from the first source that answers.

    A copy of the primary file inside its cadence is reported current without
    any request, exactly as the Rust client does.
    """
    attempts = []
    for source in sources:
        dest = dest_dir / source["name"]
        for url in source["urls"]:
            try:
                return f"{source['name']}: {_refresh(url, dest, max_age=max_age, force=force)}"
            except Exception as exc:  # noqa: BLE001 - try the next source
                attempts.append(f"{url}: {exc}")
    present = [s["name"] for s in sources if (dest_dir / s["name"]).exists()]
    kept = f"keeping existing {', '.join(present)}" if present else "no EOP file present"
    return "WARNING: EOP refresh failed (" + "; ".join(attempts) + f"); {kept}"


def fetch_all_eop(sources: list, dest_dir: Path, max_age: int = None, force: bool = False) -> list:
    """Refresh the primary Earth orientation source, and fetch the others only if missing.

    The first source is kept current exactly as ``fetch_eop`` does. The other
    sources (CelesTrak's ``EOP-All.csv``) are wanted only for their historical
    rows, so an existing copy is left alone with no request at all, even with
    ``--force-refresh``.
    """
    lines = [fetch_eop(sources[:1], dest_dir, max_age=max_age, force=force)]
    for source in sources[1:]:
        dest = dest_dir / source["name"]
        if dest.exists():
            lines.append(f"{source['name']}: present; history only, not re-requested")
            continue
        attempts = []
        for url in source["urls"]:
            try:
                lines.append(f"{source['name']}: {_refresh(url, dest, force=True)}")
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                attempts.append(f"{url}: {exc}")
        else:
            lines.append(f"WARNING: {source['name']} download failed (" + "; ".join(attempts) + "); no copy present")
    return lines


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
        "--all-eop-sources",
        action="store_true",
        help="fetch every EOP source (IERS finals2000A.all and CelesTrak EOP-All.csv), "
        "not just the first that answers",
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
    refresh_msafe(dest_dir, max_age if max_age is not None else REFRESH_MIN_AGE[MSAFE_LOCAL], ns.force_refresh)
    if manifest.get("eop"):
        if ns.all_eop_sources:
            for line in fetch_all_eop(manifest["eop"], dest_dir, max_age=max_age, force=ns.force_refresh):
                print(f"  {line}")
        else:
            print(f"  {fetch_eop(manifest['eop'], dest_dir, max_age=max_age, force=ns.force_refresh)}")


if __name__ == "__main__":
    main()

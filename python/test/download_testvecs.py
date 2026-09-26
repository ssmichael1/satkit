"""
Download SatKit test vectors to a local directory.

Usage:

* ``python python/test/download_testvecs.py [dest_dir]`` (default
  ``satkit-testvecs``) fetches every file listed in the bucket's
  ``files.json``. Each file is checked against the size and MD5 the bucket's
  object listing reports: a file already present is kept only if it matches,
  so a vector replaced in the bucket is downloaded again. If the bucket
  cannot be reached and ``dest_dir`` already holds vectors (a CI cache
  restored from an older key), it warns and keeps them instead of failing.
* ``python python/test/download_testvecs.py --print-key`` prints a short
  hash of the listed files' names, sizes and MD5s (the CI cache key), or
  nothing, with exit status 1, if the bucket cannot be reached.

Needs nothing beyond the standard library (``requests`` is used if present).
"""

import hashlib
import json
import sys
import urllib.parse
from pathlib import Path

from download_from_json import download_from_json, fetch

BUCKET = "satkit-testvecs"
BASEURL = f"https://storage.googleapis.com/{BUCKET}"
LISTING = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"


def listed_files(tree, rel=""):
    """Relative paths of the files in a files.json tree."""
    if isinstance(tree, dict):
        for key, val in tree.items():
            yield from listed_files(val, rel + key + "/")
    elif isinstance(tree, list):
        for val in tree:
            yield from listed_files(val, rel)
    elif isinstance(tree, str):
        yield rel + tree


def bucket_objects():
    """``{name: (size, md5_base64)}`` for every object in the bucket."""
    out, token = {}, None
    while True:
        q = {"fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            q["pageToken"] = token
        page = json.loads(fetch(LISTING + "?" + urllib.parse.urlencode(q)))
        for it in page.get("items", []):
            out[it["name"]] = (int(it["size"]), it.get("md5Hash", ""))
        token = page.get("nextPageToken")
        if not token:
            return out


def expected_hashes(tree):
    objs = bucket_objects()
    missing = [p for p in listed_files(tree) if p not in objs]
    if missing:
        raise SystemExit(f"files.json lists files the bucket does not hold: {missing}")
    return {p: objs[p] for p in listed_files(tree)}


def cache_key(tree) -> str:
    exp = expected_hashes(tree)
    text = "".join(f"{p} {exp[p][0]} {exp[p][1]}\n" for p in sorted(exp))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--print-key":
        try:
            print(cache_key(json.loads(fetch(BASEURL + "/files.json"))))
        except Exception as e:  # noqa: BLE001 - any failure means "no key"
            print(f"cannot compute the test-vector key: {e}", file=sys.stderr)
            sys.exit(1)
        return

    basedir = Path(args[0] if args else "satkit-testvecs")
    basedir.mkdir(exist_ok=True, parents=True)
    try:
        tree = json.loads(fetch(BASEURL + "/files.json"))
        expected = expected_hashes(tree)
    except Exception as e:  # noqa: BLE001
        if any(p.is_file() for p in basedir.rglob("*")):
            print(f"::warning::test-vector bucket unreachable ({e}); keeping the vectors in {basedir}")
            return
        raise SystemExit(f"test-vector bucket unreachable and {basedir} is empty: {e}")
    download_from_json(tree, str(basedir), BASEURL, expected)


if __name__ == "__main__":
    main()

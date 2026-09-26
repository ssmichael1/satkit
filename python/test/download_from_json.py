"""Download a tree of files described by a ``files.json`` listing.

Needs nothing beyond the standard library (``requests`` is used if present).

``download_from_json(tree, dest, base_url, expected=None)`` walks ``tree``
(dicts are directories, lists hold file names) and fetches each file from
``base_url``. ``expected`` maps a file's path relative to the root
(``"sgp4/00005.e"``) to its ``(size, md5_base64)`` as the bucket reports it;
with it, a file already on disk is kept only when it matches (a replaced
vector is downloaded again) and a downloaded file must match. Without it (the
bucket listing was unavailable), a file already on disk is kept as is.
"""

import base64
import hashlib
import os
import urllib.request
from pathlib import Path

# Seconds to wait for the server to connect / send data before giving up.
TIMEOUT = 60
USER_AGENT = "satkit-ci (+https://github.com/ssmichael1/satkit)"


def fetch(url: str) -> bytes:
    """GET ``url``; raise on any failure, including a 4xx/5xx status (so an
    error page is never saved in place of a file).

    Uses ``requests`` when it is installed (it brings certifi's CA bundle;
    some Python builds, python.org's macOS installer among them, have no
    usable system CA store), else the standard library.
    """
    try:
        import requests
    except ImportError:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read()
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def md5_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def matches(data: bytes, want) -> bool:
    size, md5 = want
    return len(data) == size and md5_b64(data) == md5


def download_from_json(jval, dname, url, expected=None, rel=""):
    if isinstance(jval, dict):
        for key, val in jval.items():
            dirname = os.path.join(dname, key)
            Path(dirname).mkdir(exist_ok=True, parents=True)
            download_from_json(val, dirname, url + "/" + key, expected, rel + key + "/")
    elif isinstance(jval, list):
        for val in jval:
            download_from_json(val, dname, url, expected, rel)
    elif isinstance(jval, str):
        fname = os.path.join(dname, jval)
        want = expected.get(rel + jval) if expected is not None else None
        if os.path.exists(fname):
            if want is None:
                print(f"{fname} exists; skipping")
                return
            with open(fname, "rb") as f:
                if matches(f.read(), want):
                    print(f"{fname} verified; skipping")
                    return
            print(f"{fname} differs from the bucket copy; replacing")
        dlurl = url + "/" + jval
        print(f"Downloading from {dlurl} to {fname}")
        data = fetch(dlurl)
        if want is not None and not matches(data, want):
            raise SystemExit(f"{dlurl}: size or MD5 does not match the bucket listing")
        part = fname + ".part"
        with open(part, "wb") as f:
            f.write(data)
        os.replace(part, fname)

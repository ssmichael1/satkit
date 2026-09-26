"""
Download SatKit test vectors
to a local directory
"""

from pathlib import Path
import requests
from download_from_json import TIMEOUT, download_from_json
import sys

if __name__ == "__main__":
    baseurl = "https://storage.googleapis.com/satkit-testvecs"

    if len(sys.argv) > 1:
        basedir = sys.argv[1]
    else:
        basedir = "satkit-testvecs"

    fileurl = baseurl + "/files.json"
    headers = {"Accept": "application/json"}

    Path(basedir).mkdir(exist_ok=True, parents=True)

    r = requests.get(fileurl, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    download_from_json(data, basedir, baseurl)

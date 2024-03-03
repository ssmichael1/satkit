import os
from pathlib import Path
import requests


def download_from_json(jval, dname, url):
    if isinstance(jval, dict):
        for key, val in jval.items():
            dirname = os.path.join(dname, key)
            Path(dirname).mkdir(exist_ok=True, parents=True)
            download_from_json(val, dirname, url + "/" + key)
    elif isinstance(jval, list):
        for val in jval:
            download_from_json(val, dname, url)
    elif isinstance(jval, str):
        fname = os.path.join(dname, jval)
        if not os.path.exists(fname):
            dlurl = url + "/" + jval
            print(f"Downloading from {dlurl} to {fname}")
            r = requests.get(url + "/" + jval)
            open(fname, "wb").write(r.content)
        else:
            print(f"{fname} exists; skipping")

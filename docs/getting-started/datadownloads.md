# Downloads and Refresh

How `satkit` fetches the files it does not compile in, how each download is
verified, how often the two daily tables are refreshed, and what happens when a
fetch fails.

## Where the files come from, and how downloads are verified

The downloadable files are described by a manifest compiled into the library
(`data/manifest.json` in the repository) that pins each file's exact size and
SHA-256 and lists where it may be downloaded from, in order of preference:

1. `SATKIT_DATA_URL` — if set, tried first for every file.
2. The GitHub release asset (`github.com/ssmichael1/satkit-data/releases/download/data-v1/…`).
3. The originating server where it serves identical bytes: JPL for the DE
   ephemerides, IERS for the `tab5.2*` tables.
4. The legacy `storage.googleapis.com/astrokit-astro-data` bucket (transitional).

A download is streamed to `<name>.part`, hashed as it goes, and only renamed
into place when both size and SHA-256 match the manifest; otherwise it is
discarded and the next source is tried. A file already present with the right
hash is never re-downloaded. The manifest is therefore what makes a given
satkit release reproducible: the same version always resolves to the same
data bytes.

Sources and attribution: DE440 / DE421 — JPL (Park et al. 2021; Folkner et al.
2009), US Government work; `tab5.2a/b/d.txt` — IERS Conventions (2010), TN 36;
EGM96, EGM2008, JGM-2, JGM-3 — NASA GSFC / NGA (US Government work), via ICGEM;
ITU_GRACE16 — Akyilmaz et al. 2016, GFZ Data Services, CC BY 4.0 (downloaded on
demand, not compiled in). The
Earth-orientation file is fetched from the IERS mirrors (CelesTrak's copy as
the fallback) and the space-weather file from CelesTrak; neither is pinned
(they change daily). See [How often they are
refreshed](#how-often-eop-and-space-weather-are-refreshed). The full table,
with licences, is in `data/README.md`.

## How often EOP and space weather are refreshed

`finals2000A.all` (or its fallback `EOP-All.csv`) and `SW-All.csv` are
whole-history tables — 1957 or 1973 to the present, several MB — and are the
only files satkit fetches more than once.
[CelesTrak's usage policy](https://celestrak.org/usage-policy.php) asks for one
download per update, so satkit makes the smallest request that keeps the local
copy current, from the IERS mirrors and CelesTrak alike:

| local copy | what happens |
|---|---|
| younger than its publication cadence (3 h for space weather, 24 h for EOP) | **no request is made** |
| older than that, unchanged upstream | a conditional `If-Modified-Since` request; the server answers `304` and no body is transferred |
| older than that, changed upstream | the new file is downloaded and installed |

So calling `update_datafiles()` at the top of every script is fine — it will
not hit the IERS mirrors or CelesTrak more than the data actually changes. `overwrite=True` skips
the gate and always transfers, which is the way to replace a copy you suspect
is damaged. The freshness state is a `<name>.http-cache` sidecar next to the
file. It also records the file's size and modification time and is ignored
once those stop matching, so a copy you replace by hand is re-fetched rather
than assumed current; deleting the sidecar (or the file) also restores a full
fetch.

Every request satkit makes also identifies itself as
`satkit/<version> (+https://github.com/ssmichael1/satkit)`, and an HTTP error
from CelesTrak is returned to you with an explanation rather than retried in a
loop — repeated retries are what gets a client firewalled.

The IERS tables and gravity models are **not** downloaded — they are compiled
in (the tables byte-identical, gravity to degree 70 — the evaluation cap, so
results are identical). The full-degree `.gfc` files remain pinned in the
manifest and hosted on the `data-v1` release; drop one into a search
directory and it takes precedence over the compiled-in copy.

## Downloads behind a TLS-inspecting proxy

Organisations that inspect outbound TLS put a gateway between satkit and the
data servers: it terminates the connection and re-signs every certificate with
a private CA. satkit verifies servers against the **operating system's trust
store** — the keychain on macOS, the certificate store on Windows, `/etc/ssl`
and friends on Unix — which is exactly where such a CA is installed, so those
downloads work with no configuration.

If they do not, the failure looks like this:

```
RuntimeError: could not fetch https://celestrak.org/SpaceData/SW-All.csv: io: invalid peer certificate: UnknownIssuer
```

The certificate presented is signed by something the machine does not trust.
Either the private CA is not installed system-wide — install it, or point
`SATKIT_CA_BUNDLE` at a PEM file containing it *together with* the public roots
— or no interception is expected on that network, in which case the certificate
really is untrusted and the download should not be forced through. satkit has no
"skip verification" switch.

A proxy that answers with a notice page instead of blocking outright cannot
corrupt the data either: `finals2000A.all`, `EOP-All.csv` and `SW-All.csv` are parsed before they
replace the copy on disk, and any download that opens with an HTML document is
rejected. The partial file is discarded and the existing one left in place, so a
blocked refresh degrades to a stale table rather than a broken one.

Note that `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` are deliberately ignored:
Python tooling routinely points them at a stock public bundle, which is the one
trust store guaranteed to fail on an intercepting network.

## Failure behaviour

| situation | what satkit does |
|---|---|
| download interrupted, or bytes don't match the manifest | the temporary `*.part.<pid>.<seq>` file is deleted and the next source is tried; a final file is only ever renamed into place after its size and SHA-256 matched |
| two processes fetch the same file at once (parallel test workers, several notebooks) | each writes its own temporary file; whichever finishes first is renamed into place, the others verify it and discard their copy — one verified file, no lock files, no partial reads |
| an ephemeris already on disk under a pinned name is corrupt or truncated | detected on first load (the file is hashed once, ~0.2 s for DE440, and a `<name>.sha256-verified` marker records the result so later loads only compare size and mtime); re-downloaded if downloads are allowed, otherwise `RuntimeError` naming the expected hash. Files not in the manifest (a user-supplied ephemeris) are trusted as-is |
| no writable location (`SATKIT_DATA` unset and no home / `%LOCALAPPDATA%`; a read-only directory) | `RuntimeError` listing the directories consulted and asking for `SATKIT_DATA`; never the current directory or a temp dir |
| an existing file cannot be replaced (Windows: another process has it open) | the rename is retried a few times, then `RuntimeError` naming the file |
| both IERS mirrors are unreachable for the Earth-orientation refresh | CelesTrak's `EOP-All.csv` is fetched instead and a warning names the URLs that failed; the loader then uses whichever of the two files on disk has the later observed record |
| every source fails (no network, all mirrors down) | `RuntimeError` listing each URL and why it failed |

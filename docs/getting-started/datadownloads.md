# Downloads and Refresh

How `satkit` fetches the files it does not compile in, how each download is
verified, how often the Earth-orientation and space-weather tables are refreshed, and what happens when a
fetch fails.

## Where the files come from, and how downloads are verified

The downloadable files are described by a manifest compiled into the library
(`data/manifest.json` in the repository) that pins each file's exact size and
SHA-256 and lists where it may be downloaded from, in order of preference:

1. `SATKIT_DATA_URL` — if set, tried first for every file in the manifest. The
   Earth-orientation and space-weather files are not in it and never use the
   mirror (see [How often EOP and space weather are refreshed](#how-often-eop-and-space-weather-are-refreshed) below).
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

What each file is, with its citation and licence, is listed under
[The files](datafiles.md#the-files); the full table is in `data/README.md`.

## How often EOP and space weather are refreshed

`finals2000A.all`, the GFZ space-weather
record (1932 to the present, several MB) and the two forecasts are the only
files satkit fetches more than once. satkit makes the smallest request that
keeps each local copy current, from every source alike:

| local copy | what happens |
|---|---|
| younger than its publication cadence (3 h for the GFZ record, 24 h for the SWPC forecast and EOP, a week for MSAFE) | **no request is made** |
| older than that, unchanged upstream | a conditional `If-Modified-Since` request; the server answers `304` and no body is transferred |
| older than that, changed upstream | the new file is downloaded and installed |

So calling `update_datafiles()` at the top of every script is fine — it will
not hit any of the sources more than the data actually changes. (MSAFE has no
stable URL — NASA names each month's file after the month — so satkit walks
back from the current month to the newest issue, and keeps the copy under the
stable name `msafe-f10-prd.txt`.) `overwrite=True` skips
the gate and always transfers, which is the way to replace a copy you suspect
is damaged. The freshness state is a `<name>.http-cache` sidecar next to the
file. It also records the file's size and modification time and is ignored
once those stop matching, so a copy you replace by hand is re-fetched rather
than assumed current; deleting the sidecar (or the file) also restores a full
fetch.

Every request satkit makes also identifies itself as
`satkit/<version> (+https://github.com/ssmichael1/satkit)`, and an HTTP error
is returned to you with an explanation rather than retried in a loop —
repeated retries are what gets a client firewalled.

## Downloads behind a TLS-inspecting proxy

Organisations that inspect outbound TLS put a gateway between satkit and the
data servers: it terminates the connection and re-signs every certificate with
a private CA. satkit verifies servers against the **operating system's trust
store** — the keychain on macOS, the certificate store on Windows, `/etc/ssl`
and friends on Unix — which is exactly where such a CA is installed, so those
downloads work with no configuration.

If they do not, the failure looks like this:

```
RuntimeError: could not fetch https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt: io: invalid peer certificate: UnknownIssuer
```

The certificate presented is signed by something the machine does not trust.
Either the private CA is not installed system-wide — install it, or point
`SATKIT_CA_BUNDLE` at a PEM file containing it *together with* the public roots
— or no interception is expected on that network, in which case the certificate
really is untrusted and the download should not be forced through. satkit has no
"skip verification" switch.

A proxy that answers with a notice page instead of blocking outright cannot
corrupt the data either: `finals2000A.all` and each of the three
space-weather files are parsed before they replace the copy on disk, and any
download that opens with an HTML document is
rejected. A `finals2000A.all` transfer cut short (inside a line, or before
its predictions) is rejected the same way. The partial file is discarded and the
existing one left in place, so a
blocked refresh degrades to a stale table rather than a broken one, and the
Earth-orientation table already loaded stays in use.

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
| every source fails (no network, all mirrors down) | `RuntimeError` listing each URL and why it failed |

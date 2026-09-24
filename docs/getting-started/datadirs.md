# Data Directories

Where `satkit` looks for its data files, where it writes downloads, and how to
provision a machine up front. For the inventory of files themselves see
[Data Files](datafiles.md).

## Where satkit looks for data, and where it writes

Two separate questions. Files are **looked up** across an ordered list of directories, any of which may be read-only; downloads are **written** to exactly one directory. `satkit.utils.data_search_dirs()` returns the first list, `satkit.utils.datadir()` the write location.

| order | searched | macOS | Linux / other Unix | Windows |
|---|---|---|---|---|
| 1 | `SATKIT_DATA` environment variable — **also the write location when set** | ✓ | ✓ | ✓ |
| 2 | directory passed to `set_datadir()` — also the write location | ✓ | ✓ | ✓ |
| 3 | directories registered with `add_search_dir()` (the `satkit` Python package registers an installed `satkit_data` bundle this way) | ✓ | ✓ | ✓ |
| 4 | `<directory of the satkit shared library>/satkit-data` | ✓ | ✓ | ✓ |
| 5 | `<site-packages>/satkit_data/data` — the optional [`satkit-data` bundle](#the-optional-satkit-data-bundle) | ✓ | ✓ | ✓ |
| 6 | **platform user-data directory — the default write location** | `~/Library/Application Support/satkit-data` | `$XDG_DATA_HOME/satkit-data`, default `~/.local/share/satkit-data` | `%LOCALAPPDATA%\satkit-data` |
| 7 | `~/.satkit-data` (legacy location, read only) | ✓ | ✓ | ✓ (`%USERPROFILE%`) |
| 8 | `/usr/share/satkit-data` (system-wide, read only) | ✓ | ✓ | — |
| 9 | `/Library/Application Support/satkit-data` (system-wide, read only) | ✓ | — | — |

A file is used from the first directory that contains it. The ephemeris is also auto-detected across all of them (highest DE version wins). satkit never creates a directory next to its own shared library or inside `site-packages` — such a directory is often not writable and is wiped on reinstall.

## Environment variables and API

| control | effect |
|---|---|
| `SATKIT_DATA=/path` | search first and write here (created if needed) |
| `SATKIT_DATA_URL=https://mirror/base` | try `"$SATKIT_DATA_URL/<name>"` before the manifest's sources for every download (plain `http://` accepted; still hash-verified) |
| `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` / `NO_PROXY` | honoured for every download (standard proxy environment variables; read by the HTTP client) |
| `SATKIT_CA_BUNDLE=/path/bundle.pem` | verify servers against the certificates in this PEM file instead of the operating system's trust store. The file replaces the trust store outright, so it must also carry the public roots (`python -m certifi` with the private CA appended). Also accepts `platform` (the default) and `webpki` (the Mozilla list compiled into satkit, for a container with no system trust store). See [Downloads behind a TLS-inspecting proxy](datadownloads.md#downloads-behind-a-tls-inspecting-proxy) |
| `SATKIT_OFFLINE=1` / `satkit.utils.set_offline(True)` | forbid **downloads** — `update_datafiles()`, the lazy ephemeris fetch, the EOP/SW refresh, any non-embedded file — with a `RuntimeError` naming the file and its sources; no connection is opened. Search locations and the compiled-in data are unaffected. The setter wins once called; otherwise the variable is read. `satkit.utils.is_offline()` reports the effective state |
| `SATKIT_JPLEPHEM_FILE=name-or-path` | which ephemeris to load — see [below](#selecting-a-jpl-ephemeris-file) |
| `SATKIT_QUIET=1` | suppress the warning printed when a corrupt or unreadable IERS table in a search directory is replaced by the compiled-in copy |
| `satkit.utils.datadir()` | the write location (`None` if none can be determined — no `SATKIT_DATA`, no home / `%LOCALAPPDATA%`) |
| `satkit.utils.data_search_dirs()` | the search list, in order |
| `satkit.utils.set_datadir(path)` / `add_search_dir(path)` | add an override / a read-only search location |
| `satkit.utils.datafiles_exist()` | whether an ephemeris file is present in any search directory (the marker of a provisioned data location) |

## The optional `satkit-data` bundle

`pip install satkit[data]` installs the `satkit-data` package (~110 MB: the
ephemeris, full-degree gravity files, IERS tables) into `site-packages`. It is
picked up automatically as a read-only search location (rows 3 and 5 above),
so no first-use download happens. It is not required — earlier releases made
it a hard dependency of `satkit`; it is now optional.

## Provisioning up front

Nothing needs to be downloaded before first use, but for a container image,
a CI job, or a machine that will later be offline:

```python
import satkit as sk
sk.utils.update_datafiles()   # ephemeris (verified) + EOP + space weather
```

Files already present with the right hash are skipped; the space-weather and
Earth-orientation files are refreshed if they are due (below).
`update_datafiles(dir="...")` writes somewhere else; `overwrite=True`
re-downloads even verified files, including those two.

## Selecting a JPL ephemeris file

By default `satkit` uses `linux_p1550p2650.440` (DE440), downloading it on first use if no ephemeris is found in any search directory. There are two ways to override that choice.

### Environment variable

Set `SATKIT_JPLEPHEM_FILE` to either an absolute path or a basename:

```bash
# Absolute path — file used directly (no download)
SATKIT_JPLEPHEM_FILE=/opt/jpl/lnxp1900p2053.421 python script.py

# Basename — found in any search directory, or downloaded to datadir() if it is a manifest file
SATKIT_JPLEPHEM_FILE=lnxp1900p2053.421 python script.py
```

**If the value is wrong.** The ephemeris is loaded on first use and the outcome is cached for the process, so a bad setting surfaces as an error from the first ephemeris query (`satkit::jplephem::Error::LoadFailed` in Rust, `RuntimeError` in Python) rather than at import. The message names the resolved path and that it was selected by `SATKIT_JPLEPHEM_FILE`:

- a path that does not exist → `… No such file or directory`;
- a bare name that is not in the data manifest → resolved against the write directory and reported as missing (only manifest-pinned names such as `linux_p1550p2650.440` and `lnxp1900p2053.421` are downloaded on demand);
- a file that exists but is not a JPL Linux-format binary ephemeris → `not a JPL binary ephemeris (header starts with …)`.

There is no silent fallback to another ephemeris: an explicit selection that fails stays failed until the setting is fixed and the process restarted.

Both DE440 and DE421 are in the manifest and can be downloaded by name; any other file must already exist.

### Autodetect

With no environment variable set, every search directory is scanned for JPL Linux-binary ephemeris files (`linux_p*.4XX`, `lnxp*.4XX`) and the highest DE version found is used, so dropping a file into the data directory is enough to switch to it.

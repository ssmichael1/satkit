# satkit data manifest

`manifest.json` is the single source of truth for the static data files
satkit downloads at runtime. It is compiled into the library
(`include_str!` in `src/utils/manifest.rs`), read by `python/test/download_data.py`
for CI, and keys the CI data cache. This note explains the design so it can be
maintained without re-deriving it.

## Why a manifest

Before this, the files lived on a Google Cloud Storage bucket and were
fetched by name with no integrity check: a satkit release did not determine
which data bytes a user got, a file could change under a fixed URL (it did —
`msis21.parm`, Feb 2026), and four hand-maintained copies (bucket, PyPI
`satkit-data` wheel, conda recipe, CI cache) drifted independently.

With the manifest:

- **A satkit build pins its data.** Every file has an exact size and SHA-256.
  A given release always resolves to the same bytes, from whichever source
  happens to work.
- **Downloads are verified before they are trusted.** The fetch streams into
  `<name>.part`, hashes as it goes, and only renames into place on a match.
  A corrupt, truncated or substituted download is discarded and the next
  source is tried; if all fail, the error lists every URL and why.
- **One artefact drives everything.** The CI cache key is
  `hashFiles('data/manifest.json')`, so changing the data invalidates the
  cache automatically (the old static keys never did).

## URL order and why

For each file, `urls` are tried in order; `SATKIT_DATA_URL` (environment) is
tried before all of them.

| order | source | rationale |
|---|---|---|
| 0 | `$SATKIT_DATA_URL/<name>` | corporate mirror, air-gapped share, or a local test server. Plain `http://` is allowed *here only*; every download is still hash-verified |
| 1 | `https://github.com/ssmichael1/satkit-data/releases/download/data-v1/<name>` | GitHub release asset: CDN-backed, stable per-tag URL, no bandwidth cost to the maintainer, no API rate limit on downloads. **Returns 404 until the release is published** (below) — the client falls through cleanly |
| 2 | origin server | only where the origin serves *byte-identical* data, verified by hash when the manifest was built: JPL for the DE files, IERS for `tab5.2*`. Zero hosting; the hash protects against silent upstream changes |
| 3 | `https://storage.googleapis.com/astrokit-astro-data/<name>` | the legacy bucket, kept as a transitional fallback for a release or two |

All manifest URLs must be `https://` (validated on load).

## What is in the manifest

| file | size | source | licence / attribution | tier |
|---|---|---|---|---|
| `linux_p1550p2650.440` | 102.3 MB | JPL | DE440 (Park et al. 2021). US Government work, public domain. Origin URL verified byte-identical | ephemeris (default) |
| `lnxp1900p2053.421` | 14.0 MB | JPL | DE421 (Folkner et al. 2009). US Government work, public domain. `default: false` — fetched only by name (`SATKIT_JPLEPHEM_FILE=lnxp1900p2053.421`) | ephemeris |
| `tab5.2a.txt`, `tab5.2b.txt`, `tab5.2d.txt` | 171 / 137 / 9 KB | IERS | IERS Conventions (2010), TN 36, Tables 5.2a/b/d. Freely redistributable. Origin URLs verified byte-identical. `default: false` — embedded byte-identical in the binary | core |
| `EGM96.gfc` | 5.6 MB | ICGEM (GFZ) | EGM96, Lemoine et al. 1998, NASA GSFC/NIMA — US Government work. `default: false` — embedded to degree 70 | core |
| `EGM2008.gfc` | 252 MB | ICGEM (GFZ) | EGM2008, Pavlis et al. 2012, NGA — US Government work. `default: false` — embedded to degree 70. Origin URL verified byte-identical; the release-asset URL is listed first for consistency but the file is not uploaded (it is only ever fetched by name, and the client falls through to ICGEM) | core |
| `JGM2.gfc`, `JGM3.gfc` | 118 / 215 KB | ICGEM (GFZ) | JGM-2 (Nerem et al. 1994), JGM-3 (Tapley et al. 1996), NASA GSFC / UT CSR — US Government work. `default: false` — embedded to degree 70 | core |
| `ITU_GRACE16.gfc` | 1.8 MB | ICGEM (GFZ) | Akyilmaz et al. 2016, GFZ Data Services, **CC BY 4.0** — the file's header block carries the attribution. **Not embedded** (the licence would otherwise attach to the library and its packages): `default: false`, fetched on first use of `GravityModel::ITUGrace16`. Origin URL verified byte-identical | core |

`tier` is informational: `core` = the small files frames and gravity need
(embedded in the binary since Phase 2, below — which is why their manifest
entries are `default: false`: pinned and fetchable by name, but pointless to
download while evaluation is capped at degree 40; ITU_GRACE16 is the one
`core` file that is not embedded and is fetched on demand), `ephemeris` =
the large JPL files. The only `default: true` entry — the only file
`update_datafiles()` downloads besides the daily refreshes — is the DE440
ephemeris.

## Files deliberately excluded

- **`msis21.parm`** — the NRLMSIS 2.1 parameter file. NRL's licence is
  academic / non-commercial and covers derived data products, which is
  incompatible with satkit's MIT / Apache-2.0 distribution. Nothing on `main`
  reads it (the NRLMSIS 2 port is on an unmerged branch); if that feature
  ships it must be an opt-in download with its own notice, not part of the
  default bundle.
- **`finals2000A.all`, `EOP-All.csv`** — Earth orientation. It changes daily,
  so it is never pinned, and CelesTrak grants no redistribution licence for
  its compiled file, so satkit does not mirror it. Earth orientation is fetched via the manifest's `eop` list,
  in order: the IERS Bulletin A combined file `finals2000A.all` from the USNO
  mirror, then from the IERS data centre, then CelesTrak's `EOP-All.csv` as
  the fallback. The loader reads both formats and, when both are on disk,
  uses the one whose observed record runs later (the CSV's 1962–1972 rows are
  kept in front of the IERS table, which starts in 1973). See
  [Refresh policy](#refresh-policy-celestrak) for how often either is fetched.
- **`Kp_ap_Ap_SN_F107_since_1932.txt`, `45-day-forecast.txt`,
  `msafe-f10-prd.txt`** — space weather, from its producers rather than a
  redistributor. The observed record is GFZ Potsdam's (CC BY 4.0; cite
  Matzka et al. 2021, *The geomagnetic Kp index*, and the data publication
  doi:10.5880/Kp.0001 in derived work; its `SN` column is CC BY-NC 4.0 from
  SILSO and is not ingested), the 45-day forecast NOAA/SWPC's and the monthly
  forecast NASA MSFC's (both US Government work). The first two are in the
  manifest's `refresh` list; MSAFE has no stable URL and is month-walked by
  the library. Never pinned. CelesTrak's merged space-weather file is no
  longer fetched.
- **`sw19571001.txt`** — an orphan on the old bucket; nothing reads it.
- **`leap-seconds.list`** — nothing reads it: the runtime leap-second table
  is a compiled-in constant (`src/time/instant.rs`). It was pinned
  (`tier: reference`) through 0.21.2 and remains an asset on the immutable
  `data-v1` tag, but the manifest entry is gone.

## Refresh policy

`finals2000A.all` (or its CelesTrak fallback `EOP-All.csv`), the GFZ
space-weather record and the two space-weather forecasts are the only files
satkit fetches repeatedly, and the first two are whole-history tables (1932 or
1973 to the present, several MB); the CelesTrak fallback is served by one
person's site. The policy below was written to [CelesTrak's usage
policy](https://celestrak.org/usage-policy.php), which asks clients to "only download
the data you need, when you are going to use it, and only download data once
per update", publishes space weather every 3 hours and EOP once a day, and
warns that machine-to-machine clients ignoring non-200 responses get
firewalled. Satkit follows it in four places, and applies the same cadence gate
and conditional request to the IERS mirrors:

| | |
|---|---|
| **cadence gate** | `utils::refresh_file` makes **no request at all** while the local copy is younger than the file's publication cadence (`refresh_min_age_secs`: 3 h for the GFZ record, 24 h for the SWPC forecast, `EOP-All.csv` and `finals2000A.all`, a week for MSAFE). `update_datafiles(overwrite=True)` (Rust: `overwrite_if_exists = true`) forces a fetch anyway |
| **conditional GET** | past the cadence the request carries `If-Modified-Since`, echoing the server's own `Last-Modified`, so an unchanged file costs a `304` and no body. State lives in a `<name>.http-cache` sidecar, which also records the file's size and whole-second mtime and is ignored once those stop matching, so a copy swapped in by hand is re-fetched rather than reported current by a `304`; delete it (or the file) to force a full fetch |
| **identification** | every request sends `User-Agent: satkit/<version> (+https://github.com/ssmichael1/satkit)` (`download::USER_AGENT`) rather than `ureq/3.x`, so a misbehaving client is traceable to the project |
| **empty-body guard** | `check_content` rejects a zero-byte response before it can replace a good file: `finals2000A.all` / `EOP-All.csv` and the three space-weather files are additionally parsed, but a feed added later would have nothing else between a broken server and a truncated table |
| **no retry loop** | an HTTP error is returned to the caller, with `celestrak_throttle_hint` explaining 403/503 and telling the user to cache rather than retry |

CI is the other half of the problem: a data-cache hit used to be followed by an
unconditional refresh in every job, which is ~12 full-file downloads per push
from GitHub's datacenter ranges. The test jobs no longer refresh at all — every
test that touches the EOP table works from the table's own bounds, so a cached
copy stays valid however old it is — and the docs and release workflows refresh
only when the cached copy is more than a week old
(`download_data.py --max-age-hours 168`), which is well inside the ~1 year of
predictions `finals2000A.all` carries.

## Publishing the release assets (maintainer)

The manifest already points at `data-v1` on the `ssmichael1/satkit-data`
repository (chosen over the main repo so satkit's Releases page stays for
software, and so that repo can purge the 100 MB ephemeris from its git
history). Until the release exists, every download falls through to the
origin / GCS URLs, so nothing breaks — but publishing makes the first URL win:

```bash
D="$HOME/Library/Application Support/satkit-data"   # or any dir holding verified copies
gh release create data-v1 --repo ssmichael1/satkit-data --latest=false \
  --title "satkit static data v1" \
  --notes "Static data files pinned by satkit's data/manifest.json (sizes and SHA-256 there). Sources and licences: see data/README.md in ssmichael1/satkit." \
  "$D/linux_p1550p2650.440" "$D/lnxp1900p2053.421" \
  "$D/tab5.2a.txt" "$D/tab5.2b.txt" "$D/tab5.2d.txt" \
  "$D/EGM96.gfc" "$D/ITU_GRACE16.gfc" "$D/JGM2.gfc" "$D/JGM3.gfc"
```

(`EGM2008.gfc` was pinned later and is deliberately not uploaded: 252 MB for
a file nothing downloads by default. Its manifest entry lists the release URL
first for consistency; the fetch falls through to the ICGEM origin, verified
by hash.)

(`lnxp1900p2053.421` can be fetched first with
`curl -O https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de421/lnxp1900p2053.421`;
its sha256 is in the manifest.) Then verify end-to-end:

```bash
SATKIT_DATA=/tmp/satkit-data-check cargo test --lib real_network -- --ignored --nocapture
```

### Release-tag policy

A satkit release pins one data tag by hash. Once published, **a data tag is
immutable**: its assets are never replaced and the tag is never deleted, so
every satkit version that pins `data-v1` keeps working indefinitely. A data
update (a new DE, a corrected table) is a *new* tag — `data-v2` — plus a
manifest change in a satkit release; nothing about the old tag changes.
`--latest=false` keeps data tags off the repository's "latest release".

## Regenerating / changing the manifest

```bash
# after replacing or adding files in a data directory:
python tools/make_manifest.py --data-dir "$D"            # recompute size + sha256
python tools/make_manifest.py --data-dir "$D" --check    # CI-style drift check (exit 1 on change)
python tools/make_manifest.py --data-dir "$D" --data-version data-v2   # new release tag
```

- **Adding a file**: add a stub entry (`name`, `urls`, `source`, `license`,
  `tier`, `default`) to `manifest.json`, put the file in the data dir, run the
  tool. The Rust unit test `embedded_manifest_is_valid` enforces the schema.
  Upload the file to the release (`gh release upload data-vN --repo
  ssmichael1/satkit-data <file>`).
- **Changing bytes of an existing file** (e.g. a corrected table): that is a
  new data version — bump `--data-version`, publish a new release tag, and
  ship the manifest change in a satkit release. Never overwrite an asset under
  an existing tag; the old satkit releases pin the old hashes.
- **Retiring GCS**: once a release or two have shipped with the release-asset
  URLs first, drop the `storage.googleapis.com` entries from `urls` and
  delete the bucket. No client change is needed.
- **conda**: there is deliberately no `satkit-data` conda package. The
  conda-forge `satkit` package (built in
  [conda-forge/satkit-feedstock](https://github.com/conda-forge/satkit-feedstock))
  relies on the embedded core data plus the on-demand, verified ephemeris
  download like the wheels do; offline conda users populate a directory with
  `satkit.utils.update_datafiles()` and set `SATKIT_DATA`, or set
  `SATKIT_DATA_URL` to a mirror.

## Client behaviour

- `satkit.utils.update_datafiles()` / `utils::update_datafiles` — fetches all
  `default: true` files (in parallel, verified), then the `refresh` files (with
  the EOP mirror walk and the MSAFE month walk beside them). `overwrite=True`
  re-downloads even verified files.
- First-use lazy loads (`jplephem`, `earthgravity`, `ierstable`) go through
  the same verified fetch by name. A file name that is not in the manifest
  (a user's alternative ephemeris, say) falls back to the old unverified
  bucket fetch with a warning.
- `utils::manifest::embedded()` exposes the parsed manifest;
  `fetch_static_file(entry, dir, force)` is the verified fetch;
  `ManifestEntry::verify(path)` checks a file on disk.
- `ManifestEntry::ensure_verified(path)` hashes an on-disk pinned file once
  (≈0.2 s for DE440) and records `<path>.sha256-verified` (hash, size, mtime)
  so later loads skip the hash; a mismatch is `download::Error::CorruptFile`.
  The lazy ephemeris load uses it: a corrupt copy is re-fetched, or — under
  offline mode — reported with the expected hash.

### Failure behaviour

| situation | behaviour |
|---|---|
| concurrent fetches of one file | per-process/per-call temporary name `<name>.part.<pid>.<seq>`, atomic rename; a process that finishes second verifies the winner's file and discards its own; no lock files (nothing to go stale). Leftover `*.part.*` from a killed process is harmless. |
| corrupt / truncated download | hash mismatch → temporary file deleted, next URL tried, `AllSourcesFailed` lists every attempt |
| corrupt file already on disk (pinned name) | `ensure_verified` → re-fetch (or `CorruptFile` when offline / without the `download` feature) |
| no writable directory | `datadir::Error::NoWriteableDirectory { detail }` — lists the search dirs consulted, says to set `SATKIT_DATA`; never falls back to the CWD |
| rename over an open file (Windows) | `download::retry_io` retries 6× at 50 ms, then `download::Error::ReplaceFailed { path }` |
| proxies | ureq's default agent reads `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY` and `NO_PROXY` |


## Phase 2: embedded core data, lazy ephemeris, optional bundle

The manifest made downloads *safe*; Phase 2 makes them *rare*. Data is
handled in three tiers:

| tier | files | how |
|---|---|---|
| **embedded** | `tab5.2a/b/d.txt`; `EGM96/EGM2008/JGM2/JGM3.gfc` truncated to degree 70 | gzip'd into `data/embedded/*.gz` (295 KB total) and compiled in with `include_bytes!` (`src/utils/embedded.rs`), inflated on first use. Frames and gravity need **no data directory and no network** |
| **ephemeris** | `linux_p1550p2650.440` (DE440, 102 MB) or `lnxp1900p2053.421` (DE421, 14 MB) | downloaded on first use through the verified manifest fetch, into the write location |
| **on demand** | `ITU_GRACE16.gfc` (1.8 MB, CC BY 4.0) | same verified fetch, on first use of `GravityModel::ITUGrace16`; not embedded so the licence does not attach to the library |
| **refreshed** | `finals2000A.all` (IERS; `EOP-All.csv` from CelesTrak as fallback); `Kp_ap_Ap_SN_F107_since_1932.txt` (GFZ), `45-day-forecast.txt` (SWPC), `msafe-f10-prd.txt` (NASA MSFC) | fetched on first use; `update_datafiles()` refreshes them, rate-limited (below) |

`tools/embed_data.py` regenerates the blobs from a data directory whose files
match `manifest.json` (it checks the source hashes) and records provenance in
`data/embedded/SOURCES.json`: full-file SHA-256, truncation degree, and the
SHA-256 of the inflated bytes. `tools/embed_data.py --check` verifies the
committed blobs; `MANIFEST.in` ships them in the sdist (`include_bytes!`
inputs must be in the sdist — the same lesson as `manifest.json`).

### Precedence

A file found in a *search directory* always wins over the embedded copy, so a
full-degree gravity file or an updated IERS table can be dropped in without a
rebuild; the embedded copy is the silent fallback (a file that exists but
cannot be parsed still produces a warning, silenced with `SATKIT_QUIET=1`).
Because the evaluator uses at most degree 40, the truncated files give
bit-identical results to the full ones.

### Search vs. write

Previously one directory was both searched and written, chosen as "the first
candidate containing `tab5.2a.txt`", with a third pass that *created* a
directory wherever it could — including `site-packages/satkit/satkit-data`
in a venv (wiped on reinstall) — and no Windows location at all (`HOME` is
unset there). Phase 2 separates the two questions
(`src/utils/datadir.rs`; `resolve()` is a pure function of the environment
with per-platform unit tests):

- **Search** (`utils::data_search_dirs()`, in order): `SATKIT_DATA`;
  `set_datadir()`; directories added with `add_search_dir()`; `<dylib dir>/satkit-data`;
  `<site-packages>/satkit_data/data` (the optional bundle); the platform user
  data dir; `~/.satkit-data` (legacy); `/usr/share/satkit-data`; macOS
  `/Library/Application Support/satkit-data`. Any of these may be read-only.
- **Write** (`utils::datadir()`, exactly one): `SATKIT_DATA` if set, else the
  `set_datadir()` directory, else the platform user data dir — macOS
  `~/Library/Application Support/satkit-data`, Linux `$XDG_DATA_HOME/satkit-data`
  (default `~/.local/share/satkit-data`), Windows `%LOCALAPPDATA%\satkit-data`.
  Never next to the shared library, never inside `site-packages`.

The sentinel file changed with the tiers: the marker of a provisioned
directory is now an ephemeris (`linux_p*.4XX` / `lnxp*.4XX`) — the one file
that is neither compiled in nor refreshed — and `data_found()` /
`utils.datafiles_exist()` report that.

### Offline mode

`SATKIT_OFFLINE=1`, or `utils::set_offline(true)` (Python
`satkit.utils.set_offline(True)`; the setter wins once called), forbids
**downloads only**: `update_datafiles()`, the lazy ephemeris fetch, the
EOP/SW refresh, and any non-embedded file. Search locations and the
embedded data are unaffected. The error is the typed
`download::Error::Offline { name, reason, urls }` — the same one a build
without the `download` feature returns — and no connection is opened.
CI runs the `offline_*` tests and `python/test/test_offline.py` with an
empty `SATKIT_DATA` and `SATKIT_OFFLINE=1` to keep this true.

### Python packaging

`satkit` no longer depends on `satkit-data`; `pip install satkit[data]`
installs it as an optional offline bundle, which the search order picks up
automatically (read-only, wherever it is installed — `satkit/__init__.py`
registers it with `add_search_dir`). This removes the 105 MB hard dependency
that sat 133 KB under PyPI's file-size cap and made every release re-upload
the ephemeris to refresh 5 MB of stale EOP/SW.

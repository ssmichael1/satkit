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
  cache automatically (the old static keys never did). The conda recipe's
  `source:` URLs + sha256 can be generated from the same file.

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
| `lnxp1900p2053.421` | 14.0 MB | JPL | DE421 (Folkner et al. 2009). US Government work, public domain. `default: false` — fetched only by name (e.g. the conda package ships this one) | ephemeris |
| `tab5.2a.txt`, `tab5.2b.txt`, `tab5.2d.txt` | 171 / 137 / 9 KB | IERS | IERS Conventions (2010), TN 36, Tables 5.2a/b/d. Freely redistributable. Origin URLs verified byte-identical. `default: false` — embedded byte-identical in the binary | core |
| `EGM96.gfc` | 5.6 MB | ICGEM (GFZ) | EGM96, Lemoine et al. 1998, NASA GSFC/NIMA — US Government work. `default: false` — embedded to degree 70 | core |
| `JGM2.gfc`, `JGM3.gfc` | 118 / 215 KB | ICGEM (GFZ) | JGM-2 (Nerem et al. 1994), JGM-3 (Tapley et al. 1996), NASA GSFC / UT CSR — US Government work. `default: false` — embedded to degree 70 | core |
| `ITU_GRACE16.gfc` | 1.8 MB | ICGEM (GFZ) | Akyilmaz et al. 2016, GFZ Data Services, **CC BY 4.0** — keep the file's header block, it carries the attribution. `default: false` — embedded to degree 70 | core |

`tier` is informational: `core` = the small files frames and gravity need
(embedded in the binary since Phase 2, below — which is why their manifest
entries are `default: false`: pinned and fetchable by name, but pointless to
download while evaluation is capped at degree 40), `ephemeris` = the large
JPL files. The only `default: true` entry — the only file
`update_datafiles()` downloads besides the daily refreshes — is the DE440
ephemeris.

## Files deliberately excluded

- **`msis21.parm`** — the NRLMSIS 2.1 parameter file. NRL's licence is
  academic / non-commercial and covers derived data products, which is
  incompatible with satkit's MIT / Apache-2.0 distribution. Nothing on `main`
  reads it (the NRLMSIS 2 port is on an unmerged branch); if that feature
  ships it must be an opt-in download with its own notice, not part of the
  default bundle.
- **`EOP-All.csv`, `SW-All.csv`** — Earth orientation and space weather.
  CelesTrak asks that its compiled files not be mirrored, and they change
  daily, so they are never pinned: `update_datafiles()` fetches them from
  CelesTrak every run via the manifest's `refresh` list.
- **`sw19571001.txt`** — an orphan on the old bucket; nothing reads it.
- **`leap-seconds.list`** — nothing reads it: the runtime leap-second table
  is a compiled-in constant (`src/time/instant.rs`). It was pinned
  (`tier: reference`) through 0.21.2 and remains an asset on the immutable
  `data-v1` tag, but the manifest entry is gone.
- **`predicted-solar-cycle.json`** — fetched directly from NOAA/SWPC by
  `solar_cycle_forecast::update()`; not a bundle file.

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
- **conda recipe** (`recipes/conda/satkit-data/recipe.yaml`): its `source:`
  list should become the manifest's release-asset (or origin) URLs with the
  manifest's sha256 values — no GCS. Not done in this branch because the
  recipe is being reworked separately.

## Client behaviour

- `satkit.utils.update_datafiles()` / `utils::update_datafiles` — fetches all
  `default: true` files (in parallel, verified), then the `refresh` files,
  then the NOAA solar-cycle forecast. `overwrite=True` re-downloads even
  verified files.
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
| **embedded** | `tab5.2a/b/d.txt`; `EGM96/ITU_GRACE16/JGM2/JGM3.gfc` truncated to degree 70 | gzip'd into `data/embedded/*.gz` (309 KB total) and compiled in with `include_bytes!` (`src/utils/embedded.rs`), inflated on first use. Frames and gravity need **no data directory and no network** |
| **ephemeris** | `linux_p1550p2650.440` (DE440, 102 MB) or `lnxp1900p2053.421` (DE421, 14 MB) | downloaded on first use through the verified manifest fetch, into the write location |
| **refreshed** | `EOP-All.csv`, `SW-All.csv` | fetched from CelesTrak on first use; `update_datafiles()` refreshes them |

`tools/embed_data.py` regenerates the blobs from a data directory whose files
match `manifest.json` (it checks the source hashes) and records provenance in
`data/embedded/SOURCES.json`: full-file SHA-256, truncation degree, and the
SHA-256 of the inflated bytes. `tools/embed_data.py --check` verifies the
committed blobs; `MANIFEST.in` ships them in the sdist (`include_bytes!`
inputs must be in the sdist — the same lesson as `manifest.json`).

### Precedence

A file found in a *search directory* always wins over the embedded copy, so a
full-degree gravity file or an updated IERS table can be dropped in without a
rebuild; the embedded copy is the fallback, and a one-time note says when it
is used (silence with `SATKIT_QUIET=1`). Because the evaluator uses at most
degree 40, the truncated files give bit-identical results to the full ones.

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

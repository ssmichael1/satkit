# Data Files

`satkit` needs three kinds of data, and handles them differently by size and by how often they change:

| tier | files | how it is provided |
|---|---|---|
| **Compiled in** | IERS Conventions (2010) Tables 5.2a/b/d (nutation and CIO series); EGM96, JGM-2, JGM-3 and ITU_GRACE16 gravity coefficients to degree 70 | gzip'd into the library (~300 KB) and inflated on first use. Frame transforms and gravity work with **no data directory and no network** |
| **Downloaded once, on first use** | JPL DE440 ephemeris `linux_p1550p2650.440` (102 MB), or DE421 `lnxp1900p2053.421` (14 MB) | fetched the first time a planet, Sun or Moon position is needed, SHA-256 verified against a manifest compiled into satkit, written to the [data directory](#where-satkit-looks-for-data-and-where-it-writes) |
| **Refreshed** | `EOP-All.csv` (Earth orientation), `SW-All.csv` (space weather) | change daily; fetched from CelesTrak on first use and refreshed by `satkit.utils.update_datafiles()` |

Everything that does not need the ephemeris or Earth orientation — gravity accelerations, the precession-nutation part of the frame chain, SGP4, time scales, Keplerian propagation, Lambert targeting — therefore works immediately after `pip install satkit`, offline. The numerical propagator needs the ephemeris (Sun and Moon) and the Earth-fixed frame chain needs the EOP file.

## The files

- **linux_p1550p2650.440** — File containing the precise ephemerides of the planets and 400 large asteroids between the years 1550 and 2650, as modelled by the Jet Propulsion Laboratory (JPL) — the DE440 ephemeris of [Park et al. (2021)](../guide/references.md#park2021). Large (~100 MB); downloaded on first use. The smaller `lnxp1900p2053.421` (DE421, [Folkner et al. 2009](../guide/references.md#folkner2009), ~14 MB, 1900–2053) is an alternative — see [Selecting a JPL ephemeris file](#selecting-a-jpl-ephemeris-file).

- **tab5.2a.txt**, **tab5.2b.txt**, **tab5.2d.txt** — Tables 5.2a, 5.2b and 5.2d of the IERS Conventions (2010), Technical Note 36 ([Petit & Luzum 2010](../guide/references.md#petit2010)): the CIP $X$, $Y$ and CIO-locator $s$ series used in the precise rotation between the inertial International Celestial Reference Frame and the Earth-fixed International Terrestrial Reference Frame. Compiled in.

- **EGM96.gfc**, **JGM2.gfc**, **JGM3.gfc**, **ITU_GRACE16.gfc** — Gravity coefficients for EGM96 ([Lemoine et al. 1998](../guide/references.md#lemoine1998)), JGM-2 ([Nerem et al. 1994](../guide/references.md#nerem1994)), JGM-3 ([Tapley et al. 1996](../guide/references.md#tapley1996)) and ITU_GRACE16 ([Akyilmaz et al. 2016](../guide/references.md#akyilmaz2016)), in the ICGEM `.gfc` format ([Ince et al. 2019](../guide/references.md#ince2019)). Compiled in, truncated to degree 70 (the evaluator uses at most degree 40, so results are identical to the full files). A full-degree copy placed in a data directory is used in preference.

- **SW-All.csv** — Space Weather. The solar flux at $\lambda = 10.7\text{cm}$ (2800 MHz) is an indication of solar activity, which in turn is an important predictor of air density at altitudes relevant for low-Earth orbits. This file is updated at [celestrak.org](https://www.celestrak.org) ([CelesTrak Space Data](../guide/references.md#celestrak-spacedata)) every 3 hours with the most-recent space weather information.

- **predicted-solar-cycle.json** — [NOAA/SWPC solar cycle forecast](https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json). Monthly predicted F10.7 solar flux values extending ~5 years into the future. Used as a fallback for atmospheric density calculations when propagating beyond the range of historical space weather data.

- **EOP-All.csv** — Earth orientation parameters. This includes $\Delta UT1$, the difference between $UT1$ and $UTC$, as well as $x_p$ and $y_p$, the polar "wander" of the Earth rotation axis. This file is updated daily with most-recent values at [celestrak.org](https://www.celestrak.org) (which repackages the IERS Bulletin A / finals series) and carries IERS predictions roughly six months ahead. For dates beyond the file, the last entry's values are used (constant extrapolation) — see [EOP coverage](#eop-coverage) below.

- **leap-seconds.list** — Downloaded by `update_datafiles()` for reference only. The UTC↔TAI leap-second table that `satkit` actually uses is compiled into the library (current through the most recent leap second, 2017-01-01, when UTC began lagging TAI by 37 s); this file is not read at runtime, and a future leap second will require a new `satkit` release. The table is transcribed from [IERS Bulletin C](../guide/references.md#bulletinc); UTC and leap seconds are defined by [ITU-R TF.460-6](../guide/references.md#itu460).

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

### Environment variables and API

| control | effect |
|---|---|
| `SATKIT_DATA=/path` | search first and write here (created if needed) |
| `SATKIT_DATA_URL=https://mirror/base` | try `"$SATKIT_DATA_URL/<name>"` before the manifest's sources for every download (plain `http://` accepted; still hash-verified) |
| `SATKIT_OFFLINE=1` / `satkit.utils.set_offline(True)` | forbid **downloads** — `update_datafiles()`, the lazy ephemeris fetch, the EOP/SW refresh, any non-embedded file — with a `RuntimeError` naming the file and its sources; no connection is opened. Search locations and the compiled-in data are unaffected. The setter wins once called; otherwise the variable is read. `satkit.utils.is_offline()` reports the effective state |
| `SATKIT_JPLEPHEM_FILE=name-or-path` | which ephemeris to load — see [below](#selecting-a-jpl-ephemeris-file) |
| `SATKIT_QUIET=1` | suppress the one-time note printed when a compiled-in file is used |
| `satkit.utils.datadir()` | the write location (`None` if none can be determined — no `SATKIT_DATA`, no home / `%LOCALAPPDATA%`) |
| `satkit.utils.data_search_dirs()` | the search list, in order |
| `satkit.utils.set_datadir(path)` / `add_search_dir(path)` | add an override / a read-only search location |
| `satkit.utils.datafiles_exist()` | whether an ephemeris file is present in any search directory (the marker of a provisioned data location) |

### Where the files come from, and how downloads are verified

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
EGM96, JGM-2, JGM-3 — NASA GSFC (public), via ICGEM; ITU_GRACE16 — Akyilmaz et
al. 2016, GFZ Data Services, CC BY 4.0; `leap-seconds.list` — IERS/IETF. The
Earth-orientation and space-weather files are fetched from CelesTrak on every
update and are not pinned (they change daily). The full table, with licences,
is in `data/README.md`.

## Provisioning up front

Nothing needs to be downloaded before first use, but for a container image,
a CI job, or a machine that will later be offline:

```python
import satkit as sk
sk.utils.update_datafiles()   # ephemeris + full-degree gravity files + IERS tables + EOP/SW, verified
```

Files already present with the right hash are skipped; the space-weather and
Earth-orientation files are always refreshed. `update_datafiles(dir="...")`
writes somewhere else; `overwrite=True` re-downloads even verified files.

### The optional `satkit-data` bundle

`pip install satkit[data]` installs the `satkit-data` package (~110 MB: the
ephemeris, full-degree gravity files, IERS tables) into `site-packages`. It is
picked up automatically as a read-only search location (rows 3 and 5 above),
so no first-use download happens. It is not required — earlier releases made
it a hard dependency of `satkit`; it is now optional.

## EOP coverage

Every Earth-fixed frame transform, every UT1-based quantity (`gmst`, `gast`, Earth rotation angle), and the high-precision propagator depend on the EOP table, so it matters where an epoch falls relative to it:

| `satkit.frametransform.eop_status(t)` | meaning | what satkit does |
|---|---|---|
| `"observed"` | on or before the last observed (`O`) row | interpolates measured values |
| `"predicted"` | after the last observed row, inside the table | interpolates IERS predictions (~6 months ahead) |
| `"extrapolated"` | after the last row | holds the last row constant and prints a **one-time warning**. Polar motion drifts ~0.1″ and $\Delta UT1$ ~10 ms over a few months — metres of position error at LEO |
| `"before_table"` | before 1962 | zeros, one-time warning |
| `"not_loaded"` | no table at all (first use offline, or the fetch failed) | zeros, one-time warning; **`propagate` refuses to run** (`RuntimeError`) |

`satkit.frametransform.eop_coverage()` returns `(first, last_observed, last)` as `satkit.time` values, or `None` if nothing is loaded. For precision work, propagate with `satkit.propsettings(require_eop_coverage=True)`: the propagator then raises instead of extrapolating past the table, and the fix is simply to refresh the file:

```python
import satkit as sk

first, last_observed, last = sk.frametransform.eop_coverage()
if sk.frametransform.eop_status(t_end) == "extrapolated":
    sk.utils.update_datafiles()   # re-downloads EOP-All.csv (and SW-All.csv)
```

The warnings can be silenced with `satkit.frametransform.disable_eop_time_warning()`.

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

Both DE440 and DE421 are in the manifest and can be downloaded by name; any other file must already exist.

### Autodetect

With no environment variable set, every search directory is scanned for JPL Linux-binary ephemeris files (`linux_p*.4XX`, `lnxp*.4XX`) and the highest DE version found is used, so dropping a file into the data directory is enough to switch to it.

# Data Files

The `satkit` package relies upon a number of data files for certain calculations:

- **leap-seconds.list** — A list of the UTC leap seconds since 1972. This is a common file on \*nix platforms and is used to keep track of the number of seconds (currently 37) that UTC lags TAI.

- **linux_p1550p2650.440** — File containing the precise ephemerides of the planets and 400 large asteroids between the years 1550 and 2650, as modelled by the Jet Propulsion Laboratory (JPL). Note: this file is large (~100 MB) and may take a long time to download. Smaller alternatives (e.g. `lnxp1900p2053.421` from JPL's DE421 release at ~13 MB, 1900–2053) work as well — see [Selecting a JPL ephemeris file](#selecting-a-jpl-ephemeris-file) below.

- **tab5.2a.txt**, **tab5.2b.txt**, **tab5.2d.txt** — Tables from IERS Conventions Technical Note 36, containing coefficients used in the precise rotation between the inertial International Celestial Reference Frame and the Earth-fixed International Terrestrial Reference Frame.

- **EGM96.gfc**, **JGM2.gfc**, **JGM3.gfc**, **ITU-GRACE16.gfc** — Files containing gravity coefficients for various gravity models. These are used to compute the precise acceleration due to Earth gravity as a function of position in the Earth-fixed ITRF frame.

- **SW-All.csv** — Space Weather. The solar flux at $\lambda = 10.7\text{cm}$ (2800 MHz) is an indication of solar activity, which in turn is an important predictor of air density at altitudes relevant for low-Earth orbits. This file is updated at [celestrak.org](https://www.celestrak.org) every 3 hours with the most-recent space weather information.

- **predicted-solar-cycle.json** — NOAA/SWPC solar cycle forecast. Monthly predicted F10.7 solar flux values extending ~5 years into the future. Used as a fallback for atmospheric density calculations when propagating beyond the range of historical space weather data.

- **EOP-All.csv** — Earth orientation parameters. This includes $\Delta UT1$, the difference between $UT1$ and $UTC$, as well as $x_p$ and $y_p$, the polar "wander" of the Earth rotation axis. This file is updated daily with most-recent values at [celestrak.org](https://www.celestrak.org). For dates beyond the file, the last entry's values are used (constant extrapolation).

## Acquiring the Data Files

The data files are included with the `satkit-data` package, a dependency of `satkit`.

The data files can also be manually downloaded with the following command:

```python
satkit.utils.update_datafiles()
```

If the files already exist, they will *not* be downloaded, with the exception of the space weather and earth orientation parameters files, as these are regularly updated.

## Download Location

The data files are all downloaded into a common directory. This directory can be queried via Python:

```python
satkit.utils.datadir()
```

The `satkit` package will search for the data files in the following locations, in order, stopping when the files are found:

- Directory pointed to by the `SATKIT_DATA` environment variable
- `$DYLIB/satkit-data` where `$DYLIB` is the directory containing the compiled satkit library
- `$SITE_PACKAGES/satkit_data/data` where `$SITE_PACKAGES` is the parent of `$DYLIB` (for the `satkit_data` pip package)
- *macOS only*: `$HOME/Library/Application Support/satkit-data`
- `$HOME/.satkit-data`
- `/usr/share/satkit-data`
- *macOS only*: `/Library/Application Support/satkit-data`

If no files are found, the `satkit` package will go through the above list of directories in order, stopping when a directory either exists and is writable, or can be created and is writable. The files will then be downloaded to that location.

## Selecting a JPL ephemeris file

By default `satkit` loads `linux_p1550p2650.440` from `datadir()`. For larger time spans, smaller binaries, or older DE releases there are two ways to override that choice.

### Environment variable

Set `SATKIT_JPLEPHEM_FILE` to either an absolute path or a basename:

```bash
# Absolute path — file used directly
SATKIT_JPLEPHEM_FILE=/opt/jpl/lnxp1900p2053.421 python script.py

# Basename — resolved under datadir()
SATKIT_JPLEPHEM_FILE=lnxp1900p2053.421 python script.py
```

### Autodetect

When `SATKIT_JPLEPHEM_FILE` is unset, `satkit` scans `datadir()` for any JPL ephemeris binary matching either of JPL's Linux naming conventions:

- `linux_p<start>p<stop>.4XX` — used for DE430 and later (DE430 / DE440 / DE441)
- `lnxp<start>p<stop>.4XX` — used for DE421 and earlier

If multiple files are present, the one with the highest DE-version suffix wins. So dropping `lnxp1900p2053.421` into `datadir()` is enough to make satkit prefer it — no env var needed unless you also have a DE440 file alongside.

The header-driven parser handles DE405, DE421, DE430, DE440, and DE441 through the same code path; choose whichever balances size and span you need. Queries outside the loaded file's coverage window return `Error::InvalidJulianDate(jd)` rather than silently extrapolating.

## Embedded / non-filesystem use

When the data files don't live on disk — for example, they're bundled as a Python package resource accessed via `importlib.resources`, stored as blobs in a SQLite database, or fetched at startup from a configuration service — every subsystem accepts an in-memory byte buffer or a specific file path through a uniform Rust API:

| Subsystem | Init function (bytes) | Init function (path) | Re-init? |
|---|---|---|---|
| `jplephem` | `init_from_bytes(&[u8])` | `init_from_path(&Path)` | once only |
| `earthgravity` | `init_from_bytes(GravityModel, &[u8])` | `init_from_path(GravityModel, &Path)` | once only |
| `frametransform::ierstable` | `init_from_bytes(IersTableId, &[u8])` | `init_from_path(IersTableId, &Path)` | once only |
| `spaceweather` | `init_from_bytes(&[u8])` | `init_from_path(&Path)` | always replaces |
| `solar_cycle_forecast` | `init_from_bytes(&[u8])` | `init_from_path(&Path)` | always replaces |
| `earth_orientation_params` | `init_from_bytes(&[u8])` | `init_from_path(&Path)` | always replaces |

The static subsystems (top three) hold mathematical-constant data — once initialized they cannot be replaced, and a second `init_from_*` call returns `Err(AlreadyInitialized)`. Initialization must happen before any position query, gravity acceleration, or frame transform that depends on the subsystem, otherwise the lazy default-load path wins and the init call is too late.

The refreshable subsystems (bottom three) hold operational data that updates daily (EOP, space weather) or monthly (solar-cycle forecast). `init_from_*` always succeeds and replaces the current contents — appropriate for long-running services that pull fresh records periodically.

```rust
// Static subsystem: init from bytes before any caller queries positions
let bytes: Vec<u8> = db.fetch_blob("de440")?;
satkit::jplephem::init_from_bytes(&bytes)?;
let pos = satkit::jplephem::geocentric_pos(SolarSystem::Moon, &t)?;

// Refreshable subsystem: replace on every poll cycle
loop {
    let csv = db.fetch_latest("eop")?;
    satkit::earth_orientation_params::init_from_bytes(&csv)?;
    // ... do work ...
}
```

These APIs are Rust-only. Python callers should use `satkit.utils.update_datafiles()` and the filesystem-based default; the embedded scenario above doesn't generalise cleanly to a notebook workflow.

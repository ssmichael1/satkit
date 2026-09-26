# Migrating to 0.24

0.24 fixes a set of long-standing defects, and some of the fixes change
results, exception types or return shapes. This page lists what to check when
you upgrade from 0.23. Each item says what changed and what to do. The
[CHANGELOG](https://github.com/ssmichael1/satkit/blob/main/CHANGELOG.md) links
each change to its pull request, where the details are.

## Check old results first

These changes move numbers. Each line says what was wrong or changed, which
releases it affects, and whose results to re-check; the linked section has
the details.

**Wrong in earlier releases:**

- **Quaternion × N×3 array** applied the *inverse* rotation (a single
  3-vector was correct). 0.14.1–0.23.1. Re-check anything that rotated a
  stacked N×3 array with one quaternion.
- **`satproperties()` positional arguments** were read as
  `(craoverm, cdaoverm)`, swapping drag and radiation pressure. Through
  0.23.1. Re-check scripts that called `satproperties(a, b)`; it is now
  [keyword-only](#python-api).
- **`lambert`** was wrong for long-way (over 180°) and retrograde transfers,
  which missed `r2` by about 2r, and for hyperbolic and near-parabolic ones
  (time of flight miscomputed). 0.14.1–0.23.1. Re-run Lambert targeting,
  delta-v budgets and pork-chop plots.
- **`kepler.from_pv`** at inclination exactly π (retrograde equatorial)
  returned elements of a different orbit. Through 0.23.1. Recompute elements
  of such orbits.
- **SGP4 reused a stale initialization** after a TLE's elements were edited,
  or when `gravconst` / `opsmode` changed between calls on the same TLE
  (thousands of km after an edit, ~120 m at LEO after 3 days with another
  gravity model). Through 0.23.1. Recompute results from such TLEs.
- **Time strings with UTC offsets**: `from_rfc3339` ignored `±HHMM` and `±HH`
  offsets, `strptime` applied `%z` with the wrong sign, and `from_string` read
  a numeric offset as microseconds. Through 0.23.1. Re-check times parsed
  from strings that carry offsets ([Time](#time)).
- **Leap-second edges**: 00:00:00 UTC right after a leap second (for example
  `time(2017, 1, 1)`) was built 1 s early, and UT1 was up to 0.5 s off on
  the day before a leap second (~250 m in Earth-fixed positions at LEO).
  Through 0.23.1. Re-check epochs and Earth-fixed results on those days.
- **TDB** was off by up to 3.3 ms (TDB − TT had a ~57-year period instead of
  one year), and the JPL ephemerides were evaluated at TT instead of TDB
  (Moon ~2 m, geocentric Sun up to ~50 m). Through 0.23.1. Regenerate stored
  TDB values and JPL-based reference results.
- **`moon.pos_gcrf`** returned mean-of-date coordinates (0.7° off in 1950
  and 2050), and **`planets.heliocentric_pos`** used the obliquity of date
  and, outside 1800–2050, mis-scaled the Jupiter–Pluto terms (tens of
  degrees). All releases through 0.23.1. Recompute stored low-precision Moon
  and planet positions ([Ephemerides and frames](#ephemerides-and-frames)).
- **`sun.rise_set`** returned the next day's events for about half of the
  inputs. Through 0.23.1. Re-check rise/set tables.
- **Density on the day after a missing F10.7** ran on the quiet-time
  defaults (35–39 % low), and `density.nrlmsise()` ignored a `datetime` time
  (up to 8× low in a storm) and read an integer latitude or longitude as 0.
  Through 0.23.1. Re-check drag propagations over those days and direct
  density calls ([Drag and density](#drag-and-density)).
- **Custom ICGEM gravity files with `norm unnormalized`** (Rust
  `Gravity::parse`) were de-normalized twice. Through 0.23.1. Re-check
  results from such files ([Rust API](#rust-api)).

**Changed by design:**

- **Rust `sgp4()` uses WGS72 / AFSPC** (was WGS84 / IMPROVED), and
  `TLE.fit_from_states()` fits under WGS72 (was WGS84). Through 0.23.1.
  Rust SGP4 users and anyone fitting TLEs ([SGP4 and TLEs](#sgp4-and-tles)).
- **Drag applies up to 1,000 km** altitude (was 700 km). All releases through
  0.23.1. Orbits that reach 700–1,000 km: high LEO, and GTO or Molniya
  perigee passes ([Drag and density](#drag-and-density)).
- **Pre-1972 UTC** follows the 1961–1971 "rubber second" model, so pre-1972
  labels convert up to 9.9 s differently. Through 0.23.1. Anyone with
  pre-1972 epochs ([Time](#time)).
- **No EOP before 1973-01-02** (CelesTrak's file, which covered 1962–1972, is
  no longer read), so UT1 = UTC there: hundreds of metres in Earth-fixed
  results. Through 0.23.1 when `EOP-All.csv` was present. Anyone with
  pre-1973 Earth-fixed results ([EOP](#earth-orientation-eop-and-data-files)).
- **ECOM coefficients (experimental)** are referred to 1 AU and scale by
  $(\text{AU}/d)^2$. Through 0.23.1. Anyone with fitted ECOM coefficients
  ([ECOM](#ecom-experimental)).
- **Smaller shifts**: a more precise Earth rotation angle (1 cm LEO, 6 cm
  GEO), no stray polar motion in the approximate TEME → GCRF rotation
  (~19 m LEO), and float → microsecond conversions that round instead of
  truncating (1 µs). Through 0.23.1. Tests that compare against 0.23 values
  at those tolerances.

## Python API

- **Python 3.11 or newer is required.** Python 3.10 reaches end of life in
  October 2026; 0.24 ships wheels for CPython 3.11–3.15. **Do this:** stay on
  satkit 0.23.x for Python 3.10.
- **`satproperties()` is keyword-only.** A positional call raises
  `TypeError`. **Do this:** write
  `sk.satproperties(cdaoverm=..., craoverm=..., thrusts=..., ecom=...)`.
- **Unknown keywords raise `TypeError`, not `ValueError`**, in `duration()`,
  `propsettings()`, `satproperties()`, `itrfcoord()`, `sgp4()`,
  `propagate()`, `satstate.propagate()`, `gravity()`,
  `gravity_and_partials()` and `nrlmsise00()`; `gravity()` and
  `gravity_and_partials()` used to ignore them. `utils.update_datafiles()`
  also rejects them now; it used to ignore them too. **Do this:** fix misspelt
  keywords, and catch `TypeError` where you caught `ValueError`.
- **A one-element list or array of times gives a one-element result**, not a
  scalar, in every vectorised function: `sgp4(tle, [t])` is `(1, 3)` (was
  `(3,)`). **Do this:** pass a scalar time, or index `[0]`.
- **A one-element list of TLEs keeps its axis in `sgp4`**: `sgp4([tle], t)` is
  `(1, 3)` (was `(3,)`), and `sgp4([tle], [t])` is `(1, 1, 3)`. Empty lists
  give empty arrays: `sgp4([], t)` is `(0, 3)` and `sgp4([tle], [])` is
  `(1, 0, 3)` (both raised `RuntimeError`). **Do this:** pass the TLE itself,
  or use `[0]` / `.squeeze()`.
- **`sgp4(..., errflag=True)` no longer raises for an element set SGP4
  cannot initialize** (decayed or eccentricity out of range at epoch,
  SGP4-XP): its rows are NaN and every time carries its init code
  (`sgp4_error.unsupported` for SGP4-XP or unusable OMM metadata), and the
  other element sets of a list are still propagated. Without `errflag` it
  still raises; for a list the message gives the index. **Do this:** check
  the error array instead of catching `RuntimeError`.
- **`TLE.from_lines()`, `TLE.from_file()` and `TLE.from_url()` always return
  `list[TLE]`**, even for a single element set. Input with no TLEs still
  raises `ValueError`. **Do this:** `tle = sk.TLE.from_lines(lines)[0]`. See
  [Loading TLEs](guide/tle.md#loading-tles) for the located parse errors and
  the new `check_checksum=True` option.
- **`TLE.from_url()` and `omm_from_url()` honour offline mode**
  (`SATKIT_OFFLINE=1` or `utils.set_offline(True)`) and raise `RuntimeError`.
  **Do this:** turn offline mode off in scripts that fetch element sets.
- **`TLE.fit_from_states()` takes keyword-only `gravconst=` and `opsmode=`**
  (default WGS72 / AFSPC); see [SGP4 and TLEs](#sgp4-and-tles).
- **`utils.build_date()` is removed** (builds are reproducible now).
  **Do this:** use `satkit.__version__`, or `utils.githash()`, which is
  `"unknown"` for builds not made from a satkit git checkout.
- **`utils.version()` returns the release version** (the same as
  `satkit.__version__`). It returned the `git describe` tag, which was
  `"unknown"` in the published wheels. **Do this:** use `utils.githash()` if
  you need the commit.
- **`frametransform.eop_source()` is deprecated** and is removed in 0.25. It
  warns, and returns `"finals2000A"` or `None`. **Do this:** use
  `frametransform.eop_coverage()`, which is `None` when no table is loaded.
- **The `as_*` conversions are removed in 0.25** (deprecated since 0.23):
  `time.as_date()`, `as_gregorian()`, `as_datetime()`, `as_mjd()`, `as_jd()`,
  `as_unixtime()`, `as_iso8601()`, `as_rfc3339()`, and
  `quaternion.as_rotation_matrix()`, `as_euler()`. **Do this:** rename them to
  `to_*`. Run with `python -W error::DeprecationWarning` to find them.

### Exception types

These calls raise a different exception than in 0.23.1, or now raise where
they used to return a meaningless value:

| call | 0.23.1 | 0.24 |
|---|---|---|
| an unknown keyword to the functions listed under [Python API](#python-api) | `ValueError`, or ignored by `gravity()`, `gravity_and_partials()` and `update_datafiles()` | `TypeError` |
| `satproperties(a, b)` (positional) | drag and SRP swapped | `TypeError` |
| `quaternion * 2.0` (or a `str`, `None`) | `RuntimeError` | `TypeError` |
| `quaternion * <array of the wrong shape>`, `quaternion.rotation_between(<wrong length>, ...)` | `RuntimeError` | `ValueError` |
| `duration + float`, `duration / str` | `RuntimeError` | `TypeError` |
| `duration / 0` | `RuntimeError` | `ZeroDivisionError` |
| `duration / 0.0`, `duration / duration(0)` | saturated duration or `inf` | `ZeroDivisionError` |
| `duration(days=nan)`, `duration.from_seconds(inf)`, ... | 0 or saturated | `ValueError` |
| `duration(days=1e300)`, `time + 1e300` | saturated or a garbage label | `OverflowError` |
| `duration(days=1e8, seconds=1e12)`, `(t + 1e8) + 1e8`, `duration + duration`: any result beyond about ±292,000 years | wrapped around (a negative duration, year −34949) | `OverflowError` |
| `time + nan`, `time - inf` (scalar, list or array) | `t` or a garbage label | `ValueError` |
| `time.strptime()`, `time.from_string()`, `time(<str>)` with an unparseable string | `RuntimeError` | `ValueError` |
| `propagate(state)` with no begin time | a result from an invalid epoch | `TypeError` |
| `propagate(state, begin)` with no end time | `RuntimeError` | `TypeError` |
| `propagate(state, begin, end, <4th positional>)` | 4th argument ignored | `TypeError` |
| `propagate(..., duration_secs=nan)` | ran to an invalid end | `ValueError` |
| `satstate.propagate(<not a time or duration>)` | `RuntimeError` | `TypeError` |
| `propsettings(gravity_degree=4, gravity_order=10)` | order clamped to 4 | `ValueError` (as the `gravity_order` setter) |
| `propresult.interp(..., output_phi=True)` without the STM | the state alone | `ValueError` |
| `gravity(<wrong length>)` / `gravity(<non-numeric>)` | `RuntimeError` | `ValueError` / `TypeError` |
| `density.nrlmsise(..., <not a time>)` (a `str`, a number, ...) | ignored: ran on the default indices | `TypeError` |
| `kepler.from_pv(<NaN or inf>)` | NaN elements | `ValueError` |
| `kepler.propagate(nan)`, `kepler.propagate(inf)` | unchanged or a garbage anomaly | `ValueError` |
| `lambert(r1, r1, tof)` | NaN velocities | `ValueError` |
| `lambert(...)` with a NaN or infinite input | "convergence failure" `ValueError` or NaN | `ValueError` naming the input |
| `TLE.epoch = [t1, t2]` | used `t1` | `TypeError` |
| `TLE.from_lines()` with a line 1 and line 2 of different satellites | a hybrid TLE | `RuntimeError` |
| `TLE.from_lines()` with a line 1 whose line 2 is missing | dropped silently | `RuntimeError` |
| `TLE.from_url()`, `omm_from_url()` in offline mode | fetched anyway | `RuntimeError` |
| `sgp4()` of an OMM whose `REF_FRAME` is not TEME or `CENTER_NAME` not EARTH | propagated | `RuntimeError` |
| `TLE.to_2line()` with `satnum >= 340000` (no Alpha-5 representation) | `RuntimeError` | `ValueError` |

Some calls that raised now work: `gravity([7e6, 0, 0])` and integer arrays,
`time - [t1, t2]` (an array of `duration`), `time + <integer or float32
array of days>`, `sgp4([], t)` and `sgp4([tle], [])` (empty arrays),
`sgp4(..., errflag=True)` for an element set that cannot be initialized
(NaN rows plus its code), and batch state transforms given integer arrays or
nested lists.

## Time

- **Time strings are parsed more strictly, and UTC offsets are applied.**
  - `time.from_rfc3339()` accepts `Z`, `±HH:MM`, `±HHMM` and `±HH`, and always
    applies the offset: `2024-01-01T12:00:00+0100` is now `11:00:00Z` (was
    `12:00:00Z`). A string with no zone is UTC.
  - `time.from_string()` applies a numeric offset after the time
    (`"2024-01-04 13:14:12 +0100"` was read as 100 µs). `HH:MM` without
    seconds keeps its time (it was dropped).
  - `time.strptime()` requires the whole string to match; `%z` requires `+`,
    `-`, `Z` or `z` and two-digit fields in range; `%m %d %H %M %S` take
    exactly two digits; `%%` is supported.
  - The `strptime` `%z` sign is fixed: `12:00:00+0100` is `11:00:00Z` (it was
    `13:00:00Z`).
  - Errors that used to be silent mis-parses: trailing input in
    `from_rfc3339` and `strptime`, a malformed or out-of-range offset, an
    empty fraction (`12:00:00.Z`), and an extra number or a lone hour in
    `from_string`. `from_string` still ignores trailing words, zone names
    included: `"2024-01-04 13:14:12 EST"` is 13:14:12 UTC.
  - **Do this:** check string inputs that carry offsets or extra text, remove
    any `%z` workaround, and catch the new errors. Every time-string parser
    (`from_rfc3339`, `strptime`, `from_string` and `time(<str>)`) raises
    `ValueError` with the reason; `strptime`, `from_string` and
    `time(<str>)` raised `RuntimeError` in 0.23.1.
- **Fractional seconds round to the nearest microsecond** instead of
  truncating, both in parsed strings with more than six digits and in float
  conversions (calendar seconds, Unix time, MJD/JD, GPS seconds of week,
  `add_utc_days`, `duration` from floats, `datetime`). Results can move by
  1 µs. **Do this:** update tests that compare exact microseconds.
- **Leap-second edges are fixed.** 00:00:00 UTC after a leap second is the
  end of the leap second: `time(2017, 1, 1)` was 1 s early, and so were
  `from_mjd`, `from_jd`, `from_unixtime` and `add_utc_days` at those
  midnights. `23:59:60.x` can be entered and round-trips. UT1 − UTC is no
  longer interpolated across the leap-second step on the preceding day
  (0.5 s off). **Do this:** recompute epochs built at those midnights.
- **UTC before 1972 follows the 1961–1971 "rubber second" model** of USNO
  `tai-utc.dat` / ERFA, and TAI − UTC is 0 before 1961. Pre-1972 labels
  convert up to 9.9 s differently. A stored pre-1972 instant (a pickle, say)
  keeps its instant but prints a label up to ~10 s different. **Do this:**
  re-derive pre-1972 epochs from their labels.
- **TDB − TT has its one-year period** (it had a ~57-year period and was off
  by up to 3.3 ms), and **the JPL ephemerides are evaluated at TDB** (was
  TT): the Moon moves by up to ~2 m, the geocentric Sun by up to ~50 m, and
  third-body propagations shift slightly. **Do this:** regenerate stored
  reference values.
- **Years outside 0000–9999 print in ISO 8601 expanded form** (`-0001`,
  `+10000`), and `from_rfc3339` / `strptime` read them back, as does
  `from_string` (`"-0044-03-15 12:00:00"` was AD 44).

## Earth orientation (EOP) and data files

- **EOP comes only from IERS `finals2000A.all`.** CelesTrak's `EOP-All.csv`
  is no longer downloaded or read. **Do this:** run
  `sk.utils.update_datafiles()` on any data directory that holds only
  `EOP-All.csv`, especially one copied to an offline machine. Otherwise frames
  run with no EOP and `propagate()` refuses to start.
- **There is no EOP before 1973-01-02.** Before that date UT1 = UTC and polar
  motion is zero, with a one-time warning (CelesTrak's file covered
  1962–1972). Frame transforms and propagations at those epochs change by up
  to ~0.9 s of Earth rotation, i.e. hundreds of metres. With
  `propsettings(require_eop_coverage=True)`, `propagate` now raises for a span
  that starts before the table as well as one that ends after it. **Do this:**
  treat pre-1973 Earth-fixed results as approximate; there is no replacement
  source.
- **The optional `satkit-data` PyPI bundle is no longer used.** The
  `satkit[data]` extra is gone, and the `satkit_data` package is no longer a
  search directory (its 0.9.0 release shipped only `EOP-All.csv`, which
  satkit no longer reads). **Do this:** `pip uninstall satkit-data`, and
  provision data with `sk.utils.update_datafiles()` plus `SATKIT_DATA` or
  `sk.utils.add_search_dir()`, as in
  [Provisioning up front](getting-started/datadirs.md#provisioning-up-front).
- **A truncated `finals2000A.all` is an error.** A line cut short (an
  interrupted copy) fails to load instead of ending the table early, and a
  download without predictions is rejected, keeping the file on disk.
  **Do this:** re-copy or refresh a file that now fails to load.
- **Stale predictions warn.** A lookup past the last observed row of a file
  whose observed data ended more than 30 days ago prints a one-time warning.
  **Do this:** refresh with `update_datafiles()`.

## Pickles

- **`TLE`, `satstate` and `satproperties` pickles now store integer
  microseconds.** Pickles from older releases still load, but **pickles
  written by 0.24 cannot be loaded by 0.23 or earlier.** **Do this:** don't
  share new pickles with older installs.

## Other result changes

### Ephemerides and frames

- **`moon.pos_gcrf` now returns GCRF coordinates.** Through 0.23 it returned
  mean-of-date coordinates, which differ by precession: about 0.7° in 1950
  and 2050 and 1.4° by 2100. The mean-of-date position is still available
  as `moon.pos_mod` (Rust `moon::pos_mod`). **Do this:** recompute stored
  low-precision Moon positions; code that rotated the result from
  mean-of-date to GCRF itself must stop doing so.
- **`planets.heliocentric_pos` is more accurate.** It rotates the J2000
  ecliptic elements with the fixed J2000 obliquity (it used the obliquity of
  date, ~47″ per century from 2000), and outside 1800–2050 the extra terms
  for Jupiter through Pluto are in the right units (these planets were off by
  up to tens of degrees). **Do this:** recompute stored planet positions.
- **`sun.rise_set` returns the sunrise and sunset of the input's UTC
  calendar date**, whatever its time of day. Through 0.23 about half of the
  inputs gave the next day's events. **Do this:** for a local date, pass a
  timezone-aware `datetime` at local noon.
- **The Earth rotation angle is more precise** (1 cm at LEO, 6 cm at GEO in
  ITRF ↔ GCRF), and the approximate TEME → GCRF rotation (`qteme2gcrf`,
  `rotation_approx`) no longer includes a stray 0.3–0.6″ polar-motion
  rotation (~19 m at LEO). **Do this:** expect small differences from 0.23
  results.

### SGP4 and TLEs

- **Rust `sgp4::sgp4()` uses WGS72 and the AFSPC ops mode** (was WGS84 and
  IMPROVED), the Python default; the ISS moves 14 m at epoch and ~280 m after
  a week. **Do this:** call `sgp4_full(..., GravConst::WGS84,
  OpsMode::IMPROVED)` to keep the old numbers.
- **`TLE.fit_from_states()` fits with WGS72** (was WGS84) and the full TEME →
  GCRF rotation. **Do this:** propagate fitted TLEs with the default
  `sgp4()`, or pass `gravconst=sgp4_gravconst.wgs84` (Rust:
  `fit_from_states_full`) to both.
- **Editing a TLE or changing `gravconst` / `opsmode` takes effect.** SGP4
  re-initializes whenever the elements or the settings differ from those of
  its cached initialization; through 0.23 it kept the first one.
  `TLE.to_2line()` wraps an element set number above 9999 and a revolution
  number above 99999, as catalog TLEs do.

### Drag and density

- **Drag is applied up to 1,000 km altitude** (above the equatorial
  radius), the upper limit of NRLMSISE-00; through 0.23 it stopped at 700 km.
  Orbits that reach 700–1,000 km (high LEO, the perigee passes of GTO and
  Molniya orbits) now feel drag there too. **Do this:** expect different
  results for those orbits.
- **Density on the day after a missing F10.7 is no longer on the quiet-time
  defaults.** The observed record has a few days without a measured flux
  (e.g. 2025-02-12, 2025-02-17, 2026-05-09); on the following day 0.23
  dropped the whole space-weather input (F10.7 = F10.7A = 150, Ap = 4, up to
  39 % low in density). The flux now comes from the nearest earlier measured
  day and the measured Ap is kept. Before 1947, when there is Ap but no F10.7,
  the measured Ap is now used too.
- **`density.nrlmsise()` uses a `datetime.datetime` time and integer
  angles.** Through 0.23 a `datetime` was silently ignored (defaults, up to
  8× low in a storm) and an `int` latitude or longitude was read as 0.

### ECOM (experimental)

- **ECOM coefficients are referred to 1 AU** and scale by
  $(\text{AU}/d)^2$ with the satellite–Sun distance $d$. **Do this:** multiply
  coefficients fitted with the old scaling by $(d/\text{AU})^2$ at their
  epoch.

## Rust API

- **`earth_orientation_params::EopSource`, `source()` and `CELESTRAK_FILE`
  are removed.** In `earth_orientation_params::Error`, `InvalidEntry` and
  `ParseFloat` are gone, and `UnsupportedCsv` is new (an `EOP-All.csv` passed
  to `init_from_path` / `init_from_bytes`). **Do this:** delete source checks
  and update exhaustive matches.
- **`DataDirReadOnly` is now `DataDirReadOnly { path, reason }`** in
  `earth_orientation_params::Error`, `spaceweather::Error` and
  `utils::update_data::Error`. These enums are not `#[non_exhaustive]`.
  **Do this:** match `Error::DataDirReadOnly { .. }`.
- **`orbitprop::Error::EopCoverage` has new `span_start` and `table_start`
  fields.** **Do this:** use `EopCoverage { .. }` in patterns.
- **`Instant` / `Duration` `+` and `-` saturate** at the ends of the
  ±292,000-year range in every build (they panicked in debug builds and
  wrapped in release). **Do this:** use the new `checked_add`, `checked_sub`
  and `Instant::checked_duration_since` where overflow must be detected.
- **`Instant::UNIX_EPOCH.raw` is `8_000_082`** (was 0), because TAI − UTC was
  8.000082 s on 1970-01-01 under the pre-1972 model. **Do this:** build
  Unix-time instants with `Instant::from_unixtime_microseconds`, not from raw
  values.
- **`Instant` → `chrono::DateTime` maps a leap second `23:59:60.x` to
  `23:59:59.x`**, as Unix time does (it was chrono's leap-second form), and
  both directions are exact to the microsecond.
- **`lambert::Error` is `#[non_exhaustive]`** and has new `NonFinite` and
  `CoincidentPositions` variants. **Do this:** add a wildcard arm to
  exhaustive matches.
- **`moon::pos_gcrf` returns GCRF**; the old mean-of-date result is the new
  `moon::pos_mod`.
- **`sgp4::sgp4()` defaults to WGS72 / AFSPC**, and `TLE::fit_from_states`
  fits under WGS72; `TLE::fit_from_states_full` takes the gravity constants
  and ops mode (see [SGP4 and TLEs](#sgp4-and-tles)).
- **`sgp4::SGP4InitArgs::jdsatepoch` is now `epoch_days_1950`** (days since
  1949-12-31 00:00 UTC, kept to sub-microsecond precision), and
  `SGP4InitArgs::from_mean_elements` takes the epoch as an `Instant`.
  `SatRec` has a private field, so it can no longer be built with a struct
  literal (use `SatRec::new()`). **Do this:** update custom `SGP4Source`
  implementations. `OMM::reset_cache` and the new `TLE::reset_cache` are no
  longer needed after edits.
- **`tle::Error` has new variants** (`Record`, `ChecksumMismatch`,
  `SatNumMismatch`, `MissingLine1`, `MissingLine2`, `Frame`, `Offline`), and
  every `TLE::from_lines` / `from_url` error is wrapped in `Record` with its
  line number and satellite. `omm::Error` gains `UnsupportedRefFrame`,
  `UnsupportedCenter` and `Offline`. Both enums are `#[non_exhaustive]`.
- **`Gravity::parse` honours the ICGEM `norm` header** (an `unnormalized`
  file was de-normalized twice), and rejects a header without a positive
  `earth_gravity_constant` or `radius` (`Error::InvalidLine`); an unrecognised
  `tide_system` is classified from C̄20. Only custom files are affected.
- **`utils::build_date()` is removed** (builds are reproducible now).
  **Do this:** use `utils::githash()`, which is `"unknown"` for builds not
  made from a satkit git checkout; the satkit version is in your
  `Cargo.lock`.

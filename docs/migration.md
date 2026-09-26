# Migrating to 0.24

0.24 fixes a set of long-standing defects, and some of the fixes change
results, exception types or return shapes. This page lists what to check when
you upgrade from 0.23. Each item says what changed and what to do. The
[CHANGELOG](https://github.com/ssmichael1/satkit/blob/main/CHANGELOG.md) links
each change to its pull request, where the details are.

## Check old results first

- **`quaternion * V` with an N×3 array was wrong in 0.14.1–0.23.1.** It
  applied the *inverse* rotation. A single 3-vector was rotated correctly.
  **Do this:** recompute anything derived from rotating a stacked N×3 array
  with one quaternion.
- **`satproperties()` positional arguments were swapped through 0.23.** They
  were read as `(craoverm, cdaoverm)`, the reverse of the documented order, so
  drag and radiation pressure were exchanged. **Do this:** re-check results
  from any script that called `satproperties(a, b)` positionally.
- **`lambert` results before 0.24 were wrong** for long-way (transfer angle
  over 180°) and retrograde transfers, which left `r1` in the wrong direction,
  and for hyperbolic and near-parabolic transfers, whose time of flight was
  miscomputed. **Do this:** re-run Lambert targeting, delta-v budgets and
  pork-chop plots.
- **`kepler.from_pv` was wrong for a retrograde equatorial orbit** (inclination
  exactly π): its elements described a different state. **Do this:** recompute
  elements of such orbits.

- **Editing a propagated TLE, or changing `gravconst` / `opsmode` between
  calls on it, had no effect through 0.23.** SGP4 kept using the
  initialization from the first call on that TLE (thousands of km after an
  element edit, ~120 m at LEO after a gravity-model change). **Do this:**
  recompute results from TLEs that were edited or re-propagated with other
  settings.

## Python API

- **`satproperties()` is keyword-only.** A positional call raises
  `TypeError`. **Do this:** write
  `sk.satproperties(cdaoverm=..., craoverm=..., thrusts=..., ecom=...)`.
- **Unknown keywords raise `TypeError`, not `ValueError`**, in `duration()`,
  `propsettings()`, `satproperties()`, `itrfcoord()`, `sgp4()`,
  `propagate()`, `satstate.propagate()`, `gravity()`,
  `gravity_and_partials()` and `nrlmsise00()`. `utils.update_datafiles()`
  also rejects them now; it used to ignore them. **Do this:** fix misspelt
  keywords, and catch `TypeError` where you caught `ValueError`.
- **A one-element list or array of times gives a one-element result**, not a
  scalar, in every vectorised function: `sgp4(tle, [t])` is `(1, 3)` (was
  `(3,)`). **Do this:** pass a scalar time, or index `[0]`.
- **A one-element list of TLEs keeps its axis in `sgp4`**: `sgp4([tle], t)` is
  `(1, 3)` (was `(3,)`), and `sgp4([tle], [t])` is `(1, 1, 3)`. Empty lists
  give empty arrays: `sgp4([], t)` is `(0, 3)` and `sgp4([tle], [])` is
  `(1, 0, 3)` (both raised `RuntimeError`). **Do this:** pass the TLE itself,
  or use `[0]` / `.squeeze()`.
- **`TLE.from_lines()`, `TLE.from_file()` and `TLE.from_url()` always return
  `list[TLE]`**, even for a single element set. Input with no TLEs still
  raises `ValueError`. **Do this:** `tle = sk.TLE.from_lines(lines)[0]`. See
  [Loading TLEs](guide/tle.md#loading-tles) for the located parse errors and
  the new `check_checksum=True` option.
- **`sgp4(..., errflag=True)` no longer raises for an element set SGP4
  cannot initialize** (decayed or eccentricity out of range at epoch,
  SGP4-XP): its rows are NaN and every time carries its init code
  (`sgp4_error.unsupported` for SGP4-XP or unusable OMM metadata), and the
  other element sets of a list are still propagated. Without `errflag` it
  still raises; for a list the message gives the index. **Do this:** check
  the error array instead of catching `RuntimeError`.
- **`TLE.from_url()` and `omm_from_url()` honour offline mode**
  (`SATKIT_OFFLINE=1` or `utils.set_offline(True)`) and raise `RuntimeError`.
  **Do this:** turn offline mode off in scripts that fetch element sets.
- **The `as_*` conversions are removed in 0.25** (deprecated since 0.23):
  `time.as_date()`, `as_gregorian()`, `as_datetime()`, `as_mjd()`, `as_jd()`,
  `as_unixtime()`, `as_iso8601()`, `as_rfc3339()`, and
  `quaternion.as_rotation_matrix()`, `as_euler()`. **Do this:** rename them to
  `to_*`. Run with `python -W error::DeprecationWarning` to find them.
- **`frametransform.eop_source()` is deprecated** and is removed in 0.25. It
  warns, and returns `"finals2000A"` or `None`. **Do this:** use
  `frametransform.eop_coverage()`, which is `None` when no table is loaded.
- **`utils.build_date()` is removed** (builds are reproducible now).
  **Do this:** use `satkit.__version__`, or `utils.githash()`, which is
  `"unknown"` for builds not made from a satkit git checkout.

### Exception types

These calls raise a different exception than in 0.23.1, or now raise where
they used to return a meaningless value:

| call | 0.23.1 | 0.24 |
|---|---|---|
| `quaternion * 2.0` (or a `str`, `None`) | `RuntimeError` | `TypeError` |
| `quaternion * <array of the wrong shape>`, `quaternion.rotation_between(<wrong length>, ...)` | `RuntimeError` | `ValueError` |
| `duration + float`, `duration / str` | `RuntimeError` | `TypeError` |
| `satstate.propagate(<not a time or duration>)` | `RuntimeError` | `TypeError` |
| `gravity(<wrong length>)` / `gravity(<non-numeric>)` | `RuntimeError` | `ValueError` / `TypeError` |
| `propagate(state, begin)` with no end time | `RuntimeError` | `TypeError` |
| `time.strptime()`, `time.from_string()`, `time(<str>)` with an unparseable string | `RuntimeError` | `ValueError` |
| `duration / 0` | `RuntimeError` | `ZeroDivisionError` |
| `duration / 0.0`, `duration / duration(0)` | saturated duration or `inf` | `ZeroDivisionError` |
| `duration(days=nan)`, `duration.from_seconds(inf)`, ... | 0 or saturated | `ValueError` |
| `time + nan`, `time - inf` (scalar, list or array) | `t` or a garbage label | `ValueError` |
| `duration(days=1e300)`, `time + 1e300` | saturated or a garbage label | `OverflowError` |
| `duration(days=1e8, seconds=1e12)`, `(t + 1e8) + 1e8`, `duration + duration`: any result beyond about ±292,000 years | wrapped around (a negative duration, year −34949) | `OverflowError` |
| `propsettings(gravity_degree=4, gravity_order=10)` | order clamped to 4 | `ValueError` (as the `gravity_order` setter) |
| `propagate(..., duration_secs=nan)` | ran to an invalid end | `ValueError` |
| `propresult.interp(..., output_phi=True)` without the STM | the state alone | `ValueError` |
| `TLE.epoch = [t1, t2]` | used `t1` | `TypeError` |
| `lambert(r1, r1, tof)` | NaN velocities | `ValueError` |
| `lambert(...)` with a NaN or infinite input | "convergence failure" `ValueError` or NaN | `ValueError` naming the input |
| `kepler.from_pv(<NaN or inf>)` | NaN elements | `ValueError` |
| `kepler.propagate(nan)`, `kepler.propagate(inf)` | unchanged or a garbage anomaly | `ValueError` |
| `density.nrlmsise(..., <not a time>)` (a `str`, a number, ...) | ignored: ran on the default indices | `TypeError` |
| `TLE.from_lines()` with a line 1 and line 2 of different satellites | a hybrid TLE | `RuntimeError` |
| `TLE.from_lines()` with a line 1 whose line 2 is missing | dropped silently | `RuntimeError` |
| `sgp4()` of an OMM whose `REF_FRAME` is not TEME or `CENTER_NAME` not EARTH | propagated | `RuntimeError` |

Some calls that raised now work: `gravity([7e6, 0, 0])` and integer arrays,
`time - [t1, t2]` (an array of `duration`), `time + <integer or float32
array of days>`, and batch state transforms given integer arrays or nested
lists.

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
  - Errors that used to be silent mis-parses: trailing input in
    `from_rfc3339` and `strptime`, a malformed or out-of-range offset, an
    empty fraction (`12:00:00.Z`), and an extra number or a lone hour in
    `from_string`. `from_string` still ignores trailing words, zone names
    included: `"2024-01-04 13:14:12 EST"` is 13:14:12 UTC.
  - **Do this:** check string inputs that carry offsets or extra text, and
    catch the new errors. Every time-string parser (`from_rfc3339`,
    `strptime`, `from_string` and `time(<str>)`) raises `ValueError` with the
    reason; `strptime`, `from_string` and `time(<str>)` raised `RuntimeError`
    in 0.23.1.
- **Fractional seconds round to the nearest microsecond** instead of
  truncating, both in parsed strings with more than six digits and in float
  conversions (calendar seconds, Unix time, MJD/JD, GPS seconds of week,
  `add_utc_days`, `duration` from floats). Results can move by 1 µs.
  **Do this:** update tests that compare exact microseconds.
- **`strptime` `%z` sign is fixed**: `12:00:00+0100` is `11:00:00Z` (it was
  `13:00:00Z`). **Do this:** remove any workaround.
- **Years outside 0000–9999 print in ISO 8601 expanded form** (`-0001`,
  `+10000`), and `from_rfc3339` / `strptime` read them back, as does
  `from_string` (`"-0044-03-15 12:00:00"` was AD 44).
- **UTC before 1972 follows the 1961–1971 "rubber second" model** of USNO
  `tai-utc.dat` / ERFA, and TAI − UTC is 0 before 1961. Pre-1972 labels
  convert up to 9.9 s differently. A stored pre-1972 instant (a pickle, say)
  keeps its instant but prints a label up to ~10 s different. **Do this:**
  re-derive pre-1972 epochs from their labels.
- **The JPL ephemerides are evaluated at TDB** (was TT), and TDB − TT has its
  correct one-year period (it was off by up to 1.7 ms). The Moon moves by up
  to ~2 m, and third-body propagations shift slightly. **Do this:** regenerate
  stored reference values.

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

- **`moon.pos_gcrf` now returns GCRF coordinates.** Through 0.23 it returned
  mean-of-date coordinates, which differ by precession: about 0.7° in 1950
  and 2050 and 1.4° by 2100. **Do this:** recompute stored low-precision Moon
  positions; code that rotated the result from mean-of-date to GCRF itself
  must stop doing so.
- **`planets.heliocentric_pos` is more accurate.** It rotates the J2000
  ecliptic elements with the fixed J2000 obliquity (it used the obliquity of
  date, ~47″ per century from 2000), and outside 1800–2050 the extra terms
  for Jupiter through Pluto are in the right units (these planets were off by
  up to tens of degrees). **Do this:** recompute stored planet positions.
- **`sun.rise_set` returns the sunrise and sunset of the input's UTC
  calendar date**, whatever its time of day. Through 0.23 about half of the
  inputs gave the next day's events. **Do this:** for a local date, pass a
  timezone-aware `datetime` at local noon.
- **ECOM coefficients (experimental) are referred to 1 AU** and scale by
  $(\text{AU}/d)^2$ with the satellite–Sun distance $d$. **Do this:** multiply
  coefficients fitted with the old scaling by $(d/\text{AU})^2$ at their
  epoch.
- **The Earth rotation angle is more precise** (1 cm at LEO, 6 cm at GEO in
  ITRF ↔ GCRF), and the approximate TEME → GCRF rotation (`qteme2gcrf`,
  `rotation_approx`) no longer includes a stray 0.3–0.6″ polar-motion
  rotation (~19 m at LEO). **Do this:** expect small differences from 0.23
  results.

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
- **Rust `sgp4::sgp4()` uses WGS72 and the AFSPC ops mode** (was WGS84 and
  IMPROVED), the Python default; the ISS moves 14 m at epoch and ~280 m after
  a week. **Do this:** call `sgp4_full(..., GravConst::WGS84,
  OpsMode::IMPROVED)` to keep the old numbers.
- **`TLE.fit_from_states()` fits with WGS72** (was WGS84) and the full TEME →
  GCRF rotation. **Do this:** propagate fitted TLEs with the default
  `sgp4()`, or pass `gravconst=sgp4_gravconst.wgs84` (Rust:
  `fit_from_states_full`) to both.

## Rust API

- **`DataDirReadOnly` is now `DataDirReadOnly { path, reason }`** in
  `earth_orientation_params::Error`, `spaceweather::Error` and
  `utils::update_data::Error`. These enums are not `#[non_exhaustive]`.
  **Do this:** match `Error::DataDirReadOnly { .. }`.
- **`earth_orientation_params::EopSource`, `source()` and `CELESTRAK_FILE`
  are removed.** In `earth_orientation_params::Error`, `InvalidEntry` and
  `ParseFloat` are gone, and `UnsupportedCsv` is new (an `EOP-All.csv` passed
  to `init_from_path` / `init_from_bytes`). **Do this:** delete source checks
  and update exhaustive matches.
- **`orbitprop::Error::EopCoverage` has new `span_start` and `table_start`
  fields.** **Do this:** use `EopCoverage { .. }` in patterns.
- **`Instant` / `Duration` `+` and `-` saturate** at the ends of the
  ±292,000-year range in every build (they panicked in debug builds and
  wrapped in release). **Do this:** use the new `checked_add`, `checked_sub`
  and `Instant::checked_duration_since` where overflow must be detected.
- **`Gravity::parse` honours the ICGEM `norm` header** (an `unnormalized`
  file was de-normalized twice), and rejects a header without a positive
  `earth_gravity_constant` or `radius` (`Error::InvalidLine`); an unrecognised
  `tide_system` is classified from C̄20. Only custom files are affected.
- **`Instant::UNIX_EPOCH.raw` is `8_000_082`** (was 0), because TAI − UTC was
  8.000082 s on 1970-01-01 under the pre-1972 model. **Do this:** build
  Unix-time instants with `Instant::from_unixtime_microseconds`, not from raw
  values.
- **`tle::Error` has new `Record` and `ChecksumMismatch` variants** (the enum
  is `#[non_exhaustive]`), and every `TLE::from_lines` / `from_url` error is
  wrapped in `Record` with its line number and satellite.
- **`utils::build_date()` is removed** (builds are reproducible now).
  **Do this:** use `utils::githash()`, which is `"unknown"` for builds not
  made from a satkit git checkout; the satkit version is in your
  `Cargo.lock`.
- **`sgp4::SGP4InitArgs::jdsatepoch` is now `epoch_days_1950`** (days since
  1949-12-31 00:00 UTC, kept to sub-microsecond precision), and
  `SGP4InitArgs::from_mean_elements` takes the epoch as an `Instant`.
  **Do this:** update custom `SGP4Source` implementations. `OMM::reset_cache`
  and the new `TLE::reset_cache` are no longer needed after edits.

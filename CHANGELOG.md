# Changelog

Only recent releases are listed. Older entries are in this file's git history (`git show vX.Y.Z:CHANGELOG.md`) and on the [GitHub Releases](https://github.com/ssmichael1/satkit/releases) page.

## Unreleased

### CI

- Docs build uses an absolute `SATKIT_DATA` and no longer installs the `satkit-data` bundle: notebooks executed from `docs/tutorials/` could not resolve the relative path and fell back to the bundle's frozen `EOP-All.csv` (ending 2026-08-23), which produced the stale-EOP warning on satkit.dev ([#148](https://github.com/ssmichael1/satkit/pull/148))
- `on_disk_file_is_verified_once_via_sidecar_marker` test sets the restored file's mtime explicitly; Windows file-time granularity (~1–15 ms) let a rewrite reuse the marker's mtime and fail intermittently ([#149](https://github.com/ssmichael1/satkit/pull/149))
- CI refreshes `EOP-All.csv` / `SW-All.csv` on every run (also on an `astro-data` cache hit) via `download_data.py --refresh-only`, so docs and tests no longer run on a stale EOP table; a failed refresh keeps the cached copy instead of failing the job ([#147](https://github.com/ssmichael1/satkit/pull/147))
- GitHub Actions updated to current major versions (checkout v7, setup-python v7, cache v6, upload-artifact v7, download-artifact v8, upload-pages-artifact v5, deploy-pages v5, sccache-action v0.0.11, cibuildwheel v4.2.0; Windows wheel repair explicitly kept off) ([#145](https://github.com/ssmichael1/satkit/pull/145))

### Fixed

- HTTP requests send a descriptive `satkit/<version>` User-Agent; a CelesTrak 503/403 from `TLE.from_url` / `omm_from_url` now explains CelesTrak's throttling of repeated identical queries instead of a bare status code; the TLE, OMM and Optical Observations tutorials fall back to a pinned element set when the live fetch is unavailable, so the docs build no longer depends on CelesTrak ([#144](https://github.com/ssmichael1/satkit/pull/144))
- Python `kepler.mean_anomaly` setter no longer hangs on NaN or `eccen >= 1` (delegates to the core capped solver); `from_pv` extracts inclination and the anomalies with `atan2` (exact down to i = 1e-9 rad, e ≤ 0.999); constructor accepts keyword arguments matching the stub (`a, eccen, incl, raan, w, nu`), `propagate` accepts `int` seconds, and a new [Keplerian Elements guide](https://satkit.dev/guide/kepler/) ([#146](https://github.com/ssmichael1/satkit/pull/146))
- Drag: NRLMSISE-00 was given geodetic latitude/longitude in radians instead of degrees (pointwise density error up to +200 %, ~6 % of the 3-day drag displacement at ISS altitude); the space-weather feed now follows the NRLMSISE-00 interface (observed rather than 1 AU-adjusted F10.7, current-day daily Ap); Python `propagate(..., satproperties=None)` is accepted; eight 3-day drag cases (constant and CelesTrak-file space weather, 250–550 km) added to the GMAT regression corpus with the measured floors documented in `tests/gmat/README.md` ([#150](https://github.com/ssmichael1/satkit/pull/150))

## 0.21.0 - 2026-08-30

### Added

- **Experimental:** ECOM (Empirical CODE Orbit Model) solar-radiation-pressure model — reduced/ECOM1/ECOM2 coefficients in the DYB frame, Rust (`EcomParams`, `SatProperties::srp_ecom`) and Python (`ecomparams`, `satproperties(ecom=...)`), with a GPS SP3 fit/prediction tutorial; the interface may change in a minor release ([#131](https://github.com/ssmichael1/satkit/pull/131))
- Static data files are downloaded from a manifest pinned to the release (`data/manifest.json`, SHA-256 verified), trying GitHub release assets, then the origin servers (JPL, IERS), then the GCS bucket; `SATKIT_DATA_URL` overrides the source for mirrors ([#137](https://github.com/ssmichael1/satkit/pull/137))
- GMAT regression corpus: 17 seven-day reference trajectories (LEO to cislunar; `j2`/`full`/`gr` force models) replayed and gated under `cargo test` and `pytest`; regenerate with `tests/gmat/generate.py` ([#127](https://github.com/ssmichael1/satkit/pull/127))
- EOP coverage is visible and enforceable: `earth_orientation_params::{coverage, status}` (Python `frametransform.eop_coverage()` / `eop_status()`), one-time warnings past the table end or with no table, and `PropSettings::require_eop_coverage` ([#133](https://github.com/ssmichael1/satkit/pull/133))
- `frametransform::ierstable::preload()`; a missing `tab5.2*.txt` now raises `RuntimeError` in Python instead of `PanicException` ([#133](https://github.com/ssmichael1/satkit/pull/133))
- Python test pinning the EME2000 frame bias (23.1 mas) against IERS 2010 values ([#132](https://github.com/ssmichael1/satkit/pull/132))

### Changed

- **Breaking (Python packaging):** core data (IERS tables, gravity models to degree 70) is compiled in and the JPL ephemeris downloads on first use (SHA-256 verified) into the platform user-data directory, so `satkit-data` is no longer a dependency (optional: `pip install satkit[data]`); `SATKIT_OFFLINE=1` / `utils.set_offline()` forbid network access, and read-only or missing data locations are typed errors ([#139](https://github.com/ssmichael1/satkit/pull/139))
- Propagator's GCRF→ITRF table now uses the full IAU 2006/2000A chain instead of the ~1″ IAU-76 approximation; removes an inclination-dependent drift of ~50 m over 7 days at LEO versus GMAT ([#127](https://github.com/ssmichael1/satkit/pull/127))
- Relativistic correction now includes geodesic precession and Lense–Thirring (all three IERS 2010 Eq. 10.12 terms); shifts results by ≤ 1 m over 7 days at 200,000 km, cm at LEO. **Breaking (Rust):** `Precomputed::interp` returns the named struct `InterpSample` instead of a tuple ([#129](https://github.com/ssmichael1/satkit/pull/129))
- **Breaking:** gravity degree/order above 40 is rejected (`Error::InvalidGravityDegree`, Python `ValueError`) instead of being silently evaluated at 40 ([#130](https://github.com/ssmichael1/satkit/pull/130))
- `Precomputed` table size is capped (`Error::PrecomputeTooLarge`) and non-finite padding rejected, instead of allocating gigabytes ([#130](https://github.com/ssmichael1/satkit/pull/130))
- A propagation with no EOP table loaded fails with `Error::EopUnavailable` instead of running with zero polar motion and UT1−UTC ([#133](https://github.com/ssmichael1/satkit/pull/133))
- Data downloads use `https://celestrak.org` and validate manifest paths and URLs ([#130](https://github.com/ssmichael1/satkit/pull/130))
- numeris 0.5.14 → 0.5.18: the adaptive Runge–Kutta integrators no longer abort at shadow-boundary kinks on eclipsing arcs ([#128](https://github.com/ssmichael1/satkit/pull/128)) ([#135](https://github.com/ssmichael1/satkit/pull/135))

- **Cannonball SRP acts along the satellite→Sun line** rather than the
  geocentric Sun direction (a ~1e-4 rad difference at LEO). `test_gps`
  residual 1.7997 → 1.7868 m.

### Fixed

- SP3 epochs are read as GPS time, not UTC (18 s error) in `test_gps`, `sp3file.py` and the validation script — `test_gps` residual 1.80 → 1.21 m; cannonball SRP now acts along the satellite→Sun line; `jgm3`/`itugrace16` documented as zero-tide models ([#131](https://github.com/ssmichael1/satkit/pull/131))
- `import satkit` no longer fails when the optional `satkit_data` bundle is installed as a namespace package (the conda layout, no `__init__.py`): its `data/` directory is discovered via `__path__` ([#140](https://github.com/ssmichael1/satkit/pull/140))
- References page rebuilt as a full bibliography and every guide, API page and tutorial now cites its primary source; wrong SGP4 (AIAA 2006-6753), JGM-3 DOI and box-wing citations corrected; drag gate, Lambert multi-revolution, leap-second and GR descriptions brought in line with the code; RK stage counts corrected (RKV98 is 21 stages) ([#136](https://github.com/ssmichael1/satkit/pull/136))
- Frame-bias docs: `EME2000` is 23.1 mas from GCRF (docs said 17); GMAT's `EarthMJ2000Eq` is an IAU-76 realization ~44 mas from ICRF, not the constant bias ([#132](https://github.com/ssmichael1/satkit/pull/132))
- Doc fixes: Gauss–Jackson dense-output note, dual licence in crate docs, CONTRIBUTING versions/paths, gravity degree limit in `docs/index.md`, leap-second table described as compiled in ([#130](https://github.com/ssmichael1/satkit/pull/130))

### CI

- Published wheels are import-tested by cibuildwheel; the NOAA network test is `#[ignore]`d and run explicitly; `doc = false` on the extension crate so `cargo doc --workspace` builds ([#130](https://github.com/ssmichael1/satkit/pull/130))

## 0.20.4 - 2026-08-28

### Added

- **`scale=` keyword on the `time(...)` constructor** (Python): Gregorian
  date/time components passed to `satkit.time(year, month, day[, hour, minute,
  second])` can now be interpreted in an explicit time scale, e.g.
  `satkit.time(2020, 1, 1, scale=satkit.timescale.TAI)`, mirroring the existing
  `time.from_mjd(..., scale=...)` / `time.from_jd(..., scale=...)` API. The
  keyword defaults to `satkit.timescale.UTC`, so all existing calls are
  unchanged. Backed by a new scale-aware core constructor
  `Instant::from_datetime_with_scale`.
- **`SGP4InitArgs::from_mean_elements`** — a constructor that performs the
  rev/day → rad/min and degree → radian conversions from catalog units. Both
  the `TLE` and CCSDS `OMM` SGP4 sources now build their init args through it,
  so the conversion factors are defined in one place.
- **`time` and `duration` are now hashable.** Both define `__eq__` but
  previously lacked `__hash__`, which made them unhashable (unusable as `dict`
  keys or `set` members). The hash is derived from the underlying microsecond
  count, so it is consistent with equality.
- **Equality and `repr` on more value types.** `TLE`, `kepler`, and `itrfcoord`
  now implement `__eq__`; `TLE`, `kepler`, `satstate`, and `propsettings` now
  implement `__repr__` (delegating to their `str` form). The three float-backed
  types that gained `__eq__` are intentionally left unhashable — a failed
  `hash()` is clearer than silently-wrong float-keyed lookups.
- **`mypy.stubtest` in CI.** The `python bindings test` job now verifies that
  the hand-written `.pyi` type stubs match the compiled PyO3 bindings. Two CLI
  flags (`--ignore-positional-only`, `--ignore-disjoint-bases`) plus
  `python/stubtest_allowlist.txt` suppress systematic PyO3 idioms (constructors,
  final classes, native submodules); everything else must agree.

### Fixed

- **Canonical physical constants corrected and documented.** The lunar
  gravitational parameter now uses the JPL DE440 value
  (`4.902800118e12 m³/s²`) instead of the erroneous `4.9048695e12`; duplicate
  lunar-GM literals in point-gravity tests now reference the shared constant.
  The solar GM, WGS 84 flattening and rotation rate, nominal solar radius,
  DE440 Earth–Moon mass ratio, and derived geosynchronous radius were updated
  to their canonical values. `JGM3_J2` now uses the conventional positive `J2`
  sign. Every constant in `consts.rs` now identifies its authoritative source.
- **Relativity documentation now describes the Schwarzschild correction
  accurately.** Removed misleading fixed position-drift-per-day estimates and
  clarified that, for a state satisfying the Newtonian circular-orbit
  relation, the post-Newtonian correction points radially outward while the
  much larger Newtonian acceleration points inward. The corresponding test
  name and Python type-stub documentation were corrected as well.
- **Type stubs now type-check and match the runtime.** The `.pyi` files had
  never been checked and contained illegal overload-implementation blocks, a
  `time.__add__` overload group split by other methods, and `time`-as-annotation
  shadowing. Filled in stub gaps (`itrfcoord.height`, `time.add_utc_days`,
  `weekday.Invalid`) and removed/fixed a phantom `time.as_gregorian(scale=)`
  parameter and a mis-declared `sgp4_opsmode.improved` property.
- **Panic hardening: malformed and edge-case input now returns errors instead
  of panicking** (PR #124), following a codebase-wide audit:
  - *TLE parsing*: day-of-year is range-checked, negative satellite numbers
    are rejected, and non-finite implied-exponent fields (bstar, nddot) error
    at parse time instead of overflowing later in epoch math, `Display`, or
    `to_2line`. `Display` also falls back to the raw satellite number rather
    than unwrapping a failed alpha5 conversion.
  - *Time*: `from_rfc3339` no longer panics on non-ASCII input while scanning
    for a timezone offset; `from_datetime` errors on extreme years (new
    `InstantError::InvalidYear`); MJD conversions, `from_gps_week_and_second`,
    and leap-second folding saturate at the i64 boundaries; chrono `DateTime`
    conversion saturates to `MIN_UTC`/`MAX_UTC`; `strftime` `%B`/`%b` handle
    out-of-range months.
  - *JPL ephemerides*: querying exactly at the file's end epoch (`jd_stop`)
    now clamps to the last Chebyshev record instead of indexing past it, and
    the parser validates header fields and declared sizes (with checked
    arithmetic, before allocating) so a corrupt or crafted file errors
    cleanly. Unpopulated bodies error instead of underflowing.
  - *Gravity models*: `Gravity::parse` rejects `.gfc` lines with order >
    degree (previously a panic for large orders and **silent coefficient
    aliasing** for moderate ones), and the evaluators clamp the requested
    degree to the loaded model's table so a custom low-degree model cannot be
    indexed out of bounds. A spec-conformant bare `end_of_head` line now
    terminates the header instead of silently yielding an all-zero model.
  - *Orbit propagation*: the force model falls back to direct computation
    when an adaptive integrator probes outside the precomputed interp table
    (previously an unwrap panic for high-altitude states);
    `propagate::<C>` with unsupported column counts, zero/negative/non-finite
    `Precomputed` steps, and invalid thrust frames constructed via pub
    fields/`Deserialize` all surface as errors instead of panics.
  - *Utilities*: `datadir()` no longer panics (and permanently poisons its
    singleton mutex) if a candidate directory cannot be stat-ed; download
    helpers return an error for paths/URLs with no valid file name.
- **Python bindings: invalid input raises clean exceptions instead of
  `PanicException`**, and non-contiguous numpy input now works:
  - Wrong-size/shape arrays to `propagate`, frame transforms,
    `kepler.from_pv`, `satstate.add_maneuver`, and `thrust.constant` raise
    `ValueError`/`RuntimeError` (via shape validation in `py_to_smatrix`).
  - Strided views and Fortran-order arrays are now accepted by `itrfcoord`,
    `gravity`, `gravity_and_partials`, and the `satstate`
    covariance/uncertainty setters (previously `as_slice().unwrap()` panics).
  - `TLE.from_lines`/`from_url` raise on input containing no valid TLEs;
    `TLE.from_file` handles non-UTF-8 files; `sgp4` rejects empty TLE
    lists/time arrays; `kepler(...)` rejects non-numeric positional args;
    `quaternion.from_axis_angle` validates axis length;
    `planets.heliocentric_pos` raises for Sun/Moon and out-of-range times
    (previously a panic with the GIL released); `time` arithmetic with ints
    too large for f64 raises `OverflowError`; `datetime.timestamp()` failures
    propagate; `utils.datadir()` handles non-UTF-8 paths.

### Changed

- **BREAKING: `sgp4::Error::SatRecInit` now carries a typed `SGP4Error`
  instead of a raw `i32`.** The raw Vallado init error code is mapped to the
  corresponding `SGP4Error` variant (eccentricity, mean motion, perturbed
  eccentricity, semi-latus rectum, orbit decay) at construction, so the
  `Display` output is now a description rather than a bare number.
  Additionally, the `SGP4Error` enum has moved from `sgp4::sgp4_impl` to
  `sgp4::error`; it remains re-exported at `satkit::sgp4::SGP4Error`, so the
  public path is unchanged.
- **`LambertError` renamed to `lambert::Error`** to match the per-module
  `module::Error` convention used everywhere else in the crate. The old name
  remains as a `#[deprecated]` type alias, so existing code compiles with a
  deprecation warning; update `satkit::lambert::LambertError` →
  `satkit::lambert::Error`. A `lambert::Result<T>` alias was also added.
- **`itrfcoord(...)`, `sgp4(...)`, and `satstate.propagate(...)` now reject
  unknown keyword arguments** instead of silently ignoring them. Previously a
  typo such as `itrfcoord(..., alttiude=100)` was dropped, leaving the ground
  station at 0 m altitude; it now raises `ValueError`. Callers passing only
  documented keywords are unaffected.
- **Renamed keyword arguments on several bindings** so the accepted Python
  keyword matches its documentation (the stubs previously advertised names the
  runtime rejected). Positional calls are unaffected; only callers passing these
  by the *old* keyword must update:
  - `duration.from_hours` / `from_minutes` / `from_seconds`: `d` → `hours` / `minutes` / `seconds`
  - `time.from_string`: `s` → `string`;  `time.from_unixtime`: `t` → `unixtime`
  - `time.from_rfc3339`: `s` → `rfc3339`;  `time.from_datetime`: `tm` → `dt`
  - `time.strftime`: `fmt` → `format`;  `time.strptime`: `(s, fmt)` → `(date_string, format)`
  - `kepler.from_pv`: `(r, v)` → `(pos, vel)`


## 0.20.3 - 2026-08-21

Released with the panic-hardening audit (PR #124), the `lambert::Error` /
typed `sgp4::Error::SatRecInit` API cleanups (PR #123) and the stubtest CI
check (PR #122). Those entries were recorded under 0.20.4 above when 0.20.4
followed a week later; see that section for details.

## 0.20.2 - 2026-07-03

### Fixed

- **`Kepler` mean→eccentric anomaly conversion was inaccurate for
  high-eccentricity orbits with unwrapped mean anomalies** (e.g.
  `Kepler::propagate` beyond one revolution at e ≳ 0.85): the naive
  `E₀ = M ± e` Newton starting guess lands in a near-flat region of Kepler's
  equation, the iteration turns chaotic, and the (0.20.0-introduced)
  iteration cap could exit with a badly wrong anomaly — up to ~0.24 rad of
  true-anomaly error observed. `mean2eccentric` now range-reduces M to
  [0, 2π) and uses Danby's initial guess, converging in <10 iterations for
  all e < 1. Found by the new property-based test suite in CI
  (`kepler_period_closure`, fresh random seed); the counterexample is pinned
  as a permanent regression test.


## 0.20.1 - 2026-07-03

### Security / dependencies

- **Resolved all 7 outstanding RustSec advisories** (surfaced by the new
  `cargo audit` CI gate): `quick-xml` 0.38 → 0.41 (two high-severity DoS
  advisories in XML parsing — relevant since OMM XML is untrusted input),
  `pyo3`/`numpy` 0.28 → 0.29 (PyList/PyTuple iterator out-of-bounds read,
  `new_closure` Sync bound), plus compatible `cargo update` bumps covering
  the `rustls-webpki`, `anyhow`, and `rand` advisories.

### Testing

- **New property-based test suite** (`tests/properties.rs`, proptest): 12
  properties over randomly generated domains with automatic shrinking —
  Kepler `from_pv ∘ to_pv` round-trip (including the exactly-circular /
  equatorial degeneracies), anomaly-conversion round-trip and termination,
  one-period orbit closure, Vincenty finiteness/non-negativity over all
  coordinate pairs, geodetic round-trip, MJD/unixtime round-trips in all
  data-independent time scales, Instant±Duration arithmetic, TLE
  format ∘ parse round-trip at field precision, TLE parser never-panics,
  and quaternion rigidity + Euler round-trip. Runs in ordinary `cargo test`
  (no data files needed); failures persist in `proptest-regressions/`.
- **`TimeScale` now derives `Clone`, `Copy`, and `Hash`** — previously each
  use of a scale value moved it, so loops/reuse required re-naming the
  variant (a long-standing ergonomic wart the new property tests
  immediately hit).

### CI

- **New gates:** `cargo clippy --workspace --all-targets -D warnings` (the
  outstanding warnings were fixed or annotated), `cargo audit` (RustSec
  advisory scan), and an **sdist-install guard** that builds the sdist and
  `pip install`s from it — the check that would have caught the 0.20.0
  source-unbuildable sdist before it shipped.
- **Weekly scheduled `cargo audit`** (`audit.yml`) — RustSec advisories are
  published on their own schedule; the cron run surfaces new ones even when
  nothing is being pushed.

### Fixed

- **PyPI sdist was unbuildable from source.** `MANIFEST.in` omitted the
  `benches/` directory while `Cargo.toml` declares the `hotpaths` bench
  target, so `cargo` failed to parse the manifest when installing from the
  sdist ("can't find `hotpaths` bench"). Wheels were unaffected; this broke
  source builds only (present since the benches landed in 0.19.0). The
  sdist now includes `benches/`.

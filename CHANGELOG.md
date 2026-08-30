# Changelog

Only recent releases are listed. Older entries are in this file's git history (`git show vX.Y.Z:CHANGELOG.md`) and on the [GitHub Releases](https://github.com/ssmichael1/satkit/releases) page.

## Unreleased

### Added

- Static data files are downloaded from a manifest pinned to the release (`data/manifest.json`, SHA-256 verified), trying GitHub release assets, then the origin servers (JPL, IERS), then the GCS bucket; `SATKIT_DATA_URL` overrides the source for mirrors ([#137](https://github.com/ssmichael1/satkit/pull/137))
- GMAT regression corpus: 17 seven-day reference trajectories (LEO to cislunar; `j2`/`full`/`gr` force models) replayed and gated under `cargo test` and `pytest`; regenerate with `tests/gmat/generate.py` ([#127](https://github.com/ssmichael1/satkit/pull/127))
- EOP coverage is visible and enforceable: `earth_orientation_params::{coverage, status}` (Python `frametransform.eop_coverage()` / `eop_status()`), one-time warnings past the table end or with no table, and `PropSettings::require_eop_coverage` ([#133](https://github.com/ssmichael1/satkit/pull/133))
- `frametransform::ierstable::preload()`; a missing `tab5.2*.txt` now raises `RuntimeError` in Python instead of `PanicException` ([#133](https://github.com/ssmichael1/satkit/pull/133))
- Python test pinning the EME2000 frame bias (23.1 mas) against IERS 2010 values ([#132](https://github.com/ssmichael1/satkit/pull/132))

### Changed

- **Breaking (Python packaging):** core data (IERS nutation tables, gravity models to degree 70) is compiled into the library, so frames and gravity work with no data files; the JPL ephemeris downloads on first use with SHA-256 verification; `satkit-data` is no longer a dependency of `satkit` (optional offline bundle: `pip install satkit[data]`). Downloads go to the platform user-data directory (macOS `~/Library/Application Support/satkit-data`, Linux `$XDG_DATA_HOME/satkit-data`, Windows `%LOCALAPPDATA%\satkit-data`) — never inside `site-packages` — while files are looked up across `utils.data_search_dirs()`; `SATKIT_OFFLINE=1` forbids network access ([#139](https://github.com/ssmichael1/satkit/pull/139))
- Propagator's GCRF→ITRF table now uses the full IAU 2006/2000A chain instead of the ~1″ IAU-76 approximation; removes an inclination-dependent drift of ~50 m over 7 days at LEO versus GMAT ([#127](https://github.com/ssmichael1/satkit/pull/127))
- Relativistic correction now includes geodesic precession and Lense–Thirring (all three IERS 2010 Eq. 10.12 terms); shifts results by ≤ 1 m over 7 days at 200,000 km, cm at LEO. **Breaking (Rust):** `Precomputed::interp` returns the named struct `InterpSample` instead of a tuple ([#129](https://github.com/ssmichael1/satkit/pull/129))
- **Breaking:** gravity degree/order above 40 is rejected (`Error::InvalidGravityDegree`, Python `ValueError`) instead of being silently evaluated at 40 ([#130](https://github.com/ssmichael1/satkit/pull/130))
- `Precomputed` table size is capped (`Error::PrecomputeTooLarge`) and non-finite padding rejected, instead of allocating gigabytes ([#130](https://github.com/ssmichael1/satkit/pull/130))
- A propagation with no EOP table loaded fails with `Error::EopUnavailable` instead of running with zero polar motion and UT1−UTC ([#133](https://github.com/ssmichael1/satkit/pull/133))
- Data downloads use `https://celestrak.org` and validate manifest paths and URLs ([#130](https://github.com/ssmichael1/satkit/pull/130))
- numeris 0.5.14 → 0.5.18: the adaptive Runge–Kutta integrators no longer abort at shadow-boundary kinks on eclipsing arcs ([#128](https://github.com/ssmichael1/satkit/pull/128)) ([#135](https://github.com/ssmichael1/satkit/pull/135))

### Fixed

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


## 0.20.0 - 2026-07-03

A large correctness, robustness, and cleanup pass. This is a pre-1.0 release, so
the breaking changes below ride in a minor bump.

### Breaking changes

- **Pickle formats for `satstate`, `satproperties`, and `TLE` changed.** All
  three now carry a leading version byte and explicit counts: `satstate` gained
  a covariance flag and maneuver count (replacing a byte-length heuristic that
  mis-read a state with ≥9 maneuvers and no covariance as *having* a covariance,
  silently dropping every maneuver), `satproperties` gained a thrust-arc count,
  and `TLE` now also serializes `element_num` and `ephem_type`. The
  maneuver/thrust frame is encoded with an exhaustive match — previously a
  `_ => 0` catch-all silently turned NTW/LVLH burns into GCRF on unpickle — and
  all byte reads are bounds-checked and alignment-safe. **Pickles written by
  ≤0.19 are not readable by this release** (they were the source of the bugs);
  loading one reports an unsupported-version error. Re-generate them.
- **`ContinuousThrust::new` now validates the frame and returns `Result`.**
  Constructing a thrust in an Earth-fixed / inertial-chain frame returns
  `Error::UnsupportedThrustFrame` at construction (mirroring the maneuver-frame
  validation in `SatState::propagate`) instead of panicking deep inside the
  force evaluation during propagation.
- **Unused UKF advertisement removed.** The numeris `estimate` feature was
  enabled but nothing was re-exported — no filter was reachable from Rust or
  Python despite the crate docs advertising a "UKF implementation". The feature
  flag and the doc claim are removed; estimation may return as a designed
  feature in a future release.
- **`quaternion.slerp` no longer takes an `epsilon` argument** (Python). The
  parameter was documented but silently ignored (numeris uses a fixed internal
  threshold); passing it now raises `TypeError`.
- **`From<i32>` for `TimeScale` and `Weekday` replaced by `TryFrom<i32>`.** The
  infallible conversion silently mapped any out-of-range integer to `Invalid`;
  the new `TryFrom` returns a typed error (`InvalidTimeScale` / `InvalidWeekday`,
  re-exported at the crate root). (`From<i32>` could not simply gain a `TryFrom`
  alongside it — the std blanket impl already provides an infallible one.)
- **`earthgravity::gravhash()` and `accel_jgm3()` removed.** `accel` /
  `accel_and_partials` now dispatch through `GravityModel::get()`, so only the
  requested model is loaded instead of eagerly parsing all four `.gfc` files
  (one missing file used to panic the whole map).
- **`GaussJackson8::interpolate` / `interpolate_batch` now take
  `&GJDenseOutput` instead of `&GJSolution`.** This removed a full deep-copy of
  the stored trajectory on every single-point interpolation query.
- **Many `frametransform` module error variants are now typed**, and `Frame`,
  `GravityModel`, `PropSettings`, and `ContinuousThrust` gained `serde`
  derives (additive).

### Fixed (correctness)

- **Python `frametransform.to_gcrf` / `from_gcrf` returned the transposed
  (inverse) rotation matrix.** numeris matrices are column-major but the
  binding flattened and reshaped them row-major, so `to_gcrf` actually
  returned the GCRF→frame rotation and vice versa. Callers that only composed
  `to_gcrf` with `from_gcrf` outputs were self-consistent, but any use of the
  matrix directly against numpy vectors was silently wrong. Caught by the new
  `rotation_with_state` cross-check test.
- **`spaceweather::update()` was a silent no-op.** It downloaded
  `sw19571001.txt` while the loader reads `SW-All.csv`, so the public refresh
  entry point re-loaded stale data forever. Now downloads `SW-All.csv`.
- **`propsettings(enable_interp=...)` was unusable** — the kwarg lookup key was
  misspelled `enable_iterp`, so both spellings raised. Fixed.
- **Lambert solver could not return hyperbolic solutions.** The Householder
  iterate was clamped to `(-0.999, 0.999)` even for the zero-revolution case,
  so short-time-of-flight / high-energy transfers walked to the clamp and
  reported `ConvergenceFailed`. The zero-rev case now clamps only the lower
  bound; multi-rev keeps the elliptic clamp.
- **`Instant::from_string` silently mis-parsed `MM/DD/YYYY` / `DD-MM-YYYY`
  dates.** The tokenizer discards punctuation, leaving ~115 lines of unreachable
  separator-matching code while numeric fields were actually consumed
  positionally (so `"12/25/2023"` became year 12). Removed the dead code and
  documented that only ISO-ordered and month-name strings parse; use `strptime`
  with an explicit format for locale-ordered numeric dates.
- **SGP4 failures returned position `(0, 0, 0)` that looked valid.** A per-time
  propagation error (e.g. decay) left the pos/vel columns zeroed — Earth's
  center — with only the error code set. Failed columns are now `NaN`, and
  `TLE::fit_from_states` rejects any trial step whose propagation failed rather
  than folding zeros into the least-squares residuals.
- **SGP4 port: recovered RAAN mean element was lost.** `satrec.om` was assigned
  twice (`nodem` then `argpm`); added a distinct `SatRec::om_node` field
  (Vallado's `satrec.Om`). Propagated state is unchanged — these are diagnostic
  fields the propagator never reads.
- **Solar-cycle-forecast fallback was dead code / `-1` sentinels could reach the
  density model.** `spaceweather::get()` extrapolates the last record forever,
  so NRLMSISE-00's forecast fallback was unreachable and trailing
  monthly-predicted rows (fields parsed as `-1`) could feed the model. The
  density path now guards against sentinel values and falls back to the forecast.

### Robustness

- **`Kepler::mean2eccentric` could loop forever** for eccentricity ≥ 1 (the
  Newton step goes non-finite); it is now capped at 30 iterations.
- **`Kepler::from_pv` produced `NaN` for exactly circular or equatorial
  orbits.** Added the standard Vallado special cases (true longitude, argument
  of latitude, longitude of periapsis) and a new `kepler::Error::Degenerate`
  for zero-angular-momentum (rectilinear) states.
- **Vincenty geodesics returned `NaN` for coincident points** (and divided by
  zero on the equatorial line). Coincident points now return `(0, 0, 0)` and
  the equatorial line is handled. Near-antipodal accuracy is a known Vincenty
  limitation and is now documented (result stays finite).
- **TLE parsing panicked on non-ASCII input and silently dropped CRLF lines.**
  `load_2line` rejects non-ASCII up front (making the byte-slicing safe), and
  `from_lines` trims trailing whitespace and accepts ≥69-char lines so
  CRLF-terminated files parse.
- **`Instant::strptime` panicked on 10+-digit fractional seconds** (i32
  overflow in the error path); it now truncates to microseconds.
- **An `ImpulsiveManeuver` in an unsupported frame panicked mid-propagation.**
  `SatState::propagate` now validates maneuver frames up front and returns
  `Error::UnsupportedManeuverFrame`.
- **`jplephem` used an alignment-unsound `Vec<u8>` → `*const f64` cast** when
  loading Chebyshev coefficients (technically UB). Replaced with a byte-wise
  `memcpy`; behavior is unchanged on little-endian hosts.
- **Malformed pickle bytes raise `ValueError` instead of a Rust panic** for
  `TLE` and `propresult` (bounds-checked reads + mapped serde errors).
- **`propsettings` and `thrust` are now picklable** (via `serde`), so they can
  cross `multiprocessing` process boundaries. (`propsettings` was previously
  unpicklable, which broke `multiprocessing`-based propagation.)
- **Gauss-Jackson 8 over a span shorter than 8 steps now returns a descriptive
  `Error::GJIntervalTooShort`** instead of the integrator's generic "step not
  finite".
- **`Instant::as_mjd_with_scale(TimeScale::Invalid)` returns `NaN`** instead of
  `0.0` (a valid MJD, 1858-11-17), so misuse poisons downstream math visibly.

### Data-file robustness

- **Downloads are now atomic.** `download_file` / `download_if_not_exist` stream
  to a sibling `.part` file and rename it into place on success, so an
  interrupted transfer (network drop, Ctrl-C) can no longer leave a truncated
  file that later runs trust as complete.
- **Data-file parsers no longer panic on truncated/corrupt input.** Added
  field-count / bounds guards to the space-weather CSV parser (skips blank
  lines, errors on short rows), the Earth-gravity `.gfc` parser (the field
  guard now matches the indices it reads), the JPL ephemeris header/constant
  reads (single header-size check plus bounds on dynamic offsets; also fixed a
  non-char-boundary slice), and the IERS table parser (propagates bad numeric
  tokens instead of `unwrap`-panicking, and bounds-checks rows/columns).
- **A missing JPL ephemeris file surfaces as an error, not a panic.** The
  public `jplephem` query functions map a cached load failure to
  `Error::LoadFailed` instead of unwrapping the singleton. The gravity-model and
  IERS-table singletons (which back non-`Result` hot paths) now panic with an
  actionable message naming `SATKIT_DATA` / `update_datafiles` rather than an
  opaque `unwrap`.

### Architecture / cleanups

- **New `frametransform::rotation_with_state(from, to, t, pos, vel)`** — a
  single front door that handles *all* frames, both the time-parameterised
  Earth chain and the orbit-dependent frames (LVLH/RTN/NTW). Pure Earth-frame
  pairs delegate to `rotation`'s shortest path through the frame graph (it does
  **not** always pivot through GCRF); only pairs involving an orbit frame
  compose through GCRF. The existing `rotation` and `to_gcrf`/`from_gcrf`
  remain as the low-level pieces. **Exposed in Python** as
  `satkit.frametransform.rotation_with_state(from_frame, to_frame, tm, pos, vel)`
  (accepts `satkit.time` or `datetime.datetime`), with stubs and documentation.
- **New `utils::RefreshableSingleton`** (`RwLock<Option<T>> + Once`) unifies the
  lazy-load/refresh scaffolding that was copy-pasted across the EOP,
  space-weather, and solar-cycle-forecast modules. Space weather no longer
  re-attempts a blocking load on *every* `get()` (which, offline, meant an HTTP
  attempt per ODE step during a drag propagation).
- **`SatState::qgcrf2lvlh` now derives from `frametransform::gcrf_to_lvlh`** so
  the LVLH axis convention lives in one place.
- **Leap-second folding factored into a single `add_leapseconds` helper** (was
  duplicated across `from_unixtime`, `from_mjd_with_scale`, `from_datetime`,
  `now`).
- **Dead code removed:** the commented-out `pyukf` module, an unregistered
  `PyQuaternionVec`, two dead `pyinstant` helpers, and unused SGP4-port locals;
  the duplicated SGP4 result-packing block and `time.from_datetime` now share a
  single implementation.

### New Python bindings

- **New `satkit.spaceweather` submodule** — `get(time)` returns the full daily
  space-weather record (Kp/Ap arrays, F10.7 observed/adjusted and 81-day
  averages, sunspot number, …) that the NRLMSISE-00 density model consumes;
  `predicted_f107(time)` exposes the NOAA/SWPC solar-cycle forecast used for
  future-epoch densities; `update()` refreshes the data. Previously none of
  this data was reachable from Python.
- **TLE catalog metadata** — `intl_desig`, `desig_year`, `desig_launch`,
  `desig_piece`, `ephem_type`, `element_num`, and `rev_num` now have Python
  getters/setters (they were pickled but unreadable).
- **`time` epoch constants** — `time.J2000`, `time.GPS_EPOCH`,
  `time.MJD_EPOCH`, `time.UNIX_EPOCH`.
- **Quaternion completion** — `from_euler` (inverse of `as_euler`),
  `identity`, `norm`, `normalize`, `inverse`, and `dot`.
- **`satstate.maneuvers`** — returns the scheduled impulsive maneuvers (time,
  delta-v, frame), not just a count.
- **`itrfcoord.distance_to(other)`** — scalar geodesic distance companion to
  `geodesic_distance`.
- **`kepler.semiparameter`**, **`duration.from_milliseconds`**,
  **`duration.microseconds`**, and **`jplephem.consts(name)`** (DE-file
  constants: AU, EMRAT, GM values).

### Python binding fixes

- **`sgp4` with a list of TLEs silently ignored the `gravconst` / `opsmode`
  kwargs** — the list path used the default configuration regardless. Now
  honored on all input paths.
- **The `sgp4` docstring example called `TLE.single_from_lines`**, a method
  that does not exist; corrected to `TLE.from_lines(lines)[0]`.
- **`nrlmsise00`** — docstring said the time kwarg was `tm` (it is `time`); the
  function now also accepts `datetime.datetime`, rejects misspelled kwargs, and
  has a type stub (it was in `__all__` with no stub).
- **`propagate` docstring** no longer documents the removed `output_dense`
  kwarg (interpolation is controlled by `propsettings.enable_interp`) and the
  force-model list now correctly includes solid Earth tides and the
  general-relativistic correction.
- **OMM-dict input validates `MEAN_ELEMENT_THEORY` / `TIME_SYSTEM`** when
  present, matching the Rust-side OMM parser (dicts from `json.load` /
  `xmltodict` are the supported route for local OMM files).

### Python type stubs

- **Time inputs accept `datetime.datetime` everywhere they accept
  `satkit.time`, and the stubs now say so.** Introduced `TimeScalar`,
  `TimeArrayLike`, and `TimeInput` type aliases (in `satkit.pyi`, re-exported to
  the submodule stubs) and applied them across `frametransform`, `sun`, `moon`,
  `planets`, and `jplephem`, fixing signatures that previously omitted
  `datetime.datetime`. `PyInstant`-direct parameters (e.g.
  `satstate(time=...)`, `sun.rise_set`) correctly still accept only
  `satkit.time`.
- **Scalar-vs-array output typing is now explicit via overloads.** Functions
  that return a scalar for a scalar time and a sequence for an array of times
  (`moon.phase`, `moon.illumination`, `moon.phase_name`) are now proper
  `@overload`s (scalar → `float` / `moonphase`, array → `list[...]`), and the
  sidereal-time and quaternion helpers' array overloads were corrected to
  `list[float]` / `list[quaternion]` (they return Python lists, not numpy
  arrays).
- **Added the missing `frametransform.eqeq` (equation of the equinoxes) stub.**

### Documentation

- **Frame-transform docs clarified** — the `frametransform` API page now has a
  "Which function do I call?" table distinguishing `rotation` (Earth frames),
  `to_gcrf`/`from_gcrf` (orbit frames), and the new `rotation_with_state` (all
  frames), with an example of the mixed Earth→orbit case.
- **Coordinate Frames tutorial expanded** — a new "Orbit-local frames in one
  call" section with a runnable `rotation_with_state` example (TEME→RTN), and
  the "Approximate vs Full" section now explains the computation the `_approx`
  variants skip (the ~2,900-term IERS CIP series plus EOP interpolation) with a
  live timing cell measuring the speedup.
- **mkdocs navigation** — added a sticky top-level menu bar (`navigation.tabs`),
  clickable section overview pages (`navigation.indexes`, removing duplicated
  section titles), collapsed subsections by default, and normalized the API
  Reference entry capitalization to Title Case.
- **Pinned `mkdocs<2.0`** in `docs/requirements.txt` — MkDocs 2.0 is
  incompatible with Material for MkDocs, so an unpinned fresh install could
  break the docs build.
- **Equations render with KaTeX instead of MathJax** (matching numeris) —
  faster page loads, same delimiters (`\(..\)`/`\[..\]` from arithmatex and
  `$..$`/`$$..$$` in tutorial notebooks).
- **New "Loading local OMM files" tutorial section** — the supported offline
  route is stdlib `json` (or `xmltodict` for CCSDS XML) → dictionary →
  `sk.sgp4`; satkit validates `MEAN_ELEMENT_THEORY`/`TIME_SYSTEM` and consumes
  the SGP4-relevant fields.

### Dependencies

- **`numeris` updated to 0.5.14** (root and Python crate; the Python crate now
  declares the `ode` feature explicitly).


